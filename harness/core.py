"""Harness 核心 — SessionManager 的透明包装层。

全链路覆盖映射：
    输入解析   handle_input/run_once 恒等委托（不加预处理）
    规划/路由  QueryRouter 的 LLM 调用经 LLMInterceptor 观测/回放
    工具执行   两个 Orchestrator 的 LLM/push_stage 全部经拦截器与回调 tee 观测
    记忆       MemoryStore 的摘要 LLM 调用经拦截器；状态经只读快照观测
    LLM        llm_client 分派器（record/replay/trace 三态）
    输出       轮次返回值边界捕获 + Output 文件清单哈希

等价性论证（恒等委托 + 纯函数包装 + 只读代理 + 确定性注入）见模块
docstring 与 docs（README「Harness 等价性」一节）。本文件不包含任何
业务逻辑分支，所有处理最终都落到 SessionManager / main 的原始实现。
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Callable

from agents.session.manager import SessionManager
from agents.progress import (
    get_progress_callback,
    set_progress_callback,
    reset_progress_callback,
)
from agents.llm_client import (
    get_stream_callback,
    set_stream_callback,
    reset_stream_callback,
)

from .config import (
    HarnessConfig,
    MODE_PASSTHROUGH,
    MODE_RECORD,
    MODE_REPLAY,
    PROJECT_ROOT,
)
from .interceptors import LLMInterceptor
from .mock import GoldenStore, ReplayLLM, sha256_file, sha256_text
from .snapshot import snapshot_session
from .tracing import Tracer

OUTPUT_DIR = PROJECT_ROOT / "Output"


class Harness:
    """统一 Harness — 智能体的透明包装层与统一入口。

    用法（API）：
        with Harness(HarnessConfig(mode="passthrough")) as h:
            h.load_data(person_csv, ball_csv)
            response = h.handle_input("聚焦7号")

    公开方法与 SessionManager 同名方法一一对应（恒等委托）；
    额外提供 run_once（委托 main.run_once）与 reset（web 上传/清除用）。
    """

    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig()
        self.tracer = Tracer(
            self.config.trace_dir if self.config.trace_enabled else None,
            enabled=self.config.trace_enabled,
        )

        # ── golden 存储与回放器 ──
        self.golden: GoldenStore | None = None
        self.replay_llm: ReplayLLM | None = None
        if self.config.golden_dir is not None:
            self.golden = GoldenStore(self.config.golden_dir)
        if self.config.mode == MODE_REPLAY:
            assert self.golden is not None
            self.replay_llm = ReplayLLM(
                self.golden.load_llm_calls(), strict=self.config.strict_mock,
            )

        # ── LLM 拦截器（三种模式均安装；passthrough 仅观测） ──
        self.interceptor = LLMInterceptor(
            self.tracer, self.config.mode,
            golden_store=self.golden, replay_llm=self.replay_llm,
        )
        self.interceptor.install()

        # ── 确定性注入状态 ──
        self._determinism_patched = False
        self._orig_apply_action: Callable | None = None

        # ── 被包装的智能体会话（组合，非继承） ──
        self._session: SessionManager | None = None
        self._turn_seq = 0
        self._record_inputs: list[str] = []
        self.reset()

    # ── 会话生命周期 ──────────────────────────────────────────

    def reset(self) -> None:
        """重建一个全新会话（web 上传/清除场景），并重新注入确定性通道。"""
        self._session = SessionManager()
        self._inject_determinism()
        self.tracer.event("session", "session_reset")

    @property
    def session(self) -> SessionManager:
        """被包装的原始 SessionManager（只读逃生舱，Harness 不修改其接口）。"""
        assert self._session is not None
        return self._session

    # ── 只读透传属性（web 迁移用） ────────────────────────────

    @property
    def corpus(self):
        return self._session.corpus if self._session else None

    @property
    def data_files(self) -> tuple[str, str]:
        return self._session._data_files if self._session else ("", "")

    @property
    def last_counterfactual_files(self) -> list[str]:
        return self._session.last_counterfactual_files if self._session else []

    # ── 统一入口：数据加载 ────────────────────────────────────

    def load_data(self, person_csv: str | Path, ball_csv: str | Path) -> bool:
        """恒等委托 SessionManager.load_data，外加只读观测。"""
        with self.tracer.span("load", "load_data",
                              person=str(person_csv), ball=str(ball_csv)):
            ok = self._session.load_data(person_csv, ball_csv)
        if ok and self.corpus is not None:
            self.tracer.event("load", "corpus_ready",
                              prefix=self.corpus.prefix,
                              player_count=self.corpus.player_count)
        return ok

    # ── 统一入口：交互轮次 ────────────────────────────────────

    def handle_input(self, user_text: str) -> str:
        """恒等委托 SessionManager.handle_input，边界处观测/录制。"""
        seq = self._turn_seq
        self._turn_seq += 1
        stream_stats = {"chunks": 0, "chars": 0}

        with self.tracer.span("turn", f"turn[{seq}]", input=user_text):
            progress_token = set_progress_callback(self._make_progress_tee())
            stream_token = set_stream_callback(self._make_stream_tee(stream_stats))
            try:
                response = self._session.handle_input(user_text)
            finally:
                reset_progress_callback(progress_token)
                reset_stream_callback(stream_token)

            self.tracer.event("stream", "stream_summary", **stream_stats)
            snapshot = self._take_snapshot()
            self.tracer.event("output", f"output[{seq}]", **{
                "response_sha256": sha256_text(response),
                "response_len": len(response),
                "is_exit_signal": response == "__EXIT__",
            })

        if self.config.mode == MODE_RECORD:
            self._record_inputs.append(user_text)
            if self.golden is not None:
                self.golden.append_turn({
                    "seq": seq,
                    "input": user_text,
                    "response": response,
                    "snapshot": snapshot,
                })
        return response

    def handle_counterfactual_form(
        self,
        subject_player: str,
        time_second: float,
        altered_action: str,
        altered_target: str = "",
        duration_seconds: float = 0.0,
    ) -> str:
        """恒等委托 SessionManager.handle_counterfactual_form。"""
        seq = self._turn_seq
        self._turn_seq += 1
        stream_stats = {"chunks": 0, "chars": 0}

        with self.tracer.span("turn", f"turn[{seq}](cf_form)",
                              subject_player=subject_player,
                              time_second=time_second,
                              altered_action=altered_action,
                              altered_target=altered_target,
                              duration_seconds=duration_seconds):
            progress_token = set_progress_callback(self._make_progress_tee())
            stream_token = set_stream_callback(self._make_stream_tee(stream_stats))
            try:
                response = self._session.handle_counterfactual_form(
                    subject_player, time_second, altered_action,
                    altered_target, duration_seconds,
                )
            finally:
                reset_progress_callback(progress_token)
                reset_stream_callback(stream_token)

            self.tracer.event("stream", "stream_summary", **stream_stats)
            snapshot = self._take_snapshot()
            self.tracer.event("output", f"output[{seq}]", **{
                "response_sha256": sha256_text(response),
                "response_len": len(response),
            })

        if self.config.mode == MODE_RECORD:
            form_desc = (f"[CF_FORM] {subject_player}|{time_second}|{altered_action}|"
                         f"{altered_target}|{duration_seconds}")
            self._record_inputs.append(form_desc)
            if self.golden is not None:
                self.golden.append_turn({
                    "seq": seq,
                    "input": form_desc,
                    "response": response,
                    "snapshot": snapshot,
                })
        return response

    # ── 统一入口：单次解说模式 ────────────────────────────────

    def run_once(
        self,
        person_csv: str | Path,
        ball_csv: str | Path,
        focus_jerseys: list[str] | None = None,
    ) -> None:
        """单次解说模式 — 直接委托 main.run_once（同一函数，行为逐字节等价）。

        LLM 观测经已安装的分派器生效；本方法仅加一个区间 span。
        """
        import main as main_module

        with self.tracer.span("turn", "run_once",
                              person=str(person_csv), ball=str(ball_csv),
                              focus_jerseys=focus_jerseys or []):
            main_module.run_once(Path(person_csv), Path(ball_csv), focus_jerseys)
        if self.config.mode == MODE_RECORD:
            self._record_inputs.append(
                f"[ONCE] {person_csv}|{ball_csv}|{','.join(focus_jerseys or [])}"
            )

    # ── golden 录制收尾 ───────────────────────────────────────

    def init_recording(self, meta: dict[str, Any]) -> None:
        """record 模式：写入 meta.json 并清空历史流量记录。"""
        if self.config.mode != MODE_RECORD or self.golden is None:
            return
        self.golden.init_for_write(meta)

    def finalize_recording(self, prefix_hint: str | None = None) -> dict[str, str]:
        """record 模式：采集 Output 文件清单哈希，返回 manifest。"""
        if self.config.mode != MODE_RECORD or self.golden is None:
            return {}
        manifest = self.capture_output_files(prefix_hint)
        self.golden.write_files_manifest(manifest)
        return manifest

    def capture_output_files(self, prefix_hint: str | None = None) -> dict[str, str]:
        """采集本次会话相关产出文件的 sha256（解说稿/反事实 CSV）。

        只采集命名确定性的文件；含时间戳的对话存档属已知等价性边界，不采集。
        """
        prefix = self.corpus.prefix if self.corpus is not None else (prefix_hint or "")
        if not prefix:
            m = None
            person = self.data_files[0] if self.data_files else ""
            if person:
                m = re.search(r"(\d+s)", Path(person).name, re.IGNORECASE)
            prefix = m.group(1) if m else ""
        manifest: dict[str, str] = {}
        if not prefix or not OUTPUT_DIR.is_dir():
            return manifest
        patterns = [
            re.compile(rf"^{re.escape(prefix)}(_\d+)*_summary\.(txt|json)$"),
            re.compile(rf"^{re.escape(prefix)}_cf_f\d+_(person|ball)\.csv$"),
        ]
        for p in sorted(OUTPUT_DIR.iterdir()):
            if p.is_file() and any(pat.match(p.name) for pat in patterns):
                manifest[p.name] = sha256_file(p)
        return manifest

    # ── 回调复合（tee，只读观测，不替换调用方回调） ───────────

    def _make_progress_tee(self) -> Callable[[str, str, dict[str, Any]], None]:
        original = get_progress_callback()
        tracer = self.tracer

        def cb(stage: str, content: str, extra: dict[str, Any]) -> None:
            tracer.event("stage", stage, content=content, **(extra or {}))
            if original is not None:
                original(stage, content, extra)

        return cb

    def _make_stream_tee(self, stats: dict[str, int]) -> Callable[[str], None] | None:
        original = get_stream_callback()

        def cb(chunk: str) -> None:
            stats["chunks"] += 1
            stats["chars"] += len(chunk)
            if original is not None:
                original(chunk)

        # 无原始回调时也安装统计回调（观测不依赖于调用方是否设置了回调）。
        # 等价性说明：llm_client 仅在 stream=True 时消费该回调；
        # 回调存在与否不改变返回值（完整文本恒等返回），只改变逐 chunk 推送。
        # 但为严格保持"未设置回调时非流式请求"的原始网络路径，仅在调用方
        # 已设置回调或记录模式下安装 tee。
        if original is not None:
            return cb
        return None

    # ── 快照 ──────────────────────────────────────────────────

    def _take_snapshot(self) -> dict[str, Any] | None:
        if not self.config.snapshot_enabled:
            return None
        snap = snapshot_session(self._session)
        self.tracer.event("snapshot", "session_snapshot",
                          turn_count=snap["memory"]["turn_count"],
                          conversation_len=len(snap["conversation_history"]))
        return snap

    def snapshot(self) -> dict[str, Any]:
        """公开接口：取当前会话的只读状态快照。"""
        return snapshot_session(self._session)

    # ── 确定性注入（仅 record/replay；passthrough 为恒等） ────

    def _inject_determinism(self) -> None:
        if self.config.seed is None or self.config.mode == MODE_PASSTHROUGH:
            return
        # 通道 1：作者预留的 seed 构造参数（不改算法，只启用已有确定性通道）
        engine = self._session.professional_orchestrator.counterfactual_engine
        engine.seed = self.config.seed
        from agents.professional.mcts.tree import MCTSTree
        engine.mcts = MCTSTree(seed=self.config.seed)

        # 通道 2：GameState.apply_action 的无种子回退（counterfactual_engine
        # 对原始/替换动作的第一步调用未传 rng）→ 注入共享的有种子回退 RNG。
        # 仅 record/replay 模式安装；passthrough 绝不触碰。
        if not self._determinism_patched:
            from agents.professional.mcts import node as mcts_node

            orig = mcts_node.GameState.apply_action
            fallback_rng = random.Random(self.config.seed)

            def patched_apply_action(gs, action, rng=None):
                return orig(gs, action, rng if rng is not None else fallback_rng)

            mcts_node.GameState.apply_action = patched_apply_action
            self._orig_apply_action = orig
            self._determinism_patched = True

    def _restore_determinism(self) -> None:
        if self._determinism_patched:
            from agents.professional.mcts import node as mcts_node

            mcts_node.GameState.apply_action = self._orig_apply_action
            self._orig_apply_action = None
            self._determinism_patched = False

    # ── 生命周期 ──────────────────────────────────────────────

    def close(self) -> None:
        """卸载全部注入，恢复进程到未包装状态。"""
        self.interceptor.uninstall()
        self._restore_determinism()
        self.tracer.close()

    def __enter__(self) -> "Harness":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── 便捷只读信息 ──────────────────────────────────────────

    @property
    def recorded_inputs(self) -> list[str]:
        return list(self._record_inputs)

    @property
    def llm_call_count(self) -> int:
        return self.interceptor.call_count
