"""Harness 等价性验证套件。

不依赖真实 API：所有用例使用确定性脚本化 mock（替换 _call_llm_impl）。

覆盖：
1. A/B 等价 —— 同一 mock 下，裸 SessionManager 与 Harness(passthrough)
   的输出 / 状态快照 / LLM 请求序列三者一致（最直接的等价证据）。
2. record → replay 闭环 —— mock 源录制 golden 后回放回归必须 PASS
   （含反事实表单 + 轨迹导出，验证确定性注入）。
3. ReplayLLM 流式契约 / 严格未命中。
4. 快照对比工具、diff 报告生成器、Tracer 结构。
5. CLI verify 子进程冒烟。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._common import BALL_CSV, PERSON_CSV

from harness import Harness, HarnessConfig
from harness.config import MODE_PASSTHROUGH, MODE_RECORD
from harness.mock import MockError, ReplayLLM
from harness.snapshot import canonical_json, compare_snapshots, snapshot_session
from harness.tracing import Tracer
from harness.verify.diff_report import render_markdown
from harness.verify.regression import (
    VerificationResult,
    execute_scripted_input,
    verify_golden,
)
from harness.verify.snapshot_diff import diff_snapshot_files

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────
# 确定性脚本化 mock（替换 _call_llm_impl，覆盖全部调用点）
# ─────────────────────────────────────────────────────────────────

def _route_json(user_text: str) -> str:
    """模拟路由器的意图 JSON 输出（确定性，不覆盖 counterfactual 路径）。"""
    def jerseys_of(text: str) -> list[str]:
        import re
        return list(dict.fromkeys(f"{m}号" for m in re.findall(r"(\d+)\s*号", text)))

    jerseys = jerseys_of(user_text)
    if "聚焦" in user_text or "看" in user_text:
        intent = "focus_player"
    elif "对比" in user_text or "谁更" in user_text:
        intent = "comparison"
    elif "风格" in user_text or "特点" in user_text:
        intent = "style_analysis"
    elif "第" in user_text and ("帧" in user_text or "秒" in user_text):
        intent = "verify_data"
    elif "整体" in user_text or "全场" in user_text:
        intent = "general_analysis"
    else:
        intent = "general_qa"
    return json.dumps({
        "intent": intent,
        "target_players": jerseys,
        "confidence": 0.9,
        "scenario": None,
    }, ensure_ascii=False)


def make_scripted_impl():
    """构造 (calls, impl)。impl 确定性：同输入必然同输出。"""
    calls: list[tuple[str, str, bool]] = []

    def impl(system_prompt, user_message, config=None, max_tokens=None,
             retries=1, stream=False):
        calls.append((system_prompt, user_message, bool(stream)))
        if "意图路由器" in system_prompt:
            return _route_json(user_message)
        digest = hashlib.sha256(
            (system_prompt + "|" + user_message).encode("utf-8")
        ).hexdigest()[:10]
        # 内容仅由输入决定（不含调用序号），保证跨运行逐字节确定
        return f"模拟回复[{digest}]"

    return calls, impl


def _patch_impl(monkeypatch: pytest.MonkeyPatch, impl) -> None:
    """同时替换裸调用链与 Harness 拦截器链的底层实现。"""
    monkeypatch.setattr("agents.llm_client._call_llm_impl", impl)
    monkeypatch.setattr("harness.interceptors._call_llm_impl", impl)


# A/B 脚本：覆盖 help/focus/style/comparison/general_analysis/verify_data/
# general_qa/clear（不含 counterfactual —— MCTS 未注入种子时本身非确定，
# 其确定性验证由 record/replay 闭环用例负责）。
AB_SCRIPT = [
    "help",
    "聚焦7号",
    "7号的跑动风格是什么",
    "对比6号和7号",
    "整体分析一下这场比赛",
    "第90帧发生了什么",
    "7号是哪个队伍的",
    "clear",
]

# 闭环脚本：额外覆盖反事实表单（生成轨迹数据）与记忆摘要触发（≥3轮）。
ROUNDTRIP_SCRIPT = [
    "help",
    "聚焦7号",
    "7号的跑动风格是什么",
    "对比6号和7号",
    "整体分析一下这场比赛",
    "第90帧发生了什么",
    "7号是哪个队伍的",
    "[CF_FORM] 7号|3.0|shoot||3",
    "clear",
]


# ─────────────────────────────────────────────────────────────────
# 1. A/B 等价
# ─────────────────────────────────────────────────────────────────

class TestABEquivalence:
    """裸 SessionManager vs Harness(passthrough)：三者一致。"""

    def test_ab_outputs_snapshots_llm_sequence(self, monkeypatch, tmp_path):
        calls, impl = make_scripted_impl()
        _patch_impl(monkeypatch, impl)

        # ── A：裸 SessionManager（dispatcher=None，原始调用链） ──
        from agents.llm_client import get_llm_dispatcher
        assert get_llm_dispatcher() is None  # 前置：无残留注入

        from agents.session.manager import SessionManager
        bare = SessionManager()
        assert bare.load_data(PERSON_CSV, BALL_CSV)
        bare_responses = [bare.handle_input(t) for t in AB_SCRIPT]
        bare_snapshot = snapshot_session(bare)
        bare_call_count = len(calls)

        # ── B：Harness(passthrough) ──
        cfg = HarnessConfig(mode=MODE_PASSTHROUGH, trace_dir=tmp_path / "traces")
        with Harness(cfg) as h:
            assert h.load_data(PERSON_CSV, BALL_CSV)
            harness_responses = [h.handle_input(t) for t in AB_SCRIPT]
            harness_snapshot = h.snapshot()
        harness_call_count = len(calls) - bare_call_count

        # ── 三者一致 ──
        assert bare_responses == harness_responses, "逐轮输出不一致"
        assert compare_snapshots(bare_snapshot, harness_snapshot) == [], "状态快照不一致"
        assert bare_call_count == harness_call_count, "LLM 调用次数不一致"
        assert calls[:bare_call_count] == calls[bare_call_count:], "LLM 请求序列不一致"

    def test_passthrough_trace_written(self, monkeypatch, tmp_path):
        """trace 观测不影响输出，且 trace 文件结构完整。"""
        _, impl = make_scripted_impl()
        _patch_impl(monkeypatch, impl)

        cfg = HarnessConfig(mode=MODE_PASSTHROUGH, trace_dir=tmp_path / "traces")
        with Harness(cfg) as h:
            h.load_data(PERSON_CSV, BALL_CSV)
            r1 = h.handle_input("聚焦7号")
            trace_path = h.tracer.trace_path
        assert r1
        assert trace_path is not None and trace_path.is_file()

        events = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        types = [e["type"] for e in events]
        assert types[0] == "run_start" and types[-1] == "run_end"
        kinds = {e.get("kind") for e in events}
        assert "load" in kinds and "turn" in kinds and "llm" in kinds
        # span 成对出现
        assert types.count("span_start") == types.count("span_end")


# ─────────────────────────────────────────────────────────────────
# 2. record → replay 闭环（含确定性注入）
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def recorded_golden(tmp_path_factory):
    """用 mock 源录制一份 golden（模块级复用）。"""
    import agents.llm_client as llm_client_module
    import harness.interceptors as interceptors_module

    tmp = tmp_path_factory.mktemp("golden_roundtrip")
    _, impl = make_scripted_impl()
    orig_impl_llm = llm_client_module._call_llm_impl
    orig_impl_harness = interceptors_module._call_llm_impl
    llm_client_module._call_llm_impl = impl
    interceptors_module._call_llm_impl = impl
    try:
        golden_dir = tmp / "golden"
        cfg = HarnessConfig(
            mode=MODE_RECORD, golden_dir=golden_dir, seed=42,
            trace_dir=tmp / "traces",
        )
        with Harness(cfg) as h:
            h.init_recording({
                "name": "roundtrip",
                "seed": 42,
                "dataset": {
                    "person": str(PERSON_CSV.resolve()),
                    "ball": str(BALL_CSV.resolve()),
                },
                "entry": "script",
            })
            assert h.load_data(PERSON_CSV, BALL_CSV)
            for line in ROUNDTRIP_SCRIPT:
                execute_scripted_input(h, line)
            manifest = h.finalize_recording(prefix_hint="12s")
        assert manifest, "文件清单不应为空（反事实轨迹应被采集）"
    finally:
        llm_client_module._call_llm_impl = orig_impl_llm
        interceptors_module._call_llm_impl = orig_impl_harness
    return golden_dir


class TestRecordReplay:
    def test_golden_files_complete(self, recorded_golden):
        meta = json.loads((recorded_golden / "meta.json").read_text(encoding="utf-8"))
        assert meta["seed"] == 42
        assert (recorded_golden / "llm_calls.jsonl").is_file()
        assert (recorded_golden / "turns.jsonl").is_file()
        turns = (recorded_golden / "turns.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(turns) == len(ROUNDTRIP_SCRIPT)

    def test_replay_verify_passes(self, recorded_golden, tmp_path):
        """回放回归：LLM 序列 / 逐轮输出 / 快照 / 文件哈希全部一致。"""
        result = verify_golden(recorded_golden, trace_dir=tmp_path / "replay_traces")
        assert not result.errors, f"回放异常: {result.errors}"
        assert result.llm_sequence_equal, result.llm_first_mismatch
        assert not result.turn_diffs, "逐轮输出或快照不一致"
        assert not result.file_diffs, f"产出文件不一致: {result.file_diffs}"
        assert result.passed

    def test_cli_verify_subprocess(self, recorded_golden):
        """CLI: python -m harness verify <golden> 退出码 0 且生成报告。"""
        report = recorded_golden / "cli_report.md"
        completed = subprocess.run(
            [sys.executable, "-m", "harness", "verify", str(recorded_golden),
             "--report", str(report)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800,
        )
        assert completed.returncode == 0, (
            f"CLI verify 失败:\n{completed.stdout}\n{completed.stderr}"
        )
        assert "PASS" in completed.stdout
        content = report.read_text(encoding="utf-8")
        assert "等价性验证报告" in content and "已知等价性边界" in content


# ─────────────────────────────────────────────────────────────────
# 3. ReplayLLM
# ─────────────────────────────────────────────────────────────────

class TestReplayLLM:
    def _calls(self):
        return [
            {"seq": 0, "stream": False, "system_prompt": "s1",
             "user_message": "u1", "response": "响应一"},
            {"seq": 1, "stream": True, "system_prompt": "s2",
             "user_message": "u2", "response": "你好，世界！" * 8},
            {"seq": 2, "stream": False, "system_prompt": "s1",
             "user_message": "u1", "response": "响应一（第二次）"},
        ]

    def test_fifo_per_key(self):
        rl = ReplayLLM(self._calls())
        assert rl("s1", "u1") == "响应一"
        assert rl("s1", "u1") == "响应一（第二次）"  # 同 key FIFO
        assert rl.exhausted() is False
        assert rl("s2", "u2", stream=True) == "你好，世界！" * 8
        assert rl.exhausted() is True

    def test_stream_contract(self):
        """流式回放：回调 chunk 拼接必须等于返回值（与真实流式契约一致）。"""
        from agents.llm_client import set_stream_callback, reset_stream_callback

        rl = ReplayLLM(self._calls())
        chunks: list[str] = []
        token = set_stream_callback(chunks.append)
        try:
            out = rl("s2", "u2", stream=True)
        finally:
            reset_stream_callback(token)
        assert out == "你好，世界！" * 8
        assert "".join(chunks) == out
        assert len(chunks) > 1  # 确实经过了分块推送

    def test_strict_miss_raises(self):
        rl = ReplayLLM(self._calls(), strict=True)
        with pytest.raises(MockError):
            rl("不存在的", "请求")

    def test_lenient_miss_returns_none(self):
        rl = ReplayLLM(self._calls(), strict=False)
        assert rl("不存在的", "请求") is None

    def test_recorded_none_response_replayed(self):
        """录制时 LLM 失败（None）也必须原样回放。"""
        rl = ReplayLLM([{"seq": 0, "stream": False, "system_prompt": "s",
                         "user_message": "u", "response": None}])
        assert rl("s", "u") is None


# ─────────────────────────────────────────────────────────────────
# 4. 快照对比 / diff 报告 / Tracer
# ─────────────────────────────────────────────────────────────────

class TestSnapshotDiff:
    def test_identical(self, tmp_path):
        snap = {"a": 1, "b": [1.5, "x"], "c": {"d": None}}
        assert compare_snapshots(snap, json.loads(canonical_json(snap))) == []

    def test_field_level_diffs(self):
        a = {"x": 1, "y": [1, 2], "z": {"k": "v"}}
        b = {"x": 2, "y": [1, 2, 3], "z": {"k": "v"}, "w": True}
        diffs = compare_snapshots(a, b)
        text = "\n".join(diffs)
        assert "$.x" in text and "$.y" in text and "$.w" in text

    def test_float_roundtrip_tolerance(self):
        assert compare_snapshots({"v": 1}, {"v": 1.0}) == []
        assert compare_snapshots({"v": 0.123456789}, {"v": 0.1235}) == []

    def test_diff_files_cli(self, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps({"k": 1}), encoding="utf-8")
        b.write_text(json.dumps({"k": 1}), encoding="utf-8")
        assert diff_snapshot_files(a, b) == []
        b.write_text(json.dumps({"k": 2}), encoding="utf-8")
        assert diff_snapshot_files(a, b) != []


class TestDiffReport:
    def test_render_sections(self):
        result = VerificationResult(golden_dir="/tmp/golden")
        result.passed = False
        result.llm_sequence_equal = False
        result.llm_first_mismatch = "第 3 次请求不同"
        result.turn_diffs.append(type("TD", (), {
            "seq": 2, "input": "聚焦7号",
            "response_equal": False,
            "response_golden": "甲\n相同行\n", "response_replay": "乙\n相同行\n",
            "snapshot_diffs": ["$.memory.turn_count: 3 != 4"],
        })())
        result.file_diffs.append("12s_cf_f90_person.csv: 哈希不同")

        md = render_markdown(result)
        assert "等价性验证报告" in md
        assert "FAIL" in md
        assert "LLM 请求序列差异" in md
        assert "逐轮差异" in md
        assert "```diff" in md
        assert "产出文件差异" in md
        assert "已知等价性边界" in md


class TestTracer:
    def test_disabled_tracer_no_file(self, tmp_path):
        tracer = Tracer(tmp_path / "nope", enabled=False)
        with tracer.span("turn", "t"):
            tracer.event("stage", "s")
        tracer.close()
        assert not (tmp_path / "nope").exists()

    def test_span_nesting_and_error_status(self, tmp_path):
        tracer = Tracer(tmp_path, enabled=True, run_id="test")
        with tracer.span("turn", "t1"):
            with tracer.span("llm", "call0"):
                pass
            with pytest.raises(ValueError):
                with tracer.span("llm", "call1"):
                    raise ValueError("boom")
        tracer.close()

        events = [json.loads(l) for l in tracer.trace_path.read_text(encoding="utf-8").splitlines()]
        starts = {e["span_id"]: e for e in events if e["type"] == "span_start"}
        ends = {e["span_id"]: e for e in events if e["type"] == "span_end"}
        turn_span = next(s for s in starts.values() if s["kind"] == "turn")
        llm_spans = [s for s in starts.values() if s["kind"] == "llm"]
        assert all(s["parent_id"] == turn_span["span_id"] for s in llm_spans)
        status_by_name = {starts[sid]["name"]: e["status"] for sid, e in ends.items()}
        assert status_by_name["call0"] == "ok"
        assert status_by_name["call1"] == "error"


# ─────────────────────────────────────────────────────────────────
# 5. llm_client 钩子健全性
# ─────────────────────────────────────────────────────────────────

class TestDispatcherHook:
    def test_default_none_and_set_clear(self):
        from agents.llm_client import (
            get_llm_dispatcher,
            set_llm_dispatcher,
        )
        assert get_llm_dispatcher() is None
        sentinel = lambda *a, **k: "x"  # noqa: E731
        set_llm_dispatcher(sentinel)
        assert get_llm_dispatcher() is sentinel
        set_llm_dispatcher(None)
        assert get_llm_dispatcher() is None

    def test_dispatcher_receives_original_args(self):
        """分派器收到的参数与 call_llm 形参一一对应。"""
        from agents import llm_client

        received = {}

        def spy(system_prompt, user_message, config=None, max_tokens=None,
                retries=1, stream=False):
            received.update({
                "system_prompt": system_prompt, "user_message": user_message,
                "max_tokens": max_tokens, "retries": retries, "stream": stream,
            })
            return "ok"

        llm_client.set_llm_dispatcher(spy)
        try:
            out = llm_client.call_llm("sys", "usr", max_tokens=100, stream=True)
        finally:
            llm_client.set_llm_dispatcher(None)
        assert out == "ok"
        assert received == {
            "system_prompt": "sys", "user_message": "usr",
            "max_tokens": 100, "retries": 1, "stream": True,
        }
