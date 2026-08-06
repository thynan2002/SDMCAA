"""Golden 数据集存储与 LLM 回放 Mock。

Golden 目录布局（record 模式生成）：
    meta.json        会话元信息（数据集路径、脚本、seed、创建时间）
    llm_calls.jsonl  全部 LLM 流量：{seq, stream, system_prompt, user_message, response}
                     工具调用轮次可含可选字段：tool_round / tools / tool_choice /
                     assistant_message（含 tool_calls 的完整 assistant 消息）
    turns.jsonl      逐轮记录：{seq, input, response, snapshot}
    files.json       产出文件清单：{相对文件名: sha256}

回放匹配规则：以 (system_prompt, user_message) 精确匹配，同一 key 多次出现
按 FIFO 顺序消费（对应记忆摘要等相同 prompt 重复调用、以及工具循环中
同一请求的多轮交互场景）。

向后兼容：旧 golden 条目无可选字段时，回放行为与纯文本时代完全一致
（返回录制的 response 文本；工具循环收到纯文本即走文本回退路径）。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

from agents.llm_client import get_stream_callback, get_tool_exchange

# 流式回放时的 chunk 大小（字符数）。契约：所有 chunk 拼接 == 完整响应，
# 与真实流式调用的可观测契约一致（cb(delta...); return "".join(deltas)）。
REPLAY_CHUNK_SIZE = 24


class MockError(RuntimeError):
    """Mock 层异常（replay 未命中 / golden 数据损坏等）。"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────
# Golden 存储
# ─────────────────────────────────────────────────────────────────

class GoldenStore:
    """golden 数据集的读写（record 写 / replay 读）。"""

    def __init__(self, golden_dir: Path | str) -> None:
        self.dir = Path(golden_dir)

    # ── 写（record） ──

    def init_for_write(self, meta: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = dict(meta)
        meta.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        # format_version=2：llm_calls 条目可含工具调用可选字段
        # （tool_round/tools/tool_choice/assistant_message）；
        # 读取端不按版本分支，旧 version=1 数据完全兼容。
        meta.setdefault("format_version", 2)
        (self.dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        # 清空旧的流量记录，防止混入历史数据
        for name in ("llm_calls.jsonl", "turns.jsonl"):
            p = self.dir / name
            if p.exists():
                p.unlink()

    def append_llm_call(self, entry: dict[str, Any]) -> None:
        with (self.dir / "llm_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def append_turn(self, entry: dict[str, Any]) -> None:
        with (self.dir / "turns.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def write_files_manifest(self, manifest: dict[str, str]) -> None:
        (self.dir / "files.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    # ── 读（replay） ──

    def _require(self, name: str) -> Path:
        p = self.dir / name
        if not p.is_file():
            raise MockError(f"golden 数据缺失: {p}")
        return p

    def load_meta(self) -> dict[str, Any]:
        return json.loads(self._require("meta.json").read_text(encoding="utf-8"))

    def load_llm_calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for line in self._require("llm_calls.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                calls.append(json.loads(line))
        return calls

    def load_turns(self) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        turns_path = self.dir / "turns.jsonl"
        if not turns_path.is_file():
            return turns
        for line in turns_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                turns.append(json.loads(line))
        return turns

    def load_files_manifest(self) -> dict[str, str]:
        p = self.dir / "files.json"
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────
# LLM 回放 Mock
# ─────────────────────────────────────────────────────────────────

class ReplayLLM:
    """确定性 LLM 回放器（作为 llm_client 分派器注入）。

    等价性论证：仅在 replay 模式注入；返回值为 golden 中同一
    (system_prompt, user_message) 的录制响应，原样转发流式回调契约，
    不向业务代码暴露任何额外行为。

    工具调用回放：golden 条目若含 assistant_message（含 tool_calls），
    且当前处于工具交换通道中，则将其回填到交换对象，使工具循环按
    录制的工具调用轨迹精确重现；旧条目无该字段时行为与纯文本一致。
    """

    def __init__(self, calls: list[dict[str, Any]], strict: bool = True) -> None:
        self.strict = strict
        # (system, user) → 完整条目队列（FIFO），保留可选的 assistant_message
        self._pool: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for c in calls:
            key = (c["system_prompt"], c["user_message"])
            self._pool.setdefault(key, []).append(c)
        self.served: list[dict[str, Any]] = []  # 已服务的请求序列（回归比对用）

    def __call__(
        self,
        system_prompt: str,
        user_message: str,
        config: Any = None,
        max_tokens: int | None = None,
        retries: int = 1,
        stream: bool = False,
    ) -> str | None:
        key = (system_prompt, user_message)
        queue = self._pool.get(key)
        entry: dict[str, Any] | None = None
        if not queue:
            if self.strict:
                raise MockError(
                    "replay 未命中 golden 记录（请求与录制时不一致，"
                    "提示词/数据可能已发生变化）:\n"
                    f"  system: {system_prompt[:120]!r}...\n"
                    f"  user  : {user_message[:120]!r}..."
                )
            response: str | None = None
        else:
            entry = queue.pop(0)
            response = entry.get("response")

        self.served.append({
            "seq": len(self.served),
            "stream": bool(stream),
            "system_prompt": system_prompt,
            "user_message": user_message,
            "response": response,
        })

        # 工具调用回放：把录制的 assistant_message 回填到交换通道，
        # 使工具循环按录制轨迹重现（旧条目无该字段 → 走文本回退）。
        exchange = get_tool_exchange()
        if exchange is not None and entry is not None:
            recorded_msg = entry.get("assistant_message")
            if recorded_msg is not None:
                exchange.assistant_message = recorded_msg

        # 流式契约回放：调用方标记 stream 且当前作用域有回调时，
        # 将完整响应切块推入回调，拼接结果与返回值一致。
        if stream and response:
            cb = get_stream_callback()
            if cb is not None:
                for chunk in _chunk_text(response):
                    cb(chunk)
        return response

    def exhausted(self) -> bool:
        """golden 中的所有记录是否都被消费（回归完整性检查）。"""
        return all(not q for q in self._pool.values())


def _chunk_text(text: str, size: int = REPLAY_CHUNK_SIZE) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i:i + size]
