"""Golden 数据集存储与 LLM 回放 Mock。

Golden 目录布局（record 模式生成）：
    meta.json        会话元信息（数据集路径、脚本、seed、创建时间）
    llm_calls.jsonl  全部 LLM 流量：{seq, stream, system_prompt, user_message, response}
    turns.jsonl      逐轮记录：{seq, input, response, snapshot}
    files.json       产出文件清单：{相对文件名: sha256}

回放匹配规则：以 (system_prompt, user_message) 精确匹配，同一 key 多次出现
按 FIFO 顺序消费（对应记忆摘要等相同 prompt 重复调用的场景）。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

from agents.llm_client import get_stream_callback

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
        meta.setdefault("format_version", 1)
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
    """

    def __init__(self, calls: list[dict[str, Any]], strict: bool = True) -> None:
        self.strict = strict
        # (system, user) → 响应队列（FIFO）
        self._pool: dict[tuple[str, str], list[str | None]] = {}
        for c in calls:
            key = (c["system_prompt"], c["user_message"])
            self._pool.setdefault(key, []).append(c.get("response"))
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
            response = queue.pop(0)

        self.served.append({
            "seq": len(self.served),
            "stream": bool(stream),
            "system_prompt": system_prompt,
            "user_message": user_message,
            "response": response,
        })

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
