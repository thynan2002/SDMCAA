"""Trace 记录 — Span 树 + JSONL 落盘。

Span 模型（全链路覆盖）：
    run                      一次 Harness 会话生命周期
    └── turn                 一轮用户交互（handle_input / 反事实表单 / once）
        ├── load             数据加载（corpus 构建）
        ├── llm              一次 LLM 调用（路由/摘要/采集/建模/报告/问答/推演…）
        ├── stage            阶段进度事件（push_stage 的点事件，非区间）
        ├── stream           流式输出统计（点事件：chunk 计数/字节数）
        ├── snapshot         状态快照采集（点事件）
        └── output           轮次输出（含响应文本哈希/长度）

线程/异步安全：
- 当前 span 栈用 contextvars 传递（asyncio.to_thread 会复制上下文，
  Web SSE 工作线程中的 LLM 调用能正确挂到对应 turn 下）。
- JSONL 文件写入用锁保护（多请求并发追加不交错）。

等价性：Tracer 只读观测，不与业务状态交互；trace_enabled=False 时
所有公开方法为空操作（no-op），连文件都不会创建。
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_harness_current_span", default=None,
)


class Tracer:
    """Span 树记录器（只读观测，无副作用）。"""

    def __init__(self, trace_dir: Path | str | None, enabled: bool = True, run_id: str | None = None) -> None:
        self.enabled = enabled and trace_dir is not None
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._file = None
        self._span_seq = 0
        self.trace_path: Path | None = None

        if self.enabled:
            trace_dir = Path(trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            self.trace_path = trace_dir / f"trace_{time.strftime('%Y%m%d_%H%M%S')}_{self.run_id}.jsonl"
            self._file = self.trace_path.open("a", encoding="utf-8")
            self._write({
                "type": "run_start",
                "run_id": self.run_id,
                "ts": time.time(),
            })

    # ── 写入 ──────────────────────────────────────────────────

    def _write(self, obj: dict[str, Any]) -> None:
        if not self.enabled or self._file is None:
            return
        line = json.dumps(obj, ensure_ascii=False, default=str)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    # ── Span ──────────────────────────────────────────────────

    @contextmanager
    def span(self, kind: str, name: str, **attrs: Any) -> Iterator[str]:
        """开启一个区间 span（上下文管理器）。异常会记录 status=error 后原样抛出。"""
        if not self.enabled:
            yield ""
            return
        self._span_seq += 1
        span_id = f"{self.run_id}-{self._span_seq:05d}"
        parent_id = _current_span.get()
        start = time.time()
        self._write({
            "type": "span_start", "run_id": self.run_id, "span_id": span_id,
            "parent_id": parent_id, "kind": kind, "name": name,
            "ts": start, "attrs": attrs,
        })
        token = _current_span.set(span_id)
        status = "ok"
        try:
            yield span_id
        except Exception as exc:
            status = "error"
            self._write({
                "type": "span_event", "run_id": self.run_id, "span_id": span_id,
                "ts": time.time(), "attrs": {"exception": repr(exc)},
            })
            raise
        finally:
            _current_span.reset(token)
            self._write({
                "type": "span_end", "run_id": self.run_id, "span_id": span_id,
                "parent_id": parent_id, "kind": kind, "name": name,
                "ts": time.time(), "dur_ms": round((time.time() - start) * 1000, 2),
                "status": status,
            })

    def event(self, kind: str, name: str, **attrs: Any) -> None:
        """记录一个点事件（挂在当前 span 下）。"""
        if not self.enabled:
            return
        self._write({
            "type": "event", "run_id": self.run_id, "span_id": _current_span.get(),
            "kind": kind, "name": name, "ts": time.time(), "attrs": attrs,
        })

    def current_span_id(self) -> str | None:
        """只读访问器：当前 span id。"""
        return _current_span.get()

    # ── 生命周期 ──────────────────────────────────────────────

    def close(self) -> None:
        if not self.enabled:
            return
        self._write({"type": "run_end", "run_id": self.run_id, "ts": time.time()})
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
