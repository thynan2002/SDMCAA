"""阶段性进度推送。

通过 ContextVar 在请求作用域内传递进度回调，供流水线各阶段
向前端推送实时状态。与 LLM 文本流式回调（stream_callback）解耦：
- stream_callback  → 最终回复的逐 token 文本增量
- progress_callback → 阶段性状态（"正在采集数据""MCTS 推演 45%"…）

回调由 web.backend 的 SSE 生成器设置，经 asyncio.to_thread 传播到
工作线程；流水线各节点调用 push_stage(...) 即可向前端推送进度，
无需关心回调是否存在（未设置时静默跳过）。
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable

_progress_callback: contextvars.ContextVar[
    Callable[[str, str, dict[str, Any]], None] | None
] = contextvars.ContextVar("_opencode_progress_callback", default=None)


def set_progress_callback(
    cb: Callable[[str, str, dict[str, Any]], None] | None,
) -> contextvars.Token:
    """设置当前作用域的进度回调。每个阶段调用一次 cb(stage, content, extra)。"""
    return _progress_callback.set(cb)


def reset_progress_callback(token: contextvars.Token) -> None:
    """恢复进度回调到之前的值。"""
    _progress_callback.reset(token)


def push_stage(stage: str, content: str, status: str = "active", **extra: Any) -> None:
    """推送一个阶段性状态。

    Args:
        stage: 阶段标识（稳定 key，前端据其聚合/更新同一进度项，
               如 "mcts"；进度更新复用同一 key）
        content: 人类可读的状态文本（如 "MCTS 推演"）
        status: "active" | "done"；active 表示进行中，done 表示完成
        **extra: 附加数据（percent/current/total 等），原样传给回调
    """
    cb = _progress_callback.get()
    if cb is None:
        return
    try:
        cb(stage, content, {"status": status, **extra})
    except Exception:
        pass
