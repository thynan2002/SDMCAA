"""模型操作事件推送（tool_calls / 思考过程 / LLM 调用 / 工具循环）。

与 progress 阶段推送（push_stage）解耦，定位更细粒度：
- progress_callback  → 流水线阶段状态（"正在采集数据""MCTS 推演 45%"…）
- op_callback        → 模型内部操作明细：每次 LLM 调用起止、模型思考
  （reasoning_content，流式增量）、发起的 tool_calls、工具执行结果、
  工具循环轮次等。Web 前端据此渲染「模型操作轨迹」面板。

回调由 web.backend 的 SSE 生成器设置，经 asyncio.to_thread 传播到
工作线程；LLM 层各关键节点调用 push_op(...) 即可推送操作事件，
未设置回调时静默跳过（与进度回调同构）。
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable

_op_callback: contextvars.ContextVar[
    Callable[[str, dict[str, Any]], None] | None
] = contextvars.ContextVar("_opcode_op_callback", default=None)


def set_op_callback(
    cb: Callable[[str, dict[str, Any]], None] | None,
) -> contextvars.Token:
    """设置当前作用域的操作事件回调。每事件调用一次 cb(event, payload)。"""
    return _op_callback.set(cb)


def reset_op_callback(token: contextvars.Token) -> None:
    """恢复操作事件回调到之前的值。"""
    _op_callback.reset(token)


def get_op_callback() -> Callable[[str, dict[str, Any]], None] | None:
    """只读访问器：返回当前作用域的操作事件回调。"""
    return _op_callback.get()


def push_op(event: str, **payload: Any) -> None:
    """推送一个模型操作事件。

    Args:
        event: 事件标识（稳定 key，前端据其渲染，如 "tool_calls"、"reasoning_delta"）
        **payload: 附加数据（round/name/arguments/result/content 等），原样传给回调
    """
    cb = _op_callback.get()
    if cb is None:
        return
    try:
        cb(event, payload)
    except Exception:
        pass
