"""LLM 调用拦截器 — Harness 对 LLM 层的唯一介入点。

等价性论证（纯函数透传包装）：
    dispatch(x) = observe(_call_llm_impl(x))     — passthrough / record
    dispatch(x) = replay_llm(x)                  — replay（确定性注入，显式启用）

observe 只读记录（哈希/长度/耗时/可选全量入出参），满足：
- 不修改返回值（response 原样返回）
- 不吞异常（_call_llm_impl 抛出的异常原样传播，span 记录 status=error）
- 不改变分支逻辑（重试、流式选择全部仍在 _call_llm_impl 内部）

安装方式：agents.llm_client.set_llm_dispatcher(dispatch)。
未安装时 dispatcher=None，call_llm 与原实现逐字节等价。

工具调用（function calling）扩展（向后兼容）：
分派签名保持不变；工具参数与响应中的 tool_calls 经 llm_client 的
tool_exchange_var 上下文变量观测。record 模式仅在出现工具交互时为
golden 条目追加可选字段（tools / tool_choice / tool_round /
assistant_message），旧格式条目与旧 golden 回放不受影响。
"""

from __future__ import annotations

import time
from typing import Any

from agents import llm_client
from agents.llm_client import _call_llm_impl, set_llm_dispatcher

from .config import MODE_RECORD, MODE_REPLAY
from .mock import GoldenStore, ReplayLLM, sha256_text
from .tracing import Tracer


class LLMInterceptor:
    """LLM 分派器：观测（trace）/ 录制（record）/ 回放（replay）。"""

    def __init__(
        self,
        tracer: Tracer,
        mode: str,
        golden_store: GoldenStore | None = None,
        replay_llm: ReplayLLM | None = None,
    ) -> None:
        self.tracer = tracer
        self.mode = mode
        self.golden = golden_store
        self.replay_llm = replay_llm
        self.call_count = 0
        self._installed = False
        # bound method 每次访问都会生成新对象（is 比较恒为 False），
        # 必须持有同一引用才能精确卸载。
        self._installed_ref: Any = None

    # ── 安装/卸载 ─────────────────────────────────────────────

    def install(self) -> None:
        self._installed_ref = self.dispatch
        set_llm_dispatcher(self._installed_ref)
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            # 仅当当前分派器仍是自己时清除，避免误删他人的注入
            if self._installed_ref is not None and \
                    llm_client.get_llm_dispatcher() is self._installed_ref:
                set_llm_dispatcher(None)
            self._installed = False
            self._installed_ref = None

    # ── 分派 ──────────────────────────────────────────────────

    def dispatch(
        self,
        system_prompt: str,
        user_message: str,
        config: Any = None,
        max_tokens: int | None = None,
        retries: int = 1,
        stream: bool = False,
    ) -> str | None:
        seq = self.call_count
        self.call_count += 1
        # 工具交换通道（激活时表示本次为工具循环中的某一轮）
        exchange = llm_client.get_tool_exchange()
        span_attrs = {
            "seq": seq,
            "stream": bool(stream),
            "system_sha256": sha256_text(system_prompt),
            "user_sha256": sha256_text(user_message),
            "system_len": len(system_prompt),
            "user_len": len(user_message),
        }
        if exchange is not None:
            span_attrs["tool_round"] = exchange.round
            span_attrs["tools"] = [
                (t.get("function") or {}).get("name", "?") for t in exchange.tools
            ]

        with self.tracer.span("llm", f"llm_call[{seq}]", **span_attrs):
            if self.mode == MODE_REPLAY:
                assert self.replay_llm is not None
                response = self.replay_llm(
                    system_prompt, user_message,
                    config=config, max_tokens=max_tokens, retries=retries, stream=stream,
                )
            else:
                start = time.time()
                response = _call_llm_impl(
                    system_prompt, user_message,
                    config=config, max_tokens=max_tokens, retries=retries, stream=stream,
                )
                _ = time.time() - start  # 耗时由 span 统一记录

            # 工具循环轮次可能无正文（仅 tool_calls）；response 可为 ""
            responded = response is not None
            has_tool_calls = bool(
                exchange is not None and exchange.assistant_message
                and exchange.assistant_message.get("tool_calls")
            )
            self.tracer.event("llm_result", f"llm_result[{seq}]", **{
                "seq": seq,
                "ok": responded,
                "tool_calls": has_tool_calls,
                "response_sha256": sha256_text(response) if response else None,
                "response_len": len(response) if response else 0,
            })

        if self.mode == MODE_RECORD and self.golden is not None:
            entry: dict[str, Any] = {
                "seq": seq,
                "stream": bool(stream),
                "system_prompt": system_prompt,
                "user_message": user_message,
                "response": response,
            }
            # 仅在出现工具交互时追加可选字段（向后兼容，旧条目格式不变）
            if exchange is not None:
                entry["tool_round"] = exchange.round
                entry["tools"] = exchange.tools
                if exchange.tool_choice is not None:
                    entry["tool_choice"] = exchange.tool_choice
                if exchange.assistant_message is not None:
                    entry["assistant_message"] = exchange.assistant_message
            self.golden.append_llm_call(entry)
        return response
