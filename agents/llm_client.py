"""LLM 调用客户端。

封装 DeepSeek Chat API，提供简单的文本生成接口。
支持流式输出（stream=True）：当调用方标记 stream=True 且设置了流式回调时，
逐 token 推送增量内容到回调，并返回完整文本。

工具调用（function calling）支持：
- call_llm(..., tools=[...]) 显式传入工具，或经 bind_prompt_tools 按系统提示词绑定；
- 多轮工具循环：LLM 发起 tool_calls → 系统经 agents.tools 注册表执行 →
  结果以 role=tool 消息回填 → 继续生成，直至模型给出最终结果；
- 同一响应内的多个独立 tool_calls 一次性并行执行；
- 终止型工具（结构化决策输出）：其调用即最终答案，立即返回
  json.dumps(参数)，供调用方既有 JSON 解析路径复用；
- 不传 tools 且无绑定时，行为与纯文本接口逐字节一致。

工具参数与多轮 messages 经 ContextVar（tool_exchange_var）传入传输层，
解析出的 assistant 消息经同一通道回传 —— 与流式回调 ContextVar 同构，
保证 dispatcher / _call_llm_impl 的既有签名与 patch 点完全不变
（harness 拦截器与测试 mock 无需感知工具层即可继续工作）。
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Generator

import requests
from agents.logging_config import get_logger

logger = get_logger("llm_client")


# ── 流式回调上下文 ──────────────────────────────────────────────
# 通过 ContextVar 在请求作用域内传递流式回调，避免侵入式参数透传。
# 只有显式标记 stream=True 的 call_llm 调用（最终用户可见输出）才会消费回调，
# 中间调用（路由解析、数据采集等）不受影响。
_stream_callback: contextvars.ContextVar[Callable[[str], None] | None] = contextvars.ContextVar(
    "_opencode_stream_callback", default=None,
)


def set_stream_callback(cb: Callable[[str], None] | None) -> contextvars.Token:
    """设置当前作用域的流式回调（每个增量 chunk 调用一次）。"""
    return _stream_callback.set(cb)


def reset_stream_callback(token: contextvars.Token) -> None:
    """恢复流式回调到之前的值。"""
    _stream_callback.reset(token)


def get_stream_callback() -> Callable[[str], None] | None:
    """只读访问器：返回当前作用域的流式回调（Harness 观测/回放用，不修改状态）。"""
    return _stream_callback.get()


# ── 调用分派器（Harness 最小钩子） ─────────────────────────────
# 默认 None：call_llm 直接执行 _call_llm_impl，行为与原实现完全一致
# （仅多一次 None 判断，语义恒等）。Harness 在 trace/record/replay 模式下
# 注入包装器；注入器接收与 call_llm 完全相同的参数，返回值语义相同（str | None）。
_LlmDispatcher = Callable[..., "str | None"]
_llm_dispatcher: _LlmDispatcher | None = None


def set_llm_dispatcher(dispatcher: _LlmDispatcher | None) -> None:
    """注入/清除 LLM 调用分派器（None = 恢复默认直调实现）。"""
    global _llm_dispatcher
    _llm_dispatcher = dispatcher


def get_llm_dispatcher() -> _LlmDispatcher | None:
    """只读访问器：返回当前分派器。"""
    return _llm_dispatcher


# ── 工具调用交换通道（ContextVar） ─────────────────────────────
# 传输层（_call_llm_impl / _call_llm_streaming）与回放器通过该通道
# 读取本轮 tools/多轮 messages，并回填解析出的 assistant 消息（含
# tool_calls）。交换对象在每个工具循环轮次开始时新建，默认 None 时
# 传输层行为与纯文本实现完全一致。
@dataclass
class ToolExchange:
    tools: list[dict[str, Any]]                      # OpenAI tools schema 列表
    tool_choice: Any = None                          # "auto"/"required"/{"type":...}
    messages: list[dict[str, Any]] | None = None     # 多轮消息（None = 首轮 system+user）
    round: int = 0                                   # 当前工具循环轮次（观测用）
    assistant_message: dict[str, Any] | None = None  # 传输层/回放器回填的完整 assistant 消息


tool_exchange_var: contextvars.ContextVar[ToolExchange | None] = contextvars.ContextVar(
    "_opencode_tool_exchange", default=None,
)


def get_tool_exchange() -> ToolExchange | None:
    """只读访问器：当前作用域的工具交换对象（harness 录制/回放用）。"""
    return tool_exchange_var.get()


# ── 提示词 → 工具契约绑定 ──────────────────────────────────────
# 按系统提示词绑定工具，使既有调用点（签名被测试 mock 固定）无需改动
# 即可迁移到工具契约。call_llm 的显式 tools 参数优先于绑定。
_prompt_tool_bindings: dict[str, tuple[list[Any], Any]] = {}


def bind_prompt_tools(
    system_prompt: str, tools: list[Any], tool_choice: Any = None,
) -> None:
    """为系统提示词绑定工具契约（tools 为 ToolSpec 或 OpenAI schema dict）。"""
    _prompt_tool_bindings[system_prompt] = (list(tools), tool_choice)


def unbind_prompt_tools(system_prompt: str) -> None:
    _prompt_tool_bindings.pop(system_prompt, None)


def get_prompt_tools(system_prompt: str) -> tuple[list[Any], Any] | None:
    return _prompt_tool_bindings.get(system_prompt)


@dataclass
class LlmConfig:
    api_key: str
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    # base_url: str = "https://opencode.ai/zen/go"
    temperature: float = 0.85
    # 推理型模型（响应含 reasoning_content）的思维链与正文共享该预算：
    # 复杂/多子问题的推理可消耗 3000+ tokens，2048 会被推理耗尽导致正文为空，
    # 因此默认预留充足空间（预算为上限，按实际用量计费）。
    # 用户确认 token 充足，按需求将上下文/输出预算提升至 81920，
    # 以支持长报告、丰富上下文与多轮复杂推理。
    # 重要：此值为全局上限，所有调用方不再传 max_tokens 覆盖它，
    # 杜绝 MCTS 推演等场景因小预算截断导致空内容报错。
    max_tokens: int = 81920
    # 大预算下生成耗时更长，超时相应放宽。
    timeout: int = 300


def load_llm_config() -> LlmConfig | None:
    """从环境变量加载 LLM 配置。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    return LlmConfig(api_key=api_key, model=model)


def call_llm(
    system_prompt: str,
    user_message: str,
    config: LlmConfig | None = None,
    max_tokens: int | None = None,
    retries: int = 1,
    stream: bool = False,
    tools: list[Any] | None = None,
    tool_choice: Any = None,
    max_rounds: int | None = None,
) -> str | None:
    """调用 DeepSeek Chat API 生成文本。

    Args:
        max_tokens: 覆盖默认输出长度上限（长报告场景使用）
        retries: 失败后的额外重试次数（应对瞬时网络/限流故障）
        stream: 标记本次调用为"最终用户可见输出"。当为 True 且当前
            作用域已设置流式回调（set_stream_callback）时，使用流式模式，
            逐增量 chunk 调用回调并返回完整文本；否则按普通非流式处理。
        tools: 工具列表（ToolSpec 或 OpenAI tools schema dict）。提供时
            启用 function calling 多轮工具循环。
        tool_choice: 工具选择策略（"auto" / "required" / 指定函数对象）。

    返回生成的文本内容，失败返回 None。

    工具模式返回值契约（保持 str | None）：
    - 模型经终止型工具提交结构化结果 → json.dumps(工具参数)，
      多个并行终止型调用时结果字典合并；
    - 数据型工具循环后模型给出文本回答 → 文本；
    - 模型直接返回文本（未发起工具调用）→ 原文（调用方文本回退解析）；
    - 失败（网络/超时/格式非法/循环超限）→ None，由调用方规则兜底。

    不传 tools 且无提示词绑定时，行为与重构前完全一致。
    """
    effective_tools = tools
    effective_choice = tool_choice
    if effective_tools is None:
        binding = _prompt_tool_bindings.get(system_prompt)
        if binding is not None:
            effective_tools, bound_choice = binding
            if effective_choice is None:
                effective_choice = bound_choice
    if effective_tools:
        return _call_llm_with_tools(
            system_prompt, user_message, effective_tools, effective_choice,
            config=config, max_tokens=max_tokens, retries=retries, stream=stream,
            max_rounds=max_rounds or _DEFAULT_MAX_TOOL_ROUNDS,
        )

    if _llm_dispatcher is not None:
        return _llm_dispatcher(
            system_prompt, user_message,
            config=config, max_tokens=max_tokens, retries=retries, stream=stream,
        )
    return _call_llm_impl(
        system_prompt, user_message,
        config=config, max_tokens=max_tokens, retries=retries, stream=stream,
    )


# ── 工具调用多轮循环 ───────────────────────────────────────────
# LLM 发起 tool_calls → 注册表执行 → 结果回填 → 继续生成。
# 终止型工具（结构化输出提交）的调用即最终答案，立即返回。
_DEFAULT_MAX_TOOL_ROUNDS = 8


def _call_llm_with_tools(
    system_prompt: str,
    user_message: str,
    tools: list[Any],
    tool_choice: Any,
    config: LlmConfig | None = None,
    max_tokens: int | None = None,
    retries: int = 1,
    stream: bool = False,
    max_rounds: int = _DEFAULT_MAX_TOOL_ROUNDS,
) -> str | None:
    """带工具循环的 LLM 调用（返回契约见 call_llm 文档）。"""
    from agents.tools.base import get_default_registry

    registry = get_default_registry()
    schemas: list[dict[str, Any]] = []
    specs_by_name: dict[str, Any] = {}
    for t in tools:
        if hasattr(t, "to_openai_schema"):
            schemas.append(t.to_openai_schema())
            specs_by_name[t.name] = t
        else:
            schemas.append(t)
            name = ((t or {}).get("function") or {}).get("name")
            if name:
                specs_by_name.setdefault(name, registry.get(name))

    messages: list[dict[str, Any]] = []  # 空 = 首轮，传输层用 [system, user]
    for rnd in range(max_rounds):
        exchange = ToolExchange(
            tools=schemas,
            tool_choice=tool_choice,
            messages=list(messages) if messages else None,
            round=rnd,
        )
        token = tool_exchange_var.set(exchange)
        try:
            if _llm_dispatcher is not None:
                text = _llm_dispatcher(
                    system_prompt, user_message,
                    config=config, max_tokens=max_tokens, retries=retries,
                    stream=stream,
                )
            else:
                text = _call_llm_impl(
                    system_prompt, user_message,
                    config=config, max_tokens=max_tokens, retries=retries,
                    stream=stream,
                )
        finally:
            tool_exchange_var.reset(token)

        assistant = exchange.assistant_message
        tool_calls = (assistant or {}).get("tool_calls") or []
        if not tool_calls:
            # 无工具调用：文本回答（或 None 失败 → 调用方规则兜底）
            return text

        messages.append(_sanitize_assistant_message(assistant))
        terminal_results: list[dict[str, Any]] = []
        # 同一响应内的多个独立工具请求：一次性全部执行（并行 tool_calls）
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            args, parse_error = _parse_tool_arguments(fn.get("arguments"))
            spec = specs_by_name.get(name) or registry.get(name)
            terminal = bool(spec is not None and getattr(spec, "terminal", False))
            if parse_error is not None:
                result: dict[str, Any] = {
                    "error": f"工具参数格式非法: {parse_error}",
                    "kind": "invalid_arguments",
                }
            elif spec is None:
                result = {"error": f"未注册的工具: {name}", "kind": "unknown_tool"}
            else:
                result = _execute_tool_spec(spec, args)
            if terminal and "error" not in result:
                terminal_results.append(result)
            messages.append({
                "role": "tool",
                "tool_call_id": str(tc.get("id") or ""),
                "content": json.dumps(result, ensure_ascii=False),
            })

        if terminal_results:
            merged = terminal_results[0]
            for extra in terminal_results[1:]:
                merged = _deep_merge_dict(merged, extra)
            return json.dumps(merged, ensure_ascii=False)

    logger.warning("工具调用循环超过 %d 轮未收敛，按失败处理", max_rounds)
    return None


def _parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """解析 tool_calls 的 arguments（JSON 字符串或 dict）。"""
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        data = json.loads(str(raw))
        if isinstance(data, dict):
            return data, None
        return {}, f"参数应为 JSON 对象，实际为 {type(data).__name__}"
    except (json.JSONDecodeError, TypeError) as exc:
        return {}, str(exc)


def _execute_tool_spec(spec: Any, args: dict[str, Any]) -> dict[str, Any]:
    """按调用点提供的 ToolSpec 直接执行（无需注册进默认注册表）。

    失败分类与 ToolRegistry.execute 一致（执行异常不回传调用方）。
    """
    from agents.tools.base import ToolExecutionError

    try:
        result = spec.handler(args)
    except ToolExecutionError as exc:
        return {"error": str(exc), "kind": exc.kind}
    except Exception as exc:
        return {"error": f"工具执行异常: {exc}", "kind": "execution_error"}
    if isinstance(result, dict):
        return result
    return {"result": result}


def _sanitize_assistant_message(message: dict[str, Any] | None) -> dict[str, Any]:
    """规整 assistant 消息用于多轮回填（仅保留 OpenAI 约定字段）。"""
    msg = message or {}
    tool_calls: list[dict[str, Any]] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args or {}, ensure_ascii=False)
        tool_calls.append({
            "id": str(tc.get("id") or ""),
            "type": str(tc.get("type") or "function"),
            "function": {
                "name": str(fn.get("name") or ""),
                "arguments": args,
            },
        })
    return {
        "role": "assistant",
        "content": msg.get("content") or "",
        "tool_calls": tool_calls,
    }


def _deep_merge_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """合并多个并行终止型工具的结果字典（同名键递归合并，后者补充前者）。"""
    merged = dict(base)
    for key, value in extra.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _call_llm_impl(
    system_prompt: str,
    user_message: str,
    config: LlmConfig | None = None,
    max_tokens: int | None = None,
    retries: int = 1,
    stream: bool = False,
) -> str | None:
    """call_llm 的默认实现。

    工具交换通道（tool_exchange_var）激活时：payload 追加 tools/tool_choice，
    messages 采用多轮历史，响应中的 tool_calls 回填到交换对象；
    通道为 None 时与原实现逐字节等价。
    """
    if config is None:
        config = load_llm_config()
    if config is None:
        return None

    cb = _stream_callback.get() if stream else None
    if cb is not None:
        return _call_llm_streaming(system_prompt, user_message, config, max_tokens, cb)

    if config is None:
        return None

    exchange = tool_exchange_var.get()

    url = f"{config.base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    if exchange is not None and exchange.messages:
        messages = exchange.messages
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
    }
    if exchange is not None:
        payload["tools"] = exchange.tools
        if exchange.tool_choice is not None:
            payload["tool_choice"] = exchange.tool_choice

    # 全局统一使用 config.max_tokens（81920），调用方不再传 max_tokens 覆盖。
    # 旧版曾在此处对 finish_reason=length 做"翻倍预算重试"，但那只是把
    # 调用方传入的小预算（2048）翻成 4096，治标不治本，且会输出误导性日志。
    # 现已移除该机制：预算足够大时不会再触发 length 截断。
    budget = max_tokens or config.max_tokens

    for attempt in range(retries + 1):
        payload["max_tokens"] = budget
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=config.timeout)
            # 降级链：部分推理型模型不接受强制 tool_choice（400 Thinking mode ...），
            # 去掉 tool_choice 后按 auto 语义重试一次（tools 保留，行为不降级）。
            if (
                response.status_code == 400
                and "tool_choice" in payload
                and attempt == 0
            ):
                body_text = response.text or ""
                if "tool_choice" in body_text or "Thinking mode" in body_text:
                    logger.warning(
                        "服务端拒绝 tool_choice（%s），降级为 auto 重试",
                        body_text[:200],
                    )
                    payload.pop("tool_choice", None)
                    continue
            response.raise_for_status()
            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls")
            finish_reason = choice.get("finish_reason")
            # 工具调用响应：回填 assistant 消息（content 可为空），由工具循环继续
            if exchange is not None and tool_calls:
                exchange.assistant_message = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
                return content
            if content:
                return content
            logger.warning(
                "LLM 返回空内容（finish_reason=%s，第%d次）", finish_reason, attempt + 1,
            )
        except Exception as e:
            logger.warning("调用失败（第%d次）: %s", attempt + 1, e)
    return None


def _merge_tool_call_delta(
    acc: dict[int, dict[str, Any]], deltas: list[dict[str, Any]],
) -> None:
    """累积流式响应中的 tool_calls 增量片段（按 index 合并）。"""
    for d in deltas or []:
        try:
            idx = int(d.get("index", 0))
        except (TypeError, ValueError):
            idx = 0
        slot = acc.setdefault(idx, {
            "id": "", "type": "function",
            "function": {"name": "", "arguments": ""},
        })
        if d.get("id"):
            slot["id"] = str(d["id"])
        if d.get("type"):
            slot["type"] = str(d["type"])
        fn = d.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += str(fn["name"])
        if fn.get("arguments"):
            slot["function"]["arguments"] += str(fn["arguments"])


def _call_llm_streaming(
    system_prompt: str,
    user_message: str,
    config: LlmConfig,
    max_tokens: int | None,
    cb: Callable[[str], None],
) -> str | None:
    """流式调用 LLM，逐增量 chunk 调用回调，返回完整文本。

    工具交换通道激活时：payload 追加 tools/tool_choice，累积流式
    tool_calls 增量并回填交换对象；正文 chunk 推送契约不变。
    """
    exchange = tool_exchange_var.get()

    url = f"{config.base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    if exchange is not None and exchange.messages:
        messages = exchange.messages
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": max_tokens or config.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    if exchange is not None:
        payload["tools"] = exchange.tools
        if exchange.tool_choice is not None:
            payload["tool_choice"] = exchange.tool_choice

    pieces: list[str] = []
    acc_tool_calls: dict[int, dict[str, Any]] = {}
    try:
        with requests.post(url, headers=headers, json=payload, timeout=config.timeout, stream=True) as resp:
            if (
                resp.status_code == 400
                and "tool_choice" in payload
                and (resp.text or "").find("tool_choice") != -1
            ):
                # 降级链：推理模式拒绝强制 tool_choice → 去 choice 重试一次
                logger.warning("流式请求 tool_choice 被拒绝（%s），降级重试", resp.text[:200])
                payload.pop("tool_choice", None)
                with requests.post(url, headers=headers, json=payload, timeout=config.timeout, stream=True) as resp2:
                    return _drain_stream(resp2, acc_tool_calls, pieces, cb)
            resp.raise_for_status()
            return _drain_stream(resp, acc_tool_calls, pieces, cb)
    except Exception as e:
        logger.warning("流式调用失败: %s", e)
        # 流式中途失败：已收到的部分仍然返回，避免完全丢失
        if pieces:
            return "".join(pieces)
        return None


def _drain_stream(
    resp: Any,
    acc_tool_calls: dict[int, dict[str, Any]],
    pieces: list[str],
    cb: Callable[[str], None],
) -> str | None:
    """消费流式响应体，累积 tool_calls 并推送正文增量 chunk。

    与纯文本时代契约一致：cb(delta...) 逐增量调用；返回值=完整文本；
    工具调用响应回填 exchange.assistant_message 后由工具循环继续。
    """
    exchange = tool_exchange_var.get()
    resp.raise_for_status()
    # 显式锁定 UTF-8：requests 默认对 text/* 响应使用 ISO-8859-1 解码，
    # 会导致中文多字节字符被错误解释为 Latin-1 → 乱码。
    resp.encoding = "utf-8"
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = (choices[0].get("delta") or {})
        _merge_tool_call_delta(acc_tool_calls, delta.get("tool_calls") or [])
        # 只转发正文 content，不转发 reasoning_content（模型内部思维链）
        text = delta.get("content") or ""
        if text:
            pieces.append(text)
            try:
                cb(text)
            except Exception:
                logger.warning("流式回调异常，已忽略", exc_info=True)
    if acc_tool_calls:
        ordered = [acc_tool_calls[i] for i in sorted(acc_tool_calls)]
        finished = [
            tc for tc in ordered
            if tc["function"]["name"] and tc["function"]["arguments"]
        ]
        if exchange is not None and finished:
            exchange.assistant_message = {
                "role": "assistant",
                "content": "".join(pieces),
                "tool_calls": finished,
            }
            return "".join(pieces)
    return "".join(pieces) if pieces else None
