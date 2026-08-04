"""LLM 调用客户端。

封装 DeepSeek Chat API，提供简单的文本生成接口。
支持流式输出（stream=True）：当调用方标记 stream=True 且设置了流式回调时，
逐 token 推送增量内容到回调，并返回完整文本。
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
) -> str | None:
    """调用 DeepSeek Chat API 生成文本。

    Args:
        max_tokens: 覆盖默认输出长度上限（长报告场景使用）
        retries: 失败后的额外重试次数（应对瞬时网络/限流故障）
        stream: 标记本次调用为"最终用户可见输出"。当为 True 且当前
            作用域已设置流式回调（set_stream_callback）时，使用流式模式，
            逐增量 chunk 调用回调并返回完整文本；否则按普通非流式处理。

    返回生成的文本内容，失败返回 None。
    """
    if _llm_dispatcher is not None:
        return _llm_dispatcher(
            system_prompt, user_message,
            config=config, max_tokens=max_tokens, retries=retries, stream=stream,
        )
    return _call_llm_impl(
        system_prompt, user_message,
        config=config, max_tokens=max_tokens, retries=retries, stream=stream,
    )


def _call_llm_impl(
    system_prompt: str,
    user_message: str,
    config: LlmConfig | None = None,
    max_tokens: int | None = None,
    retries: int = 1,
    stream: bool = False,
) -> str | None:
    """call_llm 的默认实现（原 call_llm 函数体，一字未动）。"""
    if config is None:
        config = load_llm_config()
    if config is None:
        return None

    cb = _stream_callback.get() if stream else None
    if cb is not None:
        return _call_llm_streaming(system_prompt, user_message, config, max_tokens, cb)

    if config is None:
        return None

    url = f"{config.base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": config.temperature,
    }

    # 全局统一使用 config.max_tokens（81920），调用方不再传 max_tokens 覆盖。
    # 旧版曾在此处对 finish_reason=length 做"翻倍预算重试"，但那只是把
    # 调用方传入的小预算（2048）翻成 4096，治标不治本，且会输出误导性日志。
    # 现已移除该机制：预算足够大时不会再触发 length 截断。
    budget = max_tokens or config.max_tokens

    for attempt in range(retries + 1):
        payload["max_tokens"] = budget
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=config.timeout)
            response.raise_for_status()
            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            finish_reason = choice.get("finish_reason")
            if content:
                return content
            logger.warning(
                "LLM 返回空内容（finish_reason=%s，第%d次）", finish_reason, attempt + 1,
            )
        except Exception as e:
            logger.warning("调用失败（第%d次）: %s", attempt + 1, e)
    return None


def _call_llm_streaming(
    system_prompt: str,
    user_message: str,
    config: LlmConfig,
    max_tokens: int | None,
    cb: Callable[[str], None],
) -> str | None:
    """流式调用 LLM，逐增量 chunk 调用回调，返回完整文本。"""
    url = f"{config.base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": config.temperature,
        "max_tokens": max_tokens or config.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": False},
    }

    pieces: list[str] = []
    try:
        with requests.post(url, headers=headers, json=payload, timeout=config.timeout, stream=True) as resp:
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
                # 只转发正文 content，不转发 reasoning_content（模型内部思维链）
                text = delta.get("content") or ""
                if text:
                    pieces.append(text)
                    try:
                        cb(text)
                    except Exception:
                        logger.warning("流式回调异常，已忽略", exc_info=True)
    except Exception as e:
        logger.warning("流式调用失败: %s", e)
        # 流式中途失败：已收到的部分仍然返回，避免完全丢失
        if pieces:
            return "".join(pieces)
        return None

    return "".join(pieces) if pieces else None
