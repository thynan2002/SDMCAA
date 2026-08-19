"""三个系统的 Runner —— 每次运行在独立子进程内执行（评审修订 #15/#16：
彻底隔离全局 corpus / dispatcher / ContextVar 状态）。

统一产出 RunResult（result.json）：
    answer / latency_ms / llm_calls / tokens / error / model / temperature
    / trace_file / tool_records / ts

trace 统一为 harness Tracer 的 JSONL Span 树格式（runners 自建 Tracer 实例，
多智能体则直接消费 Harness 自带的 Tracer），供 tracealign 归一化对齐。
"""

from __future__ import annotations

import dataclasses
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from . import goldref
from .config import EvalConfig, SYSTEM_BARE, SYSTEM_SINGLE, SYSTEM_MULTI, PROJECT_ROOT

SINGLE_SYSTEM_PROMPT = (
    "你是一名足球比赛数据分析师，可以通过调用工具查询比赛数据"
    "（战术事实、球路时间线、帧级快照、球员原始数据、球员画像、反事实模拟）。"
    "回答问题前先调用必要工具获取数据，仅基于工具返回的结果作答；"
    "数据不足或查询不到时必须如实声明「数据不足」，严禁编造。"
    "回答使用中文。回答规范：先给出直接的形势/战术判断（进攻态势、攻防转换、威胁程度、"
    "球员战术角色与活动区域等），再给数据依据；默认把统计数值翻译为自然语言判断"
    "（\"绝大多数时间\"\"明显更积极\"\"节奏偏快\"），严禁罗列成串的统计数字；"
    "仅当用户明确索要具体数值（如\"最快速度是多少\"\"峰值是多少\"\"具体几次\"）时，"
    "才直接引用工具返回的精确数值作答，不得含糊、不得拒绝、不得换成其他数值。"
)


def _llm_config_with_temperature(cfg: EvalConfig):
    from agents.llm_client import LlmConfig, load_llm_config

    base = load_llm_config()
    if base is None:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置（.env），无法执行真实评测")
    return LlmConfig(
        api_key=base.api_key, model=base.model, base_url=base.base_url,
        temperature=cfg.temperature,
    )


def _usage_totals() -> dict:
    """与 result schema 对齐：llm_calls 标量 + tokens 嵌套字典。"""
    from agents.llm_client import take_usage_events

    events = take_usage_events()
    return {
        "llm_calls": len(events),
        "tokens": {
            "prompt_tokens": sum(e["prompt_tokens"] for e in events),
            "completion_tokens": sum(e["completion_tokens"] for e in events),
            "total_tokens": sum(e["total_tokens"] for e in events),
        },
    }


def _latest_trace(out_dir: Path) -> str:
    traces = sorted(out_dir.glob("trace_*.jsonl"))
    return traces[-1].name if traces else ""


def _base_result(system: str, case: dict, out_dir: Path, model: str, temperature: float) -> dict:
    return {
        "system": system, "case_id": case["id"], "category": case["category"],
        "input": case["input"], "dataset": case["dataset"],
        "model": model, "temperature": temperature,
        "answer": None, "latency_ms": None, "llm_calls": 0,
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error": None, "trace_file": "", "tool_records": [],
        "out_dir": str(out_dir), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── 基线 A：单次 LLM 裸调用（全量数据入 prompt，无工具） ──────

def run_bare(case: dict, out_dir: Path, cfg: EvalConfig) -> dict:
    from agents.llm_client import call_llm
    from harness.tracing import Tracer

    config = _llm_config_with_temperature(cfg)
    result = _base_result(SYSTEM_BARE, case, out_dir, config.model, config.temperature)

    stats = goldref.compute_stats(
        str(PROJECT_ROOT / case["dataset"]["person"]), str(PROJECT_ROOT / case["dataset"]["ball"]),
    )
    summary = goldref.build_bare_summary(stats)
    user_message = f"{summary}\n\n{'=' * 40}\n【问题】{case['input']}"

    tracer = Tracer(out_dir, enabled=True)
    try:
        t0 = time.monotonic()
        with tracer.span("turn", "turn[0]", input=case["input"], system="bare"):
            with tracer.span("llm", "llm_call[0]", system_len=len(goldref.BARE_SYSTEM_PROMPT),
                             user_len=len(user_message)):
                answer = call_llm(goldref.BARE_SYSTEM_PROMPT, user_message, config=config)
        result["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        result["answer"] = answer
        if answer is None:
            result["error"] = "llm_returned_none"
        result.update(_usage_totals())
        # 与 single/multi 兜底口径对齐：裸调用必然发起 1 次 LLM 调用，
        # usage 事件缺失（服务端未返回 usage）时不将 llm_calls 计为 0。
        result["llm_calls"] = max(result["llm_calls"], 1)
        result["context_chars"] = len(user_message)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        tracer.close()
    result["trace_file"] = _latest_trace(out_dir)
    return result


# ── 基线 B：单智能体 + 同套数据工具（function calling 循环） ──

def run_single(case: dict, out_dir: Path, cfg: EvalConfig) -> dict:
    from agents.llm_client import call_llm, set_llm_dispatcher, _call_llm_impl
    from agents.player.tracker import build_corpus_from_paths
    from agents.tools import football as football_tools
    from agents.tools.base import ToolSpec
    from harness.tracing import Tracer

    config = _llm_config_with_temperature(cfg)
    result = _base_result(SYSTEM_SINGLE, case, out_dir, config.model, config.temperature)

    tracer = Tracer(out_dir, enabled=True)
    set_llm_dispatcher(None)  # 防御性重置（子进程本应为干净状态）

    call_seq = 0

    def counting_dispatch(system_prompt, user_message, **kwargs):
        nonlocal call_seq
        call_seq += 1
        with tracer.span("llm", f"llm_call[{call_seq - 1}]",
                         system_len=len(system_prompt or ""), user_len=len(user_message or ""),
                         tool_round=call_seq - 1):
            return _call_llm_impl(system_prompt, user_message, **kwargs)

    def wrap_spec(spec: ToolSpec) -> ToolSpec:
        def handler(args: dict) -> Any:
            t0 = time.monotonic()
            try:
                res = spec.handler(args)
                rec = {"name": spec.name, "args": args, "ok": "error" not in res if isinstance(res, dict) else True,
                       "dur_ms": round((time.monotonic() - t0) * 1000, 1),
                       "result_len": len(json.dumps(res, ensure_ascii=False, default=str))}
                return res
            except Exception as exc:
                rec = {"name": spec.name, "args": args, "ok": False,
                       "dur_ms": round((time.monotonic() - t0) * 1000, 1),
                       "error": str(exc)}
                raise
            finally:
                result["tool_records"].append(rec)
                tracer.event("tool", f"tool[{len(result['tool_records']) - 1}]",
                             tool=spec.name, ok=rec["ok"], dur_ms=rec["dur_ms"])
        return dataclasses.replace(spec, handler=handler)

    try:
        corpus = build_corpus_from_paths(
            PROJECT_ROOT / case["dataset"]["person"], PROJECT_ROOT / case["dataset"]["ball"],
        )
        football_tools.set_active_corpus(corpus)
        tools = [wrap_spec(spec) for spec in football_tools.get_football_tools()]

        set_llm_dispatcher(counting_dispatch)
        t0 = time.monotonic()
        with tracer.span("turn", "turn[0]", input=case["input"], system="single"):
            answer = call_llm(
                SINGLE_SYSTEM_PROMPT, case["input"], config=config,
                tools=tools, tool_choice="auto",
            )
        result["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        result["answer"] = answer
        if answer is None:
            result["error"] = "llm_returned_none"
        result.update(_usage_totals())
        result["llm_calls"] = max(result["llm_calls"], call_seq)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        set_llm_dispatcher(None)
        football_tools.set_active_corpus(None)
        tracer.close()
    result["trace_file"] = _latest_trace(out_dir)
    return result


# ── 被测系统：多智能体（SessionManager 经 Harness passthrough） ──

def run_multi(case: dict, out_dir: Path, cfg: EvalConfig) -> dict:
    import agents.llm_client as llm_mod
    from harness.config import HarnessConfig
    from harness.core import Harness

    hcfg = HarnessConfig(
        mode="passthrough", trace_enabled=True, trace_dir=out_dir,
        snapshot_enabled=False,
    )
    result = _base_result(SYSTEM_MULTI, case, out_dir, "", cfg.temperature)

    # 捕获多智能体内部全部工具执行（数据工具 + submit_* 决策工具）：
    # 包装 _execute_tool_spec（仅在本评测子进程内生效）
    _orig_exec = llm_mod._execute_tool_spec
    tool_records = result["tool_records"]

    def traced_exec(spec, args):
        t0 = time.monotonic()
        res = _orig_exec(spec, args)
        tool_records.append({
            "name": getattr(spec, "name", "unknown"), "args": args,
            "ok": "error" not in res if isinstance(res, dict) else True,
            "dur_ms": round((time.monotonic() - t0) * 1000, 1),
        })
        return res

    llm_mod._execute_tool_spec = traced_exec
    llm_call_seq = 0
    try:
        t0 = time.monotonic()
        with Harness(hcfg) as h:
            ok = h.load_data(
                PROJECT_ROOT / case["dataset"]["person"], PROJECT_ROOT / case["dataset"]["ball"],
            )
            if not ok:
                raise RuntimeError("数据加载失败")
            answer = h.handle_input(case["input"])
            # 调用次数兜底（与 run_single counting_dispatch 同思路）：
            # 流式 usage 缺失时，llm_calls 至少反映真实 LLM 调用次数。
            # Harness 拦截器对每次 dispatcher 调用计数，覆盖多智能体
            # 内部全部 LLM 调用（路由/编排/叙述，含 stream=True 路径）。
            llm_call_seq = h.llm_call_count
        result["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        result["answer"] = answer
        if answer is None:
            result["error"] = "handle_input_returned_none"
        result.update(_usage_totals())
        result["llm_calls"] = max(result["llm_calls"], llm_call_seq)
        result["model"] = _detect_model()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        llm_mod._execute_tool_spec = _orig_exec
    result["trace_file"] = _latest_trace(out_dir)
    return result


def _detect_model() -> str:
    import os

    return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


RUNNERS: dict[str, Callable[[dict, Path, EvalConfig], dict]] = {
    SYSTEM_BARE: run_bare,
    SYSTEM_SINGLE: run_single,
    SYSTEM_MULTI: run_multi,
}


def run_job(system: str, case: dict, out_dir: Path, cfg: EvalConfig) -> dict:
    """统一入口：按 system 分派到具体 Runner。"""
    runner = RUNNERS.get(system)
    if runner is None:
        raise ValueError(f"未知系统: {system!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return runner(case, out_dir, cfg)
