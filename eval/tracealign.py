"""中间步骤追溯与对齐 —— 统一 IR + 逐步骤对齐。

三个系统的 trace（均为 harness Tracer JSONL 格式）归一化为统一 IR：

    {"idx", "kind": llm|tool|stage|load|turn, "name", "dur_ms", "status",
     "phase": routing|retrieval|reasoning|synthesis|load,   # 启发式标注
     "tools": [...], "tool_round", "system_len", "user_len",
     "context_chars": 截至该步累计上下文字符数}

阶段映射为启发式（修订：UI 中明确标注「启发式对齐」，避免过度解读）：
- bare:  唯一 llm 步 → synthesis
- single: llm 步带 tool 结果回填后 → retrieval/reasoning，最后一步 → synthesis
- multi:  每个 turn 的第一个 llm → routing；带 tools 的 llm → retrieval；
          其余 → reasoning/synthesis；load span → load
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PHASE_LABELS = {
    "load": "数据加载", "routing": "意图路由", "retrieval": "工具检索",
    "reasoning": "推理分析", "synthesis": "答案生成",
}


def parse_trace(path: Path) -> list[dict[str, Any]]:
    """解析 Tracer JSONL → 有序 span/event 列表（span_start/end 配对出 dur）。"""
    if not path or not Path(path).exists():
        return []
    spans: dict[str, dict] = {}
    order: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rtype = rec.get("type")
        if rtype == "span_start":
            item = {
                "span_id": rec.get("span_id"), "parent_id": rec.get("parent_id"),
                "kind": rec.get("kind"), "name": rec.get("name"),
                "ts": rec.get("ts"), "dur_ms": None, "status": "ok",
                "attrs": rec.get("attrs") or {},
            }
            spans[rec["span_id"]] = item
            order.append(item)
        elif rtype == "span_end":
            item = spans.get(rec.get("span_id"))
            if item is not None:
                item["dur_ms"] = rec.get("dur_ms")
                item["status"] = rec.get("status", "ok")
        elif rtype == "event":
            order.append({
                "span_id": rec.get("span_id"), "parent_id": None,
                "kind": rec.get("kind"), "name": rec.get("name"),
                "ts": rec.get("ts"), "dur_ms": 0.0, "status": "ok",
                "attrs": rec.get("attrs") or {},
            })
    order.sort(key=lambda x: (x.get("ts") or 0, x.get("span_id") or ""))
    return order


def _attrs_digest(attrs: dict) -> dict:
    keep = {}
    for key in ("input", "person", "ball", "system_len", "user_len",
                "tools", "tool_round", "seq", "stream", "name", "ok", "tool_calls"):
        if key in attrs:
            value = attrs[key]
            keep[key] = (str(value)[:80] + "…") if isinstance(value, str) and len(value) > 80 else value
    return keep


def normalize(system: str, spans: list[dict]) -> list[dict]:
    """归一化为统一 IR（含启发式 phase 标注）。"""
    steps: list[dict] = []
    ctx_chars = 0
    llm_steps_idx: list[int] = []
    for sp in spans:
        kind = sp["kind"]
        attrs = sp["attrs"]
        if kind == "llm":
            ctx_chars += int(attrs.get("system_len") or 0) + int(attrs.get("user_len") or 0)
            llm_steps_idx.append(len(steps))
            steps.append({
                "idx": len(steps), "kind": "llm", "name": sp["name"],
                "dur_ms": sp["dur_ms"], "status": sp["status"], "phase": None,
                "tools": attrs.get("tools"), "tool_round": attrs.get("tool_round"),
                "context_chars": ctx_chars, "attrs": _attrs_digest(attrs),
            })
        elif kind == "tool":
            ctx_chars += 300  # 工具结果回填的近似下界
            steps.append({
                "idx": len(steps), "kind": "tool",
                "name": attrs.get("tool") or attrs.get("name", sp["name"]),
                "dur_ms": sp["dur_ms"] or attrs.get("dur_ms"), "status": "ok",
                "phase": "retrieval", "context_chars": ctx_chars,
                "attrs": _attrs_digest(attrs),
            })
        elif kind == "stage":
            steps.append({
                "idx": len(steps), "kind": "stage", "name": sp["name"],
                "dur_ms": 0.0, "status": "ok", "phase": "reasoning",
                "context_chars": ctx_chars, "attrs": _attrs_digest(attrs),
            })
        elif kind == "load":
            steps.append({
                "idx": len(steps), "kind": "load", "name": sp["name"],
                "dur_ms": sp["dur_ms"], "status": sp["status"], "phase": "load",
                "context_chars": ctx_chars, "attrs": _attrs_digest(attrs),
            })

    # ── 启发式阶段标注（对所有 llm 步统一后处理）──
    def _has_data_tools(step: dict) -> bool:
        # submit_* 为结构化决策契约（非检索）；其余视为数据检索工具
        tools = step.get("tools") or []
        return any(t and not str(t).startswith("submit_") for t in tools)

    n_llm = len(llm_steps_idx)
    for rank, step_idx in enumerate(llm_steps_idx):
        step = steps[step_idx]
        is_last = rank == n_llm - 1
        if system == "bare":
            step["phase"] = "synthesis"
        elif system == "single":
            # 单智能体工具循环：中间轮发起工具 → retrieval；最终轮 → synthesis
            step["phase"] = "synthesis" if is_last else "retrieval"
        else:  # multi
            if rank == 0:
                step["phase"] = "routing"          # 首个 llm = 意图路由
            elif is_last:
                step["phase"] = "synthesis"
            elif _has_data_tools(step):
                step["phase"] = "retrieval"
            else:
                step["phase"] = "reasoning"
    return steps


def build_case_alignment(runs_by_system: dict[str, list[dict]], repeat_idx: int = 0) -> dict:
    """构建单用例的对齐视图（每系统取指定重复的 trace）。

    runs_by_system: {system: [run result dict（按 repeat 排序）]}
    """
    alignment: dict[str, Any] = {"heuristic": True, "systems": {}, "tool_diff": {}}
    tool_seqs: dict[str, list[str]] = {}
    for sys_name, runs in runs_by_system.items():
        run = runs[repeat_idx] if repeat_idx < len(runs) else (runs[-1] if runs else None)
        if run is None:
            continue
        trace_path = Path(run.get("out_dir", "")) / (run.get("trace_file") or "")
        steps = normalize(sys_name, parse_trace(trace_path))
        tool_records = run.get("tool_records") or []
        tool_seq = [r.get("name") for r in tool_records]
        if not tool_seq:
            tool_seq = [s["name"] for s in steps if s["kind"] == "tool"]
        tool_seqs[sys_name] = tool_seq
        alignment["systems"][sys_name] = {
            "repeat": run.get("repeat"), "steps": steps,
            "latency_ms": run.get("latency_ms"),
            "llm_calls": run.get("llm_calls"),
            "answer_excerpt": (run.get("answer") or "")[:400],
        }
    # 工具序列差异（哪些调用为某系统独有）
    if len(tool_seqs) >= 2:
        from collections import Counter

        union = Counter()
        for seq in tool_seqs.values():
            union.update(set(seq))
        present_in = {t: sorted(s for s, seq in tool_seqs.items() if t in seq)
                      for t in union}
        alignment["tool_diff"] = present_in
    return alignment


def context_curves(alignment: dict) -> dict[str, list[int]]:
    """每系统上下文演变曲线（用于报告绘制）。"""
    return {
        sys_name: [s["context_chars"] for s in data["steps"]]
        for sys_name, data in alignment.get("systems", {}).items()
    }
