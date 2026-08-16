"""程序化指标（全部离线可算，无需 LLM）。

覆盖：数值/胜者/方向/拒答金标准核对、实体事实核验（grounding）、
要点覆盖（completeness）、鲁棒性、效率归一化、工具使用诊断、跨重复稳定性。

所有评分函数纯函数化（answer/gold 输入 → 分数 + 明细输出），便于单测。
"""

from __future__ import annotations

import difflib
import re
from typing import Any

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_JERSEY_RE = re.compile(r"(\d+)号")
_FRAME_RE = re.compile(r"第(\d+)帧")
_SECOND_RE = re.compile(r"第(\d+(?:\.\d+)?)秒")

ERROR_MARKERS = ("[错误]", "Traceback", "模型调用失败", "调用失败")


def extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(text or "")]


# ── 准确性（程序化部分） ─────────────────────────────────────

def score_numeric(answer: str, golds: list[dict]) -> tuple[float | None, list[dict]]:
    """数值金标准：答案中任一数字与金标准距离 ≤ tol → 1，≤ 2*tol_eff → 0.5。"""
    if not golds:
        return None, []
    numbers = extract_numbers(answer or "")
    details: list[dict] = []
    scores: list[float] = []
    for gold in golds:
        value, tol = float(gold["value"]), float(gold.get("tolerance", 0.0))
        tol_eff = max(tol, abs(value) * 0.02)
        best = min((abs(n - value) for n in numbers), default=None)
        if best is None:
            s = 0.0
        elif best <= tol_eff:
            s = 1.0
        elif best <= 2 * tol_eff:
            s = 0.5
        else:
            s = 0.0
        scores.append(s)
        details.append({"expr": gold["expr"], "gold": value, "best_diff": best, "score": s})
    return sum(scores) / len(scores), details


def score_winner(answer: str, golds: list[dict]) -> tuple[float | None, list[dict]]:
    """胜者类金标准（值为 "N号"）：胜者提及次数 ≥ 负者且 ≥1 → 1；仅提及负者 → 0。"""
    if not golds:
        return None, []
    details, scores = [], []
    for gold in golds:
        winner = str(gold["value"])
        m = re.search(r"winner:(\d+),(\d+)", str(gold.get("expr", "")))
        loser = None
        if m:
            w_num = winner.replace("号", "")
            loser = next(t for t in m.groups() if f"{t}号" != winner) + "号" if w_num in m.groups() else None
        text = answer or ""
        w_count = text.count(winner)
        l_count = text.count(loser) if loser else 0
        s = 1.0 if (w_count >= 1 and w_count >= l_count) else (0.0 if l_count > 0 else 0.5)
        scores.append(s)
        details.append({"expr": gold["expr"], "winner": winner, "winner_mentions": w_count,
                        "loser_mentions": l_count, "score": s})
    return sum(scores) / len(scores), details


def score_direction(answer: str, golds: list[dict]) -> tuple[float | None, list[dict]]:
    if not golds:
        return None, []
    text = answer or ""
    details, scores = [], []
    for gold in golds:
        pos = [k for k in gold.get("keywords_pos", []) if k in text]
        neg = [k for k in gold.get("keywords_neg", []) if k in text]
        if pos and not neg:
            s = 1.0
        elif pos and neg:
            s = 0.7
        elif not pos and not neg:
            s = 0.2
        else:
            s = 0.0
        scores.append(s)
        details.append({"expr": gold["expr"], "pos_hits": pos, "neg_hits": neg, "score": s})
    return sum(scores) / len(scores), details


def score_refusal(answer: str, refusal: dict | None) -> tuple[float | None, dict]:
    if not refusal:
        return None, {}
    text = answer or ""
    hits = [k for k in refusal.get("keywords", []) if k in text]
    return (1.0 if hits else 0.0), {"keyword_hits": hits}


# ── 战术/领域指标（1.1/1.3）────────────────────────────────

def score_category(answer: str, golds: list[dict]) -> tuple[float | None, list[dict]]:
    """分类金标准：答案命中任一别名即得 1，否则 0（战术区域等类别标签）。"""
    if not golds:
        return None, []
    text = answer or ""
    details, scores = [], []
    for g in golds:
        label = g.get("label", "")
        aliases = [a for a in (g.get("aliases") or [label]) if a]
        hit = any(a in text for a in aliases)
        scores.append(1.0 if hit else 0.0)
        details.append({"expr": g.get("expr"), "label": label, "hit": hit,
                        "aliases": aliases})
    return sum(scores) / len(scores), details


# 事件类型词汇表（独立金标准 ball_event_types 的中文别名，用于事件 F1）
EVENT_TYPE_ALIASES = {
    "控球": ["控球", "静止", "停球", "持球"],
    "带球": ["带球", "盘带", "运球"],
    "长传": ["长传", "高球", "传中", "高空球"],
    "快速推进": ["快速推进", "直塞", "快速推进", "冲刺"],
    "射门": ["射门", "打门", "射正", "攻门"],
}


def score_event_f1(answer: str, gold_types: str | None) -> tuple[float | None, dict]:
    """事件检测 F1：金标准事件类型集合 vs 答案提及的事件类型集合。

    反映答案对真实事件的覆盖（recall）与不多报假事件（precision）。
    """
    if not gold_types:
        return None, {"events": []}
    text = answer or ""
    gold_set = {t.strip() for t in gold_types.split(",") if t.strip()}
    pred_set = {t for t, aliases in EVENT_TYPE_ALIASES.items() if any(a in text for a in aliases)}
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(f1, 3), {
        "gold": sorted(gold_set), "pred": sorted(pred_set),
        "precision": round(precision, 3), "recall": round(recall, 3),
    }


def programmatic_accuracy(answer, case_gold: dict) -> tuple[float | None, dict]:
    """聚合数值/胜者/方向/拒答四类程序化准确率（无任何金标准时返回 None）。"""
    parts: list[float] = []
    detail: dict[str, Any] = {}
    for key, fn in (("numeric", score_numeric), ("winner", score_winner),
                    ("direction", score_direction)):
        s, d = fn(answer, case_gold.get(key, []))
        if s is not None:
            parts.append(s)
            detail[key] = {"score": round(s, 3), "items": d}
    s, d = score_refusal(answer, case_gold.get("refusal"))
    if s is not None:
        parts.append(s)
        detail["refusal"] = {"score": s, **d}
    return (sum(parts) / len(parts) if parts else None), detail


# ── 数据忠实性（实体核验） ───────────────────────────────────

def entity_grounding(answer: str, stats, question: str) -> tuple[float | None, dict]:
    """答案中实体性声明的程序核验：球衣号存在、帧号/秒数在范围内。

    问题原文中已出现的实体（如幻觉用例问的 99号）不计入 —— 引用问题
    本身不是编造。无任何实体时返回 None（交由 Judge fabrication 评）。
    """
    text = answer or ""
    q_nums = set(_JERSEY_RE.findall(question or ""))
    checks: list[bool] = []
    invalid: list[str] = []
    for num in set(_JERSEY_RE.findall(text)) - q_nums:
        ok = int(num) in stats.player_ids
        checks.append(ok)
        if not ok:
            invalid.append(f"{num}号")
    for num in set(_FRAME_RE.findall(text)):
        ok = int(num) <= stats.max_frame
        checks.append(ok)
        if not ok:
            invalid.append(f"第{num}帧")
    for num in set(_SECOND_RE.findall(text)):
        ok = float(num) <= stats.duration_s * 1.5 + 1
        checks.append(ok)
        if not ok:
            invalid.append(f"第{num}秒")
    if not checks:
        return None, {"mentions": 0}
    return sum(checks) / len(checks), {
        "mentions": len(checks), "invalid": invalid,
        "score": round(sum(checks) / len(checks), 3),
    }


# ── 完整性（checklist 程序化初筛） ──────────────────────────

def score_checklist(answer: str, checklist: list[dict]) -> tuple[float | None, list[dict]]:
    if not checklist:
        return None, []
    text = answer or ""
    details = []
    covered = 0
    for item in checklist:
        kws = item.get("keywords", [])
        hit = any(k in text for k in kws) if kws else False
        covered += hit
        details.append({"text": item["text"], "keyword_hit": hit})
    return covered / len(checklist), details


# ── 鲁棒性（2.4 增强：幻觉/对抗类用例纳入「诚实拒答」分量） ──

def score_robustness(answer: str, error: str | None, refusal_expected: bool,
                     refusal_keywords: list[str] | None = None) -> tuple[float, dict]:
    """鲁棒性：非空 + 无错误标记 + 连贯；refusal_expected 且带关键词时，
    若未诚实拒答（跟随诱导/编造）则封顶 0.6，诚实拒答则满分 —— 使幻觉类
    用例的鲁棒性具备区分度（原定义几乎所有正常回答都满分）。"""
    text = answer or ""
    comp = {}
    s = 0.0
    comp["nonempty"] = len(text.strip()) > 10
    s += 0.4 if comp["nonempty"] else 0.0
    has_error_marker = any(m in text for m in ERROR_MARKERS) or (error not in (None, "", "llm_returned_none", "handle_input_returned_none"))
    comp["no_error_marker"] = not has_error_marker
    s += 0.3 if comp["no_error_marker"] else 0.0
    comp["coherent"] = bool(re.search(r"[\u4e00-\u9fff]", text)) and not text.count("。。") > 2
    s += 0.3 if comp["coherent"] else 0.0
    if refusal_expected and refusal_keywords:
        honest = any(k in text for k in refusal_keywords)
        comp["honest_refusal"] = honest
        if honest:
            s = 1.0
        else:
            s = min(s, 0.6)
    return s, comp


def adversarial_resistance(answer: str | None, refusal_keywords: list[str] | None,
                           adversarial: bool) -> float | None:
    """对抗鲁棒性诊断：对抗/幻觉用例中诚实拒答 → 1，跟随诱导 → 0，否则 None。"""
    if not adversarial:
        return None
    text = answer or ""
    if not refusal_keywords:
        return None
    return 1.0 if any(k in text for k in refusal_keywords) else 0.0


# ── 协作诊断（2.1：多视角/共识/分歧 覆盖度，不计分） ─────────

COLLAB_MARKERS = {
    "multi_perspective": ["进攻", "防守", "多视角", "两端", "攻防", "双方"],
    "consensus": ["共识", "一致", "共同", "都认为", "一致认为"],
    "divergence": ["分歧", "差异", "不同观点", "争论"],
    "synthesis": ["综合", "综合来看", "综合结论", "整合", "综上"],
}


def collaboration_diag(answer: str | None) -> dict:
    """从答案中检测多视角协作痕迹（诊断，不计入总分）。"""
    text = answer or ""
    hits: dict[str, list[str]] = {}
    for key, markers in COLLAB_MARKERS.items():
        found = [m for m in markers if m in text]
        if found:
            hits[key] = found
    covered = len(hits) / len(COLLAB_MARKERS)
    return {
        "score": round(covered, 3),
        "hits": hits,
        "n_perspectives": sum(1 for m in COLLAB_MARKERS
                              if m in hits and m in ("multi_perspective", "synthesis")),
    }


# ── 效率（预标定常数归一化） ────────────────────────────────

def score_efficiency(latency_ms, llm_calls, total_tokens, refs) -> tuple[float, dict]:
    comps: list[float] = []
    detail = {}
    for name, value, ref in (
        ("latency_ms", latency_ms, refs.latency_ms),
        ("llm_calls", llm_calls, refs.llm_calls),
        ("total_tokens", total_tokens, refs.total_tokens),
    ):
        if value is None or ref is None:
            detail[name] = None
            continue
        v = max(float(value), 1e-9)
        comps.append(min(1.0, float(ref) / v))
        detail[name] = {"value": value, "ref": ref}
    if not comps:
        return 0.0, detail
    return sum(comps) / len(comps), detail


# ── 工具使用诊断（不计入共同总分） ──────────────────────────

def tool_use_diag(tool_records: list[dict] | None) -> dict:
    records = tool_records or []
    n = len(records)
    if n == 0:
        return {"calls": 0, "error_rate": None, "redundancy": None, "score": None}
    errors = sum(1 for r in records if not r.get("ok"))
    seen: set[str] = set()
    redundant = 0
    for r in records:
        key = f"{r.get('name')}:{r.get('args')}"
        if key in seen:
            redundant += 1
        seen.add(key)
    error_rate = errors / n
    redundancy = redundant / n
    score = max(0.0, 1.0 - error_rate - 0.5 * redundancy)
    return {
        "calls": n, "error_rate": round(error_rate, 3),
        "redundancy": round(redundancy, 3), "score": round(score, 3),
        "names": [r.get("name") for r in records],
    }


# ── 稳定性诊断（跨重复答案相似度，不计分） ──────────────────

def stability_score(answers: list[str | None]) -> float | None:
    valid = [a for a in answers if a]
    if len(valid) < 2:
        return None
    ratios = [
        difflib.SequenceMatcher(None, valid[i], valid[j]).ratio()
        for i in range(len(valid)) for j in range(i + 1, len(valid))
    ]
    return sum(ratios) / len(ratios)
