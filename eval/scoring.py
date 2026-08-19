"""评分汇总 —— 程序化指标 + Judge 指标融合为每 run / 每用例 / 每系统得分。

融合规则（1.3 解耦：程序化 65% + Judge 35% 总量子目标，缺失分量自动重归一化）：
- accuracy          = 0.65*程序化准确率 + 0.35*Judge accuracy（无程序金标准时 Judge 独占）
- completeness      = 0.5*checklist 覆盖 + 0.5*Judge completeness
- grounding         = 0.5*实体核验 + 0.5*(1-fabrication)
- reasoning         = Judge 1-5 折算 0-1（Judge 不可用记 None 并降权重归一化）
- efficiency        = 程序化（时延/调用数/token 预标定归一化）
- robustness        = 程序化（非空/无错误标记/连贯）
- tactical_accuracy = 程序化（战术区域分类命中率 + 威胁/球速数值容差 + 事件 F1，1.1）
共同总分 = Σ weight*metric；扩展总分额外计入 tool_use（其余等比缩放，
仅单智能体与多智能体之间比较）。
"""

from __future__ import annotations

from typing import Any

from . import metrics as M
from .config import EXTENDED_TOOL_WEIGHT, EvalConfig

JUDGE_MISSING_NOTE = "Judge 未启用或评分失败，相应分量已重归一化"

# Judge-程序分歧诊断（可信度增量）：程序化 accuracy 与 Judge accuracy 差的告警阈值
JUDGE_PROGRAM_DISAGREE_THRESHOLD = 0.3

# 战术数值类金标准（expr 前缀命中 → 计入 tactical_accuracy 而非 accuracy.numeric）
TACTICAL_NUMERIC_PREFIXES = ("threat", "ball_speed", "zone", "event")


def _blend(parts: list[tuple[float | None, float]]) -> tuple[float | None, float]:
    """带权融合，跳过 None 分量并返回剩余权重。"""
    valid = [(v, w) for v, w in parts if v is not None]
    if not valid:
        return None, 0.0
    total_w = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / total_w, total_w


def score_run(case, run: dict, stats, cfg: EvalConfig,
              judge_scores: dict | None) -> dict[str, Any]:
    """单次 run 的全指标评分。run 为 runs/.../result.json 内容。"""
    answer = run.get("answer") or ""

    def _is_tactical(expr: str) -> bool:
        return any(expr.startswith(p) for p in TACTICAL_NUMERIC_PREFIXES)

    # 数值金标准拆分：战术类（威胁/球速/区域数值）与常规类
    acc_numeric = [{"expr": g.expr, "value": g.value, "tolerance": g.tolerance}
                   for g in case.numeric
                   if isinstance(g.value, (int, float)) and not _is_tactical(g.expr)]
    tactical_numeric = [{"expr": g.expr, "value": g.value, "tolerance": g.tolerance}
                        for g in case.numeric
                        if isinstance(g.value, (int, float)) and _is_tactical(g.expr)]
    gold: dict[str, Any] = {
        "numeric": acc_numeric,
        "winner": [{"expr": g.expr, "value": g.value}
                   for g in case.numeric if isinstance(g.value, str)],
        "direction": [{"expr": g.expr, "keywords_pos": g.keywords_pos,
                       "keywords_neg": g.keywords_neg} for g in case.direction],
        "refusal": ({"keywords": case.refusal.keywords} if case.refusal else None),
    }

    prog_acc, acc_detail = M.programmatic_accuracy(answer, gold)
    ent_score, ent_detail = M.entity_grounding(answer, stats, case.input)
    checklist_score, checklist_detail = M.score_checklist(
        answer, [{"text": i.text, "keywords": i.keywords} for i in case.checklist],
    )
    robust, robust_detail = M.score_robustness(
        answer, run.get("error"), refusal_expected=bool(case.refusal),
        refusal_keywords=(case.refusal.keywords if case.refusal else None),
    )
    adv_resist = M.adversarial_resistance(
        answer, (case.refusal.keywords if case.refusal else None),
        adversarial=bool(case.refusal),
    )
    collab_diag = M.collaboration_diag(answer)
    eff, eff_detail = M.score_efficiency(
        run.get("latency_ms"), run.get("llm_calls"),
        (run.get("tokens") or {}).get("total_tokens"), cfg.refs,
    )
    js = judge_scores or {}
    judge_ok = "judge_failed" not in js

    # 战术指标（策略化）：Judge 战术判断为主 + 事件检测 F1（程序化，保留）；
    # 区域/类别与威胁/球速数值的字符串/容差扫描降级为诊断（保留在 details）。
    cat_score, cat_detail = M.score_category(
        answer, [{"expr": g.expr, "label": g.label, "aliases": g.aliases}
                 for g in case.category_gold],
    )
    t_num_score, t_num_detail = M.score_numeric(answer, tactical_numeric)
    event_type_gold = next((str(g.value) for g in case.numeric
                            if isinstance(g.value, str) and g.expr == "event_types"), None)
    event_f1, event_detail = M.score_event_f1(answer, event_type_gold)
    tact_parts: list[tuple[float | None, float]] = []
    if judge_ok and js.get("tactical") is not None:
        tact_parts.append((js["tactical"], 0.6))
    if event_f1 is not None:
        tact_parts.append((event_f1, 0.4))
    tact_accuracy, _ = _blend(tact_parts)

    # 语义对齐（混合软校验）：accuracy 由 Judge 语义比对为主；
    # 程序化数值/胜者/方向扫描降级为诊断（保留在 details，不计入 accuracy）。
    accuracy = js.get("accuracy") if (judge_ok and js.get("accuracy") is not None) else None
    completeness, _ = _blend([
        (checklist_score, 0.5),
        (js.get("completeness") if judge_ok else None, 0.5),
    ])
    grounding, _ = _blend([
        (ent_score, 0.5),
        ((1.0 - js["fabrication"]) if (judge_ok and "fabrication" in js) else None, 0.5),
    ])
    reasoning = (js.get("reasoning") / 5.0) if (judge_ok and js.get("reasoning") is not None) else None

    scores: dict[str, float | None] = {
        "accuracy": accuracy, "grounding": grounding,
        "completeness": completeness, "reasoning": reasoning,
        "efficiency": eff, "robustness": robust,
        "tactical_accuracy": tact_accuracy,
    }
    # _blend 已对可用分量重归一化（除以可用权重和），此处只需检查是否缺分量
    total, used_w = _blend([(scores[k], cfg.weights[k]) for k in scores])
    note = "" if used_w > 1.0 - 1e-9 else JUDGE_MISSING_NOTE

    return {
        "system": run["system"], "case_id": case.id, "category": case.category,
        "repeat": run.get("repeat"), "level": case.level,
        "scores": {k: (round(v, 4) if v is not None else None)
                   for k, v in scores.items()},
        "total": round(total, 4) if total is not None else None,
        "judge_failure_mode": js.get("failure_mode") if judge_ok else None,
        # 诊断性附加字段（不影响 scores/total 计算）：程序化 accuracy（现仅写入
        # details.program_accuracy_score）与 Judge accuracy 的交叉对照，
        # 供 results.json 的 judge_disagreements 汇总定位 Judge 可疑用例。
        "judge_program_agreement": judge_program_agreement(prog_acc, accuracy),
        "adversarial_resistance": adv_resist,
        "collaboration": collab_diag,
        "details": {
            "program_accuracy": acc_detail,
            "program_accuracy_score": prog_acc,
            "entity_grounding": ent_detail,
            "checklist": checklist_detail, "robustness": robust_detail,
            "efficiency": eff_detail, "tactical": {
                "category": cat_detail, "tactical_numeric": t_num_detail,
                "event_f1": event_detail,
            },
        },
        "judge_note": note,
    }


def judge_program_agreement(program_score: float | None,
                            judge_accuracy: float | None) -> dict[str, Any]:
    """程序化 accuracy 与 Judge accuracy 的分歧诊断（仅诊断，不参与计分）。

    两侧均已算出时比较差值：> JUDGE_PROGRAM_DISAGREE_THRESHOLD 记
    flag="disagreement"，否则 "ok"；任一分量缺失（无程序金标准或 Judge
    未评分/失败）记 flag="n/a"。
    """
    if program_score is None or judge_accuracy is None:
        return {"program_accuracy": program_score, "judge_accuracy": judge_accuracy,
                "diff": None, "flag": "n/a"}
    diff = round(abs(program_score - judge_accuracy), 4)
    flag = "disagreement" if diff > JUDGE_PROGRAM_DISAGREE_THRESHOLD else "ok"
    return {"program_accuracy": program_score, "judge_accuracy": judge_accuracy,
            "diff": diff, "flag": flag}


def extended_total(score_row: dict, tool_diag: dict) -> float | None:
    """扩展总分：tool_use 以固定权重并入，其余等比缩放。"""
    base = score_row.get("total")
    tu = tool_diag.get("score")
    if base is None or tu is None:
        return None
    return round(base * (1 - EXTENDED_TOOL_WEIGHT) + tu * EXTENDED_TOOL_WEIGHT, 4)
