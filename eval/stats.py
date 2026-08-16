"""统计分析 —— 两级置信区间、配对检验、效应量、多重比较校正。

层级（评审修订 #7）：
- 用例级：case × system × metric 跨重复的均值/标准差（描述重复方差）；
- 系统级：跨用例 bootstrap CI（对用例重采样）—— 对比结论的依据。

检验（主检验按用例配对，n=用例数；评审修订 #6）：
- Wilcoxon 符号秩（小样本稳健，主检验）+ 配对 t（正态假设，参考）；
- 效应量：paired Cohen's d + Cliff's delta；
- 多指标家族 Holm 校正；
- 检验力说明：power_note() 给出 n 与 α 下的可检效应量参考。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as sps

from .config import SCORED_METRICS

BOOTSTRAP_B = 2000
ALPHA = 0.05


def describe(values: list[float | None]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0,
    }


def bootstrap_ci(values: list[float], b: int = BOOTSTRAP_B,
                 alpha: float = ALPHA, seed: int = 42) -> tuple[float, float] | None:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.array([
        arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(b)
    ])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)


def paired_tests(x: list[float], y: list[float]) -> dict[str, Any]:
    """x - y 配对（同用例）；返回 Wilcoxon / t / 效应量。"""
    xa = np.array([v for v in x if v is not None], dtype=float)
    ya = np.array([v for v in y if v is not None], dtype=float)
    n = min(xa.size, ya.size)
    if n < 3:
        return {"n": n, "note": "配对样本不足（<3），无法检验"}
    xa, ya, diff = xa[:n], ya[:n], xa[:n] - ya[:n]
    out: dict[str, Any] = {"n": int(n), "mean_diff": round(float(diff.mean()), 4)}

    nz = diff[diff != 0]
    if nz.size >= 3:
        try:
            w = sps.wilcoxon(xa, ya, zero_method="wilcox", alternative="two-sided",
                             method="approx")
            out["wilcoxon_p"] = round(float(w.pvalue), 5)
        except ValueError:
            out["wilcoxon_p"] = None
    else:
        out["wilcoxon_p"] = None
    t = sps.ttest_rel(xa, ya)
    out["ttest_p"] = round(float(t.pvalue), 5)

    sd = diff.std(ddof=1)
    out["cohens_d_paired"] = round(float(diff.mean() / sd), 3) if sd > 1e-12 else None
    # Cliff's delta（配对数据按非严格配对比较）
    gt = sum(1 for xi in xa for yi in ya if xi > yi)
    lt = sum(1 for xi in xa for yi in ya if xi < yi)
    out["cliffs_delta"] = round(float((gt - lt) / (n * n)), 3)
    return out


def holm_correction(p_values: dict[str, float | None]) -> dict[str, float | None]:
    """Holm-Bonferroni step-down 校正，返回调整后 p。"""
    valid = {k: p for k, p in p_values.items() if p is not None}
    if not valid:
        return {k: None for k in p_values}
    items = sorted(valid.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        adjusted[key] = round(running, 5)
    return {k: adjusted.get(k) for k in p_values}


def power_note(n: int, alpha: float = ALPHA) -> str:
    """检验力说明（近似）：配对 t 在 80% power 下可检的效应量。"""
    if n < 3:
        return "样本不足，无法给出检验力参考"
    z_a = sps.norm.ppf(1 - alpha / 2)
    z_b = sps.norm.ppf(0.80)
    d_detect = (z_a + z_b) / math.sqrt(n)
    return (
        f"n={n}（配对用例数），α={alpha}，配对 t 检验在 80% power 下约可检出 "
        f"|d|≥{d_detect:.2f} 的效应；更小差异仅作描述性参考。"
    )


def _mean_or_none(values: list | None) -> float | None:
    if not values:
        return None
    cleaned = [v for v in values if v is not None]
    return float(np.mean(cleaned)) if cleaned else None


def aggregate(per_case_scores: dict[str, dict[str, dict]]) -> dict[str, Any]:
    """聚合入口。

    per_case_scores: {case_id: {system: {
        "total": [每重复总分], "extended_total": [...],
        "metrics": {metric: [每重复得分]},
        "stability": float | None,            # 每用例单值
        "tool_use": [每重复工具得分 | None],
    }}}
    返回：系统级总分/分项（均值、跨用例 bootstrap CI）、两两配对检验（Holm 校正）。
    """
    systems: list[str] = sorted({s for case in per_case_scores.values() for s in case})
    metrics_all: list[str] = list(SCORED_METRICS)

    def case_metric(case_sys: dict, metric: str) -> list | None:
        if metric == "total":
            return case_sys.get("total")
        return (case_sys.get("metrics") or {}).get(metric)

    sys_scores: dict[str, dict[str, Any]] = {}
    for sys_name in systems:
        totals = [_mean_or_none(case[sys_name].get("total"))
                  for case in per_case_scores.values() if sys_name in case]
        totals = [v for v in totals if v is not None]
        entry: dict[str, Any] = {
            "n_cases": len(totals),
            "total_mean": round(float(np.mean(totals)), 4) if totals else None,
            "total_ci95": bootstrap_ci(totals),
            "metrics": {},
        }
        for metric in metrics_all:
            vals = [_mean_or_none(case_metric(case[sys_name], metric))
                    for case in per_case_scores.values()
                    if sys_name in case]
            vals = [v for v in vals if v is not None]
            entry["metrics"][metric] = {**describe(vals), "ci95": bootstrap_ci(vals)}
        stab = [case[sys_name].get("stability")
                for case in per_case_scores.values()
                if sys_name in case and case[sys_name].get("stability") is not None]
        entry["stability"] = describe(stab)
        tu = [_mean_or_none(case[sys_name].get("tool_use"))
              for case in per_case_scores.values() if sys_name in case]
        tu = [v for v in tu if v is not None]
        entry["tool_use"] = describe(tu)
        ext = [_mean_or_none(case[sys_name].get("extended_total"))
               for case in per_case_scores.values() if sys_name in case]
        ext = [v for v in ext if v is not None]
        entry["extended_total_mean"] = round(float(np.mean(ext)), 4) if ext else None
        entry["extended_total_ci95"] = bootstrap_ci(ext)
        sys_scores[sys_name] = entry

    # 两两配对检验（总分 + 各分项，Holm 校正）
    comparisons: dict[str, dict] = {}
    raw_p: dict[str, float | None] = {}
    for i, s1 in enumerate(systems):
        for s2 in systems[i + 1:]:
            pair_key = f"{s1}_vs_{s2}"
            comp: dict[str, Any] = {}
            for metric in ["total"] + metrics_all:
                x, y = [], []
                for case in per_case_scores.values():
                    if s1 in case and s2 in case:
                        v1 = _mean_or_none(case_metric(case[s1], metric))
                        v2 = _mean_or_none(case_metric(case[s2], metric))
                        if v1 is not None and v2 is not None:
                            x.append(v1)
                            y.append(v2)
                comp[metric] = paired_tests(x, y)
                raw_p[f"{pair_key}:{metric}"] = comp[metric].get("wilcoxon_p")
            comparisons[pair_key] = comp
    adjusted = holm_correction(raw_p)
    for key, p in adjusted.items():
        pair_key, metric = key.split(":", 1)
        comparisons[pair_key][metric]["wilcoxon_p_holm"] = p

    return {
        "systems": sys_scores,
        "comparisons": comparisons,
        "power_note_total": power_note(len(per_case_scores)),
        "n_cases": len(per_case_scores),
    }
