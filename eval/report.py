"""报告渲染 —— Markdown + Matplotlib 图 + 自包含 HTML。

- Markdown：与 results.json 同构的静态报告，便于归档 docs/；
- figures/：总分条形+CI、指标雷达、分指标箱线、类别热力、成本、上下文演变；
- HTML：单文件自包含（图 base64 内嵌 + 原生 <details> 交互 +
  三轨步骤对齐时间线 + 上下文曲线 SVG），离线可打开。
"""

from __future__ import annotations

import base64
import html as H
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SCORED_METRICS, SYSTEM_LABELS
from .judge import RUBRIC_VERSION
from .tracealign import PHASE_LABELS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SYS_COLORS = {"bare": "#4C78A8", "single": "#F58518", "multi": "#54A24B"}
STEP_COLORS = {"llm": "#4C78A8", "tool": "#F58518", "stage": "#72B7B2",
               "load": "#B279A2", "turn": "#999999"}


def _load_metrics_rows(event_dir: Path) -> list[dict]:
    path = event_dir / "metrics.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _load_meta(event_dir: Path) -> dict:
    path = event_dir / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ── Matplotlib 图 ───────────────────────────────────────────

def make_figures(event_dir: Path, results: dict) -> list[Path]:
    figs: list[Path] = []
    figdir = event_dir / "figures"
    figdir.mkdir(exist_ok=True)
    systems = list(results["aggregate"]["systems"].keys())
    rows = _load_metrics_rows(event_dir)

    # 1. 总分条形 + CI
    fig, ax = plt.subplots(figsize=(6, 4.2))
    means, errs = [], []
    for s in systems:
        m = results["aggregate"]["systems"][s]["total_mean"] or 0
        ci = results["aggregate"]["systems"][s]["total_ci95"] or (m, m)
        means.append(m)
        errs.append([max(0, m - ci[0]), max(0, ci[1] - m)])
    bars = ax.bar([SYSTEM_LABELS.get(s, s) for s in systems], means,
                  color=[SYS_COLORS.get(s, "#666") for s in systems], width=0.5)
    ax.errorbar(range(len(systems)), means,
                yerr=np.array(errs).T if errs else None, fmt="none",
                ecolor="#222", capsize=5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("共同总分（跨用例均值 ± 95% bootstrap CI）")
    ax.set_title("系统总分对比")
    fig.tight_layout()
    fig.savefig(figdir / "fig_total.png", dpi=150)
    plt.close(fig)
    figs.append(figdir / "fig_total.png")

    # 2. 指标雷达
    metrics_ok = [m for m in SCORED_METRICS
                  if all(results["aggregate"]["systems"][s]["metrics"][m]["mean"] is not None
                         for s in systems)]
    if len(metrics_ok) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(metrics_ok), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(5.6, 5.2), subplot_kw={"polar": True})
        for s in systems:
            vals = [results["aggregate"]["systems"][s]["metrics"][m]["mean"] or 0
                    for m in metrics_ok]
            vals += vals[:1]
            ax.plot(angles, vals, color=SYS_COLORS.get(s, "#666"),
                    label=SYSTEM_LABELS.get(s, s), linewidth=2)
            ax.fill(angles, vals, color=SYS_COLORS.get(s, "#666"), alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_ok)
        ax.set_ylim(0, 1)
        ax.set_title("分指标雷达")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08), fontsize=9)
        fig.tight_layout()
        fig.savefig(figdir / "fig_radar.png", dpi=150)
        plt.close(fig)
        figs.append(figdir / "fig_radar.png")

    # 3. 分指标箱线（跨用例分布，动态网格适配指标数）
    if rows:
        n_metrics = len(SCORED_METRICS)
        ncols, nrows = (3, (n_metrics + 2) // 3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.1 * nrows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        for mi, metric in enumerate(SCORED_METRICS):
            ax = axes[mi]
            data, labels, colors = [], [], []
            for s in systems:
                vals = [r["scores"][metric] for r in rows
                        if r["system"] == s and r["scores"].get(metric) is not None]
                if vals:
                    data.append(vals)
                    labels.append(SYSTEM_LABELS.get(s, s))
                    colors.append(SYS_COLORS.get(s, "#666"))
            bp = ax.boxplot(data, patch_artist=True, tick_labels=labels) if data else None
            if bp:
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.6)
            ax.set_title(metric)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(axis="x", labelsize=8)
        for extra in axes[n_metrics:]:
            extra.set_visible(False)
        fig.suptitle("分指标得分分布（每点 = 一次运行）")
        fig.tight_layout()
        fig.savefig(figdir / "fig_box.png", dpi=150)
        plt.close(fig)
        figs.append(figdir / "fig_box.png")

    # 4. 类别 × 系统热力
    cat_data = results.get("category_breakdown", {})
    if cat_data:
        cats = sorted(cat_data)
        mat = np.full((len(cats), len(systems)), np.nan)
        for ci, cat in enumerate(cats):
            for si, s in enumerate(systems):
                v = cat_data[cat].get(s)
                if v is not None:
                    mat[ci, si] = v
        fig, ax = plt.subplots(figsize=(5.2, 0.5 * len(cats) + 2))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([SYSTEM_LABELS.get(s, s) for s in systems], fontsize=9)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=9)
        for ci in range(len(cats)):
            for si in range(len(systems)):
                if not np.isnan(mat[ci, si]):
                    ax.text(si, ci, f"{mat[ci, si]:.2f}", ha="center",
                            va="center", fontsize=8)
        fig.colorbar(im, shrink=0.8)
        ax.set_title("类别 × 系统总分热力")
        fig.tight_layout()
        fig.savefig(figdir / "fig_category.png", dpi=150)
        plt.close(fig)
        figs.append(figdir / "fig_category.png")

    # 5. 成本对比
    cost = results.get("cost", {})
    if cost:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for ai, (key, title) in enumerate((
            ("latency_ms", "时延 (ms)",), ("llm_calls", "LLM 调用数"),
            ("total_tokens", "总 token"),
        )):
            vals = [cost.get(s, {}).get(key, (None,))[0] for s in systems]
            axes[ai].bar([SYSTEM_LABELS.get(s, s) for s in systems],
                         [v or 0 for v in vals],
                         color=[SYS_COLORS.get(s, "#666") for s in systems])
            axes[ai].set_title(title)
            axes[ai].tick_params(axis="x", labelsize=8)
        fig.suptitle("成本对比（均值）")
        fig.tight_layout()
        fig.savefig(figdir / "fig_cost.png", dpi=150)
        plt.close(fig)
        figs.append(figdir / "fig_cost.png")

    # 6. 上下文演变（对齐用例）
    align = results.get("alignment") or {}
    curves = {
        s: [st.get("context_chars", 0) for st in d.get("steps", [])]
        for s, d in align.get("systems", {}).items()
    }
    if any(curves.values()):
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for s, ys in curves.items():
            if ys:
                ax.plot(range(len(ys)), ys, marker="o", markersize=3,
                        color=SYS_COLORS.get(s, "#666"),
                        label=SYSTEM_LABELS.get(s, s))
        ax.set_xlabel("步骤序号")
        ax.set_ylabel("累计上下文字符数")
        ax.set_title(f"上下文演变（用例 {results.get('alignment_case', '?')}，启发式对齐）")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(figdir / "fig_context.png", dpi=150)
        plt.close(fig)
        figs.append(figdir / "fig_context.png")

    return figs


# ── Markdown ────────────────────────────────────────────────

def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"


# ── 同源风险披露与 Judge 一致性（可信度增量，仅披露不改分） ──

# Judge 一致性告警阈值：正逆序 accuracy 差超过该值的用例标黄/加告警标记
JUDGE_CONSISTENCY_WARN = 0.15


def _vendor_prefix(model: str) -> str:
    """模型名前缀 → 厂商族粗判（如 deepseek-v4-pro → deepseek）。

    仅作同源风险的粗粒度披露；同族不同模型仍属同厂商，如需精确结论
    请人工核对厂商归属（可通过部署配置自行补充说明）。
    """
    name = (model or "").lower().split("/")[-1].strip()
    for sep in ("-", "_", ":"):
        if sep in name:
            name = name.split(sep, 1)[0]
    return name or "?"


def _judge_disclosure_lines(meta: dict) -> list[str]:
    """报告头部固定披露段：judge_model / rubric 版本 / 同厂商判定 / 风险声明。"""
    jcfg = (meta.get("config", {}) or {}).get("judge", {}) or {}
    judge_model = str(jcfg.get("model") or "(same-as-system)")
    import os

    # 机器口径与展示口径分离：同源判定用真实生效的模型名（环境变量
    # 未配置时即默认模型 deepseek-v4-flash），展示仍用带说明的兜底文案；
    # judge_model 为 "(same-as-system)"（engine 未单独配置 judge）时直接判同厂商。
    effective_model = os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash"
    sys_model_display = os.getenv("DEEPSEEK_MODEL") or "(未配置，默认 deepseek-v4-flash)"
    same = judge_model == "(same-as-system)" or (
        _vendor_prefix(judge_model) == _vendor_prefix(effective_model)
    )
    return [
        f"- 【同源风险披露】Judge 模型: {judge_model}；rubric 版本: {RUBRIC_VERSION}；"
        f"被测系统模型: {sys_model_display}；Judge 与被测系统"
        f"{'同厂商（模型名前缀粗判: ' + _vendor_prefix(effective_model) + '）' if same else '非同厂商（模型名前缀粗判）'}。"
        "同源 Judge 存在自我偏好风险，结论需人工抽检。"
    ]


def _judge_consistency_rows(results: dict) -> list[tuple[str, str, float | None]]:
    """从 results.judge_meta 提取每用例每系统的正逆序 accuracy 差。"""
    rows: list[tuple[str, str, float | None]] = []
    for cid, m in (results.get("judge_meta") or {}).items():
        cons = (m or {}).get("consistency") or {}
        if not cons:
            continue
        for sys_name, diff in cons.items():
            rows.append((cid, sys_name, diff))
    return rows


def _level_breakdown(results: dict, meta: dict, systems: list[str]) -> list[list]:
    """按业务分层 L1-L7 聚合共同总分（3.3：可解释性分层展示）。"""
    from eval.cases import LEVEL_NAMES

    level_of = {c["id"]: int(c.get("level", 1)) for c in meta.get("cases", [])}
    pcs = results.get("per_case_summary", {})
    grouped: dict[int, dict[str, list[float]]] = {}
    for cid, per_sys in pcs.items():
        lv = level_of.get(cid, 1)
        for s, d in per_sys.items():
            if d.get("total") is not None:
                grouped.setdefault(lv, {}).setdefault(s, []).append(d["total"])
    rows: list[list] = []
    for lv in sorted(grouped):
        cells: list[float] = [lv, len(grouped[lv])]
        counts = [len(grouped[lv][s]) for s in systems if s in grouped[lv]]
        if counts:
            cells[1] = max(counts)
        for s in systems:
            vals = grouped[lv].get(s, [])
            cells.append(round(sum(vals) / len(vals), 3) if vals else "—")
        rows.append(cells)
    return rows


def render_markdown(event_dir: Path, results: dict) -> str:
    meta = _load_meta(event_dir)
    agg = results["aggregate"]
    systems = list(agg["systems"].keys())
    L: list[str] = []

    L.append(f"# 评测报告：单 LLM 直接调用 vs 多智能体系统\n")
    L.append(f"- 事件目录: `{results['event_dir']}`")
    L.append(f"- 用例数: {results['n_cases']}，重复: {meta.get('config', {}).get('repeats')}，"
             f"采样温度: {meta.get('config', {}).get('temperature')}")
    L.append(f"- 模型: {meta.get('config', {}).get('judge', {}).get('model', '')}；"
             f"git: `{meta.get('git_commit', '')[:8]}`")
    L.append(f"- 权重: {meta.get('config', {}).get('weights')}")
    L.append(f"- Judge: {meta.get('config', {}).get('judge')}")
    L.extend(_judge_disclosure_lines(meta))
    L.append("")

    L.append("## 1. 总分\n")
    L.append("| 系统 | 共同总分 | 95% CI | 扩展总分(+工具) | n |")
    L.append("|---|---|---|---|---|")
    for s in systems:
        d = agg["systems"][s]
        L.append(f"| {SYSTEM_LABELS.get(s, s)} | {d['total_mean']:.3f} "
                 f"| {_fmt_ci(d['total_ci95'])} | "
                 f"{d['extended_total_mean'] if d['extended_total_mean'] is None else round(d['extended_total_mean'], 3)} "
                 f"| {d['n_cases']} |")
    L.append("")

    # 1b. 按业务分层总分（3.3 可解释性）
    level_rows = _level_breakdown(results, meta, systems)
    if level_rows:
        L.append("### 1b. 按业务分层（L1-L7）总分\n")
        L.append("| 层 | 用例数 | " + " | ".join(SYSTEM_LABELS.get(s, s) for s in systems) + " |")
        L.append("|" + "---|" * (len(systems) + 2))
        for row in level_rows:
            lv, n, *cells = row
            L.append(f"| L{lv} | {n} | " + " | ".join(str(c) for c in cells) + " |")
        L.append("")

    L.append("## 2. 分指标\n")
    header = "| 指标 | " + " | ".join(SYSTEM_LABELS.get(s, s) for s in systems) + " |"
    L.append(header)
    L.append("|---" * (len(systems) + 1) + "|")
    for m in SCORED_METRICS:
        cells = []
        for s in systems:
            d = agg["systems"][s]["metrics"][m]
            cells.append(f"{d['mean']:.3f} ± {_fmt_ci(d['ci95'])}" if d["mean"] is not None else "—")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 3. 配对显著性检验（按用例配对，Wilcoxon 主检验）\n")
    L.append(agg.get("power_note_total", ""))
    L.append("")
    for pair, comp in agg.get("comparisons", {}).items():
        s1, s2 = pair.split("_vs_")
        c = comp.get("total", {})
        L.append(f"### {SYSTEM_LABELS.get(s1, s1)} vs {SYSTEM_LABELS.get(s2, s2)}\n")
        L.append(f"- 均值差: {c.get('mean_diff')}，paired Cohen's d: {c.get('cohens_d_paired')}，"
                 f"Cliff's δ: {c.get('cliffs_delta')}")
        L.append(f"- Wilcoxon p={c.get('wilcoxon_p')}（Holm 校正后 p={c.get('wilcoxon_p_holm')}），"
                 f"配对 t p={c.get('ttest_p')}")
        L.append("")
        L.append("| 指标 | p (Wilcoxon) | p (Holm) | d | δ |")
        L.append("|---|---|---|---|---|")
        for m in SCORED_METRICS:
            cm = comp.get(m, {})
            L.append(f"| {m} | {cm.get('wilcoxon_p')} | {cm.get('wilcoxon_p_holm')} "
                     f"| {cm.get('cohens_d_paired')} | {cm.get('cliffs_delta')} |")
        L.append("")

    L.append("## 4. 类别 × 系统热力\n")
    cat = results.get("category_breakdown", {})
    if cat:
        L.append("| 类别 | " + " | ".join(SYSTEM_LABELS.get(s, s) for s in systems) + " |")
        L.append("|---" * (len(systems) + 1) + "|")
        for c in sorted(cat):
            L.append(f"| {c} | " + " | ".join(
                (f"{cat[c].get(s):.3f}" if cat[c].get(s) is not None else "—")
                for s in systems) + " |")
        L.append("")

    L.append("## 5. 诊断指标\n")
    L.append("| 系统 | 稳定性(跨重复相似度) | 工具使用得分 | 工具调用数 |")
    L.append("|---|---|---|---|")
    for s in systems:
        d = agg["systems"][s]
        st_ = d.get("stability", {})
        tu = d.get("tool_use", {})
        L.append(f"| {SYSTEM_LABELS.get(s, s)} | "
                 f"{st_.get('mean') if st_.get('mean') is None else round(st_['mean'], 3)} "
                 f"| {tu.get('mean') if tu.get('mean') is None else round(tu['mean'], 3)} "
                 f"| {tu.get('n', 0)} |")
    L.append("")

    L.append("## 6. 失败模式（Judge 标注 top-3）\n")
    fm = results.get("failure_modes", {})
    if fm:
        for mode, info in fm.items():
            L.append(f"- **{mode}** ×{info['count']}（样例: {', '.join(info['samples'])}）")
    else:
        L.append("- 无显著失败模式")
    jd = results.get("judge_disagreements") or {}
    if jd.get("top"):
        L.append("")
        L.append("Judge-程序分歧 top（详见 results.json judge_disagreements）:")
        for item in jd["top"]:
            L.append(f"- {item.get('system')}/{item.get('case_id')}"
                     f"#r{item.get('repeat')}：程序 {item.get('program_accuracy')}"
                     f" vs Judge {item.get('judge_accuracy')}（差 {item.get('diff')}）")
    L.append("")

    # 6b. Judge 一致性（位置互换正逆序差，仅告警不改分）
    cons_rows = _judge_consistency_rows(results)
    if cons_rows:
        L.append("## 6b. Judge 一致性（位置互换正逆序差）\n")
        L.append(f"阈值：正逆序差 > {JUDGE_CONSISTENCY_WARN} 的用例标 ⚠，建议人工抽检（仅告警不改分）。")
        L.append("")
        L.append("| 用例 | 系统 | 正逆序差 | 告警 |")
        L.append("|---|---|---|---|")
        for cid, sys_name, diff in cons_rows:
            warn = diff is not None and diff > JUDGE_CONSISTENCY_WARN
            mark = "⚠" if warn else ""
            diff_txt = f"{diff:.3f}" if diff is not None else "—"
            L.append(f"| {cid} | {SYSTEM_LABELS.get(sys_name, sys_name)} "
                     f"| {diff_txt} | {mark} |")
        flagged = sorted({cid for cid, _s, d in cons_rows
                          if d is not None and d > JUDGE_CONSISTENCY_WARN})
        L.append("")
        if flagged:
            L.append(f"⚠ 一致性告警用例（{len(flagged)}）: {', '.join(flagged)}")
        else:
            L.append("全部用例正逆序差在阈值内。")
        L.append("")

    L.append("## 7. 成本\n")
    cost = results.get("cost", {})
    L.append("| 系统 | 时延 ms | LLM 调用 | 总 token |")
    L.append("|---|---|---|---|")
    for s in systems:
        c = cost.get(s, {})
        L.append(f"| {SYSTEM_LABELS.get(s, s)} | {c.get('latency_ms', ('—',))[0]} "
                 f"| {c.get('llm_calls', ('—',))[0]} | {c.get('total_tokens', ('—',))[0]} |")
    L.append("")

    L.append("## 8. 方法与局限说明\n")
    L.append("- 共同总分 = Σ 权重×分项；效率按预标定常数归一化；缺失分量自动重归一化。")
    L.append("- 系统级 CI 为跨用例 bootstrap（B=2000）；显著性按用例配对（Wilcoxon），"
             "Judge 主观分指标有族内相关，Holm 校正仅作参考下界。")
    L.append("- Judge 模型（" + str(meta.get("config", {}).get("judge", {}).get("model"))
             + "）与被测系统解耦，采用盲评+位置互换缓解偏好，结论需人工抽检。")
    L.append("- 数值金标准由 pandas 独立重算（不经被测系统工具层，单位米制）；反事实类因 MCTS "
             "随机性仅用方向性金标准。")
    L.append("- 阶段对齐为启发式（llm 序号/工具属性推断），时间线仅供定性比较。")
    return "\n".join(L)


# ── 自包含 HTML ─────────────────────────────────────────────

def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _timeline_html(align: dict) -> str:
    if not align or not align.get("systems"):
        return "<p>（无可用 trace）</p>"
    parts = ['<div class="tl-note">⚠ 阶段为启发式对齐（由 llm 序号与工具属性推断），'
             '仅供定性比较</div>']
    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{k}</span>'
        for k, c in STEP_COLORS.items())
    parts.append(f'<div class="legend">{legend}</div>')
    for s, data in align["systems"].items():
        steps = data.get("steps", [])
        total_dur = sum(st.get("dur_ms") or 0 for st in steps) or 1
        segs = []
        for st in steps:
            w = max(0.6, (st.get("dur_ms") or 0) / total_dur * 100)
            color = STEP_COLORS.get(st["kind"], "#999")
            tip = H.escape(f"{st['kind']}:{st['name']} | {st.get('phase','')} | "
                           f"{st.get('dur_ms')}ms | ctx={st.get('context_chars')}")
            segs.append(f'<div class="seg" style="width:{w:.2f}%;background:{color}" '
                        f'title="{tip}"></div>')
        parts.append(
            f'<div class="tl-row"><div class="tl-label">{H.escape(SYSTEM_LABELS.get(s, s))}'
            f'<br><small>{len(steps)}步 · {data.get("llm_calls", "?")}次LLM</small></div>'
            f'<div class="tl-track">{"".join(segs)}</div></div>')
    return "".join(parts)


def _context_svg(align: dict) -> str:
    curves = {
        s: [st.get("context_chars", 0) for st in d.get("steps", [])]
        for s, d in (align.get("systems") or {}).items()
    }
    live = {s: ys for s, ys in curves.items() if ys}
    if not live:
        return ""
    w, h, pad = 560, 160, 24
    max_len = max(len(ys) for ys in live.values())
    max_v = max(max(ys) for ys in live.values()) or 1
    lines = []
    for s, ys in live.items():
        pts = " ".join(
            f"{pad + i / max(max_len - 1, 1) * (w - 2 * pad):.1f},"
            f"{h - pad - v / max_v * (h - 2 * pad):.1f}"
            for i, v in enumerate(ys))
        lines.append(f'<polyline fill="none" stroke="{SYS_COLORS.get(s, "#666")}" '
                     f'stroke-width="2" points="{pts}"/>')
    return (f'<svg width="{w}" height="{h}" style="background:#fafafa">'
            f'{"".join(lines)}</svg>')


def _judge_consistency_html(results: dict) -> str:
    """Judge 一致性表（HTML）：正逆序差 > 阈值行标黄加 ⚠（仅告警不改分）。"""
    rows = _judge_consistency_rows(results)
    if not rows:
        return ("<p class='sub'>（本次事件无 Judge 位置互换一致性数据，"
                "如 Judge 未启用或未开启 order_swap）</p>")
    parts = [f'<p class="sub">位置互换正逆序 accuracy 差；'
             f'差 &gt; {JUDGE_CONSISTENCY_WARN} 标黄加 ⚠，建议人工抽检（仅告警不改分）。</p>',
             '<table><tr><th>用例</th><th>系统</th><th>正逆序差</th><th>告警</th></tr>']
    for cid, sys_name, diff in rows:
        warn = diff is not None and diff > JUDGE_CONSISTENCY_WARN
        diff_txt = f"{diff:.3f}" if diff is not None else "—"
        cls = ' class="warn"' if warn else ""
        parts.append(
            f"<tr{cls}><td>{H.escape(cid)}</td>"
            f"<td>{H.escape(SYSTEM_LABELS.get(sys_name, sys_name))}</td>"
            f"<td>{diff_txt}</td><td>{'⚠' if warn else ''}</td></tr>")
    parts.append("</table>")
    return "".join(parts)


def render_html(event_dir: Path, results: dict) -> str:
    agg = results["aggregate"]
    systems = list(agg["systems"].keys())
    figs = sorted((event_dir / "figures").glob("*.png"))
    img_tags = "".join(
        f'<figure><img src="data:image/png;base64,{_b64(f)}" alt="{f.name}"/>'
        f"<figcaption>{f.stem}</figcaption></figure>" for f in figs)

    cards = []
    for s in systems:
        d = agg["systems"][s]
        ci = d["total_ci95"]
        ci_txt = f"{ci[0]:.3f}–{ci[1]:.3f}" if ci else "—"
        cards.append(
            f'<div class="card" style="border-top:4px solid {SYS_COLORS.get(s, "#666")}">'
            f'<h3>{H.escape(SYSTEM_LABELS.get(s, s))}</h3>'
            f'<div class="big">{d["total_mean"]:.3f}</div>'
            f'<div class="sub">95% CI {ci_txt}</div>'
            f'<div class="sub">扩展总分 '
            f'{d["extended_total_mean"] if d["extended_total_mean"] is None else round(d["extended_total_mean"], 3)}</div>'
            f"</div>")

    rows_html = []
    for m in SCORED_METRICS:
        cells = []
        for s in systems:
            d = agg["systems"][s]["metrics"][m]
            cells.append(f"{d['mean']:.3f}" if d["mean"] is not None else "—")
        rows_html.append(f"<tr><td>{m}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    metric_table = ("<table><tr><th>指标</th>" +
                    "".join(f"<th>{SYSTEM_LABELS.get(s, s)}</th>" for s in systems) +
                    "</tr>" + "".join(rows_html) + "</table>")

    comp_html = []
    for pair, comp in agg.get("comparisons", {}).items():
        s1, s2 = pair.split("_vs_")
        c = comp.get("total", {})
        comp_html.append(
            f"<tr><td>{SYSTEM_LABELS.get(s1, s1)} vs {SYSTEM_LABELS.get(s2, s2)}</td>"
            f"<td>{c.get('mean_diff')}</td><td>{c.get('wilcoxon_p')}</td>"
            f"<td>{c.get('wilcoxon_p_holm')}</td><td>{c.get('cohens_d_paired')}</td>"
            f"<td>{c.get('cliffs_delta')}</td></tr>")
    comp_table = ("<table><tr><th>配对</th><th>均值差</th><th>Wilcoxon p</th>"
                  "<th>Holm p</th><th>Cohen's d</th><th>Cliff's δ</th></tr>"
                  + "".join(comp_html) + "</table>")

    fm = results.get("failure_modes", {})
    fm_html = "".join(
        f"<li><b>{H.escape(mode)}</b> ×{info['count']}（{', '.join(info['samples'])}）</li>"
        for mode, info in fm.items()) or "<li>无显著失败模式</li>"

    per_case_rows = []
    for cid, per_sys in results.get("per_case_summary", {}).items():
        cells = "".join(
            f"<td>{per_sys[s]['total'] if per_sys.get(s) and per_sys[s]['total'] is not None else '—'}</td>"
            for s in systems)
        stab = next((f"{per_sys[s]['stability']:.3f}" for s in systems
                     if per_sys.get(s) and per_sys[s].get("stability") is not None), "—")
        per_case_rows.append(f"<tr><td>{cid}</td>{cells}<td>{stab}</td></tr>")
    per_case_table = ("<table><tr><th>用例</th>" +
                      "".join(f"<th>{SYSTEM_LABELS.get(s, s)}</th>" for s in systems) +
                      "<th>稳定性(多)</th></tr>" + "".join(per_case_rows) + "</table>")

    align = results.get("alignment") or {}
    align_case = results.get("alignment_case", "")
    ctx_svg = _context_svg(align)

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>评测报告 {H.escape(str(event_dir.name))}</title>
<style>
body{{font-family:"Microsoft YaHei",sans-serif;margin:24px auto;max-width:1080px;
     color:#222;line-height:1.5;padding:0 16px}}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
.card{{flex:1;min-width:220px;background:#f8f9fa;border-radius:8px;padding:14px;
      text-align:center}}
.big{{font-size:2.2em;font-weight:700}}
.sub{{color:#666;font-size:.9em}}
table{{border-collapse:collapse;margin:12px 0;width:100%}}
th,td{{border:1px solid #ddd;padding:6px 10px;font-size:.9em;text-align:center}}
th{{background:#f0f0f0}}
figure{{margin:16px 0;text-align:center}}
figure img{{max-width:100%}}
figcaption{{color:#888;font-size:.85em}}
.tl-row{{display:flex;align-items:center;gap:8px;margin:6px 0}}
.tl-label{{width:130px;font-size:.85em;text-align:right;flex-shrink:0}}
.tl-track{{flex:1;display:flex;height:22px;background:#eee;border-radius:4px;
          overflow:hidden}}
.seg{{height:100%;min-width:2px}}
.legend .lg{{margin-right:12px;font-size:.85em}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;
          margin-right:4px}}
.tl-note{{color:#a05a00;font-size:.85em;margin-bottom:6px}}
.disclosure{{background:#f8f9fa;border-left:4px solid #a05a00;padding:10px 14px;
            font-size:.9em;color:#444;margin:12px 0}}
tr.warn td,td.warn{{background:#fff3cd}}
h2{{border-bottom:2px solid #eee;padding-bottom:6px;margin-top:36px}}
details{{margin:6px 0}} summary{{cursor:pointer;color:#36c}}
</style></head><body>
<h1>评测报告：单 LLM 直接调用 vs 多智能体系统</h1>
<p>事件 <code>{H.escape(str(event_dir.name))}</code> ·
用例 {results['n_cases']} · 重复 {_load_meta(event_dir).get('config', {}).get('repeats')} ·
模型 {H.escape(str(_load_meta(event_dir).get('config', {}).get('judge', {}).get('model', '')))}</p>
<div class="disclosure">{H.escape("".join(_judge_disclosure_lines(_load_meta(event_dir))).lstrip("- "))}</div>
<div class="cards">{"".join(cards)}</div>
{img_tags}
<h2>分指标</h2>{metric_table}
<h2>配对显著性检验</h2>
<p class="sub">{H.escape(agg.get("power_note_total", ""))}</p>
{comp_table}
<h2>类别 × 系统</h2>
{f'<img src="data:image/png;base64,{_b64(event_dir / "figures" / "fig_category.png")}" style="max-width:100%">' if (event_dir / "figures" / "fig_category.png").exists() else "<p class='sub'>（无类别数据）</p>"}
<h2>失败模式（Judge 标注）</h2><ul>{fm_html}</ul>
<h2>Judge 一致性（位置互换正逆序差）</h2>
{_judge_consistency_html(results)}
<h2>逐步骤对齐（用例 {H.escape(align_case)}，重复 #0）</h2>
{_timeline_html(align)}
{ctx_svg and f'<h3>上下文演变</h3>{ctx_svg}'}
<h2>逐用例明细</h2>{per_case_table}
<h2>方法与局限</h2>
<ul>
<li>共同总分 = Σ 权重×分项；效率按预标定常数归一化；Judge 缺失分量自动重归一化。</li>
<li>系统级 CI 为跨用例 bootstrap（B=2000）；Wilcoxon 按用例配对；Holm 校正控制族错误率。</li>
<li>Judge 模型（deepseek-v4-pro）与被测系统解耦（盲评+位置互换已缓解），结论建议人工抽检。</li>
<li>数值金标准由 pandas 独立重算（单位米制）；反事实类因 MCTS 随机性仅用方向性金标准。</li>
<li>阶段对齐为启发式，时间线仅供定性比较。</li>
</ul>
</body></html>"""


def render_all(event_dir: Path, results: dict) -> None:
    make_figures(event_dir, results)
    (event_dir / "report.md").write_text(render_markdown(event_dir, results),
                                         encoding="utf-8")
    (event_dir / "report.html").write_text(render_html(event_dir, results),
                                           encoding="utf-8")
