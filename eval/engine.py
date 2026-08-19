"""执行引擎 —— 交错调度、子进程作业、断点续跑、评分聚合、报告生成。

执行顺序（评审修订 #8）：按重复轮分层，轮内 (case × system) 随机交错
（种子记录于 meta.json），避免 API 时延波动的时间混淆。

产物布局（事件目录 Score/{event}_{ts}/）：
    meta.json            配置快照 / git commit / 用例清单 / 随机种子
    plan.json            交错执行计划
    runs/{sys}/{case}/{repeat}/   result.json + trace_*.jsonl + stdout.log
    judge/{case}.json    Judge 评分（缓存复用）
    metrics.json         每 run 评分明细
    results.json         系统级聚合 + 配对检验
    report.md / report.html / figures/*.png
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import metrics as M
from . import stats as ST
from .cases import TestCase, load_cases
from .config import ALL_SYSTEMS, CASES_DIR, EvalConfig, PROJECT_ROOT, SCORED_METRICS, SYSTEM_LABELS
from .judge import cache_key, load_cached, save_cached, score_case_with_judge
from .scoring import extended_total, score_run
from .tracealign import build_case_alignment


# ── 事件目录 ────────────────────────────────────────────────

def create_event(event_name: str, cfg: EvalConfig, cases: list[TestCase],
                 preset: str | None = None) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    event_dir = cfg.score_root / f"{event_name}_{ts}"
    for sub in ("runs", "judge", "figures"):
        (event_dir / sub).mkdir(parents=True, exist_ok=True)
    meta = {
        "event": event_name, "ts": ts,
        "config": {
            "weights": cfg.weights, "repeats": cfg.repeats,
            "systems": list(cfg.systems), "temperature": cfg.temperature,
            "seed": cfg.seed, "job_timeout_s": cfg.job_timeout_s,
            "refs": asdict(cfg.refs),
            "judge": {"model": cfg.judge.model or "(same-as-system)",
                      "sample_repeats": cfg.judge.sample_repeats,
                      "order_swap": cfg.judge.order_swap,
                      "temperature": cfg.judge.temperature},
            "judge_enabled": cfg.judge_enabled,
        },
        "cases": [{"id": c.id, "category": c.category, "level": c.level}
                  for c in cases],
        "case_selection": [c.id for c in cases],
        "python": sys.version.split()[0],
    }
    meta["git_commit"] = _git_commit()
    # 口径标记：v1 起流式调用经 stream_options.include_usage 采集真实
    # token usage，llm_calls 另含调用次数兜底；历史事件无此字段，
    # 效率指标跨口径不可直接比较。
    meta["usage_fix_version"] = 1
    # 分层评测预设标记（如 smoke），附加性字段，历史事件无此字段
    if preset:
        meta["preset"] = preset
    (event_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return event_dir


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def find_event(event_or_dir: str, cfg: EvalConfig) -> Path:
    p = Path(event_or_dir)
    if p.is_dir() and (p / "meta.json").exists():
        return p
    matches = sorted(cfg.score_root.glob(f"{event_or_dir}*"))
    if not matches:
        raise FileNotFoundError(f"找不到事件目录: {event_or_dir}（Score/ 下）")
    return matches[-1]


# ── 计划与执行 ──────────────────────────────────────────────

def sample_smoke_cases(cases: list[TestCase]) -> list[TestCase]:
    """smoke 预设抽样：每个 category 取 1 个用例。

    确定性规则：按 case_id 排序后每类取第一个，结果按 category 排序，
    保证同一用例集下抽样结果可复现（不依赖随机种子）。
    """
    picked: dict[str, TestCase] = {}
    for case in sorted(cases, key=lambda c: c.id):
        picked.setdefault(case.category, case)
    return [picked[cat] for cat in sorted(picked)]


def build_plan(cases: list[TestCase], cfg: EvalConfig) -> list[dict]:
    """重复轮分层 + 轮内 (case×system) 随机交错。"""
    rng = random.Random(cfg.seed)
    jobs: list[dict] = []
    for rep in range(cfg.repeats):
        units = [(c.id, s) for c in cases for s in cfg.systems]
        rng.shuffle(units)
        for order, (cid, sys_name) in enumerate(units):
            jobs.append({"system": sys_name, "case_id": cid, "repeat": rep,
                         "round": rep, "order_in_round": order})
    return jobs


def run_dir(event_dir: Path, system: str, case_id: str, repeat: int) -> Path:
    return event_dir / "runs" / system / case_id / str(repeat)


def _run_job_subprocess(cfg: EvalConfig, job: dict, out: Path,
                        marker: Path) -> dict:
    """执行单个作业（子进程），返回状态。"""
    cmd = [sys.executable, "-m", "eval.exec_job",
           "--system", job["system"], "--case-id", job["case_id"],
           "--repeat", str(job["repeat"]), "--out-dir", str(out),
           "--cases-dir", str(cfg.cases_dir)]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, timeout=cfg.job_timeout_s,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        ok = proc.returncode == 0 and marker.exists()
        status = {
            **job, "status": "ok" if ok else f"fail(rc={proc.returncode})",
            "wall_s": round(time.monotonic() - t0, 1),
        }
        if not ok:
            _write_error_result(out, job, f"subprocess_failed rc={proc.returncode}: "
                               f"{(proc.stderr or '')[-500:]}")
    except subprocess.TimeoutExpired:
        status = {**job, "status": "timeout",
                  "wall_s": round(time.monotonic() - t0, 1)}
        _write_error_result(out, job, f"timeout after {cfg.job_timeout_s}s")
    return status


def execute_jobs(event_dir: Path, plan: list[dict], cfg: EvalConfig,
                 resume: bool = True, dry_run: bool = False) -> list[dict]:
    """执行计划（3.1：max_workers>1 时并行调度子进程，保持交错计划语义）。"""
    statuses: list[dict] = []
    pending: list[dict] = []
    for i, job in enumerate(plan, 1):
        out = run_dir(event_dir, job["system"], job["case_id"], job["repeat"])
        marker = out / "result.json"
        if resume and marker.exists():
            try:
                json.loads(marker.read_text(encoding="utf-8"))
                statuses.append({**job, "status": "cached"})
                continue
            except json.JSONDecodeError:
                pass
        if dry_run:
            statuses.append({**job, "status": "planned"})
            continue
        pending.append(job)
    if not pending:
        return statuses

    def _submit(job: dict) -> dict:
        out = run_dir(event_dir, job["system"], job["case_id"], job["repeat"])
        return _run_job_subprocess(cfg, job, out, out / "result.json")

    total_pending = len(pending)
    workers = max(1, cfg.max_workers or 1)
    if workers <= 1:
        for i, job in enumerate(pending, 1):
            status = _submit(job)
            statuses.append(status)
            print(f"[{i}/{total_pending}] {status['status']}: "
                  f"{job['system']}/{job['case_id']}/#{job['repeat']}", flush=True)
        return statuses

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_submit, job): job for job in pending}
        done = 0
        for fut in as_completed(futures):
            status = fut.result()
            statuses.append(status)
            done += 1
            job = status
            print(f"[{done}/{total_pending}] {status['status']}: "
                  f"{job['system']}/{job['case_id']}/#{job['repeat']}", flush=True)
    return statuses


def _write_error_result(out: Path, job: dict, message: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps({"system": job["system"], "case_id": job["case_id"],
                    "repeat": job["repeat"], "answer": None, "error": message,
                    "out_dir": str(out)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── 收集与评分 ──────────────────────────────────────────────

def collect_runs(event_dir: Path, cases: list[TestCase],
                 cfg: EvalConfig) -> dict[str, dict[str, list[dict]]]:
    """{case_id: {system: [run result（按 repeat 排序）]}}"""
    result: dict[str, dict[str, list[dict]]] = {}
    for case in cases:
        per_sys: dict[str, list[dict]] = {}
        for sys_name in cfg.systems:
            runs = []
            for rep in range(cfg.repeats):
                marker = run_dir(event_dir, sys_name, case.id, rep) / "result.json"
                if marker.exists():
                    try:
                        runs.append(json.loads(marker.read_text(encoding="utf-8")))
                    except json.JSONDecodeError:
                        pass
            runs.sort(key=lambda r: r.get("repeat") if r.get("repeat") is not None else 0)
            per_sys[sys_name] = runs
        result[case.id] = per_sys
    return result


def run_judge_phase(event_dir: Path, cases: list[TestCase],
                    runs_by_case: dict, cfg: EvalConfig,
                    force: bool = False) -> dict[str, dict]:
    """逐用例盲评（缓存复用，修订：resume 不重复烧 Judge 费用）。"""
    judge_dir = event_dir / "judge"
    rng = random.Random(cfg.seed + 7)
    scores: dict[str, dict] = {}
    if not cfg.judge_enabled:
        return scores
    for case in cases:
        answers = {
            sys_name: [r.get("answer") or "" for r in runs]
            for sys_name, runs in runs_by_case.get(case.id, {}).items() if runs
        }
        if len(answers) < 2:
            continue
        key = cache_key(case, answers)
        cached = None if force else load_cached(judge_dir, case.id, key)
        if cached is not None:
            scores[case.id] = dict(cached)
            print(f"[judge] {case.id}: cached", flush=True)
            continue
        try:
            case_scores, meta = score_case_with_judge(case, answers, cfg.judge, rng)
            save_cached(judge_dir, case.id, key, case_scores, meta)
            scores[case.id] = dict(case_scores)
            print(f"[judge] {case.id}: done", flush=True)
        except Exception as exc:
            print(f"[judge] {case.id}: FAILED {exc}", flush=True)
    return scores


def load_judge_meta(event_dir: Path, cases: list[TestCase]) -> dict[str, dict]:
    """按用例读回 Judge 缓存中的 meta（consistency / orders / judge_model /
    rubric_version 等，由 judge.score_case_with_judge 产出）。

    附加性字段：历史事件缓存若无 meta 则跳过，不影响既有流程。
    """
    meta_by_case: dict[str, dict] = {}
    judge_dir = event_dir / "judge"
    for case in cases:
        path = judge_dir / f"{case.id}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = data.get("meta")
        if isinstance(meta, dict):
            meta_by_case[case.id] = meta
    return meta_by_case


def score_all(event_dir: Path, cases: list[TestCase], runs_by_case: dict,
              judge_scores: dict, cfg: EvalConfig) -> dict[str, dict]:
    """全量评分：每 run 明细（metrics.json）+ 每用例聚合（stats.aggregate 输入）。"""
    rows: list[dict] = []
    per_case_scores: dict[str, dict[str, dict]] = {}
    for case in cases:
        case_runs = runs_by_case.get(case.id, {})
        stats_ref = case.resolve_dataset_stats()
        per_sys: dict[str, dict] = {}
        for sys_name, runs in case_runs.items():
            if not runs:
                continue
            js = (judge_scores.get(case.id) or {}).get(sys_name)
            sys_rows = [score_run(case, run, stats_ref, cfg, js) for run in runs]
            for run, row in zip(runs, sys_rows):
                row["tool_use"] = M.tool_use_diag(run.get("tool_records"))
                row["extended_total"] = extended_total(row, row["tool_use"])
                rows.append(row)
            answers = [r.get("answer") for r in runs]
            per_sys[sys_name] = {
                "total": [r["total"] for r in sys_rows],
                "extended_total": [r["extended_total"] for r in sys_rows],
                "metrics": {m: [r["scores"][m] for r in sys_rows] for m in SCORED_METRICS},
                "stability": M.stability_score(answers),
                "tool_use": [r["tool_use"].get("score") for r in sys_rows],
                "judge_failure_mode": sys_rows[0].get("judge_failure_mode"),
            }
        per_case_scores[case.id] = per_sys
    (event_dir / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    return per_case_scores


def category_breakdown(cases: list[TestCase],
                       per_case_scores: dict) -> dict[str, dict[str, float | None]]:
    """类别 × 系统总分热力表数据。"""
    cat_of = {c.id: c.category for c in cases}
    out: dict[str, dict[str, list[float]]] = {}
    for case_id, per_sys in per_case_scores.items():
        cat = cat_of.get(case_id, "?")
        out.setdefault(cat, {})
        for sys_name, data in per_sys.items():
            out[cat].setdefault(sys_name, [])
            totals = [t for t in (data.get("total") or []) if t is not None]
            if totals:
                out[cat][sys_name].append(sum(totals) / len(totals))
    return {
        cat: {s: (round(sum(v) / len(v), 3) if v else None)
              for s, v in sys_map.items()}
        for cat, sys_map in out.items()
    }


def failure_modes(metrics_rows: list[dict]) -> dict[str, dict]:
    """Judge 标注的失败模式汇总（top-3 + 样例）。"""
    counter: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for row in metrics_rows:
        mode = row.get("judge_failure_mode")
        if mode and mode != "无":
            counter[mode] = counter.get(mode, 0) + 1
            samples.setdefault(mode, [])
            if len(samples[mode]) < 3:
                samples[mode].append(f"{row['system']}/{row['case_id']}#r{row['repeat']}")
    top = sorted(counter.items(), key=lambda kv: -kv[1])[:3]
    return {mode: {"count": n, "samples": samples.get(mode, [])} for mode, n in top}


def judge_disagreements(metrics_rows: list[dict], top_n: int = 5) -> dict:
    """Judge 与程序化 accuracy 分歧汇总（仅诊断，不改分）。

    基于每 run 的 judge_program_agreement 字段（scoring.score_run 产出），
    列出 top 分歧用例及差值，供人工抽检 Judge 可信度。
    """
    items: list[dict] = []
    counts = {"ok": 0, "disagreement": 0, "n/a": 0}
    for row in metrics_rows:
        ag = row.get("judge_program_agreement") or {}
        flag = ag.get("flag")
        if flag not in counts:
            continue
        counts[flag] += 1
        if flag == "disagreement":
            items.append({
                "system": row.get("system"), "case_id": row.get("case_id"),
                "repeat": row.get("repeat"),
                "program_accuracy": ag.get("program_accuracy"),
                "judge_accuracy": ag.get("judge_accuracy"),
                "diff": ag.get("diff"),
            })
    items.sort(key=lambda x: (-(x["diff"] or 0.0),
                              str(x.get("case_id")), str(x.get("system"))))
    return {"counts": counts, "top": items[:top_n]}


def cost_summary(runs_by_case: dict, cfg: EvalConfig) -> dict[str, dict]:
    """成本透明度：每系统时延/调用数/token 均值。"""
    out: dict[str, dict[str, list[float]]] = {s: {} for s in cfg.systems}
    for per_sys in runs_by_case.values():
        for sys_name, runs in per_sys.items():
            acc = out.setdefault(sys_name, {})
            for run in runs:
                for key in ("latency_ms", "llm_calls"):
                    if run.get(key) is not None:
                        acc.setdefault(key, []).append(float(run[key]))
                tok = (run.get("tokens") or {}).get("total_tokens")
                if tok:
                    acc.setdefault("total_tokens", []).append(float(tok))
    return {
        sys_name: {k: (round(sum(v) / len(v), 1), f"n={len(v)}")
                   for k, v in data.items()}
        for sys_name, data in out.items()
    }


# ── 一条龙 ──────────────────────────────────────────────────

def evaluate(event_name: str, cfg: EvalConfig, select: str | None = None,
             dry_run: bool = False, resume: bool = True,
             preset: str | None = None) -> Path:
    cases = load_cases(cfg.cases_dir, select)
    event_dir = create_event(event_name, cfg, cases, preset=preset)
    plan = build_plan(cases, cfg)
    (event_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"事件目录: {event_dir}")
    print(f"用例 {len(cases)} × 系统 {len(cfg.systems)} × 重复 {cfg.repeats} "
          f"= {len(plan)} 个作业")
    if dry_run:
        for job in plan[:30]:
            print(f"  [dry] r{job['repeat']} #{job['order_in_round']:02d} "
                  f"{job['system']:6s} {job['case_id']}")
        print(f"  ...（共 {len(plan)} 条）Judge={'on' if cfg.judge_enabled else 'off'}")
        return event_dir
    statuses = execute_jobs(event_dir, plan, cfg, resume=resume)
    ok = sum(1 for s in statuses if s["status"] in ("ok", "cached"))
    print(f"作业完成: {ok}/{len(plan)}")
    finalize_event(event_dir, cfg, select)
    return event_dir


def finalize_event(event_dir: Path, cfg: EvalConfig, select: str | None = None,
                   align_case: str | None = None) -> dict:
    """从 runs + judge 缓存重算评分与报告（可独立反复调用）。"""
    from . import report as R

    cases = load_cases(cfg.cases_dir, select)
    # 以事件目录实际存在的 run 为准（避免 select 与录制的错位）
    runs_by_case = collect_runs(event_dir, cases, cfg)
    present = [c for c in cases if any(runs_by_case.get(c.id, {}).values())]
    judge_scores = run_judge_phase(event_dir, present, runs_by_case, cfg)
    per_case_scores = score_all(event_dir, present, runs_by_case, judge_scores, cfg)
    agg = ST.aggregate(per_case_scores)

    metrics_rows = json.loads((event_dir / "metrics.json").read_text(encoding="utf-8"))
    results = {
        "event_dir": str(event_dir),
        "n_cases": len(present),
        "aggregate": agg,
        "category_breakdown": category_breakdown(present, per_case_scores),
        "failure_modes": failure_modes(metrics_rows),
        # 可信度增量（附加字段）：Judge 元信息与 Judge-程序分歧诊断
        "judge_meta": load_judge_meta(event_dir, present),
        "judge_disagreements": judge_disagreements(metrics_rows),
        "cost": cost_summary(runs_by_case, cfg),
        "per_case_summary": {
            cid: {s: {"total": (round(sum(t for t in d['total'] if t is not None) /
                                len([t for t in d['total'] if t is not None]), 3)
                              if any(t is not None for t in d['total']) else None),
                      "stability": d.get("stability")}
                  for s, d in per_sys.items()}
            for cid, per_sys in per_case_scores.items()
        },
        "system_labels": SYSTEM_LABELS,
    }
    # trace 对齐（默认取第一个有 trace 的用例）
    align_target = align_case or next(
        (c.id for c in present
         if any((run_dir(event_dir, s, c.id, 0) /
                 (runs and (runs[0].get("trace_file") or ""))).exists()
                for s, runs in runs_by_case[c.id].items() if runs)),
        None,
    )
    if align_target:
        results["alignment_case"] = align_target
        results["alignment"] = build_case_alignment(
            runs_by_case.get(align_target, {}), repeat_idx=0,
        )
    (event_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    R.render_all(event_dir, results)
    print(f"报告: {event_dir / 'report.html'}")
    print(f"      {event_dir / 'report.md'}")
    return results
