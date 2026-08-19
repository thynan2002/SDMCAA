"""评测 CLI。

子命令：
    run       执行完整评测（计划 → 子进程作业 → Judge → 评分 → 报告）
    report    从事件目录重算评分与报告（Judge 走缓存）
    judge     仅重跑 Judge（--force 忽略缓存）
    cases     list / lint 用例集
    gold      打印各用例独立重算的金标准（人工核对清单，评审修订 #1）

示例：
    python -m eval run --event demo_v1
    python -m eval run --event smoke --cases cf_7s_shoot --repeats 1 --no-judge
    python -m eval run --event smoke_v2 --smoke          # 分层冒烟预设
    python -m eval report Score/demo_v1_20260815_1030
    python -m eval cases lint
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cases import lint_cases, lint_warnings, load_cases
from .config import ALL_SYSTEMS, CASES_DIR, SYSTEM_MULTI, SYSTEM_SINGLE, EvalConfig, SCORE_ROOT
from .cases import LEVEL_NAMES


def _apply_overrides(cfg: EvalConfig, args) -> EvalConfig:
    if args.systems:
        cfg.systems = tuple(s.strip() for s in args.systems.split(","))
        bad = set(cfg.systems) - set(ALL_SYSTEMS)
        if bad:
            raise SystemExit(f"未知系统: {bad}")
    if args.repeats is not None:
        cfg.repeats = args.repeats
    if args.no_judge:
        cfg.judge_enabled = False
    if args.timeout:
        cfg.job_timeout_s = args.timeout
    if args.workers is not None:
        cfg.max_workers = args.workers
    if args.quick:
        cfg.repeats = 1
    if args.smoke:
        # smoke 预设优先于显式 --systems/--repeats：每 category 抽 1 例
        # （抽样在 cmd_run 中完成），repeats=1，仅 single+multi
        cfg.repeats = 1
        cfg.systems = (SYSTEM_SINGLE, SYSTEM_MULTI)
    cfg.__post_init__()
    return cfg


def cmd_run(args) -> int:
    from .engine import evaluate, sample_smoke_cases

    cfg = EvalConfig.load(args.config)
    preset = None
    if args.smoke:
        preset = "smoke"
        # 在（可选 --cases 过滤后的）用例集上确定性抽样，以 id 列表回传 select
        cases = load_cases(cfg.cases_dir, args.cases)
        picked = sample_smoke_cases(cases)
        if not picked:
            raise SystemExit("--smoke: 过滤后无用例可抽")
        args.cases = ",".join(c.id for c in picked)
    cfg = _apply_overrides(cfg, args)
    event_dir = evaluate(
        args.event, cfg, select=args.cases,
        dry_run=args.dry_run, resume=not args.no_resume,
        preset=preset,
    )
    if not args.dry_run:
        print(f"\n完成。结果: {event_dir}")
    return 0


def cmd_report(args) -> int:
    from .engine import finalize_event, find_event

    cfg = EvalConfig.load(args.config)
    event_dir = find_event(args.event, cfg)
    finalize_event(event_dir, cfg, align_case=args.align_case)
    return 0


def cmd_judge(args) -> int:
    from .engine import finalize_event, find_event

    cfg = EvalConfig.load(args.config)
    event_dir = find_event(args.event, cfg)
    # 先强制重评 Judge
    from . import engine

    cases = load_cases(cfg.cases_dir)
    runs = engine.collect_runs(event_dir, cases, cfg)
    present = [c for c in cases if any(runs.get(c.id, {}).values())]
    engine.run_judge_phase(event_dir, present, runs, cfg, force=args.force)
    finalize_event(event_dir, cfg)
    return 0


def cmd_cases(args) -> int:
    if args.action == "lint":
        problems = lint_cases(CASES_DIR)
        if problems:
            print("[FAIL] 用例集存在问题:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("[ok] 用例集校验通过")
        # 刷分点卫生检查（仅 warnings，不阻断；不发 LLM）
        warns = lint_warnings(CASES_DIR)
        if warns:
            print(f"[warn] 刷分点卫生检查（{len(warns)} 条，不影响 lint 结论）:")
            for w in warns:
                print(f"  - {w}")
        else:
            print("[warn] 刷分点卫生检查无告警")
        return 0
    # list
    cases = load_cases(CASES_DIR)
    print(f"{len(cases)} 个用例:")
    for c in cases:
        gold_desc = []
        if c.numeric:
            gold_desc.append(f"numeric×{len(c.numeric)}")
        if c.direction:
            gold_desc.append(f"direction×{len(c.direction)}")
        if c.refusal:
            gold_desc.append("refusal")
        if c.category_gold:
            gold_desc.append(f"category×{len(c.category_gold)}")
        level = LEVEL_NAMES.get(c.level, f"L{c.level}")
        print(f"  [{c.category:14s}][{level:10s}] {c.id:24s} {', '.join(gold_desc) or 'judge-only'}")
    return 0


def cmd_gold(args) -> int:
    """打印独立重算金标准（供人工核对）。"""
    cases = load_cases(CASES_DIR)
    for c in cases:
        stats = c.resolve_dataset_stats()
        if not c.numeric:
            continue
        print(f"[{c.id}]  数据: {c.dataset_person} "
              f"({stats.player_count}人, {stats.duration_s:.1f}s, max_frame={stats.max_frame})")
        for g in c.numeric:
            print(f"    {g.expr:28s} = {g.value}  (tol={g.tolerance})")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(
        prog="eval", description="单 LLM vs 多智能体对比评测框架",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="覆盖默认 eval/config.yaml 的配置文件")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="执行完整评测")
    p_run.add_argument("--event", required=True, help="事件名（Score/{event}_{ts}）")
    p_run.add_argument("--cases", default=None,
                       help="用例/类别过滤（逗号分隔 id 或 category）")
    p_run.add_argument("--systems", default=None,
                       help=f"系统过滤（逗号分隔，可选 {ALL_SYSTEMS}）")
    p_run.add_argument("--repeats", type=int, default=None)
    p_run.add_argument("--timeout", type=int, default=None, help="单作业超时（秒）")
    p_run.add_argument("--workers", type=int, default=None,
                       help="并行执行作业数（3.1：>1 启用多进程并发）")
    p_run.add_argument("--quick", action="store_true",
                       help="快速分层预设：1 重复（3.1）")
    p_run.add_argument("--smoke", action="store_true",
                       help="冒烟分层预设：每 category 确定性抽 1 例（按 id 排序取首个），"
                            "repeats=1，systems 仅 single+multi")
    p_run.add_argument("--dry-run", action="store_true", help="只打印执行计划")
    p_run.add_argument("--no-resume", action="store_true")
    p_run.add_argument("--no-judge", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="重算评分与报告")
    p_rep.add_argument("event", help="事件目录或事件名前缀")
    p_rep.add_argument("--align-case", default=None, help="trace 对齐用例 id")
    p_rep.set_defaults(func=cmd_report)

    p_jdg = sub.add_parser("judge", help="重跑 Judge 评分")
    p_jdg.add_argument("event")
    p_jdg.add_argument("--force", action="store_true", help="忽略缓存")
    p_jdg.set_defaults(func=cmd_judge)

    p_cs = sub.add_parser("cases", help="用例集管理")
    p_cs.add_argument("action", choices=["list", "lint"])
    p_cs.set_defaults(func=cmd_cases)

    p_gold = sub.add_parser("gold", help="打印独立重算金标准（人工核对）")
    p_gold.set_defaults(func=cmd_gold)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
