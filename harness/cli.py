"""Harness 统一 CLI 入口。

用法：
    # 交互模式（默认，等价 python main.py <person> <ball>）
    python -m harness run <person.csv> <ball.csv>
    python -m harness run <person.csv> <ball.csv> --script turns.txt   # 批量脚本
    python -m harness run <person.csv> <ball.csv> --once [球衣号...]   # 单次解说
    python -m harness run <person.csv> <ball.csv> --focus 6 10         # 聚焦解说

    # 录制 golden（真实 API）
    python -m harness run <person.csv> <ball.csv> --script turns.txt \
        --mode record --golden-dir harness/golden/standard --seed 42

    # 回放回归（不触网）
    python -m harness verify harness/golden/standard --report report.md

    # 快照对比
    python -m harness snapshot-diff A.json B.json

脚本文件语法（--script，逐行执行；空行与 # 开头行忽略）：
    聚焦7号
    [CF_FORM] 7号|3.0|shoot||0        → 反事实表单通道
    [ONCE] person.csv|ball.csv|6,10   → 单次解说通道
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import (
    HarnessConfig,
    MODE_PASSTHROUGH,
    MODE_RECORD,
    PROJECT_ROOT,
    VALID_MODES,
)
from .core import Harness
from .verify.regression import execute_scripted_input


def _read_script(path: str | Path) -> list[str]:
    """读取脚本文件：忽略空行与 # 注释行。"""
    lines: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _prefix_hint(person_csv: str | Path) -> str:
    m = re.search(r"(\d+s)", Path(person_csv).name, re.IGNORECASE)
    return m.group(1) if m else Path(person_csv).stem


def run_repl(harness: Harness) -> None:
    """交互式 REPL — 与 SessionManager.run_repl 相同的外层行为，
    但每一轮都经过 Harness 边界（trace/快照/录制）。"""
    session = harness.session
    corpus = session.corpus
    if corpus is None:
        print("[错误] 未加载数据，无法启动交互模式。")
        return

    print(f"\n{'=' * 50}")
    print(f"  足球战术分析 Multi-Agent 交互系统 (Harness)")
    print(f"  数据: {corpus.prefix} ({corpus.player_count}名球员)")
    print(f"  输入 'help' 查看可用命令，'exit' 退出")
    print(f"{'=' * 50}\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            session._save_conversation()
            break

        response = harness.handle_input(user_input)
        if response == "__EXIT__":
            print("再见！")
            session._save_conversation()
            break
        if response:
            print(f"\n{response}\n")


def _cmd_run(args: argparse.Namespace) -> int:
    person_csv = Path(args.person_csv)
    ball_csv = Path(args.ball_csv)
    for p in (person_csv, ball_csv):
        if not p.exists():
            print(f"错误：文件不存在 — {p}")
            return 1

    mode = args.mode
    golden_dir = Path(args.golden_dir) if args.golden_dir else None
    if mode == MODE_RECORD and golden_dir is None:
        # 缺省 golden 目录：harness/golden/<name 或 时间戳>
        golden_dir = PROJECT_ROOT / "harness" / "golden" / (args.name or "default")

    cfg = HarnessConfig(
        mode=mode,
        golden_dir=golden_dir,
        seed=args.seed,
        trace_enabled=not args.no_trace,
        trace_dir=Path(args.trace_dir) if args.trace_dir else PROJECT_ROOT / "Output" / "traces",
        strict_mock=not args.lenient,
        snapshot_enabled=not args.no_snapshot,
    )

    # 解析单次模式参数（与 main.py 相同的语义）
    focus_jerseys: list[str] | None = None
    once = args.once
    if args.focus is not None:
        once = True
        focus_jerseys = args.focus
    elif args.once_jerseys:
        focus_jerseys = args.once_jerseys

    with Harness(cfg) as harness:
        if mode == MODE_RECORD:
            harness.init_recording({
                "name": args.name or golden_dir.name,
                "seed": args.seed,
                "dataset": {
                    "person": str(person_csv.resolve()),
                    "ball": str(ball_csv.resolve()),
                },
                "entry": "once" if once else ("script" if args.script else "interactive"),
                "script_file": str(args.script) if args.script else None,
            })

        if once:
            harness.run_once(person_csv, ball_csv, focus_jerseys)
        else:
            if not harness.load_data(person_csv, ball_csv):
                print("数据加载失败，退出。")
                return 1
            if harness.corpus:
                print(f"球员数: {harness.corpus.player_count}"
                      f"（含门将 {len(harness.corpus.goalkeepers())} 人）")

            if args.script:
                for line in _read_script(args.script):
                    print(f">> {line}")
                    response = execute_scripted_input(harness, line)
                    if response == "__EXIT__":
                        print("再见！（脚本触发 exit）")
                        break
                    if response:
                        print(f"\n{response}\n")
            else:
                run_repl(harness)

        if mode == MODE_RECORD:
            manifest = harness.finalize_recording(prefix_hint=_prefix_hint(person_csv))
            print(f"\n[Harness] golden 已录制: {golden_dir}")
            print(f"[Harness] LLM 调用 {harness.llm_call_count} 次，"
                  f"产出文件清单 {len(manifest)} 项")
        if harness.tracer.trace_path:
            print(f"[Harness] trace: {harness.tracer.trace_path}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .verify.regression import main as regression_main

    argv = [args.golden_dir]
    if args.report:
        argv += ["--report", args.report]
    if args.lenient:
        argv += ["--lenient"]
    return regression_main(argv)


def _cmd_snapshot_diff(args: argparse.Namespace) -> int:
    from .verify.snapshot_diff import main as diff_main

    return diff_main([args.snapshot_a, args.snapshot_b])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="足球战术分析智能体 — 统一 Harness（透明包装层）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──
    p_run = sub.add_parser("run", help="运行智能体（passthrough/record/replay）")
    p_run.add_argument("person_csv")
    p_run.add_argument("ball_csv")
    p_run.add_argument("--once", action="store_true", help="单次解说模式")
    p_run.add_argument("once_jerseys", nargs="*", help="--once 时聚焦的球衣号")
    p_run.add_argument("--focus", nargs="+", default=None, help="聚焦球员解说")
    p_run.add_argument("-i", "--interactive", action="store_true", help="交互模式（默认）")
    p_run.add_argument("--script", default=None, help="批量脚本文件（逐行执行）")
    p_run.add_argument("--mode", default=MODE_PASSTHROUGH, choices=VALID_MODES)
    p_run.add_argument("--golden-dir", default=None)
    p_run.add_argument("--trace-dir", default=None)
    p_run.add_argument("--seed", type=int, default=None, help="确定性注入种子（record/replay）")
    p_run.add_argument("--name", default=None, help="golden 名称（record）")
    p_run.add_argument("--no-trace", action="store_true")
    p_run.add_argument("--no-snapshot", action="store_true")
    p_run.add_argument("--lenient", action="store_true", help="replay 未命中不抛错")
    p_run.set_defaults(func=_cmd_run)

    # ── verify ──
    p_verify = sub.add_parser("verify", help="golden 回放回归")
    p_verify.add_argument("golden_dir")
    p_verify.add_argument("--report", default=None)
    p_verify.add_argument("--lenient", action="store_true")
    p_verify.set_defaults(func=_cmd_verify)

    # ── snapshot-diff ──
    p_sd = sub.add_parser("snapshot-diff", help="状态快照字段级对比")
    p_sd.add_argument("snapshot_a")
    p_sd.add_argument("snapshot_b")
    p_sd.set_defaults(func=_cmd_snapshot_diff)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
