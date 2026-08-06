"""程序入口 — 足球战术分析 Multi-Agent 交互系统。

用法:

  交互模式（推荐）:
    python main.py <person.csv> <ball.csv>
    python main.py <person.csv> <ball.csv> -i

  单次解说模式（兼容旧版）:
    python main.py <person.csv> <ball.csv> --once [球衣号...]
    python main.py <person.csv> <ball.csv> [球衣号...] --once

  聚焦球员解说:
    python main.py <person.csv> <ball.csv> --focus 6 10 1

示例:
  python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv
  python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --focus 6
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv


def run_once(person_csv: Path, ball_csv: Path, focus_jerseys: list[str] | None = None) -> None:
    """单次解说模式（兼容旧版 CLI）。"""
    from agents.player.tracker import build_corpus_from_paths
    from agents.player.orchestrator import PlayerTrackingOrchestrator

    print(f"读取: {person_csv.name} + {ball_csv.name}")
    if focus_jerseys:
        print(f"聚焦球员: {', '.join(f'{j}号' for j in focus_jerseys)}")

    corpus = build_corpus_from_paths(person_csv, ball_csv)
    print(f"球员数: {corpus.player_count}（含门将 {len(corpus.goalkeepers())} 人）")

    # 数据工具的语料来源（单次解说路径）
    from agents.tools.football import set_active_corpus
    set_active_corpus(corpus)

    orch = PlayerTrackingOrchestrator()
    decision = orch.run_single(corpus, focus_jerseys=focus_jerseys)

    # 写文件
    output_dir = Path(__file__).resolve().parent / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_" + "_".join(focus_jerseys) if focus_jerseys else ""
    txt_path = output_dir / f"{corpus.prefix}{suffix}_summary.txt"
    txt_path.write_text(decision.composed_summary, encoding="utf-8")

    print("\n" + decision.composed_summary)
    print(f"\n解说稿已写入: {txt_path}")


def run_interactive(person_csv: Path, ball_csv: Path) -> None:
    """交互式 REPL 模式。"""
    from agents.session.manager import SessionManager

    session = SessionManager()

    print(f"加载数据: {person_csv.name} + {ball_csv.name}")
    if not session.load_data(person_csv, ball_csv):
        print("数据加载失败，退出。")
        sys.exit(1)

    if session.corpus:
        print(f"球员数: {session.corpus.player_count}"
              f"（含门将 {len(session.corpus.goalkeepers())} 人）")

    session.run_repl()


def main() -> None:
    load_dotenv()

    if len(sys.argv) < 3:
        print("用法:")
        print("  交互模式: python main.py <person.csv> <ball.csv>")
        print("  单次解说: python main.py <person.csv> <ball.csv> --once [球衣号...]")
        print("  聚焦球员: python main.py <person.csv> <ball.csv> --focus 6 10")
        print("\n示例:")
        print("  python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv")
        print("  python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --focus 6")
        sys.exit(1)

    person_csv = Path(sys.argv[1])
    ball_csv = Path(sys.argv[2])

    for path in [person_csv, ball_csv]:
        if not path.exists():
            print(f"错误：文件不存在 — {path}")
            sys.exit(1)

    # 解析模式参数
    args = sys.argv[3:]
    mode = "interactive"  # 默认交互模式
    focus_jerseys: list[str] | None = None

    if "--once" in args:
        mode = "once"
        args.remove("--once")
        # 剩余的是球衣号
        if args:
            focus_jerseys = args
    elif "--focus" in args:
        mode = "once"
        idx = args.index("--focus")
        focus_jerseys = args[idx + 1:]
    elif "-i" in args or "--interactive" in args:
        mode = "interactive"
    elif args:
        # 有额外参数但没有 --once/--focus 标志 → 可能是球衣号
        # 检测：如果所有参数都是数字 → 视为单次模式
        all_digits = all(a.isdigit() for a in args)
        if all_digits:
            mode = "once"
            focus_jerseys = args

    # 去重保持顺序
    if focus_jerseys:
        seen: set[str] = set()
        focus_jerseys = [x for x in focus_jerseys if not (x in seen or seen.add(x))]

    if mode == "once":
        run_once(person_csv, ball_csv, focus_jerseys)
    else:
        run_interactive(person_csv, ball_csv)


if __name__ == "__main__":
    main()
