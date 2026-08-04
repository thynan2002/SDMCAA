"""状态快照对比工具。

CLI: python -m harness.verify.snapshot_diff <A.json> <B.json>
输出字段级差异；无差异时输出 IDENTICAL。退出码 0=一致 / 1=有差异。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..snapshot import compare_snapshots


def diff_snapshot_files(path_a: str | Path, path_b: str | Path) -> list[str]:
    """对比两个快照 JSON 文件，返回字段级差异列表（空列表 = 一致）。"""
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    return compare_snapshots(a, b)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="状态快照字段级对比")
    parser.add_argument("snapshot_a")
    parser.add_argument("snapshot_b")
    args = parser.parse_args(argv)

    diffs = diff_snapshot_files(args.snapshot_a, args.snapshot_b)
    if not diffs:
        print("IDENTICAL — 两个快照完全一致")
        return 0
    print(f"DIFFERENT — {len(diffs)} 处差异:")
    for d in diffs:
        print(f"  {d}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
