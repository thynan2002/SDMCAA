"""反事实数据拼接导出。

将"干预帧之前的原始真实数据"与"干预帧之后生成的关键帧"拼接，
输出与输入格式严格一致的两个 CSV 文件（球员文件、足球文件）。

- 仅写入状态发生变化的关键帧（直线段只保留端点，中间插值交给下游）
- 导出前对全部生成段执行物理校验（运动学限速 / 场地边界）
"""

from __future__ import annotations

import csv
from pathlib import Path

from agents.constants import PIXELS_PER_METER
from agents.player.tracker import PrefixPlayerCorpus, _read_csv_rows
from . import physics
from .engine import SimulationResult


def export_simulation(
    corpus: PrefixPlayerCorpus,
    result: SimulationResult,
    output_dir: str | Path,
    tag: str = "cf",
) -> tuple[Path, Path, list[str]]:
    """拼接原始数据与生成轨迹并写出 CSV。

    Args:
        corpus: 比赛语料（需含 source_files 原始路径与球员颜色信息）
        result: 模拟结果（关键帧序列）
        output_dir: 输出目录
        tag: 文件名标记

    Returns:
        (球员CSV路径, 足球CSV路径, 物理校验警告列表)
    """
    if len(corpus.source_files) < 2 or not all(corpus.source_files):
        raise ValueError("corpus 缺少原始数据文件路径（source_files），无法拼接导出")

    person_csv, ball_csv = Path(corpus.source_files[0]), Path(corpus.source_files[1])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = _validate_result(result)

    person_path = out_dir / f"{corpus.prefix}_{tag}_f{result.start_frame}_person.csv"
    ball_path = out_dir / f"{corpus.prefix}_{tag}_f{result.start_frame}_ball.csv"

    _write_person_csv(person_csv, person_path, corpus, result)
    _write_ball_csv(ball_csv, ball_path, result)

    return person_path, ball_path, warnings


# ── 物理校验 ───────────────────────────────────────────────────

def _validate_result(result: SimulationResult) -> list[str]:
    """校验全部生成段的运动学合理性，返回警告列表（不阻断导出）。"""
    warnings: list[str] = []

    for tid, kfs in result.player_keyframes.items():
        for i in range(1, len(kfs)):
            f1, x1, y1 = kfs[i - 1]
            f2, x2, y2 = kfs[i]
            if not physics.validate_player_segment(x1, y1, f1, x2, y2, f2):
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                warnings.append(
                    f"球员{tid} 帧{f1}→{f2} 移动 {dist:.0f}px "
                    f"（{(dist * physics.FPS / max(1, f2 - f1)):.0f}px/s）超过运动学上限"
                )
            if not physics.in_field(x2, y2, margin=1.0):
                warnings.append(f"球员{tid} 帧{f2} 坐标越界 ({x2}, {y2})")

    bkf = result.ball_keyframes
    for i in range(1, len(bkf)):
        f1, x1, y1, _ = bkf[i - 1]
        f2, x2, y2, _ = bkf[i]
        if f2 > f1:
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            speed = dist * physics.FPS / (f2 - f1)
            # 球速上限：射门速度 352 px/s，留 15% 余量
            if speed > physics.SHOT_SPEED * 1.15:
                warnings.append(f"球 帧{f1}→{f2} 速度 {speed:.0f}px/s 异常")

    return warnings


# ── CSV 拼接写出 ───────────────────────────────────────────────

def _write_person_csv(
    src: Path, dst: Path,
    corpus: PrefixPlayerCorpus, result: SimulationResult,
) -> None:
    headers, rows = _read_csv_rows(src)
    cut = result.start_frame

    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # 1) 干预帧之前的原始行（原样保留字符串，格式零失真）
        for row in rows:
            try:
                fr = int(float(row.get("frame_num", "")))
            except (TypeError, ValueError):
                continue
            if fr < cut:
                writer.writerow([row.get(h, "") for h in headers])

        # 2) 干预帧及之后的生成关键帧
        gen_rows: list[tuple[int, int, list[str]]] = []
        for tid, kfs in result.player_keyframes.items():
            color = corpus.players[tid].color if tid in corpus.players else ""
            for fr, x, y in kfs:
                if fr < cut:
                    continue
                values = {
                    "frame_num": str(fr),
                    "track_id": str(tid),
                    "x": f"{x:.1f}",
                    "y": f"{y:.1f}",
                    "color": color,
                }
                gen_rows.append((fr, _tid_sort_key(tid), [values.get(h, "") for h in headers]))

        gen_rows.sort(key=lambda r: (r[0], r[1]))
        for _, _, line in gen_rows:
            writer.writerow(line)


def _write_ball_csv(src: Path, dst: Path, result: SimulationResult) -> None:
    headers, rows = _read_csv_rows(src)
    cut = result.start_frame

    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in rows:
            try:
                fr = int(float(row.get("frame_num", "")))
            except (TypeError, ValueError):
                continue
            if fr < cut:
                writer.writerow([row.get(h, "") for h in headers])

        for fr, x, y, z in result.ball_keyframes:
            if fr < cut:
                continue
            # 模拟内部 z 为像素，导出时换算回米（与原始 CSV 的 z 列单位一致）
            values = {
                "frame_num": str(fr),
                "x": f"{x:.1f}",
                "y": f"{y:.1f}",
                "z": f"{z / PIXELS_PER_METER:.1f}",
            }
            writer.writerow([values.get(h, "") for h in headers])


def _tid_sort_key(tid: str) -> int:
    try:
        return int(tid)
    except (TypeError, ValueError):
        return 10**9
