"""金标准独立重算路径 + 裸调用基线数据摘要模板。

设计原则（对应评审修订 #1）：
- **绝不使用被测系统的工具层/corpus 代码**，只用 pandas/NumPy 直接从原始 CSV
  重算，避免「金标准与多智能体共用同一实现 → 循环验证」的系统性偏袒；
- 数值类金标准在用例加载时按 expr 自动解析（eval.goldref.resolve），始终与
  当前数据文件一致，无陈旧快照问题；
- 裸调用基线的数据摘要（评审修订 #3）：确定性模板 —— 全量球轨迹 + 全量球员
  轨迹原始行 + 独立重算的统计表，遵循「信息给足、不因信息饥饿而输」原则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

TEAM_NAMES = {"A": "A队（红）", "B": "B队（蓝）", "C": "门将"}

# ── 米制标定（与 agents/constants.py 保持一致）：1200×700 px → 105×68 m ──
# x 轴（1200px）= 长度（门线到门线，105m）；y 轴（700px）= 宽度（边线到边线，68m）
_FIELD_WIDTH_PX = 1200.0
_FIELD_HEIGHT_PX = 700.0
_PX_PER_M_X = _FIELD_WIDTH_PX / 105.0   # ≈ 11.43 px/m
_PX_PER_M_Y = _FIELD_HEIGHT_PX / 68.0   # ≈ 10.29 px/m


@dataclass(frozen=True)
class DatasetStats:
    """一份 (person, ball) 数据对的独立统计结果。"""

    person_csv: Path
    ball_csv: Path
    player_ids: list[int]                 # 有序 track_id
    player_teams: dict[int, str]          # track_id -> A/B/C
    max_frame: int                        # 两文件最大帧号
    duration_s: float                     # max_frame / 30
    ball_frames: list[int]
    ball_pos: dict[int, tuple[float, float, float]]   # 原始帧 -> (x, y, z)
    ball_z_max: float
    player_distance: dict[int, float]     # 累计位移（像素单位，原始相邻行差分）
    player_avg_speed: dict[int, float]    # 像素/秒（距离/时间跨度）
    player_top_speed: dict[int, float]    # 滑动窗口(5帧)峰值速度 像素/秒
    # ── 米制指标（单位米 / 米每秒） ──
    player_distance_m: dict[int, float]
    player_avg_speed_ms: dict[int, float]
    player_top_speed_ms: dict[int, float]
    ball_speed_max_ms: float
    ball_speed_avg_ms: float
    # ── 领域量化指标（战术区域/威胁/球速，1.1 新增） ──
    pitch: tuple[float, float, float, float]          # (xmin, xmax, ymin, ymax) 由球员轨迹包围盒
    ball_speed_max: float                             # 球峰值速度 像素/秒
    ball_speed_avg: float                             # 球平均速度 像素/秒
    ball_speed_segments: tuple[tuple[float, float, float], ...]  # (f0, f1, speed) 相邻原始帧段
    player_mean_pos: dict[int, tuple[float, float]]   # track_id -> 平均位置 (x, y)

    @property
    def player_count(self) -> int:
        return len(self.player_ids)

    def jersey(self, tid: int) -> str:
        return f"{tid}号"


# ── 独立统计 ────────────────────────────────────────────────

@lru_cache(maxsize=16)
def compute_stats(person_csv: str, ball_csv: str) -> DatasetStats:
    """用 pandas 直接重算（不经被测系统任何代码路径）。"""
    person_csv, ball_csv = Path(person_csv), Path(ball_csv)
    dfp = pd.read_csv(person_csv)
    dfb = pd.read_csv(ball_csv)

    # 球员：按 track_id 分组，帧排序后逐行差分求位移
    teams: dict[int, str] = {}
    distance: dict[int, float] = {}
    avg_speed: dict[int, float] = {}
    top_speed: dict[int, float] = {}
    distance_m: dict[int, float] = {}
    avg_speed_ms: dict[int, float] = {}
    top_speed_ms: dict[int, float] = {}
    for tid, g in dfp.groupby("track_id"):
        g = g.sort_values("frame_num")
        teams[int(tid)] = str(g["color"].mode().iloc[0])
        xy = g[["x", "y"]].to_numpy(dtype=float)
        frames = g["frame_num"].to_numpy(dtype=float)
        if len(xy) >= 2:
            dxy = xy[1:] - xy[:-1]
            steps = np.hypot(dxy[:, 0], dxy[:, 1])
            steps_m = np.hypot(dxy[:, 0] / _PX_PER_M_X, dxy[:, 1] / _PX_PER_M_Y)
            distance[int(tid)] = float(steps.sum())
            distance_m[int(tid)] = float(steps_m.sum())
            span_s = max((frames[-1] - frames[0]) / 30.0, 1e-9)
            avg_speed[int(tid)] = distance[int(tid)] / span_s
            avg_speed_ms[int(tid)] = distance_m[int(tid)] / span_s
            # 滑动窗口峰值速度：窗口 5 个采样行
            win = min(5, len(steps))
            if win >= 1 and len(steps) >= win:
                csum = np.concatenate([[0.0], np.cumsum(steps)])
                csum_m = np.concatenate([[0.0], np.cumsum(steps_m)])
                window_sum = csum[win:] - csum[:-win]
                window_sum_m = csum_m[win:] - csum_m[:-win]
                window_t = (frames[win:] - frames[:-win]) / 30.0
                ok = window_t > 0.05
                top_speed[int(tid)] = float(
                    (window_sum[ok] / window_t[ok]).max() if ok.any() else steps.max() * 30
                )
                top_speed_ms[int(tid)] = float(
                    (window_sum_m[ok] / window_t[ok]).max() if ok.any() else steps_m.max() * 30
                )
            else:
                top_speed[int(tid)] = float(steps.max() * 30)
                top_speed_ms[int(tid)] = float(steps_m.max() * 30)
        else:
            distance[int(tid)] = 0.0
            avg_speed[int(tid)] = 0.0
            top_speed[int(tid)] = 0.0
            distance_m[int(tid)] = 0.0
            avg_speed_ms[int(tid)] = 0.0
            top_speed_ms[int(tid)] = 0.0

    # 球：原始帧位置 + z 峰值
    ball_pos: dict[int, tuple[float, float, float]] = {
        int(r.frame_num): (float(r.x), float(r.y), float(r.z))
        for r in dfb.itertuples(index=False)
    }
    ball_frames = sorted(ball_pos)
    max_frame = int(max(dfp["frame_num"].max(), dfb["frame_num"].max()))

    # ── 领域量化指标（1.1） ──
    # 球场包围盒：球员轨迹覆盖范围（近似比赛区域）
    pitch = (
        float(dfp["x"].min()), float(dfp["x"].max()),
        float(dfp["y"].min()), float(dfp["y"].max()),
    )
    # 球速：相邻原始帧差分，帧差折算为秒（像素 + 米制两套）
    bf = dfb["frame_num"].to_numpy(float)
    bx = dfb["x"].to_numpy(float)
    by = dfb["y"].to_numpy(float)
    dbfx = np.diff(bx)
    dbfy = np.diff(by)
    steps = np.hypot(dbfx, dbfy)
    steps_m = np.hypot(dbfx / _PX_PER_M_X, dbfy / _PX_PER_M_Y)
    dt = np.diff(bf) / 30.0
    ok = dt > 0
    seg = [(float(f0), float(f1), float(s / t))
           for f0, f1, s, t in zip(bf[:-1][ok], bf[1:][ok], steps[ok], dt[ok])]
    ball_speed_max = float(max((s for _, _, s in seg), default=0.0))
    ball_speed_avg = float(sum(s for _, _, s in seg) / len(seg)) if seg else 0.0
    seg_ms = [(float(f0), float(f1), float(s / t))
              for f0, f1, s, t in zip(bf[:-1][ok], bf[1:][ok], steps_m[ok], dt[ok])]
    ball_speed_max_ms = float(max((s for _, _, s in seg_ms), default=0.0))
    ball_speed_avg_ms = float(sum(s for _, _, s in seg_ms) / len(seg_ms)) if seg_ms else 0.0
    # 球员平均位置（用于主活动区域）
    mean_pos: dict[int, tuple[float, float]] = {}
    for tid, g in dfp.groupby("track_id"):
        mean_pos[int(tid)] = (float(g["x"].mean()), float(g["y"].mean()))

    return DatasetStats(
        person_csv=person_csv, ball_csv=ball_csv,
        player_ids=sorted(teams), player_teams=teams,
        max_frame=max_frame, duration_s=max_frame / 30.0,
        ball_frames=ball_frames, ball_pos=ball_pos,
        ball_z_max=float(dfb["z"].max()) if len(dfb) else 0.0,
        player_distance=distance, player_avg_speed=avg_speed,
        player_top_speed=top_speed,
        player_distance_m=distance_m, player_avg_speed_ms=avg_speed_ms,
        player_top_speed_ms=top_speed_ms,
        ball_speed_max_ms=round(ball_speed_max_ms, 1),
        ball_speed_avg_ms=round(ball_speed_avg_ms, 1),
        pitch=pitch, ball_speed_max=ball_speed_max,
        ball_speed_avg=ball_speed_avg, ball_speed_segments=tuple(seg),
        player_mean_pos=mean_pos,
    )


def ball_at_frame(stats: DatasetStats, frame: int) -> tuple[float, float, float] | None:
    """球位置（相邻原始帧线性插值 — 独立实现）。"""
    if frame in stats.ball_pos:
        return stats.ball_pos[frame]
    frames = stats.ball_frames
    if not frames or frame < frames[0] or frame > frames[-1]:
        return None
    import bisect

    i = bisect.bisect_left(frames, frame)
    f0, f1 = frames[i - 1], frames[i]
    p0, p1 = np.array(stats.ball_pos[f0]), np.array(stats.ball_pos[f1])
    t = (frame - f0) / (f1 - f0)
    return tuple(p0 + t * (p1 - p0))  # type: ignore[return-value]


def player_position_at_second(stats: DatasetStats, tid: int, second: float) -> tuple[float, float] | None:
    """球员位置（秒 → 帧 30fps，行内线性插值 — 独立实现）。"""
    import pandas as pd

    dfp = pd.read_csv(stats.person_csv)
    g = dfp[dfp["track_id"] == tid].sort_values("frame_num")
    if g.empty:
        return None
    frame = second * 30.0
    xf = np.interp(frame, g["frame_num"].to_numpy(float), g["x"].to_numpy(float))
    yf = np.interp(frame, g["frame_num"].to_numpy(float), g["y"].to_numpy(float))
    return float(xf), float(yf)


def nearest_player_at_second(stats: DatasetStats, second: float) -> str | None:
    """离球最近的球员球衣号（独立插值 + 欧氏距离，用于 verify 类金标准）。"""
    bp = ball_at_frame(stats, int(round(second * 30)))
    if bp is None:
        return None
    best_tid, best_d = None, None
    for tid in stats.player_ids:
        pos = player_position_at_second(stats, tid, second)
        if pos is None:
            continue
        d = np.hypot(pos[0] - bp[0], pos[1] - bp[1])
        if best_d is None or d < best_d:
            best_d, best_tid = d, int(tid)
    return stats.jersey(best_tid) if best_tid is not None else None


def ball_player_distance_at_second(stats: DatasetStats, tid: int, second: float) -> float | None:
    """球与指定球员在给定秒的欧氏距离（米，独立插值）。"""
    bp = ball_at_frame(stats, int(round(second * 30)))
    pos = player_position_at_second(stats, tid, second)
    if bp is None or pos is None:
        return None
    dx = (pos[0] - bp[0]) / _PX_PER_M_X
    dy = (pos[1] - bp[1]) / _PX_PER_M_Y
    return round(float(np.hypot(dx, dy)), 1)


# ── 领域量化指标（1.1）：战术区域 / 威胁值 / 球速 ──────────────
# 方法论参照：
# - 战术区域：按距球门线距离与横向位置划分为 禁区/前场/中场 × 中路/边路
#   （足球战术分析的通用空间划分，如 Metrica / StatsBomb 的进攻三区概念）；
# - 威胁值：参考 Karun Singh 的 xT（Expected Threat, 2018）思路 —— 球越接近
#   球门、越靠中路，对得分的威胁越高；此处用无训练数据的确定性启发式模型
#   （距门线距离线性衰减 + 中路加成），保证独立可重算、不依赖被测系统。

def _normalize(stats: DatasetStats, x: float, y: float) -> tuple[float, float]:
    """球场内点归一化到 [0,1]×[0,1]：t=进攻轴（x，门线到门线），s=横向（y，边线到边线）。

    固定 1200×700 场地（与 agents/constants 一致），双球门对称、不依赖进攻方向。
    """
    t = x / _FIELD_WIDTH_PX
    s = y / _FIELD_HEIGHT_PX
    return min(max(t, 0.0), 1.0), min(max(s, 0.0), 1.0)


def zone_of(stats: DatasetStats, x: float, y: float) -> str:
    """战术区域：禁区内 / 前场中路 / 前场边路 / 中场中路 / 中场边路。

    near = 距最近球门线距离（0=门线, 0.5=中线）：<0.15 禁区，<0.40 前场，否则中场；
    横向 s<0.25 或 >0.75 为边路。
    """
    t, s = _normalize(stats, x, y)
    near = min(t, 1.0 - t)
    side = "边路" if (s < 0.25 or s > 0.75) else "中路"
    if near < 0.15:
        return "禁区"
    if near < 0.40:
        return f"前场{side}"
    return f"中场{side}"


def threat_of(stats: DatasetStats, x: float, y: float) -> float:
    """威胁值 [0,1]：参考 xT 方法论的中性启发式（不需方向假设）。

    threat = 中路加成 × max(0, 1 - 2·near)；near=0（门线）→1.0，中线→0。
    中路加成 s=0.5 时为 1.0，贴边 s∈{0,1} 时 0.65。
    """
    t, s = _normalize(stats, x, y)
    near = min(t, 1.0 - t)
    mid_bonus = 1.0 - 0.35 * (2.0 * s - 1.0) ** 2
    return round(mid_bonus * max(0.0, 1.0 - 2.0 * near), 3)


def ball_speed_at_frame(stats: DatasetStats, frame: int) -> float:
    """球速（像素/秒）在指定帧所在相邻段的值；越界取最近段。"""
    if not stats.ball_speed_segments:
        return 0.0
    f0, f1, sp = stats.ball_speed_segments[0]
    for a, b, spd in stats.ball_speed_segments:
        if a <= frame <= b:
            return round(spd, 1)
        f0, f1, sp = a, b, spd
    return round(sp, 1)


def ball_zone_at_frame(stats: DatasetStats, frame: int) -> str | None:
    pos = ball_at_frame(stats, frame)
    if pos is None:
        return None
    return zone_of(stats, pos[0], pos[1])


def ball_side_at_frame(stats: DatasetStats, frame: int) -> str | None:
    pos = ball_at_frame(stats, frame)
    if pos is None:
        return None
    _, s = _normalize(stats, pos[0], pos[1])
    return "边路" if (s < 0.25 or s > 0.75) else "中路"


def ball_threat_at_frame(stats: DatasetStats, frame: int) -> float | None:
    pos = ball_at_frame(stats, frame)
    if pos is None:
        return None
    return threat_of(stats, pos[0], pos[1])


def threat_max(stats: DatasetStats) -> float:
    """球路全程最大威胁值（对全部原始球帧）。"""
    vals = [threat_of(stats, *stats.ball_pos[f][:2]) for f in stats.ball_frames]
    return round(max(vals), 3) if vals else 0.0


def player_primary_zone(stats: DatasetStats, tid: int) -> str:
    """球员主要活动区域（平均位置映射到战术区域）。"""
    pos = stats.player_mean_pos.get(int(tid))
    if pos is None:
        return "无数据"
    return zone_of(stats, pos[0], pos[1])


# ── 独立事件检测（1.1：事件类金标准，不经被测系统） ──────────
# 基于球速/高度的确定性分段；速度阈值与事件类型对应足球常见事件。
# 阈值说明（像素/秒，启发式）：<40 视为控球/静止；40-120 视为带球/短传；
# ≥120 且球高度高（z≥0.5m）视为长传，≥120 且低球视为快速推进；
# 高速且接近球门线（威胁≥0.8）额外标记射门。
SPEED_STILL = 40.0
SPEED_DRIVE = 120.0
HIGH_BALL_Z = 0.5
SHOT_THREAT = 0.8


def ball_event_types(stats: DatasetStats) -> list[str]:
    """独立检测的数据集内发生的事件类型（排序后的集合）。"""
    types: set[str] = set()
    for f0, _f1, spd in stats.ball_speed_segments:
        z0 = stats.ball_pos.get(int(f0), (0.0, 0.0, 0.0))[2]
        if spd < SPEED_STILL:
            types.add("控球")
        elif spd < SPEED_DRIVE:
            types.add("带球")
        else:
            types.add("长传" if z0 >= HIGH_BALL_Z else "快速推进")
    for f in stats.ball_frames:
        x, y, _z = stats.ball_pos[f]
        if threat_of(stats, x, y) >= SHOT_THREAT and ball_speed_at_frame(stats, f) >= SPEED_DRIVE:
            types.add("射门")
            break
    return sorted(types)


# ── 金标准表达式解析（cases JSON 中 value: "auto:<expr>"） ──

def resolve_gold(expr: str, stats: DatasetStats) -> float | int | str:
    """把 auto 表达式解析为具体金标准值。支持：

    player_count / duration_s / ball_zmax
    ball_x@F ball_y@F ball_z@F        （F=帧号，插值；x/y 为米，z 为米）
    distance:T / avg_speed:T / top_speed:T   （T=track_id；米 / 米每秒）
    distance_winner:T1,T2             （累计位移更多者 → "7号"）
    speed_winner:T1,T2
    ball_speed_max / ball_speed_avg           （球速 米/秒，1.1）
    ball_zone:F / ball_side:F                 （战术区域/横位，1.1）
    threat_at_frame:F / threat_max            （威胁值 [0,1]，1.1）
    player_primary_zone:T                     （球员主活动区域，1.1）
    """
    expr = expr.strip()
    if expr == "player_count":
        return stats.player_count
    if expr == "duration_s":
        return round(stats.duration_s, 1)
    if expr == "ball_zmax":
        return round(stats.ball_z_max, 1)
    if expr == "ball_speed_max":
        return stats.ball_speed_max_ms
    if expr == "ball_speed_avg":
        return stats.ball_speed_avg_ms
    if expr == "threat_max":
        return threat_max(stats)
    if expr == "event_types":
        return ",".join(ball_event_types(stats))
    m = re.fullmatch(r"ball_([xyz])@(\d+)", expr)
    if m:
        pos = ball_at_frame(stats, int(m.group(2)))
        if pos is None:
            raise ValueError(f"帧 {m.group(2)} 超出球数据范围")
        axis = m.group(1)
        if axis == "x":
            return round(pos[0] / _PX_PER_M_X, 1)
        if axis == "y":
            return round(pos[1] / _PX_PER_M_Y, 1)
        return round(pos[2], 1)
    m = re.fullmatch(r"ball_zone@(\d+)", expr)
    if m:
        z = ball_zone_at_frame(stats, int(m.group(1)))
        if z is None:
            raise ValueError(f"帧 {m.group(1)} 超出球数据范围")
        return z
    m = re.fullmatch(r"ball_side@(\d+)", expr)
    if m:
        z = ball_side_at_frame(stats, int(m.group(1)))
        if z is None:
            raise ValueError(f"帧 {m.group(1)} 超出球数据范围")
        return z
    m = re.fullmatch(r"threat_at_frame@(\d+)", expr)
    if m:
        v = ball_threat_at_frame(stats, int(m.group(1)))
        if v is None:
            raise ValueError(f"帧 {m.group(1)} 超出球数据范围")
        return v
    m = re.fullmatch(r"player_primary_zone:(\d+)", expr)
    if m:
        return player_primary_zone(stats, int(m.group(1)))
    m = re.fullmatch(r"player_([xy])_at_second:(\d+),([\d.]+)", expr)
    if m:
        pos = player_position_at_second(stats, int(m.group(2)), float(m.group(3)))
        if pos is None:
            raise ValueError(f"无球员 {m.group(2)} 位置数据")
        axis = m.group(1)
        scale = _PX_PER_M_X if axis == "x" else _PX_PER_M_Y
        return round(pos[0 if axis == "x" else 1] / scale, 1)
    m = re.fullmatch(r"nearest_player_at_second:([\d.]+)", expr)
    if m:
        v = nearest_player_at_second(stats, float(m.group(1)))
        if v is None:
            raise ValueError(f"秒 {m.group(1)} 无球/球员数据")
        return v
    m = re.fullmatch(r"ball_player_distance_at_second:(\d+),([\d.]+)", expr)
    if m:
        v = ball_player_distance_at_second(stats, int(m.group(1)), float(m.group(2)))
        if v is None:
            raise ValueError(f"无球员 {m.group(1)} 或球在秒 {m.group(2)} 的数据")
        return v
    m = re.fullmatch(r"(distance|avg_speed|top_speed):(\d+)", expr)
    if m:
        table = {"distance": stats.player_distance_m, "avg_speed": stats.player_avg_speed_ms,
                 "top_speed": stats.player_top_speed_ms}[m.group(1)]
        return round(table.get(int(m.group(2)), 0.0), 1)
    m = re.fullmatch(r"(distance|speed)_winner:(\d+),(\d+)", expr)
    if m:
        table = (stats.player_distance_m if m.group(1) == "distance"
                 else stats.player_top_speed_ms)
        t1, t2 = int(m.group(2)), int(m.group(3))
        return stats.jersey(t1) if table.get(t1, 0) >= table.get(t2, 0) else stats.jersey(t2)
    if expr in ("speed_winner_top", "distance_winner_top"):
        table = (stats.player_top_speed_ms if expr.startswith("speed")
                 else stats.player_distance_m)
        best = max(table, key=lambda t: table[t])
        return stats.jersey(best)
    raise ValueError(f"无法解析金标准表达式: {expr!r}")


# ── 裸调用基线摘要（确定性模板，修订 #3） ───────────────────

BARE_SYSTEM_PROMPT = (
    "你是一名足球比赛数据分析专家。用户将提供一段比赛的球员追踪数据与足球轨迹数据"
    "（原始 CSV 及统计摘要）。请仅基于这些数据回答问题；数据不足以回答时，"
    "必须如实说明「数据不足/数据中没有该信息」，严禁编造。"
    "回答使用中文。回答规范：先给出直接的形势/战术判断（进攻态势、攻防转换、威胁程度、"
    "球员战术角色与活动区域等），再给出数据依据；默认把统计数值翻译为自然语言判断"
    "（\"绝大多数时间\"\"明显更积极\"\"节奏偏快\"），严禁罗列成串的统计数字；"
    "仅当用户明确索要具体数值（如\"最快速度是多少\"\"峰值是多少\"\"具体几次\"）时，"
    "才直接引用摘要中的精确数值作答（速度单位米/秒、距离单位米、高度单位米）。"
)


def build_bare_summary(stats: DatasetStats) -> str:
    """确定性摘要模板：全量原始数据 + 独立统计表（信息给足原则）。

    固定字段顺序与内容，保证跨运行逐字节一致（可复现）。
    """
    lines: list[str] = []
    lines.append("【比赛数据总览】")
    lines.append(f"- 球员人数: {stats.player_count}")
    lines.append(f"- 帧号范围: 0~{stats.max_frame}（采样率 30fps，时长约 {stats.duration_s:.1f} 秒）")
    lines.append(f"- 球员列表（track_id=球衣号）: {', '.join(stats.jersey(t) for t in stats.player_ids)}")
    roster = ", ".join(
        f"{stats.jersey(t)}={TEAM_NAMES.get(c, c)}" for t, c in stats.player_teams.items()
    )
    lines.append(f"- 球员所属: {roster}（C=门将）")
    lines.append(f"- 球高度峰值 z_max: {stats.ball_z_max:.1f}")

    lines.append("")
    lines.append("【球员统计表】（位移单位米、速度单位米/秒；由原始相邻采样差分累计")
    lines.append("  并按帧跨度折算，x/y 轴分别按 105m/68m 场地标定）")
    lines.append("球衣号 | 累计位移(m) | 平均速度(m/s) | 峰值速度(m/s)")
    for t in stats.player_ids:
        lines.append(
            f"{stats.jersey(t)} | {stats.player_distance_m.get(t, 0):.1f} "
            f"| {stats.player_avg_speed_ms.get(t, 0):.1f} | {stats.player_top_speed_ms.get(t, 0):.1f}"
        )

    lines.append("")
    lines.append("【足球轨迹原始数据】（frame_num,x,y,z；z=球心离地高度，米）")
    for f in stats.ball_frames:
        x, y, z = stats.ball_pos[f]
        lines.append(f"{f},{x:.1f},{y:.1f},{z:.1f}")

    lines.append("")
    lines.append("【球员轨迹原始数据】（frame_num,track_id,x,y,color）")
    lines.append(stats.person_csv.read_text(encoding="utf-8").strip())

    return "\n".join(lines)
