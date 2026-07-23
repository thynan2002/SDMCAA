"""球员轨迹追踪器。

从 CSV 中按 track_id 提取每个球员的完整运动轨迹，
计算跑动距离、速度、活动热区、与球的关系等指标。
球场尺寸：1200 × 700。
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from math import hypot
from pathlib import Path
import csv
import re
from statistics import mean
from typing import Any

from agents.constants import (
    MAX_POSSESSION_DIST,
    GROUND_PROXIMITY_RADIUS,
    AERIAL_Z_THRESHOLD,
    PIXELS_PER_METER,
)

# 长传事件的最小距离（px）：约 20 米，短于此的高速移动归为"传球"
LONG_PASS_MIN_DIST = 20 * PIXELS_PER_METER


# ── 轻量 CSV 读取（替代已删除的 legacy.input_loader） ──────────

# 文件名模式：{prefix}-{kind}.csv，如 "10s-person.csv", "10s-ball.csv"
FILE_PATTERN = re.compile(r"^(?P<prefix>\d+s)[-_](?P<kind>person\d*d|ball|soccer\d*d)\.csv$", re.IGNORECASE)


def _safe_int(value: Any) -> int | None:
    """安全整数转换。"""
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


REQUIRED_PERSON_COLS = {"frame_num", "track_id", "x", "y"}
REQUIRED_BALL_COLS = {"frame_num", "x", "y"}


def _validate_csv_rows(
    headers: list[str],
    rows: list[dict[str, str]],
    required_cols: set[str],
    kind: str,
) -> None:
    """校验 CSV 的必需列和非空性，失败时抛出 ValueError。"""
    header_set = set(headers)
    missing = required_cols - header_set
    if missing:
        raise ValueError(f"{kind} CSV 缺少必需列: {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError(f"{kind} CSV 无数据行")


def _read_csv_rows(file_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取 CSV 返回 (headers, rows)。"""
    path = Path(file_path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows


@dataclass
class CsvAuditResult:
    """CSV 审核结果（占位，保持兼容）。"""
    file_path: str = ""
    row_count: int = 0
    columns: list[str] = field(default_factory=list)


@dataclass
class TestInputAudit:
    """测试输入审核（占位，保持兼容）。"""
    csv_dir: Path = field(default_factory=Path)
    files: list[str] = field(default_factory=list)
    audit_results: list[CsvAuditResult] = field(default_factory=list)


def resolve_test_input_layout(test_input_dir: str | Path) -> TestInputAudit:
    """解析 TestInput 目录结构。"""
    base = Path(test_input_dir)
    files_dir = base / "Files"
    csv_dir = files_dir if files_dir.is_dir() else base
    return TestInputAudit(csv_dir=csv_dir)


# ── 球场分区：3×3 网格 ──────────────────────────────────────────
# x 轴三等分（基于 1200 宽度）
X_ZONES = [
    ("左路", 0, 400),
    ("中路", 400, 800),
    ("右路", 800, 1200),
]
# y 轴三等分（基于 700 高度）
Y_ZONES = [
    ("后场", 0, 233),
    ("中场", 233, 466),
    ("前场", 466, 700),
]

ZONE_LABELS: dict[tuple[int, int], str] = {}
for xi, (x_label, _, _) in enumerate(X_ZONES):
    for yi, (y_label, _, _) in enumerate(Y_ZONES):
        ZONE_LABELS[(xi, yi)] = f"{x_label}{y_label}"


def classify_position(avg_y: float) -> str:
    """按平均 y 坐标分类前后位置。"""
    if avg_y < 220:
        return "后卫"
    elif avg_y < 380:
        return "中场"
    return "前锋"


def classify_side(avg_x: float) -> str:
    """按平均 x 坐标分类边路/中路。"""
    if avg_x < 400:
        return "左"
    elif avg_x > 800:
        return "右"
    return "中"


def _which_zone(x: float, y: float) -> str:
    xi = 0
    yi = 0
    for i, (_, lo, hi) in enumerate(X_ZONES):
        if lo <= x < hi:
            xi = i
            break
    else:
        xi = 2 if x >= X_ZONES[-1][1] else 0
    for i, (_, lo, hi) in enumerate(Y_ZONES):
        if lo <= y < hi:
            yi = i
            break
    else:
        yi = 2 if y >= Y_ZONES[-1][1] else 0
    return ZONE_LABELS.get((xi, yi), "未知区域")


@dataclass
class PlayerTrajectory:
    """单个球员（track_id）的完整轨迹。"""
    prefix: str
    track_id: str
    color: str
    # 逐帧数据（按 frame_num 排序）
    frames: list[int] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    # 衍生指标
    total_distance: float = 0.0          # 总跑动距离
    avg_speed: float = 0.0               # 平均速度（单位/秒）
    max_speed: float = 0.0               # 峰值速度
    speed_curve: list[float] = field(default_factory=list)  # 逐帧速度
    x_range: tuple[float, float] = (0, 0)
    y_range: tuple[float, float] = (0, 0)
    field_coverage_pct: float = 0.0      # 覆盖面积占全场百分比
    dominant_zone: str = ""              # 最常出现的区域
    zone_distribution: dict[str, float] = field(default_factory=dict)  # 各区域时间占比
    ball_distances: list[float] = field(default_factory=list)  # 逐帧与球距离
    ball_proximity_pct: float = 0.0      # 接近球的时间占比（<30单位）
    avg_ball_distance: float = 0.0       # 平均球距
    is_goalkeeper: bool = False          # 是否为守门员

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def jersey_label(self) -> str:
        tid = str(self.track_id).strip()
        return f"{tid}号" if tid else "未知号"

    def avg_x(self) -> float:
        return mean(self.xs) if self.xs else 0.0

    def avg_y(self) -> float:
        return mean(self.ys) if self.ys else 0.0


@dataclass
class BallEvent:
    """足球关键事件。"""
    event_type: str           # "传球"/"射门"/"带球"/"解围"/"争顶"/"静止"
    start_frame: int
    end_frame: int
    start_pos: tuple[float, float, float]
    end_pos: tuple[float, float, float]
    speed: float              # 移动速度
    distance: float           # 移动距离
    direction: str            # 方向描述
    nearest_player: str       # 起点最近球员的 jersey_label（传球发起者候选）
    nearest_player_end: str   # 终点最近球员的 jersey_label（接球者候选）
    nearest_dist: float = 0.0      # 起点最近球员实际距离（"未知"判定的核查依据）
    end_nearest_dist: float = 0.0  # 终点最近球员实际距离（>MAX_POSSESSION_DIST 时终点标"未知"）


@dataclass
class BallTrajectoryAnalysis:
    """足球轨迹的完整运动分析。"""
    # 整体
    total_frames: int = 0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    total_distance: float = 0.0
    # 方向统计
    direction_counts: dict[str, int] = field(default_factory=dict)
    # 事件列表
    events: list[BallEvent] = field(default_factory=list)
    # 连续帧范围（检测跳帧）
    frame_ranges: list[tuple[int, int]] = field(default_factory=list)
    # 高空球占比（z > 1 的帧）
    aerial_pct: float = 0.0


@dataclass
class PrefixPlayerCorpus:
    """一个前缀下所有球员的轨迹集合。"""
    prefix: str
    players: dict[str, PlayerTrajectory] = field(default_factory=dict)  # track_id → trajectory
    ball_frames: dict[int, tuple[float, float, float]] = field(default_factory=dict)  # frame → (x, y, z)
    ball_analysis: BallTrajectoryAnalysis | None = None
    audit: TestInputAudit | None = None
    frame_player_index: dict[int, dict[str, tuple[float, float]]] = field(default_factory=dict)  # frame → {track_id: (x, y)}
    source_files: tuple[str, str] = ("", "")  # (球员CSV, 球CSV) 原始路径，供反事实数据拼接导出使用

    @property
    def player_count(self) -> int:
        return len(self.players)

    def sorted_players(self) -> list[PlayerTrajectory]:
        return sorted(self.players.values(), key=lambda p: p.total_distance, reverse=True)

    def field_players(self) -> list[PlayerTrajectory]:
        """非守门员球员。"""
        return [p for p in self.players.values() if not p.is_goalkeeper]

    def goalkeepers(self) -> list[PlayerTrajectory]:
        return [p for p in self.players.values() if p.is_goalkeeper]


def _compute_player_trajectory(
    prefix: str,
    track_id: str,
    person_rows: list[dict[str, str]],
    ball_by_frame: dict[int, tuple[float, float, float]],
) -> PlayerTrajectory:
    """从一个人物的所有行构建 PlayerTrajectory。"""
    # 按 frame_num 排序
    sorted_rows = sorted(person_rows, key=lambda r: _safe_int(r.get("frame_num")) or 0)

    frames: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    color = ""

    for row in sorted_rows:
        frame = _safe_int(row.get("frame_num"))
        if frame is None:
            continue
        try:
            x = float(row.get("x", ""))
            y = float(row.get("y", ""))
        except (TypeError, ValueError):
            continue
        frames.append(frame)
        xs.append(x)
        ys.append(y)
        if not color:
            color = str(row.get("color", "")).strip()

    if not frames:
        return PlayerTrajectory(prefix=prefix, track_id=track_id, color=color)

    # 速度曲线（帧间速度 * 30 / 帧差 = 每秒速度）
    raw_speed: list[float] = []
    total_distance = 0.0
    for i in range(1, len(xs)):
        dist = hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        # 按实际帧差归一化，避免跳帧时速度被高估
        frame_diff = max(1, frames[i] - frames[i - 1])
        raw_speed.append(dist * 30 / frame_diff)
        total_distance += dist

    # 5 点滑动窗口均值平滑（半径 W=3，覆盖 i-2 到 i+2），减少 tracking 噪声对速度曲线的影响
    speed_curve: list[float] = []
    W = 3
    for i in range(len(raw_speed)):
        lo = max(0, i - W + 1)
        hi = min(len(raw_speed), i + W)
        speed_curve.append(sum(raw_speed[lo:hi]) / (hi - lo))

    avg_speed = mean(speed_curve) if speed_curve else 0.0
    max_speed = max(speed_curve) if speed_curve else 0.0

    # 活动范围
    x_range = (min(xs), max(xs))
    y_range = (min(ys), max(ys))
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    field_coverage_pct = round((width * height) / (1200 * 700) * 100, 2)

    # 区域分布
    zone_counter: dict[str, int] = defaultdict(int)
    for x, y in zip(xs, ys):
        zone_counter[_which_zone(x, y)] += 1
    total_positions = sum(zone_counter.values()) or 1
    zone_distribution = {zone: round(cnt / total_positions * 100, 1) for zone, cnt in zone_counter.items()}
    dominant_zone = max(zone_counter, key=zone_counter.get) if zone_counter else ""

    # 与球距离
    ball_distances: list[float] = []
    for frame, x, y in zip(frames, xs, ys):
        ball_pos = ball_by_frame.get(frame)
        if ball_pos:
            ball_distances.append(hypot(x - ball_pos[0], y - ball_pos[1]))
    avg_ball_distance = mean(ball_distances) if ball_distances else 0.0
    ball_proximity_pct = round(
        sum(1 for d in ball_distances if d < GROUND_PROXIMITY_RADIUS) / max(1, len(ball_distances)) * 100, 1
    )

    return PlayerTrajectory(
        prefix=prefix,
        track_id=track_id,
        color=color,
        frames=frames,
        xs=xs,
        ys=ys,
        total_distance=round(total_distance, 2),
        avg_speed=round(avg_speed, 2),
        max_speed=round(max_speed, 2),
        speed_curve=[round(s, 2) for s in speed_curve],
        x_range=x_range,
        y_range=y_range,
        field_coverage_pct=field_coverage_pct,
        dominant_zone=dominant_zone,
        zone_distribution=zone_distribution,
        ball_distances=[round(d, 2) for d in ball_distances],
        ball_proximity_pct=ball_proximity_pct,
        avg_ball_distance=round(avg_ball_distance, 2),
    )


def build_player_corpus(test_input_dir: str | Path) -> dict[str, PrefixPlayerCorpus]:
    """读入 TestInput 中所有 CSV，按前缀构建每位球员的轨迹。

    返回 {prefix: PrefixPlayerCorpus}。
    """
    layout = resolve_test_input_layout(test_input_dir)
    csv_dir = layout.csv_dir

    # 按前缀收集文件
    prefix_files: dict[str, dict[str, Path]] = defaultdict(dict)
    for file_path in sorted(csv_dir.iterdir()):
        if not file_path.is_file():
            continue
        match = FILE_PATTERN.match(file_path.name)
        if not match:
            continue
        prefix = match.group("prefix")
        kind = match.group("kind")
        prefix_files[prefix][kind] = file_path

    all_corpora: dict[str, PrefixPlayerCorpus] = {}

    for prefix, files in sorted(prefix_files.items()):
        # 读取所有 person 行
        all_person_rows: list[dict[str, str]] = []
        person_files = {k: v for k, v in files.items() if k.startswith("person")}
        for file_path in person_files.values():
            _, rows = _read_csv_rows(file_path)
            all_person_rows.extend(rows)

        # 读取所有 ball/soccer 行，构建 frame→(x,y,z) 映射
        ball_by_frame: dict[int, tuple[float, float, float]] = {}
        ball_files = {k: v for k, v in files.items() if k in ("ball", "soccer", "soccer3d")}
        for file_path in ball_files.values():
            _, rows = _read_csv_rows(file_path)
            for row in rows:
                frame = _safe_int(row.get("frame_num"))
                if frame is None:
                    continue
                try:
                    x = float(row.get("x", ""))
                    y = float(row.get("y", ""))
                    z = float(row.get("z", "0") or 0)
                except (TypeError, ValueError):
                    continue
                ball_by_frame[frame] = (x, y, z)

        # 校验数据
        _validate_coord_range(all_person_rows, "球员")
        _validate_coord_range(
            [{"x": str(v[0]), "y": str(v[1])} for v in ball_by_frame.values()],
            "球",
        )

        # 按 track_id 分组
        by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in all_person_rows:
            tid = str(row.get("track_id", "")).strip()
            if not tid:
                continue
            by_track[tid].append(row)

        # 为每个 track_id 构建轨迹
        players: dict[str, PlayerTrajectory] = {}
        for tid, rows in by_track.items():
            players[tid] = _compute_player_trajectory(prefix, tid, rows, ball_by_frame)

        # 标记守门员
        _detect_goalkeepers(players)

        # 构建帧索引
        frame_player_index = _build_frame_player_index(players)

        # 分析球轨迹
        ball_analysis = _analyze_ball_trajectory(ball_by_frame, players)

        all_corpora[prefix] = PrefixPlayerCorpus(
            prefix=prefix,
            players=players,
            ball_frames=ball_by_frame,
            ball_analysis=ball_analysis,
            frame_player_index=frame_player_index,
        )

    return all_corpora


def _validate_coord_range(
    rows: list[dict[str, str]],
    kind: str,
    x_max: float = 1200,
    y_max: float = 700,
) -> None:
    """校验坐标值是否为有效数值。

    注意：不再拒绝超出场地边界（x>1200 或 y>700）的坐标。
    实际足球场景中，发球位置、界外球事件完全可能发生在标准场地
    之外，这些越界坐标是合理的，必须予以保留。此处仅校验坐标是否
    为可解析的数值（防止 CSV 损坏导致 NaN/字符串混入下游计算）。
    """
    for i, row in enumerate(rows):
        try:
            x = float(row.get("x", ""))
            y = float(row.get("y", ""))
            # 仅拒绝非有限值（NaN / Inf），允许任何有限坐标（含越界）
            if not (x == x and y == y):  # NaN != NaN
                raise ValueError(f"{kind} CSV 第 {i + 2} 行坐标为 NaN")
        except (TypeError, ValueError) as e:
            if "NaN" in str(e):
                raise ValueError(f"{kind} CSV 第 {i + 2} 行坐标无效: {row.get('x')}, {row.get('y')}")
            raise ValueError(f"{kind} CSV 第 {i + 2} 行坐标无效: {row.get('x')}, {row.get('y')}")


def build_corpus_from_paths(
    person_csv: str | Path,
    ball_csv: str | Path,
) -> PrefixPlayerCorpus:
    """从两个显式 CSV 路径构建单场比赛语料。

    Args:
        person_csv: 球员轨迹 CSV（frame_num, track_id, x, y, color）
        ball_csv: 球轨迹 CSV（frame_num, x, y, z）
    """
    person_path = Path(person_csv)
    ball_path = Path(ball_csv)

    # 推断前缀
    m = re.search(r"(\d+s)", person_path.name, re.IGNORECASE)
    prefix = m.group(1) if m else person_path.stem

    person_headers, person_rows = _read_csv_rows(person_path)
    ball_headers, ball_rows = _read_csv_rows(ball_path)
    _validate_csv_rows(person_headers, person_rows, REQUIRED_PERSON_COLS, "球员")
    _validate_csv_rows(ball_headers, ball_rows, REQUIRED_BALL_COLS, "球")
    _validate_coord_range(person_rows, "球员")
    _validate_coord_range(ball_rows, "球")

    ball_by_frame: dict[int, tuple[float, float, float]] = {}
    for row in ball_rows:
        frame = _safe_int(row.get("frame_num"))
        if frame is None:
            continue
        try:
            ball_by_frame[frame] = (
                float(row.get("x", "")),
                float(row.get("y", "")),
                float(row.get("z", "0") or 0),
            )
        except (TypeError, ValueError):
            continue

    by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in person_rows:
        tid = str(row.get("track_id", "")).strip()
        if not tid:
            continue
        by_track[tid].append(row)

    players: dict[str, PlayerTrajectory] = {}
    for tid, rows in by_track.items():
        players[tid] = _compute_player_trajectory(prefix, tid, rows, ball_by_frame)

    _detect_goalkeepers(players)
    ball_analysis = _analyze_ball_trajectory(ball_by_frame, players)

    # 构建帧索引
    frame_player_index = _build_frame_player_index(players)

    return PrefixPlayerCorpus(
        prefix=prefix,
        players=players,
        ball_frames=ball_by_frame,
        ball_analysis=ball_analysis,
        frame_player_index=frame_player_index,
        source_files=(str(person_path), str(ball_path)),
    )


def _detect_goalkeepers(players: dict[str, PlayerTrajectory]) -> None:
    """多维度标记守门员：位置 + 跑动量 + 活动范围 + 颜色稀有度。"""
    # 先收集所有球员的指标，避免重复计算
    player_metrics: dict[str, dict] = {}
    for pid, p in players.items():
        avg_y = p.avg_y()
        total_dist = p.total_distance
        coverage = p.field_coverage_pct
        x_range = p.x_range
        y_range = p.y_range
        player_metrics[pid] = {
            "avg_y": avg_y,
            "total_distance": total_dist,
            "coverage": coverage,
            "near_goal": avg_y < 180 and 450 <= p.avg_x() <= 750,
            "range_small": (x_range[1] - x_range[0]) < 300 and (y_range[1] - y_range[0]) < 150,
            "rare_color": any(
                p.color != other.color
                for other in players.values()
                if other.track_id != p.track_id
            ) if len(players) > 1 else True,
        }

    for pid, p in players.items():
        m = player_metrics[pid]

        # 强信号：在球门附近且活动范围极小 → 门将
        if m["near_goal"] and m["range_small"]:
            p.is_goalkeeper = True
            continue

        # 强信号：平均 y < 120（靠近底线）且跑动量很低
        if m["avg_y"] < 120 and m["total_distance"] < 100:
            p.is_goalkeeper = True
            continue

        # 中等信号：颜色稀有 + y 靠后 + 低跑动量
        if m["rare_color"] and m["avg_y"] < 250 and m["total_distance"] < 150:
            p.is_goalkeeper = True
            continue

        # 兜底：颜色 C 仅在 1-2 人时 → 门将（兼容已知数据格式）
        if p.color == "C" and sum(1 for o in players.values() if o.color == "C") <= 2:
            p.is_goalkeeper = True

        # 兜底：任何颜色只有 1 人且跑动量极小 → 门将
        if not p.is_goalkeeper and m["rare_color"] and m["total_distance"] < 50:
            p.is_goalkeeper = True


def _build_frame_player_index(
    players: dict[str, PlayerTrajectory],
) -> dict[int, dict[str, tuple[float, float]]]:
    """构建 {frame: {track_id: (x, y)}} 索引，O(1) 球员位置查询。

    注意：该索引只含"恰好有检测行"的帧，跟踪数据通常是稀疏不规则采样的，
    因此它只适合精确帧查询。需要"任意帧全员位置"（如球路事件的最近球员
    判定）时，必须使用 position_at_frame / positions_at_frame 的插值结果，
    否则大部分球员在大多数帧会被误判为"不在场"。
    """
    index: dict[int, dict[str, tuple[float, float]]] = {}
    for tid, player in players.items():
        for frame, x, y in zip(player.frames, player.xs, player.ys):
            index.setdefault(frame, {})[tid] = (x, y)
    return index


def position_at_frame(traj: PlayerTrajectory, frame: int) -> tuple[float, float] | None:
    """估计球员在任意帧的位置（相邻样本线性插值）。

    语义（与模拟引擎的"在场"判定约定一致）：
    - frame 早于首个样本 → None（尚未进入视野）
    - frame 位于两个样本之间 → 线性插值
    - frame 晚于最后样本 → 最后已知位置（last-known 语义）
    """
    frames, xs, ys = traj.frames, traj.xs, traj.ys
    if not frames or frame < frames[0]:
        return None
    if frame >= frames[-1]:
        return xs[-1], ys[-1]
    i = bisect_right(frames, frame)
    f0, f1 = frames[i - 1], frames[i]
    if frame == f0:
        return xs[i - 1], ys[i - 1]
    t = (frame - f0) / (f1 - f0)
    return xs[i - 1] + (xs[i] - xs[i - 1]) * t, ys[i - 1] + (ys[i] - ys[i - 1]) * t


def positions_at_frame(
    players: dict[str, PlayerTrajectory], frame: int,
) -> dict[str, tuple[float, float]]:
    """全体在场球员在某帧的插值位置 {track_id: (x, y)}。"""
    result: dict[str, tuple[float, float]] = {}
    for tid, p in players.items():
        pos = position_at_frame(p, frame)
        if pos is not None:
            result[tid] = pos
    return result


_DIRECTION_NAMES = {
    (1, 0): "向右", (-1, 0): "向左",
    (0, 1): "向前", (0, -1): "向后",
    (1, 1): "向右前", (-1, 1): "向左前",
    (1, -1): "向右后", (-1, -1): "向左后",
}


def _ball_direction(dx: float, dy: float) -> str:
    """球移动方向。"""
    tx = 1 if dx > 2 else -1 if dx < -2 else 0
    ty = 1 if dy > 2 else -1 if dy < -2 else 0
    return _DIRECTION_NAMES.get((tx, ty), "原地")


def _analyze_ball_trajectory(
    ball_by_frame: dict[int, tuple[float, float, float]],
    players: dict[str, PlayerTrajectory],
) -> BallTrajectoryAnalysis:
    """分析球的运动轨迹事件：传球、射门、带球等。"""
    if not ball_by_frame:
        return BallTrajectoryAnalysis()

    sorted_frames = sorted(ball_by_frame.items())
    frames = [f for f, _ in sorted_frames]
    positions = [pos for _, pos in sorted_frames]

    if len(positions) < 2:
        return BallTrajectoryAnalysis(total_frames=len(positions))

    # 基础统计
    speeds: list[float] = []
    distances: list[float] = []
    for i in range(1, len(positions)):
        d = hypot(positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
        distances.append(d)
        # 按实际帧差归一化，避免跳帧时速度被高估
        frame_diff = max(1, frames[i] - frames[i - 1])
        speeds.append(d * 30 / frame_diff)

    total_distance = sum(distances)
    avg_speed = mean(speeds) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0

    # 方向统计
    dir_counts: dict[str, int] = defaultdict(int)
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        dir_counts[_ball_direction(dx, dy)] += 1

    # 连续帧范围
    frame_ranges: list[tuple[int, int]] = []
    if frames:
        range_start = frames[0]
        prev = frames[0]
        for f in frames[1:]:
            if f - prev > 5:  # 跳帧 >5，认为球出界或丢失
                frame_ranges.append((range_start, prev))
                range_start = f
            prev = f
        frame_ranges.append((range_start, prev))

    # 高空球占比（z 单位为米，阈值见 agents.constants）
    aerial = sum(1 for _, _, z in positions if z > AERIAL_Z_THRESHOLD)
    aerial_pct = round(aerial / len(positions) * 100, 1) if positions else 0.0

    # ── 事件检测 ──
    events: list[BallEvent] = []

    # 从数据分布计算自适应阈值（排序分位数，避免硬编码）
    sorted_speeds = sorted(speeds)
    sorted_distances = sorted(distances)
    n_speed = len(sorted_speeds)
    n_dist = len(sorted_distances)

    def pctile(arr, n, p):
        idx = max(0, min(n - 1, int(n * p / 100)))
        return arr[idx]

    speed_high = pctile(sorted_speeds, n_speed, 85) if n_speed >= 3 else 500.0
    speed_medium = pctile(sorted_speeds, n_speed, 60) if n_speed >= 3 else 150.0
    speed_low = pctile(sorted_speeds, n_speed, 25) if n_speed >= 3 else 30.0
    dist_medium = pctile(sorted_distances, n_dist, 70) if n_dist >= 3 else 15.0

    # 任意帧全员位置（线性插值），避免 O(players × frames) 嵌套扫描。
    # 不能用 frame_player_index 的精确帧匹配：跟踪数据稀疏不规则，
    # 精确匹配会让绝大多数球员在事件帧"不可见"，导致传球人/接球人误判。
    # 缓存 track_id→jersey_label 映射
    jersey_map: dict[str, str] = {}
    for tid, p in players.items():
        jersey_map[tid] = p.jersey_label

    i = 1
    while i < len(positions):
        d = distances[i-1]
        spd = speeds[i-1]
        f = frames[i]

        # 找到起点最近球员（传球发起者候选），超出控球距离则标记为未知
        prev_f = frames[i - 1]
        bx_start, by_start, _bz_start = positions[i - 1]
        nearest = "?"
        nearest_dist = float("inf")
        players_at_prev = positions_at_frame(players, prev_f)
        for pid, (px, py) in players_at_prev.items():
            dist = hypot(bx_start - px, by_start - py)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = jersey_map.get(pid, "?")
        # 最近球员距离超过合理控球范围，标记为未知（可能是跟踪丢失或球在空中）
        if nearest_dist > MAX_POSSESSION_DIST:
            nearest = "?"

        # 找到终点最近球员（接球者候选），同样加距离阈值
        bx_end, by_end, _bz_end = positions[i]
        end_nearest = "?"
        end_nearest_dist = float("inf")
        players_at_f = positions_at_frame(players, f)
        for pid, (px, py) in players_at_f.items():
            dist = hypot(bx_end - px, by_end - py)
            if dist < end_nearest_dist:
                end_nearest_dist = dist
                end_nearest = jersey_map.get(pid, "?")
        if end_nearest_dist > MAX_POSSESSION_DIST:
            end_nearest = "?"

        # 事件判定（使用数据驱动的自适应阈值）
        if spd > speed_high:
            if positions[i][1] > 500 and abs(positions[i][1] - positions[i-1][1]) > 50:
                etype = "射门"
            elif d >= LONG_PASS_MIN_DIST:
                # 长传需同时满足高速 + 长距离，避免大力短传被误判
                etype = "长传"
            else:
                etype = "传球"
        elif spd > speed_medium and d > dist_medium:
            etype = "传球"
        elif spd < speed_low and d < 5 and nearest_dist < 20:
            etype = "带球"
        elif d < 1:
            etype = "静止"
        else:
            etype = "移动"

        dir_name = _ball_direction(positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])

        events.append(BallEvent(
            event_type=etype,
            start_frame=f,
            end_frame=f,
            start_pos=positions[i-1],
            end_pos=positions[i],
            speed=round(spd, 1),
            distance=round(d, 1),
            direction=dir_name,
            nearest_player=nearest,
            nearest_player_end=end_nearest,
            nearest_dist=round(nearest_dist, 1) if nearest_dist != float("inf") else -1.0,
            end_nearest_dist=round(end_nearest_dist, 1) if end_nearest_dist != float("inf") else -1.0,
        ))
        i += 1

    # 合并相邻同类事件
    merged: list[BallEvent] = []
    last_from_longball = False  # 追踪上一个合并事件是否来自长传
    for e in events:
        if merged and merged[-1].event_type == e.event_type and e.start_frame - merged[-1].end_frame <= 2:
            merged[-1].end_frame = e.end_frame
            merged[-1].end_pos = e.end_pos
            merged[-1].distance += e.distance
            merged[-1].speed = max(merged[-1].speed, e.speed)
            # 合并时终点最近球员更新为最后一段的终点最近球员
            merged[-1].nearest_player_end = e.nearest_player_end
            merged[-1].end_nearest_dist = e.end_nearest_dist
        else:
            # 如果上一个合并事件是长传且当前事件是地面事件，
            # 当前事件的起点球员可能是追球者，标记为"?"留待继承逻辑处理
            if last_from_longball and e.event_type in {"传球", "带球", "移动", "静止"}:
                e.nearest_player = "?"
            merged.append(e)
            last_from_longball = (e.event_type == "长传")

    # ── 控球连续性追踪：当起点球员未知时，继承上一次已知的控球者 ──
    # 场景：球从 10 号脚下飞出，但下一采样帧 10 号跟踪丢失，
    # 此时应推断传球者仍是 10 号，而非标记为"未知"
    # 添加时间窗约束：超过 5 帧未匹配到已知控球者则放弃继承
    # 注意：对"长传"事件，终点最近球员可能是追球者而非实际接球者，
    # 不应用来更新 last_known，避免污染后续事件的控球者标签
    last_known = "?"
    frames_since_known = 0
    for e in merged:
        if e.nearest_player != "?":
            last_known = e.nearest_player
            frames_since_known = 0
        else:
            frames_since_known += 1
            if frames_since_known <= 5:
                e.nearest_player = last_known
            else:
                last_known = "?"
        # 终点球员已知则更新最后已知控球者
        # 但"长传"事件的终点最近球员可能是追球者而非实际接球者，不更新
        if e.nearest_player_end != "?" and e.event_type != "长传":
            last_known = e.nearest_player_end
            frames_since_known = 0

    return BallTrajectoryAnalysis(
        total_frames=len(positions),
        avg_speed=round(avg_speed, 2),
        max_speed=round(max_speed, 2),
        total_distance=round(total_distance, 2),
        direction_counts=dict(dir_counts),
        events=merged,
        frame_ranges=frame_ranges,
        aerial_pct=aerial_pct,
    )
