"""球员与足球三维位置关系分析智能体。

增加 Z 轴高度维度、相对运动趋势、争顶检测等融合分析，
输出供解说链路（commentary.build_user_message）消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

from agents.constants import (
    GROUND_Z_THRESHOLD, LOW_BALL_Z_MAX,
    GROUND_PROXIMITY_RADIUS, LOW_BALL_PROXIMITY_RADIUS, HIGH_BALL_PROXIMITY_RADIUS,
    APPROACH_DELTA, RETREAT_DELTA, PIXELS_PER_METER,
)
from .tracker import PlayerTrajectory, PrefixPlayerCorpus, positions_at_frame


@dataclass
class BallPositionReport:
    """单球员与球的三维位置关系分析报告。"""

    track_id: str
    jersey: str
    color: str

    # ── 3D 球距统计 ──
    avg_3d_distance: float = 0.0         # 含 z 轴高度差的平均距离
    avg_2d_distance: float = 0.0         # 仅 xy 水平面的平均距离
    min_3d_distance: float = float("inf")
    min_3d_frame: int = 0                # 最接近球的帧号

    # ── 按球高度分层的接近度 ──
    ground_proximity_pct: float = 0.0    # z≤1 且水平距<30 的帧占比
    low_ball_proximity_pct: float = 0.0  # 1<z≤50 且水平距<50 的帧占比
    high_ball_proximity_pct: float = 0.0 # z>50 且水平距<60 的帧占比

    # ── 空中球争顶 ──
    aerial_contest_count: int = 0        # 高空球帧中水平距<60 的次数

    # ── 相对运动趋势（帧间距离变化） ──
    approaching_pct: float = 0.0         # 距离缩小帧占比
    retreating_pct: float = 0.0          # 距离增大帧占比
    stable_pct: float = 0.0              # 距离不变帧占比

    # ── 最近球员统计（按球高度分层） ──
    nearest_high_ball_frames: int = 0    # z>1 时作为全场最近球员的帧数
    nearest_ground_ball_frames: int = 0  # z≤1 时作为全场最近球员的帧数
    nearest_high_ball_total: int = 0     # 高空球总帧数（用于算比例）
    nearest_ground_ball_total: int = 0   # 地面球总帧数（用于算比例）

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "jersey": self.jersey,
            "color": self.color,
            "avg_3d_distance": round(self.avg_3d_distance, 1),
            "avg_2d_distance": round(self.avg_2d_distance, 1),
            "min_3d_distance": round(self.min_3d_distance, 1),
            "min_3d_frame": self.min_3d_frame,
            "ground_proximity_pct": round(self.ground_proximity_pct, 1),
            "low_ball_proximity_pct": round(self.low_ball_proximity_pct, 1),
            "high_ball_proximity_pct": round(self.high_ball_proximity_pct, 1),
            "aerial_contest_count": self.aerial_contest_count,
            "approaching_pct": round(self.approaching_pct, 1),
            "retreating_pct": round(self.retreating_pct, 1),
            "stable_pct": round(self.stable_pct, 1),
            "nearest_high_ball_pct": round(
                self.nearest_high_ball_frames / max(1, self.nearest_high_ball_total) * 100, 1
            ),
            "nearest_ground_ball_pct": round(
                self.nearest_ground_ball_frames / max(1, self.nearest_ground_ball_total) * 100, 1
            ),
        }


@dataclass
class BallHeightSummary:
    """球的高度阶段整体摘要（跨全部球员聚合）。"""

    ground_pct: float = 0.0      # 地面球时间占比
    low_ball_pct: float = 0.0    # 低平球时间占比
    high_ball_pct: float = 0.0   # 高空球时间占比
    dominant_style: str = ""     # "地面渗透"/"高低结合"/"长传冲吊"
    total_ball_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "地面球占比": f"{round(self.ground_pct, 1)}%",
            "低平球占比": f"{round(self.low_ball_pct, 1)}%",
            "高空球占比": f"{round(self.high_ball_pct, 1)}%",
            "比赛风格": self.dominant_style,
        }


class BallPositionRelationAgent:
    """球员与球的三维位置关系分析智能体。

    接收 PrefixPlayerCorpus，为每位球员计算：
    - 逐帧 3D 球距（含 z 高度差）
    - 不同球高度层下的接近度
    - 空中球争顶检测
    - 球员-球相对运动趋势
    """

    name = "BallPositionRelationAgent"

    def summarize_ball_height(self, corpus: PrefixPlayerCorpus) -> BallHeightSummary:
        """汇总球的整体高度分布特征。"""
        ball_frames = corpus.ball_frames
        if not ball_frames:
            return BallHeightSummary()

        total = len(ball_frames)
        ground = sum(1 for _, _, z in ball_frames.values() if z <= GROUND_Z_THRESHOLD)
        high = sum(1 for _, _, z in ball_frames.values() if z > LOW_BALL_Z_MAX)
        low = total - ground - high

        ground_pct = ground / total * 100
        low_pct = low / total * 100
        high_pct = high / total * 100

        if high_pct > 30:
            style = "长传冲吊为主，高空球频繁"
        elif high_pct > 15:
            style = "高低结合，有一定空中对抗"
        else:
            style = "地面渗透为主，球基本在脚下运行"

        return BallHeightSummary(
            ground_pct=ground_pct,
            low_ball_pct=low_pct,
            high_ball_pct=high_pct,
            dominant_style=style,
            total_ball_frames=total,
        )

    def analyze(self, player: PlayerTrajectory, corpus: PrefixPlayerCorpus) -> BallPositionReport:
        """分析单个球员与球的三维位置关系。

        需要 corpus 以获取球的逐帧 (x, y, z) 和进行全队比较。
        """
        ball_frames = corpus.ball_frames
        report = BallPositionReport(
            track_id=player.track_id,
            jersey=player.jersey_label,
            color=player.color,
        )

        if not player.frames or not ball_frames:
            return report

        # ── 逐帧分析 ──
        distances_3d: list[float] = []
        distances_2d: list[float] = []
        prev_dist_3d: float | None = None

        # 按球高度分层的接近计数
        ground_proximity = 0
        low_ball_proximity = 0
        high_ball_proximity = 0
        ground_total = 0
        low_ball_total = 0
        high_ball_total = 0

        # 相对运动
        approaching = 0
        retreating = 0
        stable = 0

        # 争顶
        aerial_contests = 0

        # 与球员 vs 其他球员比较（最近球员统计）
        nearest_high = 0
        nearest_ground = 0

        for fi, (frame, px, py) in enumerate(zip(player.frames, player.xs, player.ys)):
            ball_pos = ball_frames.get(frame)
            if ball_pos is None:
                continue
            bx, by, bz = ball_pos

            # 3D 距离（z 为米，统一换算成像素后再与 xy 合成，避免单位混算）
            dz = bz * PIXELS_PER_METER  # 球员 z=0（地面）
            d3d = hypot(px - bx, py - by) if dz == 0 else hypot(hypot(px - bx, py - by), dz)
            d2d = hypot(px - bx, py - by)

            distances_3d.append(d3d)
            distances_2d.append(d2d)

            if d3d < report.min_3d_distance:
                report.min_3d_distance = d3d
                report.min_3d_frame = frame

            # 按高度分层
            if bz <= GROUND_Z_THRESHOLD:
                ground_total += 1
                if d2d < GROUND_PROXIMITY_RADIUS:
                    ground_proximity += 1
            elif bz <= LOW_BALL_Z_MAX:
                low_ball_total += 1
                if d2d < LOW_BALL_PROXIMITY_RADIUS:
                    low_ball_proximity += 1
            else:
                high_ball_total += 1
                if d2d < HIGH_BALL_PROXIMITY_RADIUS:
                    high_ball_proximity += 1
                    aerial_contests += 1

            # 相对运动趋势
            if prev_dist_3d is not None:
                delta = prev_dist_3d - d3d
                if delta > APPROACH_DELTA:
                    approaching += 1
                elif delta < -RETREAT_DELTA:
                    retreating += 1
                else:
                    stable += 1
            prev_dist_3d = d3d

            # 最近球员比较：用任意帧插值位置替代精确帧索引（稀疏数据下
            # 精确帧索引会让绝大多数球员"不可见"，导致最近球员误判）
            players_at_frame = positions_at_frame(corpus.players, frame)
            this_tid = player.track_id
            is_nearest = True
            competitors_found = 0
            for other_tid, (ox, oy) in players_at_frame.items():
                if other_tid == this_tid:
                    continue
                competitors_found += 1
                if hypot(ox - bx, oy - by) < d2d:
                    is_nearest = False
                    # 不必继续——已确定不是最近，但 competitor_found 仍然更新
            if is_nearest and competitors_found > 0:
                # 计数口径与分母严格一致：高空(z>2.5m) / 地面(z≤0.2m)，低平球不计入
                if bz > LOW_BALL_Z_MAX:
                    nearest_high += 1
                elif bz <= GROUND_Z_THRESHOLD:
                    nearest_ground += 1

        # ── 汇总 ──
        if distances_3d:
            report.avg_3d_distance = sum(distances_3d) / len(distances_3d)
        if distances_2d:
            report.avg_2d_distance = sum(distances_2d) / len(distances_2d)

        motion_total = approaching + retreating + stable
        if motion_total > 0:
            report.approaching_pct = approaching / motion_total * 100
            report.retreating_pct = retreating / motion_total * 100
            report.stable_pct = stable / motion_total * 100

        report.ground_proximity_pct = ground_proximity / max(1, ground_total) * 100
        report.low_ball_proximity_pct = low_ball_proximity / max(1, low_ball_total) * 100
        report.high_ball_proximity_pct = high_ball_proximity / max(1, high_ball_total) * 100
        report.aerial_contest_count = aerial_contests

        report.nearest_high_ball_frames = nearest_high
        report.nearest_ground_ball_frames = nearest_ground
        report.nearest_high_ball_total = high_ball_total
        report.nearest_ground_ball_total = ground_total

        return report

    def summarize_all(
        self, corpus: PrefixPlayerCorpus
    ) -> list[dict[str, Any]]:
        """对 corpus 中所有球员执行分析，返回 dict 列表。"""
        results: list[dict[str, Any]] = []
        for player in corpus.sorted_players():
            report = self.analyze(player, corpus)
            results.append(report.to_dict())
        return results

    def summarize_aggregated(
        self, corpus: PrefixPlayerCorpus
    ) -> dict[str, Any]:
        """跨球员聚合的队级摘要。"""
        reports = [self.analyze(p, corpus) for p in corpus.players.values()]
        height_summary = self.summarize_ball_height(corpus)

        if not reports:
            return {"球高度概况": height_summary.to_dict()}

        # 按争顶排名
        by_contests = sorted(reports, key=lambda r: r.aerial_contest_count, reverse=True)
        # 按 3D 最近排名
        by_3d = sorted(reports, key=lambda r: r.avg_3d_distance)
        # 按地面接近排名
        by_ground = sorted(reports, key=lambda r: r.ground_proximity_pct, reverse=True)

        return {
            "球高度概况": height_summary.to_dict(),
            "争顶最多": [
                {"球员": r.jersey, "空中争顶次数": r.aerial_contest_count}
                for r in by_contests[:3]
            ],
            "3D综合最近": [
                {"球员": r.jersey, "平均3D距离": round(r.avg_3d_distance, 1)}
                for r in by_3d[:3]
            ],
            "地面球最紧密": [
                {"球员": r.jersey, "地面接近率": f"{round(r.ground_proximity_pct, 1)}%"}
                for r in by_ground[:3]
            ],
        }
