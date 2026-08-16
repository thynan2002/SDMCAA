"""数据采集与分析智能体。

从 PrefixPlayerCorpus 中提取指定球员的详细数据，
构建 PlayerProfile（包含空间特征、进攻/防守风格、战术角色）。
"""

from __future__ import annotations

from math import hypot
from statistics import mean, stdev
from typing import Any

from agents.constants import PX_PER_M_X, PX_PER_M_Y
from agents.player.tracker import (
    PlayerTrajectory,
    PrefixPlayerCorpus,
    BallTrajectoryAnalysis,
)
from agents.llm_client import call_llm
from ..types import PlayerProfile, AttackingProfile, DefendingProfile
from ..models.attacking_style import build_attacking_profile
from ..models.defending_style import build_defending_profile


class DataCollectorAgent:
    """球员数据采集与分析智能体。

    收集并处理指定球员的历史跑动轨迹数据，
    包括位置坐标、移动速度、加速度等时空特征，
    分析球员站位偏好、区域覆盖范围及战术角色定位。
    """

    name = "DataCollectorAgent"

    def collect(
        self,
        corpus: PrefixPlayerCorpus,
        target_players: list[str] | None = None,
    ) -> list[PlayerProfile]:
        """采集指定球员的详细数据。

        Args:
            corpus: 比赛语料
            target_players: 目标球员 jersey_label 列表，None 表示全部

        Returns:
            球员画像列表
        """
        if target_players:
            players = [
                p for p in corpus.players.values()
                if p.jersey_label in target_players
            ]
        else:
            players = list(corpus.players.values())

        profiles: list[PlayerProfile] = []
        for player in players:
            profile = self._build_profile(player, corpus)
            profiles.append(profile)

        return profiles

    def _build_profile(
        self,
        player: PlayerTrajectory,
        corpus: PrefixPlayerCorpus,
    ) -> PlayerProfile:
        """为单个球员构建综合画像。"""
        ball = corpus.ball_analysis

        # 战术角色
        tactical_role = self._infer_tactical_role(player)

        # 进攻风格
        attacking_data = build_attacking_profile(player, ball, corpus.players)

        # 防守风格
        defending_data = build_defending_profile(player, ball, corpus.players)

        # 空间特征
        speed_stats = self._build_speed_stats(player)
        accel_dist = self._build_acceleration_distribution(player)
        total_distance_m = self._total_distance_m(player)

        profile = PlayerProfile(
            track_id=player.track_id,
            jersey_label=player.jersey_label,
            color=player.color,
            team_name=self._get_team_name(player.color),
            avg_position=(player.avg_x(), player.avg_y()),
            coverage_area=player.field_coverage_pct,
            zone_distribution=player.zone_distribution,
            dominant_zone=player.dominant_zone,
            speed_stats=speed_stats,
            total_distance_m=total_distance_m,
            ball_proximity_pct=player.ball_proximity_pct,
            acceleration_distribution=accel_dist,
            tactical_role=tactical_role,
            attacking=AttackingProfile(
                pass_tendency=attacking_data.get("传球倾向", {}),
                shoot_preference={
                    k: v for k, v in attacking_data.get("射门选择", {}).items()
                    if k != "总射门次数"
                },
                shoot_total=int(attacking_data.get("射门选择", {}).get("总射门次数", 0)),
                dribble_frequency=attacking_data.get("盘带频率", 0.1),
                dribble_direction_bias=attacking_data.get("盘带方向偏好", {}),
                off_ball_run_type=attacking_data.get("无球跑动模式", {}),
                attacking_contribution_score=attacking_data.get("进攻贡献评分", 50),
            ),
            defending=DefendingProfile(
                intercept_tendency=defending_data.get("拦截倾向", 0.5),
                tackle_frequency=defending_data.get("抢断频率", 0.05),
                defensive_line_preference=defending_data.get("防守站位偏好", "中位防守"),
                marking_style=defending_data.get("盯人风格", "混合防守"),
                cover_awareness_score=defending_data.get("协防意识评分", 50),
                recovery_speed_tier=defending_data.get("回追速度", "中等"),
                defensive_contribution_score=defending_data.get("防守贡献评分", 50),
            ),
        )

        return profile

    def _build_speed_stats(self, player: PlayerTrajectory) -> dict[str, float]:
        """构建速度统计。"""
        if not player.speed_curve:
            return {"avg": 0, "max": 0, "sprint_count": 0, "median": 0}

        sprint_count = sum(1 for s in player.speed_curve if s > 300)
        median_speed = sorted(player.speed_curve)[len(player.speed_curve) // 2] if player.speed_curve else 0

        return {
            "avg": round(player.avg_speed, 1),
            "max": round(player.max_speed, 1),
            "sprint_count": sprint_count,
            "median": round(median_speed, 1),
        }

    @staticmethod
    def _total_distance_m(player: PlayerTrajectory) -> float:
        """累计跑动距离（米）：x/y 分轴标定（1200px→105m、700px→68m），
        与 eval/goldref 独立重算同口径。"""
        if len(player.xs) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(player.xs)):
            dx = (player.xs[i] - player.xs[i - 1]) / PX_PER_M_X
            dy = (player.ys[i] - player.ys[i - 1]) / PX_PER_M_Y
            total += hypot(dx, dy)
        return round(total, 2)

    def _build_acceleration_distribution(self, player: PlayerTrajectory) -> dict[str, float]:
        """构建加速度分布。"""
        if len(player.speed_curve) < 2:
            return {"high": 0.1, "medium": 0.3, "low": 0.6}

        accels = []
        for i in range(1, len(player.speed_curve)):
            accels.append(player.speed_curve[i] - player.speed_curve[i - 1])

        if not accels:
            return {"high": 0.1, "medium": 0.3, "low": 0.6}

        high = sum(1 for a in accels if abs(a) > 150)
        medium = sum(1 for a in accels if 50 < abs(a) <= 150)
        low = sum(1 for a in accels if abs(a) <= 50)
        total = len(accels) or 1

        return {
            "high": round(high / total, 3),
            "medium": round(medium / total, 3),
            "low": round(low / total, 3),
        }

    def _infer_tactical_role(self, player: PlayerTrajectory) -> str:
        """推断战术角色（使用 tracker.py 共享分类器）。"""
        # 门将优先检测
        if player.is_goalkeeper:
            return "门将"

        from agents.player.tracker import classify_position, classify_side

        avg_y = player.avg_y()
        avg_x = player.avg_x()
        coverage = player.field_coverage_pct

        position = classify_position(avg_y)
        side = classify_side(avg_x)

        # 细化
        if position == "后卫":
            if side == "中":
                role = "中后卫"
            else:
                role = f"{side}后卫"
        elif position == "中场":
            if coverage > 8:
                role = f"全能中场（{side}路）" if side != "中" else "全能中场"
            elif player.total_distance > 250:
                role = f"进攻型中场（{side}路）" if side != "中" else "进攻型中场"
            else:
                role = f"防守型中场（{side}路）" if side != "中" else "防守型中场"
        else:
            if side != "中":
                role = f"{side}边锋"
            elif coverage > 5:
                role = "中锋（跑动型）"
            else:
                role = "中锋（站位型）"

        return role

    def _get_team_name(self, color: str) -> str:
        """颜色 → 队名。"""
        mapping = {"A": "A队", "B": "B队", "C": "门将"}
        return mapping.get(color, f"{color}队")
