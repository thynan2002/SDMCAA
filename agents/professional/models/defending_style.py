"""防守风格模型构建。

从球员轨迹数据中提取防守风格特征。
"""

from __future__ import annotations

from typing import Any

from agents.player.tracker import PlayerTrajectory, BallTrajectoryAnalysis, BallEvent


def build_defending_profile(
    player: PlayerTrajectory,
    ball_analysis: BallTrajectoryAnalysis | None,
    all_players: dict[str, PlayerTrajectory],
) -> dict[str, Any]:
    """从轨迹数据构建防守风格画像。

    Returns:
        dict 包含拦截倾向、防守站位、协防意识、回追速度等。
    """
    profile: dict[str, Any] = {
        "拦截倾向": _analyze_intercept_tendency(player, ball_analysis),
        "抢断频率": _analyze_tackle_frequency(player, ball_analysis),
        "防守站位偏好": _analyze_defensive_line(player),
        "盯人风格": _analyze_marking_style(player, all_players),
        "协防意识评分": _analyze_cover_awareness(player, ball_analysis, all_players),
        "回追速度": _analyze_recovery_speed(player),
    }

    # 防守贡献评分
    profile["防守贡献评分"] = _compute_defending_score(profile)

    return profile


def _analyze_intercept_tendency(
    player: PlayerTrajectory,
    ball: BallTrajectoryAnalysis | None,
) -> float:
    """分析拦截倾向 0-1。

    判断依据：
    - 高速帧占比（积极上抢型）
    - 与球距离的波动（频繁靠近又远离 = 频繁上抢）
    - 区域覆盖范围（大范围覆盖 = 扫荡型）
    """
    score = 0.5  # 默认中性

    # 速度分布
    if player.speed_curve:
        high_speed_pct = sum(1 for s in player.speed_curve if s > 300) / len(player.speed_curve)
        score += high_speed_pct * 0.3

    # 球距波动
    if len(player.ball_distances) >= 2:
        fluctuations = 0
        for i in range(1, len(player.ball_distances)):
            if abs(player.ball_distances[i] - player.ball_distances[i - 1]) > 50:
                fluctuations += 1
        fluct_pct = fluctuations / len(player.ball_distances)
        score += fluct_pct * 0.2

    # 覆盖范围大 → 扫荡型
    score += (player.field_coverage_pct / 20) * 0.2  # 归一化

    return round(max(0.1, min(0.95, score)), 3)


def _analyze_tackle_frequency(
    player: PlayerTrajectory,
    ball: BallTrajectoryAnalysis | None,
) -> float:
    """分析抢断频率（每百帧归一化）。"""
    if not ball or not ball.events:
        return 0.05

    # 用"接近球 + 高速"作为抢断代理指标
    pid = player.jersey_label
    near_ball_events = [
        e for e in ball.events
        if e.nearest_player == pid
    ]

    total_frames = len(player.frames) or 1
    freq = len(near_ball_events) / total_frames * 100
    return round(min(freq / 8, 1.0), 3)


def _analyze_defensive_line(player: PlayerTrajectory) -> str:
    """根据球员平均 y 位置判断防守站位偏好。"""
    avg_y = player.avg_y()

    if avg_y > 380:
        return "高位压迫"
    elif avg_y > 250:
        return "中位防守"
    else:
        return "低位退守"


def _analyze_marking_style(
    player: PlayerTrajectory,
    all_players: dict[str, PlayerTrajectory],
) -> str:
    """分析盯人风格。

    判断依据：
    - 区域集中度高 → 区域防守
    - 活动范围大、与多个对手距离近 → 紧逼盯人
    """
    zone_pct = max(player.zone_distribution.values()) if player.zone_distribution else 50

    if zone_pct > 60:
        return "区域防守"
    elif zone_pct > 35:
        return "混合防守"
    else:
        return "紧逼盯人"


def _analyze_cover_awareness(
    player: PlayerTrajectory,
    ball: BallTrajectoryAnalysis | None,
    all_players: dict[str, PlayerTrajectory],
) -> float:
    """分析协防意识评分 0-100。

    判断依据：
    - 横向移动幅度（左右补位）
    - 与同队球员的间距稳定性
    - 对球的反应速度（球速快时是否加速）
    """
    score = 50.0  # 基础分

    # 横向覆盖范围
    x_range = player.x_range[1] - player.x_range[0] if player.x_range else 0
    score += min(x_range / 800 * 25, 25)  # 宽幅移动加分

    # 速度响应：球速变化时球员速度是否相应变化
    if player.speed_curve:
        avg_speed = player.avg_speed
        if avg_speed > 300:
            score += 15
        elif avg_speed > 150:
            score += 8

    # 团队位置稳定性
    # 计算与同色球员的平均距离方差
    same_color = [
        p for p in all_players.values()
        if p.color == player.color and p.track_id != player.track_id
    ]
    if same_color:
        dists = []
        for other in same_color:
            dx = player.avg_x() - other.avg_x()
            dy = player.avg_y() - other.avg_y()
            dists.append((dx * dx + dy * dy) ** 0.5)
        if dists:
            from statistics import stdev
            try:
                std = stdev(dists)
                # 方差小 = 位置稳定 = 协防意识好
                score += max(0, 10 - std / 50)
            except Exception:
                pass

    return round(min(score, 100), 1)


def _analyze_recovery_speed(player: PlayerTrajectory) -> str:
    """分析回追速度特征。

    判断依据：后场球员 + 最高速度。
    """
    max_speed = player.max_speed

    if max_speed > 600:
        return "极快"
    elif max_speed > 400:
        return "快"
    elif max_speed > 200:
        return "中等"
    else:
        return "慢"


def _compute_defending_score(profile: dict[str, Any]) -> float:
    """综合防守贡献评分 0-100。"""
    intercept = profile.get("拦截倾向", 0.5) * 30
    tackle = profile.get("抢断频率", 0.1) * 25
    cover = profile.get("协防意识评分", 50) * 0.3

    speed_tier = profile.get("回追速度", "中等")
    speed_score = {"极快": 15, "快": 12, "中等": 8, "慢": 3}.get(speed_tier, 8)

    return round(min(intercept + tackle + cover + speed_score, 100), 1)
