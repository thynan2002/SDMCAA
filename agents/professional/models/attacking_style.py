"""进攻风格模型构建。

从球员轨迹数据中提取进攻风格特征。
"""

from __future__ import annotations

from typing import Any

from agents.player.tracker import PlayerTrajectory, BallTrajectoryAnalysis, BallEvent


def build_attacking_profile(
    player: PlayerTrajectory,
    ball_analysis: BallTrajectoryAnalysis | None,
    all_players: dict[str, PlayerTrajectory],
) -> dict[str, Any]:
    """从轨迹数据构建进攻风格画像。

    Returns:
        dict 包含传球倾向、射门选择、盘带习惯、无球跑动模式等。
    """
    profile: dict[str, Any] = {
        "传球倾向": _analyze_pass_tendency(player, ball_analysis),
        "射门选择": _analyze_shoot_preference(player, ball_analysis),
        "盘带频率": _analyze_dribble_frequency(player, ball_analysis),
        "盘带方向偏好": _analyze_dribble_direction(player, ball_analysis),
        "无球跑动模式": _analyze_off_ball_movement(player, ball_analysis, all_players),
    }

    # 进攻贡献评分
    score = _compute_attacking_score(profile)
    profile["射门频率"] = profile["射门选择"].get("总射门次数", 0) / max(1, len(player.frames)) * 1000
    profile["进攻贡献评分"] = score

    return profile


def _analyze_pass_tendency(
    player: PlayerTrajectory,
    ball: BallTrajectoryAnalysis | None,
) -> dict[str, float]:
    """分析传球倾向分布。"""
    result: dict[str, float] = {
        "短传": 0.4, "长传": 0.2, "直塞": 0.15,
        "回传": 0.15, "横传": 0.1,
    }

    if not ball or not ball.events:
        return result

    pid = player.jersey_label
    player_events = [
        e for e in ball.events
        if e.nearest_player == pid and e.event_type in ("传球", "长传")
    ]

    if not player_events:
        return result

    total = len(player_events)
    # 按速度分：高速 → 长传/直塞，低速 → 短传/回传
    short = sum(1 for e in player_events if e.speed < 200)
    long_pass = sum(1 for e in player_events if e.speed >= 500)
    through = sum(1 for e in player_events if 300 <= e.speed < 500 and e.distance > 30)

    # 方向判断
    backward = sum(1 for e in player_events if "后" in e.direction)
    lateral = sum(1 for e in player_events if "左" in e.direction or "右" in e.direction)

    remaining = total - short - long_pass - through
    result = {
        "短传": round(short / total, 3) if total else 0.4,
        "长传": round(long_pass / total, 3) if total else 0.2,
        "直塞": round(through / total, 3) if total else 0.15,
        "回传": round(backward / total, 3) if total else 0.15,
        "横传": round(lateral / total, 3) if total else 0.1,
    }
    return result


def _analyze_shoot_preference(
    player: PlayerTrajectory,
    ball: BallTrajectoryAnalysis | None,
) -> dict[str, float]:
    """分析射门选择偏好。"""
    result: dict[str, float] = {
        "远射": 0.3, "禁区内射门": 0.5, "头球": 0.2, "总射门次数": 0,
    }

    if not ball or not ball.events:
        return result

    pid = player.jersey_label
    shoots = [e for e in ball.events if e.nearest_player == pid and e.event_type == "射门"]
    result["总射门次数"] = len(shoots)

    if not shoots:
        return result

    # 按射门位置判断类型
    far = sum(1 for s in shoots if _dist_to_goal(s.end_pos) > 200)
    header = sum(1 for s in shoots if s.end_pos[2] > 50)

    total = len(shoots)
    result = {
        "远射": round(far / total, 3),
        "禁区内射门": round((total - far - header) / total, 3),
        "头球": round(header / total, 3),
        "总射门次数": float(total),
    }
    return result


def _analyze_dribble_frequency(
    player: PlayerTrajectory,
    ball: BallTrajectoryAnalysis | None,
) -> float:
    """计算盘带频率（每百帧盘带次数归一化到0-1）。"""
    if not ball or not ball.events:
        return 0.1

    pid = player.jersey_label
    dribbles = [e for e in ball.events if e.nearest_player == pid and e.event_type == "带球"]
    total_frames = len(player.frames) or 1
    freq = len(dribbles) / total_frames * 100  # 每百帧
    return round(min(freq / 5, 1.0), 3)  # 归一化（>5视为高频）


def _analyze_dribble_direction(
    player: PlayerTrajectory,
    ball: BallTrajectoryAnalysis | None,
) -> dict[str, float]:
    """分析盘带方向偏好。"""
    result: dict[str, float] = {"向前": 0.5, "横向": 0.3, "向后": 0.2}

    if not ball or not ball.events:
        return result

    pid = player.jersey_label
    dribbles = [e for e in ball.events if e.nearest_player == pid and e.event_type == "带球"]

    if not dribbles:
        return result

    forward = sum(1 for d in dribbles if "前" in d.direction)
    lateral = sum(1 for d in dribbles if "左" in d.direction or "右" in d.direction)
    backward = sum(1 for d in dribbles if "后" in d.direction)
    total = len(dribbles)

    return {
        "向前": round(forward / total, 3),
        "横向": round(lateral / total, 3),
        "向后": round(backward / total, 3),
    }


def _analyze_off_ball_movement(
    player: PlayerTrajectory,
    ball: BallTrajectoryAnalysis | None,
    all_players: dict[str, PlayerTrajectory],
) -> dict[str, float]:
    """分析无球跑动模式。"""
    result: dict[str, float] = {
        "前插": 0.3, "回撤接应": 0.25, "拉边": 0.2,
        "横向拉扯": 0.15, "原地等待": 0.1,
    }

    # 基于速度方向和 y 轴变化
    if len(player.speed_curve) < 2:
        return result

    # 分析帧间移动方向
    forward_moves = 0
    backward_moves = 0
    lateral_moves = 0
    static = 0

    # 判断是否持球（通过球距）
    for i in range(len(player.ys) - 1):
        dy = player.ys[i + 1] - player.ys[i]
        dx = player.xs[i + 1] - player.xs[i]
        speed = player.speed_curve[i] if i < len(player.speed_curve) else 0

        if speed < 50:
            static += 1
        elif abs(dy) > abs(dx) and dy > 2:
            forward_moves += 1
        elif abs(dy) > abs(dx) and dy < -2:
            backward_moves += 1
        else:
            lateral_moves += 1

    total = forward_moves + backward_moves + lateral_moves + static or 1
    return {
        "前插": round(forward_moves / total, 3),
        "回撤接应": round(backward_moves / total, 3),
        "拉边": round(lateral_moves * 0.6 / total, 3),
        "横向拉扯": round(lateral_moves * 0.4 / total, 3),
        "原地等待": round(static / total, 3),
    }


def _compute_attacking_score(profile: dict[str, Any]) -> float:
    """综合进攻贡献评分 0-100。"""
    pass_t = profile.get("传球倾向", {})
    shoot_p = profile.get("射门选择", {})
    dribble_f = profile.get("盘带频率", 0)
    off_ball = profile.get("无球跑动模式", {})

    score = 50.0  # 基础分

    # 传球多样性加分
    score += (pass_t.get("直塞", 0) + pass_t.get("长传", 0)) * 30

    # 射门频率加分
    shoot_count = shoot_p.get("总射门次数", 0)
    score += min(shoot_count * 8, 20)

    # 盘带频率加分
    score += dribble_f * 20

    # 无球跑动加分
    score += off_ball.get("前插", 0) * 15
    score += off_ball.get("回撤接应", 0) * 10

    return round(min(score, 100), 1)


def _dist_to_goal(pos: tuple[float, float, float]) -> float:
    """计算位置到对方球门的近似距离。

    假设进攻方向为 +y 方向，球门在 (600, 700)。
    """
    from math import hypot
    return hypot(pos[0] - 600, pos[1] - 700)
