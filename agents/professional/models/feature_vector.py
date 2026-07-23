"""球员行为特征向量构建。

将 PlayerProfile 的结构化数据映射为 N 维数值特征向量，
供 MCTS 决策策略和风格匹配使用。
"""

from __future__ import annotations

from typing import Any


# 特征向量维度
FEATURE_DIM = 8

FEATURE_NAMES = [
    "传球侵略性",       # [0] pass_aggressiveness
    "射门信心",         # [1] shoot_confidence
    "盘带倾向",         # [2] dribble_frequency
    "防守侵略性",       # [3] defensive_aggressiveness
    "站位纪律",         # [4] positional_discipline
    "风险承受",         # [5] risk_tolerance
    "团队倾向",         # [6] team_orientation
    "前插倾向",         # [7] forward_bias
]


def build_feature_vector(profile: dict[str, Any]) -> list[float]:
    """从 PlayerProfile.to_dict() 构建归一化特征向量。

    返回值: 8 维 [0-1] 范围的特征列表。
    """
    features = [0.5] * FEATURE_DIM  # 默认中性值

    # ── [0] 传球侵略性 ──────────────────────────────────────────
    attacking = profile.get("进攻风格", {})
    pass_tendency = attacking.get("传球倾向", {})
    # 直塞+长传 比例反映出球侵略性
    pass_aggr = pass_tendency.get("直塞", 0) + pass_tendency.get("长传", 0)
    features[0] = _clamp(pass_aggr, 0, 1)

    # ── [1] 射门信心 ────────────────────────────────────────────
    shoot_pref = attacking.get("射门选择", {})
    total_shoot = sum(shoot_pref.values()) or 0.01
    # 远射占比高 → 射门信心足
    long_shoot_pct = shoot_pref.get("远射", 0) / total_shoot
    # 射门频率按每百帧归一化，避免少量样本直接饱和
    shoot_frequency = attacking.get("射门频率", 0)
    normalized_frequency = min(shoot_frequency / 8, 1.0)
    # 射门频率 + 远射比例 → 综合信心
    shoot_score = (normalized_frequency * 0.6 + long_shoot_pct * 0.4)
    features[1] = _clamp(shoot_score, 0, 1)

    # ── [2] 盘带倾向 ────────────────────────────────────────────
    dribble_freq = attacking.get("盘带频率", 0)
    # 盘带方向中"向前"占比
    dribble_dir = attacking.get("盘带方向偏好", {})
    forward_dribble = dribble_dir.get("向前", 0)
    features[2] = _clamp(dribble_freq * 0.7 + forward_dribble * 0.3, 0, 1)

    # ── [3] 防守侵略性 ──────────────────────────────────────────
    defending = profile.get("防守风格", {})
    intercept = defending.get("拦截倾向", 0.5)
    tackle = defending.get("抢断频率", 0)
    features[3] = _clamp(intercept * 0.6 + tackle * 0.4, 0, 1)

    # ── [4] 站位纪律 ────────────────────────────────────────────
    # 区域集中度（dominant_zone 占比越高 = 站位越固定）
    zone_dist = profile.get("区域分布", {})
    if zone_dist:
        max_zone_pct = max(zone_dist.values()) / 100
        # 后卫默认站位纪律偏高
        role = profile.get("战术角色", "")
        role_bonus = 0.2 if "后卫" in role else 0
        features[4] = _clamp(max_zone_pct * 0.8 + role_bonus, 0, 1)
    else:
        features[4] = 0.5

    # ── [5] 风险承受 ────────────────────────────────────────────
    # 盘带频率高 + 传球侵略性高 + 射门远射多 → 风险承受高
    risk = (
        features[2] * 0.4 +         # 盘带倾向
        features[1] * 0.35 +        # 射门信心
        features[0] * 0.25          # 传球侵略性
    )
    features[5] = _clamp(risk, 0, 1)

    # ── [6] 团队倾向 ────────────────────────────────────────────
    # 无球跑动中"回撤接应"占比高 → 团队型
    off_ball = attacking.get("无球跑动模式", {})
    support_pct = off_ball.get("回撤接应", 0)
    # 防守协防意识也体现团队性
    cover = defending.get("协防意识评分", 50) / 100
    features[6] = _clamp(support_pct * 0.5 + cover * 0.5, 0, 1)

    # ── [7] 前插倾向 ────────────────────────────────────────────
    # 无球跑动"前插"占比 + 平均位置偏前场
    forward_run = off_ball.get("前插", 0)
    avg_y = profile.get("平均纵向位置", 350) / 700  # 归一化
    features[7] = _clamp(forward_run * 0.6 + avg_y * 0.4, 0, 1)

    return features


def features_to_action_weights(features: list[float]) -> dict[str, float]:
    """将特征向量映射为 MCTS 动作初始权重。

    返回值: {"pass": w1, "shoot": w2, "dribble": w3, "clear": w4, "hold": w5}
    """
    if len(features) < FEATURE_DIM:
        features = features + [0.5] * (FEATURE_DIM - len(features))

    pass_aggr, shoot_conf, dribble, def_aggr, pos_disc, risk, team, forward = features

    # 传球权重：团队倾向 + 传球侵略性
    pass_weight = 0.25 + team * 0.2 + pass_aggr * 0.15
    # 射门权重：射门信心 + 前插倾向 + 风险承受
    shoot_weight = 0.1 + shoot_conf * 0.3 + forward * 0.15 + risk * 0.1
    # 盘带权重：盘带倾向 + 风险承受 - 站位纪律（纪律高不爱带）
    dribble_weight = 0.1 + dribble * 0.3 + risk * 0.15 - pos_disc * 0.1
    # 解围权重：防守侵略性 + (1-传球侵略性)
    clear_weight = 0.05 + def_aggr * 0.2 + (1 - pass_aggr) * 0.1
    # 控球等待权重：站位纪律 - 风险承受
    hold_weight = 0.05 + pos_disc * 0.15 - risk * 0.1

    # 确保非负
    weights = {
        "pass": max(0.05, pass_weight),
        "shoot": max(0.02, shoot_weight),
        "dribble": max(0.02, dribble_weight),
        "clear": max(0.02, clear_weight),
        "hold": max(0.02, hold_weight),
    }

    # 归一化
    total = sum(weights.values())
    return {k: round(v / total, 4) for k, v in weights.items()}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
