"""风格建模智能体（精简版）。

基于球员画像构建行为预测模型：特征向量、确定性风格标签、MCTS 决策权重、
传球目标偏好。原 LLM 风格标注步骤已随中间层精简移除 —— 风格标签/描述
由特征向量确定性推导（消除 LLM 中间文本的夸大与误差累积），风格描述
在报告中仅作辅助，报告以硬数据（距离/速度/冲刺/区域）为依据。
"""

from __future__ import annotations

from typing import Any

from ..types import PlayerProfile, PlayerBehaviorModel
from ..models.feature_vector import (
    build_feature_vector,
    features_to_action_weights,
    FEATURE_NAMES,
)


class StyleModelingAgent:
    """风格学习与建模智能体。

    通过分析球员画像，构建：
    - N 维行为特征向量
    - 球员风格标签与描述（LLM 生成）
    - MCTS 决策权重矩阵
    - 传球目标偏好
    """

    name = "StyleModelingAgent"

    def build_model(
        self,
        profile: PlayerProfile,
        all_profiles: list[PlayerProfile] | None = None,
    ) -> PlayerBehaviorModel:
        """为单个球员构建行为预测模型。

        Args:
            profile: 球员画像
            all_profiles: 全体球员画像（用于计算传球偏好等相对指标）

        Returns:
            球员行为模型
        """
        profile_dict = profile.to_dict()

        # 1. 构建特征向量
        feature_vec = build_feature_vector(profile_dict)

        # 2. 特征向量 → 动作权重
        action_weights = features_to_action_weights(feature_vec)

        # 3. 传球目标偏好
        pass_preferences = self._build_pass_preferences(profile, all_profiles)

        # 4. 确定性风格标签和描述（基于特征向量极值，不再经 LLM）
        style_label, style_desc = _deterministic_style(profile_dict, feature_vec)

        # 5. 模型置信度评估
        confidence = self._evaluate_confidence(profile, feature_vec)

        return PlayerBehaviorModel(
            track_id=profile.track_id,
            jersey_label=profile.jersey_label,
            feature_vector=feature_vec,
            style_label=style_label,
            style_description=style_desc,
            action_weights=action_weights,
            pass_target_preference=pass_preferences,
            confidence_score=confidence,
        )

    def build_models_batch(
        self,
        profiles: list[PlayerProfile],
        all_profiles: list[PlayerProfile] | None = None,
    ) -> list[PlayerBehaviorModel]:
        """批量构建球员行为模型。

        all_profiles：全体球员画像（传球偏好等相对指标参照）；
        缺省时用 profiles 自身（兼容原有调用）。

        风格标签/描述为确定性推导（每名球员零 LLM 调用）。
        """
        reference = all_profiles if all_profiles is not None else profiles

        # 1. 预计算纯 Python 部分（特征向量 / 风格标注 / 传球偏好 / 置信度）
        prepared: list[tuple[PlayerProfile, dict, list[float], dict, float]] = []
        for profile in profiles:
            profile_dict = profile.to_dict()
            feature_vec = build_feature_vector(profile_dict)
            pass_pref = self._build_pass_preferences(profile, reference)
            confidence = self._evaluate_confidence(profile, feature_vec)
            prepared.append((profile, profile_dict, feature_vec, pass_pref, confidence))

        # 2. 组装行为模型（确定性风格标注）
        models: list[PlayerBehaviorModel] = []
        for (profile, profile_dict, feature_vec, pass_pref, confidence) in prepared:
            label, desc = _deterministic_style(profile_dict, feature_vec)
            models.append(PlayerBehaviorModel(
                track_id=profile.track_id,
                jersey_label=profile.jersey_label,
                feature_vector=feature_vec,
                style_label=label,
                style_description=desc,
                action_weights=features_to_action_weights(feature_vec),
                pass_target_preference=pass_pref,
                confidence_score=confidence,
            ))
        return models

    def _build_pass_preferences(
        self,
        profile: PlayerProfile,
        all_profiles: list[PlayerProfile] | None,
    ) -> dict[str, float]:
        """计算传球目标偏好。"""
        preferences: dict[str, float] = {}

        if not all_profiles:
            return preferences

        # 同队球员
        teammates = [
            p for p in all_profiles
            if p.color == profile.color and p.track_id != profile.track_id
        ]

        if not teammates:
            return preferences

        for teammate in teammates:
            # 基于位置、距离、进攻贡献等计算偏好
            dist = _euclidean(
                profile.avg_position[0], profile.avg_position[1],
                teammate.avg_position[0], teammate.avg_position[1],
            )
            # 距离适中（不太近不太远）偏好高
            dist_score = 1.0 / max(1, abs(dist - 200) / 100 + 1)

            # 进攻贡献高 + 前场位置 → 更愿意传
            attack_score = teammate.attacking.attacking_contribution_score / 100
            forward_score = teammate.avg_position[1] / 700

            weight = dist_score * 0.4 + attack_score * 0.3 + forward_score * 0.3
            preferences[teammate.jersey_label] = round(weight, 3)

        # 归一化
        total = sum(preferences.values())
        if total > 0:
            preferences = {k: round(v / total, 3) for k, v in preferences.items()}

        return preferences

    def _evaluate_confidence(
        self,
        profile: PlayerProfile,
        feature_vec: list[float],
    ) -> float:
        """评估模型置信度。

        基于：
        - 数据完整性（轨迹帧数）
        - 特征区分度（向量方差）
        - 风格极值性（极端值说明特征明确）
        """
        confidence = 0.6  # 基础置信度

        # 特征方差大 → 区分度高
        if len(feature_vec) >= 2:
            try:
                vec_mean = sum(feature_vec) / len(feature_vec)
                variance = sum((v - vec_mean) ** 2 for v in feature_vec) / len(feature_vec)
                confidence += min(variance * 2, 0.2)
            except Exception:
                pass

        # 极端值加分
        extreme_count = sum(1 for v in feature_vec if v > 0.8 or v < 0.2)
        confidence += extreme_count * 0.03

        return round(min(confidence, 0.95), 3)


# ── 确定性风格标注（替代原 LLM 标注步骤）────────────────────────
# 特征维度 → 风格特点词，取最突出的两个维度组成标签；
# 描述为确定性文本（标注数据来源，不夸大、不编造）。

_DIM_TRAITS = {
    "传球侵略性": "组织", "射门信心": "抢点", "盘带倾向": "突破",
    "防守侵略性": "压迫", "站位纪律": "站位", "风险承受": "冒险",
    "团队倾向": "团队", "前插倾向": "前插",
}


def _deterministic_style(
    profile_dict: dict[str, Any],
    feature_vec: list[float],
) -> tuple[str, str]:
    """基于特征向量极值确定性推导风格标签与描述（零 LLM 调用）。"""
    role = str(profile_dict.get("战术角色") or "球员")
    order = sorted(range(len(feature_vec)), key=lambda i: -feature_vec[i])
    top = order[:2]
    traits = "".join(_DIM_TRAITS[FEATURE_NAMES[i]] for i in top)
    label = f"{traits}型{role}"
    desc = (
        f"基于数据特征向量的确定性风格标注："
        f"{FEATURE_NAMES[top[0]]}与{FEATURE_NAMES[top[1]]}维度最突出，"
        f"战术角色为{role}。"
    )
    return label, desc


def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    from math import hypot
    return hypot(x2 - x1, y2 - y1)
