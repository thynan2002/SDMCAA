"""风格学习与建模智能体。

基于球员画像构建行为预测模型，
生成特征向量、风格标签和决策权重。
所有输出文本由 LLM 生成。
"""

from __future__ import annotations

import json
from typing import Any

from agents.llm_client import call_llm
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

        # 4. LLM 生成风格标签和描述
        style_label, style_desc = self._generate_style_with_llm(profile_dict, feature_vec)

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
    ) -> list[PlayerBehaviorModel]:
        """批量构建球员行为模型。"""
        return [self.build_model(p, profiles) for p in profiles]

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

    def _generate_style_with_llm(
        self,
        profile_dict: dict[str, Any],
        feature_vec: list[float],
    ) -> tuple[str, str]:
        """使用 LLM 生成风格标签和详细描述。"""
        system_prompt = """## 角色
你是足球战术分析师 (Style Labeler)。职责是为球员生成风格标签和描述。你的输出是下游分析的重要基础——风格标签会被 MCTS 用于决策模拟，描述会出现在最终报告中。

## 输入
- 球员数据：球衣号、战术角色、主要活动区域、进攻贡献评分
- 行为特征向量：8 个维度的 0-1 数值（传球侵略性、射门信心、盘带倾向、防守侵略性、站位纪律、风险承受、团队倾向、前插倾向）

## 输出格式（严格二行结构）
第一行: 风格标签（2-8 个汉字，不要前缀）
第二行起: 风格描述（80-120 字，不使用具体数值）

## 风格标签规则
- 长度：2-8 个汉字。**不是必须 5 字以内**——"出球型中后卫"(6字)、"抢点型边锋"(5字)、"防守工兵"(4字) 都允许
- 结构：特点/动作 + 位置。如"扫荡型后腰"、"组织型中场"、"突破型边锋"
- 精准度优先：宁可略长也要准确，不要为凑字数用模糊标签
- 禁止使用：球员号、队伍名、具体数值

## 风格描述规则
- 分析维度：技术特点 → 战术角色 → 对比赛的影响
- 语言风格：像资深球探报告，专业但不晦涩
- 禁止使用：任何具体数值（坐标、距离、百分比）
- 长度：80-120 字

## 注意
- 你的输出会被程序自动解析：第一行截取为标签，剩余行为描述
- 不要添加"风格标签："、"描述："等前缀
- 如果不确定，宁可选更通用的标签（如"中场球员"）也不要强行创造不准确的标签"""

        user_msg = json.dumps(
            {
                "球员数据": profile_dict,
                "特征向量": dict(zip(FEATURE_NAMES, [round(v, 3) for v in feature_vec])),
            },
            ensure_ascii=False,
            indent=2,
        )
        result = call_llm(system_prompt, user_msg)
        if result:
            lines = result.strip().split("\n")
            label = lines[0].strip() if lines else "未分类"
            desc = "\n".join(lines[1:]).strip() if len(lines) > 1 else result.strip()
            return label, desc

        # 回退
        return profile_dict.get("战术角色", "球员"), f"基于数据推断的{profile_dict.get('战术角色', '球员')}"

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


def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    from math import hypot
    return hypot(x2 - x1, y2 - y1)
