"""MCTS Rollout 策略。

基于球员行为模型进行动作采样，驱动模拟推演。
"""

from __future__ import annotations

from typing import Any
import random
from math import hypot

from .node import Action, ActionType, GameState


class RolloutPolicy:
    """基于球员行为模型的 rollout 策略。

    使用球员风格特征向量加权的动作选择概率分布。
    """

    def __init__(
        self,
        action_weights: dict[str, float] | None = None,
        pass_preferences: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        """初始化策略。

        Args:
            action_weights: 动作权重 {"pass": 0.4, "shoot": 0.15, ...}
            pass_preferences: 传球目标偏好 {jersey_label: weight}
            seed: 随机种子
        """
        self.action_weights = action_weights or {
            "pass": 0.4, "shoot": 0.15, "dribble": 0.25,
            "clear": 0.1, "hold": 0.1,
        }
        self.pass_preferences = pass_preferences or {}
        self.rng = random.Random(seed)

    def select_action(
        self,
        legal_actions: list[Action],
        state: GameState | None = None,
    ) -> Action:
        """从合法动作中按权重采样一个动作。

        当提供 state 时，融合领域启发式偏置：靠近球门提升射门权重，
        己方半场远端提升解围权重，使 rollout 更贴近真实决策逻辑。
        """
        if not legal_actions:
            return Action(ActionType.HOLD)

        # 按动作类型分组
        action_groups: dict[ActionType, list[Action]] = {}
        for a in legal_actions:
            action_groups.setdefault(a.action_type, []).append(a)

        # 获取每种类型的基础权重，乘以启发式偏置
        heuristic_bias = self._compute_heuristic_bias(state) if state else {}
        weights: list[tuple[list[Action], float]] = []
        for atype, actions_list in action_groups.items():
            w = self.action_weights.get(atype.value, 0.05)
            w *= heuristic_bias.get(atype, 1.0)
            weights.append((actions_list, w))

        # 归一化并采样类型
        total_w = sum(w for _, w in weights) or 1.0
        roll = self.rng.random() * total_w
        cum = 0.0
        selected_group = weights[-1][0]  # 默认最后一种
        for actions_list, w in weights:
            cum += w
            if roll <= cum:
                selected_group = actions_list
                break

        # 从选中组中再选具体动作
        return self._select_from_group(selected_group, state)

    def _compute_heuristic_bias(self, state: GameState) -> dict[ActionType, float]:
        """基于足球领域知识计算动作类型启发式偏置乘数。

        根据控球球员与球门距离、所在半场，调整动作权重：
        - 靠近对方球门 → 大幅提升射门权重
        - 对方半场 → 略增盘带/传球权重
        - 己方半场远端 → 提升解围权重
        """
        bias: dict[ActionType, float] = {}
        if state.possession not in state.player_positions:
            return bias

        px, py = state.player_positions[state.possession]
        goal_y = 700.0 if state.attack_dir > 0 else 0.0
        dist_to_goal = hypot(px - 600, py - goal_y)

        # 靠近球门 → 偏向射门
        if dist_to_goal < 200:
            bias[ActionType.SHOOT] = 3.0
        elif dist_to_goal < 350:
            bias[ActionType.SHOOT] = 1.8

        # 对方半场 → 偏向向前传球和盘带
        in_attacking_half = (py - 350) * state.attack_dir > 0
        if in_attacking_half:
            bias[ActionType.DRIBBLE] = bias.get(ActionType.DRIBBLE, 1.0) * 1.3
            bias[ActionType.PASS] = bias.get(ActionType.PASS, 1.0) * 1.2
        else:
            # 己方半场 → 偏向传球和解围
            bias[ActionType.PASS] = bias.get(ActionType.PASS, 1.0) * 1.2
            if dist_to_goal > 450:
                bias[ActionType.CLEAR] = 1.8

        return bias

    def _select_from_group(
        self,
        actions: list[Action],
        state: GameState | None = None,
    ) -> Action:
        """从同类型动作中选择具体目标。

        PASS 动作优先按球员偏好加权，同时结合位置启发式：
        向前传球（朝进攻方向）获得额外权重加成。
        """
        if not actions:
            return Action(ActionType.HOLD)

        # PASS 动作按偏好 + 向前偏置加权
        if actions[0].action_type == ActionType.PASS:
            weighted_actions: list[tuple[Action, float]] = []
            for a in actions:
                w = self.pass_preferences.get(a.target_player, 0.1)
                # 启发式：向进攻方向传球获得加成
                if state and a.target_player in state.player_positions:
                    tx, ty = state.player_positions[a.target_player]
                    if state.possession in state.player_positions:
                        _, py = state.player_positions[state.possession]
                        if (ty - py) * state.attack_dir > 0:
                            w *= 1.5  # 向前传球加成
                weighted_actions.append((a, w))
            total = sum(w for _, w in weighted_actions) or 1.0
            roll = self.rng.random() * total
            cum = 0.0
            for a, w in weighted_actions:
                cum += w
                if roll <= cum:
                    return a

        # 随机选择
        return self.rng.choice(actions)

    def rollout(
        self,
        state: GameState,
        max_depth: int = 50,
    ) -> float:
        """从给定状态开始一次 rollout，返回累积奖励。

        使用启发式动作选择（非纯随机）并在每步累积
        带折扣的即时奖励，使不同动作路径产生可区分的 Q 值。
        """
        current = state.copy()
        total_reward = 0.0
        discount = 0.95
        gamma = 1.0

        for _ in range(max_depth):
            # 用大 max_depth 调用 is_terminal，仅依赖 for 循环控制 rollout 深度，
            # 避免 tree 深度叠加导致 rollout 被过早截断
            if current.is_terminal(max_depth=9999):
                total_reward += current.get_reward() * gamma
                break

            legal_actions = current.get_legal_actions()
            if not legal_actions:
                break

            action = self.select_action(legal_actions, state=current)
            current = current.apply_action(action, self.rng)

            # 即时奖励：射门结果给全额奖励，非终结状态给 0.3 系数
            # （此前 0.1 系数过小，导致不同动作路径 Q 值几乎不可区分）
            if current.last_shot_result:
                total_reward += current.get_reward() * gamma
            else:
                total_reward += current.get_reward() * gamma * 0.3

            gamma *= discount

        return total_reward

    def rollout_many(
        self,
        state: GameState,
        num_rollouts: int = 100,
        max_depth: int = 50,
    ) -> list[float]:
        """执行多次 rollout，返回奖励列表。"""
        return [self.rollout(state, max_depth) for _ in range(num_rollouts)]


def create_policy_from_model(
    action_weights: dict[str, float],
    pass_preferences: dict[str, float] | None = None,
    seed: int | None = None,
) -> RolloutPolicy:
    """从球员行为模型的权重创建 rollout 策略。"""
    defaults = {
        "pass": 0.4, "shoot": 0.15, "dribble": 0.25,
        "clear": 0.1, "hold": 0.1,
    }
    # 合并，确保所有键存在
    merged = {**defaults, **action_weights}
    # 归一化
    total = sum(merged.values())
    merged = {k: v / total for k, v in merged.items()}

    return RolloutPolicy(
        action_weights=merged,
        pass_preferences=pass_preferences,
        seed=seed,
    )
