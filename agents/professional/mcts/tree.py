"""MCTS 搜索树。

实现 UCB1 选择、扩展、模拟、反向传播。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, log
import random

from .node import Action, GameState
from .policy import RolloutPolicy
from agents.progress import push_stage


@dataclass
class MCTSNode:
    """搜索树节点。"""
    state: GameState
    parent: MCTSNode | None = None
    action_from_parent: Action | None = None
    children: list[MCTSNode] = field(default_factory=list)
    untried_actions: list[Action] = field(default_factory=list)

    # 统计
    visit_count: int = 0
    total_reward: float = 0.0


class MCTSTree:
    """蒙特卡洛树搜索。

    输入初始状态和多位球员的行为策略（用于不同的 rollout 视角），
    执行 UCB1 搜索返回最优动作。
    """

    def __init__(
        self,
        seed: int | None = None,
        exploration_weight: float = 1.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.exploration_weight = exploration_weight

    def search(
        self,
        root_state: GameState,
        policy: RolloutPolicy,
        num_iterations: int = 500,
        max_rollout_depth: int = 30,
    ) -> MCTSNode:
        """执行 MCTS 搜索，返回根节点（含扩展后的子节点信息）。

        Args:
            root_state: 初始游戏状态
            policy: rollout 策略
            num_iterations: 搜索迭代次数
            max_rollout_depth: 每次 rollout 最大深度

        Returns:
            根节点（children 中包含各动作的统计信息）
        """
        root = MCTSNode(state=root_state.copy())
        root.untried_actions = root_state.get_legal_actions()

        # 进度推送间隔：约 10 次更新（避免过于频繁阻塞队列）
        report_interval = max(1, num_iterations // 10)
        for i in range(num_iterations):
            # 1. 选择：从根到叶
            node = self._select(root)

            # 2. 扩展：若叶节点未终止，展开一个子节点
            if not node.state.is_terminal() and node.untried_actions:
                node = self._expand(node)

            # 3. 模拟：从新节点 rollout
            reward = policy.rollout(node.state, max_rollout_depth)

            # 4. 反向传播
            self._backpropagate(node, reward)

            # 阶段性进度推送（复用 "mcts" key，前端更新同一进度项）
            if (i + 1) % report_interval == 0 or (i + 1) == num_iterations:
                pct = round((i + 1) / num_iterations * 100)
                push_stage(
                    "mcts", "MCTS 推演",
                    detail=f"{pct}%", percent=pct,
                    current=i + 1, total=num_iterations,
                )

        return root

    def get_best_action(self, root: MCTSNode) -> Action | None:
        """从根节点的子节点中选出访问次数最多的动作。"""
        if not root.children:
            return None
        best = max(root.children, key=lambda c: c.visit_count)
        return best.action_from_parent

    def get_action_statistics(self, root: MCTSNode) -> list[dict]:
        """获取各动作的访问统计。"""
        if not root.children:
            return []

        stats = []
        for child in root.children:
            avg_reward = child.total_reward / max(1, child.visit_count)
            stats.append({
                "action": child.action_from_parent,
                "visit_count": child.visit_count,
                "avg_reward": round(avg_reward, 4),
                "total_reward": round(child.total_reward, 4),
            })
        # 按访问次数降序
        stats.sort(key=lambda s: s["visit_count"], reverse=True)
        return stats

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 选择：从根沿着树向下选择最佳子节点，直到叶节点。"""
        while not node.state.is_terminal() and not node.untried_actions and node.children:
            node = self._ucb1_select_child(node)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """从未尝试动作中随机选一个展开。"""
        action = self.rng.choice(node.untried_actions)
        node.untried_actions.remove(action)

        new_state = node.state.apply_action(action, self.rng)
        child = MCTSNode(
            state=new_state,
            parent=node,
            action_from_parent=action,
            untried_actions=new_state.get_legal_actions(),
        )
        node.children.append(child)
        return child

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """反向传播奖励。"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent

    def _ucb1_select_child(self, node: MCTSNode) -> MCTSNode:
        """UCB1 公式选择最佳子节点。"""
        total_visits = node.visit_count
        log_total = log(max(1, total_visits))

        best_child = None
        best_score = float("-inf")

        for child in node.children:
            if child.visit_count == 0:
                return child  # 优先探索未访问节点

            exploitation = child.total_reward / child.visit_count
            exploration = self.exploration_weight * sqrt(log_total / child.visit_count)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child if best_child else node.children[0]
