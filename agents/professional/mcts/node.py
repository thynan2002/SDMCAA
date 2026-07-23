"""MCTS 节点定义。

定义游戏状态节点和可用动作空间。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot, sqrt, log
from typing import Any
import random


class ActionType(Enum):
    """足球动作类型。"""
    PASS = "pass"         # 传球给指定队友
    SHOOT = "shoot"       # 射门
    DRIBBLE = "dribble"   # 盘带
    CLEAR = "clear"       # 解围
    HOLD = "hold"         # 控球等待


@dataclass
class Action:
    """MCTS 动作。"""
    action_type: ActionType
    target_player: str = ""  # 传球目标 jersey_label（仅 PASS）
    direction: tuple[float, float] = (0, 0)  # 盘带方向（仅 DRIBBLE）


@dataclass
class GameState:
    """比赛状态节点。

    球场尺寸: 1200 × 700。控球方进攻方向由 attack_dir 给出：
    +1 攻 (600, 700) 球门，-1 攻 (600, 0) 球门（由初始状态按真实数据推断，
    不再硬编码固定方向）。
    """
    # 当前帧
    frame: int = 0

    # 球
    ball_x: float = 0.0
    ball_y: float = 0.0
    ball_z: float = 0.0

    # 球员位置 {jersey_label: (x, y)}
    player_positions: dict[str, tuple[float, float]] = field(default_factory=dict)

    # 球员队伍 {jersey_label: team_name}
    # 用于 MCTS 中区分同队/对手球员，使传球目标选择更真实
    player_teams: dict[str, str] = field(default_factory=dict)

    # 球权
    possession: str = ""  # 控球球员 jersey_label

    # 控球方进攻方向：+1 攻 +y 球门，-1 攻 -y 球门
    attack_dir: int = 1

    # 比分/时间
    score_diff: int = 0       # 当前球队领先分数（正=领先）
    time_remaining: int = 90  # 剩余帧数

    # 回合数
    turn: int = 0

    # 动作历史
    action_history: list[Action] = field(default_factory=list)

    # 最近射门结果
    last_shot_result: str = ""  # "goal" / "save" / "miss" / ""

    def get_possession_team(self) -> str:
        """获取控球方所属队伍。"""
        if self.possession and self.possession in self.player_teams:
            return self.player_teams[self.possession]
        return "unknown"

    def _attack_goal_y(self) -> float:
        """控球方进攻目标球门的 y 坐标。"""
        return 700.0 if self.attack_dir > 0 else 0.0

    def get_legal_actions(self) -> list[Action]:
        """获取当前状态下合法动作列表。"""
        actions: list[Action] = []

        # 球必须在场上
        if not (0 <= self.ball_x <= 1200 and 0 <= self.ball_y <= 700):
            return actions

        # 传球只能给同队队友（绝不能传给对手）。
        # player_teams 存在的意义就是区分敌我——此前未做该过滤，
        # 导致模拟中大量"传球给对手"的荒谬动作进入搜索树。
        poss_team = self.player_teams.get(self.possession, "")
        for pid, pos in self.player_positions.items():
            if pid == self.possession:
                continue
            if poss_team and self.player_teams.get(pid, "") != poss_team:
                continue  # 对手球员不是合法传球目标
            actions.append(Action(ActionType.PASS, target_player=pid))

        # 射门：控球球员靠近对方球门（球门方向由 attack_dir 决定）
        if self.possession in self.player_positions:
            px, py = self.player_positions[self.possession]
            goal_y = self._attack_goal_y()
            dist_to_goal = hypot(px - 600, py - goal_y)
            # 在对方半场且距球门 < 400 可射门
            if (py - 350) * self.attack_dir > 0 and dist_to_goal < 400:
                actions.append(Action(ActionType.SHOOT))

        # 盘带（向进攻方向）
        fwd = 30 * self.attack_dir
        actions.append(Action(ActionType.DRIBBLE, direction=(0, fwd)))
        actions.append(Action(ActionType.DRIBBLE, direction=(30, fwd)))
        actions.append(Action(ActionType.DRIBBLE, direction=(-30, fwd)))

        # 解围
        actions.append(Action(ActionType.CLEAR))

        # 等待
        actions.append(Action(ActionType.HOLD))

        return actions

    def apply_action(self, action: Action, rng: random.Random | None = None) -> GameState:
        """应用动作，返回新状态（不修改原状态）。"""
        if rng is None:
            rng = random.Random()

        new_state = GameState(
            frame=self.frame + 10,  # 每个动作约 10 帧
            ball_x=self.ball_x,
            ball_y=self.ball_y,
            ball_z=self.ball_z,
            player_positions=dict(self.player_positions),
            player_teams=dict(self.player_teams),
            possession=self.possession,
            attack_dir=self.attack_dir,
            score_diff=self.score_diff,
            time_remaining=self.time_remaining - 10,
            turn=self.turn + 1,
            action_history=self.action_history + [action],
        )

        if action.action_type == ActionType.PASS:
            # 球移向目标球员（带噪声）
            if action.target_player in new_state.player_positions:
                tx, ty = new_state.player_positions[action.target_player]
                new_state.ball_x = tx + rng.uniform(-20, 20)
                new_state.ball_y = ty + rng.uniform(-10, 10)
                new_state.ball_z = 0
                new_state.possession = action.target_player
                # 球权易主（如反事实设定的"误传给对手"）→ 进攻方向参照翻转
                if self.player_teams.get(action.target_player, "") != \
                        self.player_teams.get(self.possession, ""):
                    new_state.attack_dir = -self.attack_dir

        elif action.action_type == ActionType.SHOOT:
            # 射门判定（校准为更真实的概率），目标球门由 attack_dir 决定
            dist_to_goal = hypot(self.ball_x - 600, self.ball_y - self._attack_goal_y())
            # 射正概率：距离越近越容易射正
            on_target = max(0.05, 1.0 - dist_to_goal / 400)
            # 射正后进球的概率：距离越近进球率越高
            goal_ratio = max(0.05, 0.4 - dist_to_goal / 1000)
            roll = rng.random()

            if roll < on_target * goal_ratio:
                new_state.last_shot_result = "goal"
                new_state.score_diff = self.score_diff + 1
            elif roll < on_target:
                new_state.last_shot_result = "save"
            else:
                new_state.last_shot_result = "miss"

            # 球出界/被没收，重置到中场
            new_state.ball_x = 600 + rng.uniform(-100, 100)
            new_state.ball_y = 350
            new_state.ball_z = 0
            new_state.possession = self._random_pass_target(rng)
            if self.player_teams.get(new_state.possession, "") != \
                    self.player_teams.get(self.possession, ""):
                new_state.attack_dir = -self.attack_dir

        elif action.action_type == ActionType.DRIBBLE:
            # 球随球员移动
            dx, dy = action.direction
            new_x = self.ball_x + dx + rng.uniform(-15, 15)
            new_y = self.ball_y + dy + rng.uniform(-10, 10)
            new_state.ball_x = max(10, min(1190, new_x))
            new_state.ball_y = max(10, min(690, new_y))
            new_state.ball_z = 0
            # 更新控球球员位置
            if self.possession in new_state.player_positions:
                new_state.player_positions[self.possession] = (new_state.ball_x, new_state.ball_y)

        elif action.action_type == ActionType.CLEAR:
            # 解围：球踢向前场（进攻方向由 attack_dir 决定）
            new_state.ball_x = rng.uniform(200, 1000)
            if self.attack_dir > 0:
                new_state.ball_y = rng.uniform(400, 680)
            else:
                new_state.ball_y = rng.uniform(20, 300)
            new_state.ball_z = rng.uniform(0, 5)  # z 单位：米（与输入数据一致）
            new_state.possession = self._random_pass_target(rng)
            if self.player_teams.get(new_state.possession, "") != \
                    self.player_teams.get(self.possession, ""):
                new_state.attack_dir = -self.attack_dir

        elif action.action_type == ActionType.HOLD:
            # 原地控球，球微动
            new_state.ball_x += rng.uniform(-5, 5)
            new_state.ball_y += rng.uniform(-3, 3)

        return new_state

    def is_terminal(self, max_depth: int = 50) -> bool:
        """判断是否终止状态。"""
        # 进球
        if self.last_shot_result == "goal":
            return True
        # 时间到
        if self.time_remaining <= 0:
            return True
        # 深度限制
        if self.turn >= max_depth:
            return True
        # 球出界
        if not (0 <= self.ball_x <= 1200 and 0 <= self.ball_y <= 700):
            return True
        return False

    def get_reward(self) -> float:
        """获取当前状态的奖励值。"""
        reward = 0.0

        # 进球奖励
        if self.last_shot_result == "goal":
            reward += 2.0
        elif self.last_shot_result == "save":
            reward += 0.3
        elif self.last_shot_result == "miss":
            reward -= 0.1

        # 位置奖励：球越靠近对方球门越好（球门方向由 attack_dir 决定）
        goal_dist = hypot(self.ball_x - 600, self.ball_y - self._attack_goal_y())
        position_reward = max(0, (350 - goal_dist) / 350 * 0.5)
        reward += position_reward

        # 控球奖励
        if self.possession:
            reward += 0.1

        return reward

    def _random_pass_target(self, rng: random.Random) -> str:
        """从当前球员中随机选一个同队球员作为新的控球者。"""
        # 优先选择同队球员，避免把球权交给对手
        possession_team = self.player_teams.get(self.possession, "")
        if possession_team:
            teammates = [
                p for p in self.player_positions
                if p != self.possession
                and self.player_teams.get(p, "") == possession_team
            ]
            if teammates:
                return rng.choice(teammates)
        # 无同队球员时才允许交给任意球员
        options = [p for p in self.player_positions if p != self.possession]
        if not options:
            return self.possession
        return rng.choice(options)

    def copy(self) -> GameState:
        """深拷贝。"""
        return GameState(
            frame=self.frame,
            ball_x=self.ball_x,
            ball_y=self.ball_y,
            ball_z=self.ball_z,
            player_positions=dict(self.player_positions),
            player_teams=dict(self.player_teams),
            possession=self.possession,
            attack_dir=self.attack_dir,
            score_diff=self.score_diff,
            time_remaining=self.time_remaining,
            turn=self.turn,
            action_history=list(self.action_history),
            last_shot_result=self.last_shot_result,
        )
