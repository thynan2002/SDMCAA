"""反事实分析引擎。

基于 MCTS 决策模拟系统，预测球员在反事实场景下的可能决策。
生成多维度分析报告，包括决策概率分布、预期结果评估、关键影响因素分析。
"""

from __future__ import annotations

import time
import json
from typing import Any

from agents.player.tracker import PrefixPlayerCorpus
from agents.llm_client import call_llm
from agents.progress import push_stage
from ..types import (
    CounterfactualScenario,
    CounterfactualReport,
    DecisionDistribution,
    PlayerBehaviorModel,
    ScenarioType,
)
from ..mcts.node import GameState, Action, ActionType
from ..mcts.tree import MCTSTree
from ..mcts.policy import RolloutPolicy, create_policy_from_model


# ═══════════════════════════════════════════════════════════════════
# 智能体
# ═══════════════════════════════════════════════════════════════════

class CounterfactualEngineAgent:
    """反事实分析引擎。

    接收反事实场景定义和球员行为模型，
    通过 MCTS 模拟预测球员在不同决策路径下的可能结果。
    """

    name = "CounterfactualEngineAgent"

    def __init__(
        self,
        num_simulations: int = 300,
        seed: int | None = None,
        llm_engine: Any | None = None,
    ) -> None:
        self.num_simulations = num_simulations
        self.seed = seed
        self.mcts = MCTSTree(seed=seed)
        # LLM 决策引擎：为 MCTS rollout 生成策略表（替代公式权重）
        self.llm_engine = llm_engine

    def analyze(
        self,
        scenario: CounterfactualScenario,
        behavior_models: dict[str, PlayerBehaviorModel],
        corpus: PrefixPlayerCorpus,
    ) -> CounterfactualReport:
        """执行反事实分析。

        Args:
            scenario: 反事实场景定义
            behavior_models: {jersey_label: PlayerBehaviorModel} 全体球员模型
            corpus: 比赛语料（用于获取初始状态）

        Returns:
            反事实分析报告
        """
        start_time = time.time()

        # ── 构建初始游戏状态 ──
        subject_model = behavior_models.get(scenario.subject_player)
        if subject_model is None:
            return self._empty_report(scenario)

        push_stage("build_state", "正在构建初始比赛状态…")
        # 从 corpus 提取关键帧的状态
        root_state = self._build_initial_state(scenario, corpus)
        push_stage("build_state", "初始状态构建完成", status="done")

        # ── 创建 subject 球员的 rollout 策略 ──
        # 优先由 LLM 决策引擎一次性生成策略表；不可用时回退行为模型公式权重
        action_weights = subject_model.action_weights
        pass_prefs = subject_model.pass_target_preference
        if self.llm_engine and self.llm_engine.enabled:
            push_stage("strategy", "正在生成 LLM 策略表…")
            teammates = [
                jl for jl in behavior_models.keys()
                if jl != scenario.subject_player
            ]
            strategy = self.llm_engine.generate_strategy_table(
                subject_model.to_dict(),
                teammates,
                f"反事实情境：{scenario.describe_altered()}",
            )
            if strategy:
                action_weights = strategy["action_weights"]
                pass_prefs = strategy["pass_preferences"] or pass_prefs
            push_stage("strategy", "策略表生成完成", status="done")

        policy = create_policy_from_model(
            action_weights,
            pass_prefs,
            seed=self.seed,
        )

        # ── MCTS 搜索 ──
        push_stage(
            "mcts", f"MCTS 推演（{self.num_simulations}次模拟）",
            detail="0%", percent=0, total=self.num_simulations,
        )
        root_node = self.mcts.search(
            root_state=root_state,
            policy=policy,
            num_iterations=self.num_simulations,
            max_rollout_depth=30,
        )
        push_stage("mcts", "MCTS 推演", detail="完成", percent=100, status="done")
        actual_depth_avg = self._compute_avg_depth(root_node)

        # ── 提取动作统计 ──
        action_stats = self.mcts.get_action_statistics(root_node)
        total_visits = sum(s["visit_count"] for s in action_stats) or 1

        # 构建决策分布
        decision_dist: list[DecisionDistribution] = []
        for stat in action_stats[:5]:  # Top 5
            action = stat["action"]
            prob = stat["visit_count"] / total_visits
            action_name = self._describe_action(action)
            decision_dist.append(DecisionDistribution(
                action=action_name,
                probability=round(prob, 4),
                expected_value=round(stat["avg_reward"], 4),
                description="",
            ))

        # ── 原始决策 vs 反事实决策的对比模拟 ──
        push_stage("simulate_original", "正在模拟原始决策路径…")
        original_result = self._simulate_original(scenario, policy, root_state)
        push_stage("simulate_original", "原始路径模拟完成", status="done")
        push_stage("simulate_altered", "正在模拟反事实决策路径…")
        altered_result = self._simulate_altered(scenario, policy, root_state)
        push_stage("simulate_altered", "反事实路径模拟完成", status="done")

        # ── LLM 生成分析文本 ──
        elapsed = time.time() - start_time
        push_stage("llm_analysis", "正在生成分析报告…")
        analysis_text, key_factors = self._generate_analysis_with_llm(
            scenario, decision_dist, original_result, altered_result, elapsed,
        )
        push_stage("llm_analysis", "分析报告生成完成", status="done")

        return CounterfactualReport(
            scenario=scenario,
            decision_distribution=decision_dist,
            total_simulations=self.num_simulations,
            simulation_depth_avg=round(actual_depth_avg, 1),
            goal_probability_original=original_result.get("goal", 0),
            goal_probability_altered=altered_result.get("goal", 0),
            possession_retention_original=original_result.get("possession", 0),
            possession_retention_altered=altered_result.get("possession", 0),
            analysis_summary=analysis_text,
            key_factors=key_factors,
            confidence=self._compute_confidence(action_stats, total_visits),
        )

    def _compute_avg_depth(self, root_node: Any) -> float:
        """从 MCTS 搜索树计算真实平均模拟深度。"""
        if root_node is None:
            return 0.0
        depths: list[int] = []
        self._collect_depths(root_node, 0, depths)
        return sum(depths) / max(1, len(depths))

    def _collect_depths(self, node: Any, depth: int, depths: list[int]) -> None:
        """递归收集所有叶子节点的深度。"""
        if not hasattr(node, 'children') or not node.children:
            depths.append(depth)
            return
        for child in node.children:
            self._collect_depths(child, depth + 1, depths)

    def _compute_confidence(
        self,
        action_stats: list[dict],
        total_visits: int,
    ) -> float:
        """基于 MCTS 实际搜索结果动态计算置信度。

        综合四个维度：
        1. 最优动作访问占比：访问越集中于最优动作 → 置信度越高
        2. 访问充分性：最优动作访问次数需达到阈值才可信赖，
           防止"低访问次数 + 高胜率"节点获得虚高置信度
        3. Q 值分离度：最优与次优动作的 avg_reward 差距越大 → 置信度越高
        4. 模拟规模充分性：总访问数越多 → 置信度越高（平方根衰减）

        与旧版的关键差异：
        - 用"最优动作访问占比"替代"归一化熵"，更直接反映决策确信程度
        - 新增"访问充分性"维度，防止低访问次数的高胜率节点获得虚高置信度
        - Q 值分离度归一化阈值从 0.5 降至 0.15，适配实际奖励范围
        - 模拟规模基于实际总访问数而非配置迭代次数

        返回 [0.1, 0.95] 区间的置信度。
        """
        import math

        if not action_stats or total_visits <= 0:
            return 0.1

        n_actions = len(action_stats)
        best = action_stats[0]  # 已按 visit_count 降序排列

        # ── 维度 1：最优动作访问占比（权重 0.35）──
        # 访问越集中于最优动作 → 置信度越高
        best_visit_share = best["visit_count"] / total_visits

        # ── 维度 2：访问充分性（权重 0.15）──
        # 最优动作至少需 50 次访问才具备统计可靠性
        visit_sufficiency = min(1.0, best["visit_count"] / 50.0)

        # ── 维度 3：Q 值分离度（权重 0.30）──
        if n_actions >= 2:
            q_gap = abs(best["avg_reward"] - action_stats[1]["avg_reward"])
        else:
            q_gap = 0.15  # 仅一个可选动作 → 中等分离度
        # 归一化阈值 0.15：rollout 奖励通常在 [0, 1.5] 区间，
        # 0.15 的 avg_reward 差距已具有显著区分意义
        q_separation = min(1.0, q_gap / 0.15)

        # ── 维度 4：模拟规模充分性（权重 0.20）──
        # 基于实际总访问数（而非配置迭代次数），平方根衰减避免线性饱和
        sim_factor = min(1.0, math.sqrt(total_visits / 1000.0))

        # ── 加权融合 ──
        confidence = (
            best_visit_share * 0.35
            + visit_sufficiency * 0.15
            + q_separation * 0.30
            + sim_factor * 0.20
        )

        return round(min(0.95, max(0.1, confidence)), 4)

    # ── 辅助方法 ──

    def _build_initial_state(
        self,
        scenario: CounterfactualScenario,
        corpus: PrefixPlayerCorpus,
    ) -> GameState:
        """从 corpus 提取关键帧的比赛状态。"""
        frame = scenario.frame
        ball_frames = corpus.ball_frames

        # 球位置：取最近观测帧（球数据稀疏，精确帧常常缺失；
        # 绝不能回退到硬编码的场地中心 (600,350)，那会向模拟输入错误局势）
        ball_pos = self._nearest_ball_pos(ball_frames, frame)
        bx, by, bz = ball_pos

        # 球员位置和队伍信息
        player_positions: dict[str, tuple[float, float]] = {}
        player_teams: dict[str, str] = {}
        for pid, player in corpus.players.items():
            # 查找该帧的位置
            pos = self._find_position_at_frame(player, frame)
            if pos:
                player_positions[player.jersey_label] = pos
                # 用 color 作为队伍标识
                player_teams[player.jersey_label] = player.color or "unknown"

        # 控球判定：谁最近
        possession = scenario.subject_player
        min_dist = float("inf")
        for jlabel, (px, py) in player_positions.items():
            d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
            if d < min_dist:
                min_dist = d
                possession = jlabel

        # 控球方进攻方向：按真实数据推断（与模拟引擎同一约定——
        # 各队平均 y 排序交替分配，平均 y 小的队攻 +y），不再硬编码 +y
        attack_dir = self._infer_attack_dir(corpus, player_teams.get(possession, ""))

        return GameState(
            frame=frame,
            ball_x=bx,
            ball_y=by,
            ball_z=bz,
            player_positions=player_positions,
            player_teams=player_teams,
            possession=possession,
            attack_dir=attack_dir,
            score_diff=scenario.score_diff,
            time_remaining=scenario.time_remaining_frames or 300,
        )

    @staticmethod
    def _nearest_ball_pos(
        ball_frames: dict[int, tuple[float, float, float]],
        frame: int,
    ) -> tuple[float, float, float]:
        """取 ≤ frame 的最近球观测；无则取全片最近观测；再无可用的空数据才回退场心。"""
        if not ball_frames:
            return (600, 350, 0)
        best = -1
        for bf in ball_frames:
            if bf <= frame and bf > best:
                best = bf
        if best >= 0:
            return ball_frames[best]
        nearest = min(ball_frames, key=lambda bf: abs(bf - frame))
        return ball_frames[nearest]

    @staticmethod
    def _infer_attack_dir(corpus: PrefixPlayerCorpus, team_color: str) -> int:
        """推断指定队伍的进攻方向（+1 攻 +y / -1 攻 -y）。

        与 TrajectorySimulator._attack_directions 同一约定：
        按各队平均 y 升序交替分配方向，平均 y 偏小的队攻 +y。
        """
        sums: dict[str, list[float]] = {}
        for p in corpus.players.values():
            sums.setdefault(p.color or "unknown", []).append(p.avg_y())
        if not sums:
            return 1
        avgs = {c: sum(ys) / len(ys) for c, ys in sums.items()}
        sign = 1
        for color in sorted(avgs, key=lambda c: avgs[c]):
            if color == team_color:
                return sign
            sign = -sign
        return 1

    def _find_position_at_frame(
        self,
        player: Any,
        target_frame: int,
    ) -> tuple[float, float] | None:
        """球员在任意帧的估计位置（线性插值，跟踪数据稀疏时不再失真）。"""
        from agents.player.tracker import position_at_frame
        return position_at_frame(player, target_frame)

    def _simulate_original(
        self,
        scenario: CounterfactualScenario,
        policy: RolloutPolicy,
        root_state: GameState,
    ) -> dict[str, float]:
        """模拟原始决策路径的结果。"""
        # 从初始状态开始，首先应用原始动作
        original_action = self._action_from_scenario(scenario, altered=False)
        state = root_state.apply_action(original_action)

        # 继续 rollout
        rewards = policy.rollout_many(state, num_rollouts=50, max_depth=20)
        return self._compute_outcome_stats(rewards)

    def _simulate_altered(
        self,
        scenario: CounterfactualScenario,
        policy: RolloutPolicy,
        root_state: GameState,
    ) -> dict[str, float]:
        """模拟反事实决策路径的结果。"""
        altered_action = self._action_from_scenario(scenario, altered=True)
        state = root_state.apply_action(altered_action)

        rewards = policy.rollout_many(state, num_rollouts=50, max_depth=20)
        return self._compute_outcome_stats(rewards)

    def _action_from_scenario(
        self,
        scenario: CounterfactualScenario,
        altered: bool,
    ) -> Action:
        """从场景描述构建 Action 对象。"""
        action_name = scenario.altered_action if altered else scenario.original_action

        if action_name == "pass":
            target = scenario.altered_target if altered else scenario.original_target
            return Action(ActionType.PASS, target_player=target)
        elif action_name == "shoot":
            return Action(ActionType.SHOOT)
        elif action_name == "dribble":
            return Action(ActionType.DRIBBLE, direction=(0, 30))
        elif action_name == "clear":
            return Action(ActionType.CLEAR)
        else:
            return Action(ActionType.HOLD)

    def _compute_outcome_stats(self, rewards: list[float]) -> dict[str, float]:
        """从奖励列表计算结果统计。"""
        if not rewards:
            return {"goal": 0, "possession": 0.5}

        # 高奖励 (>1.5) ≈ 进球
        goal_count = sum(1 for r in rewards if r > 1.5)
        # 正奖励 ≈ 保持控球优势
        possession_count = sum(1 for r in rewards if r > 0.2)

        return {
            "goal": round(goal_count / len(rewards), 4),
            "possession": round(possession_count / len(rewards), 4),
        }

    def _describe_action(self, action: Action) -> str:
        """生成动作的可读描述。"""
        if action.action_type == ActionType.PASS:
            return f"传球给{action.target_player}" if action.target_player else "传球"
        elif action.action_type == ActionType.SHOOT:
            return "射门"
        elif action.action_type == ActionType.DRIBBLE:
            dx, dy = action.direction
            if abs(dx) < 10:
                return "向前盘带"
            elif dx > 0:
                return "向右前盘带"
            else:
                return "向左前盘带"
        elif action.action_type == ActionType.CLEAR:
            return "解围"
        return "控球等待"

    def _generate_analysis_with_llm(
        self,
        scenario: CounterfactualScenario,
        decision_dist: list[DecisionDistribution],
        original_result: dict[str, float],
        altered_result: dict[str, float],
        elapsed: float,
    ) -> tuple[str, list[str]]:
        """使用 LLM 生成反事实分析报告和关键因素（所有输出由 LLM 完成）。"""
        system_prompt = """## 角色
你是足球战术分析师 (Counterfactual Analyst)，专门负责"反事实"战术推演。你的职责是基于 MCTS（蒙特卡洛树搜索）模拟结果，对比分析原始决策与替代决策的优劣。

## 输入
- **反事实场景**: 原始发生了什么 vs 反事实假设是什么
- **MCTS 模拟结果**: 基于球员行为模型进行了 N 次推演后的统计（决策概率分布、进球概率对比、控球率对比）
- **球员风格背景**: 模拟中使用的球员行为特征

## MCTS 模拟说明（重要）
- MCTS 是基于位置、速度、风格特征向量的简化模拟，**不是真实比赛**
- 模拟假设球员按历史风格模式行动，不包含偶然因素（伤病、情绪等）
- 模拟次数越多（如 300 次），结果越稳定，但仍是一种**概率性估计**
- 你的任务是在认可模拟结果合理性的同时，指出其局限性

## 输出格式
```
【分析报告】
（分析文本，对比原始与反事实决策的优劣；篇幅根据内容自然展开，不设硬性上限，该详则详、该简则简）

【关键因素】
- 因素1
- 因素2
- ...
```

## 核心要求
1. **语言风格**: 像资深足球评论员的分析，专业但不晦涩
2. **禁止数值**: 不要输出任何具体概率/数值（用"更可能"、"显著提升"、"略微下降"、"几乎一样"等表达）
3. **深度分析**: 解释为什么某决策更好/更差——基于球员风格特点、场上位置、比赛态势
4. **诚实说明**: 如果两个决策差异极小（模拟结果接近），说明"差异不大，取决于临场判断"，不要强行制造差异"""

        # 构建顶部分布描述
        top_actions = "\n".join(
            f"  {d.action}: 概率{d.probability:.1%}, 预期收益{d.expected_value:.3f}"
            for d in decision_dist[:3]
        )

        user_msg = f"""反事实场景：
  原始决策: {scenario.describe_original()}
  反事实决策: {scenario.describe_altered()}

模拟结果（基于{self.num_simulations}次MCTS模拟）：
  
球员最可能的决策Top-3：
{top_actions}

原始决策路径预期：
  进球概率: {original_result.get('goal', 0):.1%}
  控球保持率: {original_result.get('possession', 0):.1%}

反事实决策路径预期：
  进球概率: {altered_result.get('goal', 0):.1%}
  控球保持率: {altered_result.get('possession', 0):.1%}

请生成分析报告。"""

        result = call_llm(system_prompt, user_msg, stream=True)
        if result:
            report_text, factors = self._parse_llm_report(result)
            return report_text, factors

        # 回退
        return (
            f"模拟分析完成（{self.num_simulations}次模拟）。"
            f"\n原始决策: {scenario.describe_original()}，预期进球率{original_result.get('goal', 0):.1%}。"
            f"\n反事实决策: {scenario.describe_altered()}，预期进球率{altered_result.get('goal', 0):.1%}。",
            ["球员风格", "场上位置", "比赛态势"],
        )

    def _parse_llm_report(self, text: str) -> tuple[str, list[str]]:
        """解析 LLM 输出的报告文本。"""
        report = ""
        factors: list[str] = []

        # 简单分段解析
        in_factors = False
        for line in text.strip().split("\n"):
            line = line.strip()
            if "关键因素" in line or "【关键因素】" in line:
                in_factors = True
                continue
            if "分析报告" in line or "【分析报告】" in line:
                continue
            if in_factors:
                clean = line.lstrip("-•·0123456789. ").strip()
                if clean:
                    factors.append(clean)
            else:
                if line:
                    report += line + "\n"

        return report.strip(), factors[:5]

    def _empty_report(self, scenario: CounterfactualScenario) -> CounterfactualReport:
        """生成空报告。"""
        return CounterfactualReport(
            scenario=scenario,
            total_simulations=0,
            analysis_summary="无法找到主体球员的行为模型，分析终止。",
            key_factors=[],
            confidence=0.0,
        )
