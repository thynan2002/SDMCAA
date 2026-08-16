"""专业分析子系统数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# 场景类型枚举
# ═══════════════════════════════════════════════════════════════════

class ScenarioType(Enum):
    """反事实场景类型。"""
    PASS_TARGET_CHANGE = "pass_target_change"       # 队友传球目标改变
    OPPONENT_PATH_CHANGE = "opponent_path_change"    # 对手传球路径变化
    SELF_CHOICE_CHANGE = "self_choice_change"        # 球员自身选择变化
    ENVIRONMENT_CHANGE = "environment_change"         # 比赛环境变量调整


# ═══════════════════════════════════════════════════════════════════
# 意图类型枚举
# ═══════════════════════════════════════════════════════════════════

class IntentType(Enum):
    """用户意图类型。"""
    FOCUS_PLAYER = "focus_player"            # 聚焦某球员
    STYLE_ANALYSIS = "style_analysis"         # 风格分析
    COUNTERFACTUAL = "counterfactual"         # 反事实分析
    COMPARISON = "comparison"                # 球员对比
    GENERAL_ANALYSIS = "general_analysis"     # 整体分析
    VERIFY_DATA = "verify_data"              # 数据核验/质疑/回查
    GENERAL_QA = "general_qa"                # 综合问答（兜底）
    HELP = "help"                            # 帮助
    EXIT = "exit"                            # 退出
    CLEAR = "clear"                          # 清除上下文


# ═══════════════════════════════════════════════════════════════════
# 球员画像相关
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AttackingProfile:
    """球员进攻风格画像。"""
    pass_tendency: dict[str, float] = field(default_factory=dict)
    # 传球倾向分布，如 {"短传": 0.5, "长传": 0.2, "直塞": 0.15, "回传": 0.15}

    shoot_preference: dict[str, float] = field(default_factory=dict)
    # 射门选择分布，如 {"远射": 0.3, "禁区内射门": 0.5, "头球": 0.2}

    shoot_total: int = 0
    # 射门总次数

    dribble_frequency: float = 0.0
    # 盘带频率（每百帧盘带次数归一化）

    dribble_direction_bias: dict[str, float] = field(default_factory=dict)
    # 盘带方向偏好，如 {"向前": 0.6, "横向": 0.25, "向后": 0.15}

    off_ball_run_type: dict[str, float] = field(default_factory=dict)
    # 无球跑动模式，如 {"前插": 0.4, "回撤接应": 0.3, "拉边": 0.2, "原地等待": 0.1}

    attacking_contribution_score: float = 0.0
    # 进攻贡献综合评分 0-100


@dataclass
class DefendingProfile:
    """球员防守风格画像。"""
    intercept_tendency: float = 0.0
    # 拦截倾向 0-1（0=站位防守，1=积极上抢）

    tackle_frequency: float = 0.0
    # 抢断频率（每百帧抢断次数归一化）

    defensive_line_preference: str = ""
    # 防守站位偏好："高位压迫" / "中位防守" / "低位退守"

    marking_style: str = ""
    # 盯人风格："紧逼盯人" / "区域防守" / "混合防守"

    cover_awareness_score: float = 0.0
    # 协防意识评分 0-100

    recovery_speed_tier: str = ""
    # 回追速度特征："极快" / "快" / "中等" / "慢"

    defensive_contribution_score: float = 0.0
    # 防守贡献综合评分 0-100


@dataclass
class PlayerProfile:
    """球员综合画像。"""
    track_id: str = ""
    jersey_label: str = ""
    color: str = ""
    team_name: str = ""

    # ── 时空特征 ──
    avg_position: tuple[float, float] = (0, 0)
    coverage_area: float = 0.0
    zone_distribution: dict[str, float] = field(default_factory=dict)
    dominant_zone: str = ""
    speed_stats: dict[str, float] = field(default_factory=dict)
    # {"avg": ..., "max": ..., "sprint_count": ...}
    total_distance_m: float = 0.0       # 累计跑动距离（米，x/y 分轴标定，与独立金标准同口径）
    ball_proximity_pct: float = 0.0     # 接近球时间占比（%）

    acceleration_distribution: dict[str, float] = field(default_factory=dict)
    # {"high": 0.2, "medium": 0.5, "low": 0.3}

    # ── 战术角色 ──
    tactical_role: str = ""
    # "中后卫"/"边后卫"/"防守中场"/"组织中场"/"边锋"/"中锋" 等

    # ── 进攻/防守风格 ──
    attacking: AttackingProfile = field(default_factory=AttackingProfile)
    defending: DefendingProfile = field(default_factory=DefendingProfile)

    # ── LLM 生成的语义描述（所有输出由 LLM 完成） ──
    style_summary: str = ""
    # LLM 生成的球员风格一句话概括

    def to_dict(self) -> dict[str, Any]:
        """转为字典供 LLM 消费。"""
        return {
            "球衣号": self.jersey_label,
            "所属队伍": self.team_name,
            "战术角色": self.tactical_role,
            "主要活动区域": self.dominant_zone,
            "区域分布": self.zone_distribution,
            "平均纵向位置": self.avg_position[1],  # y 坐标，供特征向量使用
            "跑动距离(米)": self.total_distance_m,
            "场地覆盖率(%)": self.coverage_area,
            "接近球时间占比(%)": self.ball_proximity_pct,
            "速度统计": self.speed_stats,
            "加速度分布": self.acceleration_distribution,
            "进攻风格": {
                "传球倾向": self.attacking.pass_tendency,
                "射门选择": self.attacking.shoot_preference,
                "射门频率": self.attacking.shoot_total,  # 供特征向量射门信心维度使用
                "盘带频率": self.attacking.dribble_frequency,
                "盘带方向偏好": self.attacking.dribble_direction_bias,
                "无球跑动模式": self.attacking.off_ball_run_type,
                "进攻贡献评分": self.attacking.attacking_contribution_score,
            },
            "防守风格": {
                "拦截倾向": self.defending.intercept_tendency,
                "抢断频率": self.defending.tackle_frequency,
                "防守站位偏好": self.defending.defensive_line_preference,
                "盯人风格": self.defending.marking_style,
                "协防意识评分": self.defending.cover_awareness_score,
                "回追速度": self.defending.recovery_speed_tier,
                "防守贡献评分": self.defending.defensive_contribution_score,
            },
            "风格概括": self.style_summary,
        }


# ═══════════════════════════════════════════════════════════════════
# 行为模型相关
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PlayerBehaviorModel:
    """球员决策行为预测模型。"""
    track_id: str = ""
    jersey_label: str = ""

    # ── N 维行为特征向量 ──
    feature_vector: list[float] = field(default_factory=list)
    # 维度说明：
    # [0] pass_aggressiveness      传球侵略性 (0-1)
    # [1] shoot_confidence         射门信心 (0-1)
    # [2] dribble_frequency        盘带倾向 (0-1)
    # [3] defensive_aggressiveness 防守侵略性 (0-1)
    # [4] positional_discipline    站位纪律 (0-1)
    # [5] risk_tolerance           风险承受 (0-1)
    # [6] team_orientation         团队倾向 vs 个人主义 (0-1)
    # [7] forward_bias             前插倾向 (0-1)

    FEATURE_DIM = 8  # 特征向量维度
    FEATURE_NAMES = [
        "传球侵略性", "射门信心", "盘带倾向", "防守侵略性",
        "站位纪律", "风险承受", "团队倾向", "前插倾向",
    ]

    # ── LLM 生成 ──
    style_label: str = ""
    # 风格标签，如 "组织型中场"、"抢点型前锋"、"扫荡型后腰"

    style_description: str = ""
    # LLM 生成的详细风格描述

    # ── MCTS 决策权重 ──
    action_weights: dict[str, float] = field(default_factory=dict)
    # 各动作的初始权重 {"pass": 0.4, "shoot": 0.15, "dribble": 0.25, "clear": 0.1, "hold": 0.1}

    pass_target_preference: dict[str, float] = field(default_factory=dict)
    # 传球目标偏好 {jersey_label: weight}

    # ── 模型评估 ──
    confidence_score: float = 0.0
    # 模型置信度 0-1

    def to_dict(self) -> dict[str, Any]:
        return {
            "球衣号": self.jersey_label,
            "风格标签": self.style_label,
            "风格描述": self.style_description,
            "特征向量": {
                name: round(val, 3)
                for name, val in zip(self.FEATURE_NAMES, self.feature_vector)
            },
            "动作权重": self.action_weights,
            "模型置信度": round(self.confidence_score, 3),
        }


# ═══════════════════════════════════════════════════════════════════
# 反事实分析相关
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CounterfactualScenario:
    """反事实场景定义。"""
    scenario_type: ScenarioType = ScenarioType.SELF_CHOICE_CHANGE

    # 主体球员
    subject_player: str = ""  # jersey_label

    # 时间/帧范围
    frame: int = 0            # 关键帧号
    time_second: float = 0.0  # 对应秒数

    # 原始动作
    original_action: str = ""  # "pass"/"shoot"/"dribble"/"clear"/"hold"
    original_target: str = ""  # 原传球目标（如有）

    # 替换动作
    altered_action: str = ""
    altered_target: str = ""   # 替换传球目标（如有）

    # 比赛上下文
    score_diff: int = 0
    time_remaining_frames: int = 0

    # 环境变量调整（ENVIRONMENT_CHANGE 场景）
    environment_params: dict[str, Any] = field(default_factory=dict)

    # 反事实轨迹数据生成（用户要求"生成数据/轨迹/文件"时启用）
    generate_data: bool = False
    # 指定生成时长（秒）；<=0 表示补全到原始数据结束帧
    duration_seconds: float = 0.0

    # 原始用户输入
    raw_query: str = ""

    def describe_original(self) -> str:
        """LLM 友好的原始场景描述。"""
        parts = [f"{self.subject_player}在第{self.time_second:.1f}秒时"]
        if self.original_action == "pass":
            parts.append(f"选择传球给{self.original_target}")
        elif self.original_action == "shoot":
            parts.append("选择了射门")
        elif self.original_action == "dribble":
            parts.append("选择了盘带突破")
        else:
            parts.append(f"执行了{self.original_action}")
        return "".join(parts)

    def describe_altered(self) -> str:
        """LLM 友好的反事实场景描述。"""
        parts = [f"如果{self.subject_player}改为"]
        if self.altered_action == "pass":
            target = self.altered_target or "另一名球员"
            parts.append(f"传球给{target}")
        elif self.altered_action == "shoot":
            parts.append("直接射门")
        elif self.altered_action == "dribble":
            parts.append("选择盘带突破")
        else:
            parts.append(f"执行{self.altered_action}")
        return "".join(parts)


@dataclass
class DecisionDistribution:
    """单个决策选项的概率分布。"""
    action: str = ""
    probability: float = 0.0
    expected_value: float = 0.0  # 预期收益
    description: str = ""        # LLM 生成的描述


@dataclass
class CounterfactualReport:
    """反事实分析报告。"""
    scenario: CounterfactualScenario = field(default_factory=CounterfactualScenario)

    # 决策概率分布（Top-N）
    decision_distribution: list[DecisionDistribution] = field(default_factory=list)

    # MCTS 统计
    total_simulations: int = 0
    simulation_depth_avg: float = 0.0

    # 预期结果
    goal_probability_original: float = 0.0    # 原始决策进球概率
    goal_probability_altered: float = 0.0     # 反事实决策进球概率
    possession_retention_original: float = 0.0
    possession_retention_altered: float = 0.0

    # LLM 生成的分析文本（所有输出由 LLM 完成）
    analysis_summary: str = ""
    # LLM 生成的完整分析报告

    key_factors: list[str] = field(default_factory=list)
    # LLM 识别的关键影响因素

    confidence: float = 0.0
    # 分析置信度

    # ── 反事实轨迹数据生成结果（simulation 子系统） ──
    generated_files: list[str] = field(default_factory=list)
    # 导出的数据文件路径（球员CSV、足球CSV）

    trajectory_events: list[str] = field(default_factory=list)
    # 生成轨迹的关键事件时间线（帧级中文描述）

    outcome: str = ""
    # 模拟结局：goal / save_held / miss / out / completed

    def to_dict(self) -> dict[str, Any]:
        return {
            "场景描述": {
                "原始": self.scenario.describe_original(),
                "反事实": self.scenario.describe_altered(),
            },
            "模拟统计": {
                "总模拟次数": self.total_simulations,
                "平均模拟深度": round(self.simulation_depth_avg, 1),
            },
            "决策概率分布": [
                {
                    "动作": d.action,
                    "概率": f"{d.probability:.1%}",
                    "预期收益": round(d.expected_value, 3),
                    "说明": d.description,
                }
                for d in self.decision_distribution[:5]
            ],
            "结果对比": {
                "原始决策进球概率": f"{self.goal_probability_original:.1%}",
                "反事实决策进球概率": f"{self.goal_probability_altered:.1%}",
                "原始决策控球保持率": f"{self.possession_retention_original:.1%}",
                "反事实决策控球保持率": f"{self.possession_retention_altered:.1%}",
            },
            "关键因素": self.key_factors,
            "分析报告": self.analysis_summary,
            "置信度": round(self.confidence, 3),
            "生成数据文件": self.generated_files,
            "轨迹事件时间线": self.trajectory_events,
            "模拟结局": self.outcome,
        }


# ═══════════════════════════════════════════════════════════════════
# 查询路由
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RoutedQuery:
    """LLM 意图解析后的结构化查询。"""
    intent: IntentType = IntentType.GENERAL_QA
    target_players: list[str] = field(default_factory=list)
    # 涉及球员的 jersey_label 列表

    scenario: CounterfactualScenario | None = None
    # 反事实场景（仅 COUNTERFACTUAL 意图）

    original_text: str = ""
    # 原始用户输入

    confidence: float = 0.0
    # 意图识别置信度

    def to_dict(self) -> dict[str, Any]:
        return {
            "意图": self.intent.value,
            "目标球员": self.target_players,
            "原始输入": self.original_text,
            "识别置信度": round(self.confidence, 3),
        }
