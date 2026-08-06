"""多智能体提示词统一管理。

所有 LLM 智能体的系统提示词集中在此，按 agent_name 索引，方便修改。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.professional.simulation.llm_brain import SemanticTierBatcher

# ═══════════════════════════════════════════════════════════════════
# TrajectoryCompositionAgent  — 最终输出解说稿
# ═══════════════════════════════════════════════════════════════════

COMPOSITION_SYSTEM_PROMPT = """你是一位专业的足球比赛解说员。你的风格是：

1. **生动但不夸张**：用自然的语言描述场上发生的事，像真正的直播解说。
2. **严禁输出任何具体数值**：这是最重要的规则。绝对不要在解说中出现任何坐标数字（如"从844到1195"）、距离数字（如"跑动距离487"、"球距56"）、百分比数字（如"60%的活动"、"覆盖3.4%"）、帧序号等。你需要把数值翻译成足球解说员的感受性语言——
   - 坐标范围 → "覆盖了整个右路"、"从中场一直压到禁区"
   - 跑动距离大 → "跑动量惊人"、"覆盖面积非常大"
   - 跑动距离小 → "站位相对固定"、"活动范围紧凑"
   - 百分比高 → "绝大部分时间"、"几乎完全"
   - 百分比低 → "偶尔"、"零星"
   - 冲刺次数多 → "频繁爆发"、"反复冲击"
   - 冲刺次数少 → "偶尔提速"、"以匀速为主"
3. **关注人与球的互动**：谁在控球、谁在跑位、阵型如何变化、球是怎么流转的。
4. **语言多变**：每次表达都要有所不同，不要重复相同的句式。
5. **篇幅自然**：每段解说根据事件信息量自然决定篇幅——事件丰富时充分展开，事件简单时保持精炼，直击要点，不要为凑字数而啰嗦，也不要为求短而丢细节。
6. **纯文本输出**：不要使用 markdown 格式，不要加前缀，直接输出解说文本。
7. **个人表现解读**：如果数据中包含"重点关注球员"，请围绕他们的跑位线路、触球时机、冲刺爆发、活动范围、对比赛节奏的影响来分析。务必把球员的跑动和球的运行结合起来解读。如果只有一个关注球员，深挖他的个人表现；如果有多个，还要分析他们之间的传切配合。
8. **按时间分段输出**：你的解说必须按时间分段，每段以"Xs~Ys："（如"0~3s："）开头标记该段对应的时间区间。
   - 数据中的帧率是30fps，帧号除以30即为秒数（取整）。根据提供的"赛段时间"中的秒数范围来标注每段解说的时间。
   - 开场第一段从0s开始；中间各段根据赛段时间自然衔接；最后一段到数据结束时间。
   - 每段解说覆盖一个完整的技术动作或战术片段，通常为2-5秒。
   - 分段不宜过粗（一次覆盖超过8秒会丢失细节），也不宜过细（每1秒一段会显得零碎）。
   - 当赛段之间无明显事件时，可以合并为较长的过渡段落，描述整体态势。
   - 输出格式示例：
0~3s：开场后B队迅速在中路控球，17号球员从后场启动，开始向对方半场渗透。
3~7s：看这次进攻，B队整体压上，17号在禁区前沿接球后没有贪功，而是果断将球分给边路插上的队友。
7~12s：请注意这次长传配合，17号与队友3号完成了一次中距离联动，随后立刻向球门方向冲刺，反复冲击对方防线肋部。
9. **自行判断球路事件类型**：
   - 球路事件中的「球路」字段描述了球的移动轨迹，例如「6号→7号」表示球从6号附近移动到7号附近，「7号控球移动」表示球随7号移动。
   - 「方向」「速度档位」「距离档位」「球高度」是辅助判断的客观指标，请根据足球常识自行判断每个球路对应的事件类型：
     - 球快速从一名球员飞向另一名球员，距离中等以上 → **传球**（距离长则为长传）
     - 球高速飞向球门方向且距离较长 → **射门**
     - 球随同一名球员低速短距离移动 → **盘带/控球**
     - 球从防守方脚下飞出远离危险区域 → **解围**
     - 高空球起跳争夺 → **争顶**
   - 「球高度」提示：地面球=球贴地运行，低平球=球在半空，高空球=球高高飞起
   - 解说时请根据以上规则，自然地描述事件（如"6号一记长传找到7号"），不要提及速度/距离的具体数值
   - 「未知」标记表示该端球员因跟踪数据缺失或球在空中无法确定，请根据上下文（前一段的终点球员）合理推断"""


# ═══════════════════════════════════════════════════════════════════
# PlayerDecisionAgent  — 比赛态势判断
# ═══════════════════════════════════════════════════════════════════

DECISION_FAST_TEMPO = "本段比赛节奏很快，球员跑动非常积极，攻防转换频繁。"
DECISION_NORMAL_TEMPO = "比赛节奏正常，双方阵型保持得比较有组织。"
DECISION_SLOW_TEMPO = "这个片段整体节奏偏慢，队形变化不大。"


# ═══════════════════════════════════════════════════════════════════
# PlayerComparisonAgent  — 球员对比摘要
# ═══════════════════════════════════════════════════════════════════

COMPARISON_SUMMARY_TEMPLATE = "{active}是本段跑动最积极的球员"
COMPARISON_BALL_ADDON = "，而{ball_player}与球的距离最近、最有可能在控球"


# ═══════════════════════════════════════════════════════════════════
# 辅助：档位标签化（数值 → 语义）
# ═══════════════════════════════════════════════════════════════════

def _tier_from_ranges(value: float, tiers: list[tuple[float, str]]) -> str:
    for threshold, label in tiers:
        if value > threshold:
            return label
    return tiers[-1][1] if tiers else "未知"


def speed_tier(speed: float, batcher: SemanticTierBatcher | None = None) -> str:
    """速度语义描述（LLM 优先，回退阈值）。"""
    if batcher is not None:
        desc = batcher.tier(speed, "speed")
        if desc is not None:
            return desc
    return _tier_from_ranges(speed, [(800, "极高"), (400, "高"), (100, "中"), (0, "低")])


# 距离档位阈值（px，PIXELS_PER_METER ≈ 11）：
#   <165px(15m)  短距离
#   165-330px    中距离
#   >330px(30m)  长距离转移
# 「跨越半场」必须真正跨越中线(y=350)且距离超过 400px(36m)
MID_LINE_Y = 350.0


def distance_tier(
    dist: float,
    y1: float | None = None,
    y2: float | None = None,
    batcher: SemanticTierBatcher | None = None,
) -> str:
    """传球距离语义描述（LLM 优先，回退阈值）。

    Args:
        dist: 移动距离（px）
        y1, y2: 起点/终点 y 坐标（可选；提供时用于「跨越半场」判定）
    """
    if batcher is not None:
        extra = None
        if y1 is not None and y2 is not None:
            extra = {"起点y": round(y1, 0), "终点y": round(y2, 0)}
        desc = batcher.tier(dist, "distance", extra)
        if desc is not None:
            return desc
    if (
        y1 is not None and y2 is not None
        and (y1 - MID_LINE_Y) * (y2 - MID_LINE_Y) < 0
        and dist > 400
    ):
        return "跨越半场的长传"
    return _tier_from_ranges(dist, [(330, "长距离转移"), (165, "中距离"), (0, "短距离")])


def coverage_tier(pct: float, batcher: SemanticTierBatcher | None = None) -> str:
    """场地覆盖率语义描述（LLM 优先，回退阈值）。"""
    if batcher is not None:
        desc = batcher.tier(pct, "coverage")
        if desc is not None:
            return desc
    return _tier_from_ranges(pct, [(8, "极大"), (3, "较大"), (1, "中等"), (0, "较小")])


def activity_tier(pct: float, batcher: SemanticTierBatcher | None = None) -> str:
    """活跃度语义描述（LLM 优先，回退阈值）。"""
    if batcher is not None:
        desc = batcher.tier(pct, "activity")
        if desc is not None:
            return desc
    return _tier_from_ranges(pct, [(70, "极高"), (40, "高"), (15, "中"), (0, "低")])


def sprint_tier(count: int, batcher: SemanticTierBatcher | None = None) -> str:
    """冲刺特征语义描述（LLM 优先，回退阈值）。"""
    if batcher is not None:
        desc = batcher.tier(float(count), "sprint")
        if desc is not None:
            return desc
    return _tier_from_ranges(float(count), [(8, "频繁爆发"), (3, "多次冲刺"), (1, "偶尔提速"), (0, "以匀速为主")])


def ball_height_tier(aerial_pct: float, batcher: SemanticTierBatcher | None = None) -> str:
    """球高度风格语义描述（LLM 优先，回退阈值）。"""
    if batcher is not None:
        desc = batcher.tier(aerial_pct, "ball_height")
        if desc is not None:
            return desc
    return _tier_from_ranges(aerial_pct, [(30, "长传冲吊"), (15, "高低结合"), (0, "地面渗透")])


def motion_trend_tier(approaching_pct: float, batcher: SemanticTierBatcher | None = None) -> str:
    """球员相对球运动趋势语义描述（LLM 优先，回退阈值）。"""
    if batcher is not None:
        desc = batcher.tier(approaching_pct, "motion_trend")
        if desc is not None:
            return desc
    if approaching_pct > 55:
        return "积极靠拢"
    elif approaching_pct > 45:
        return "保持距离"
    elif approaching_pct > 35:
        return "略微拉开"
    else:
        return "远离球路"
