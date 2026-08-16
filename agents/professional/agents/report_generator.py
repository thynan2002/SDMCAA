"""报告生成智能体。

将所有上游分析结果通过 LLM 整合为自然语言分析报告。
所有输出内容由 LLM 生成，不硬编码任何文案。
"""

from __future__ import annotations

import json
from typing import Any

from agents.llm_client import call_llm
from ..types import (
    PlayerProfile,
    PlayerBehaviorModel,
    CounterfactualReport,
    CounterfactualScenario,
)


class ReportGeneratorAgent:
    """分析报告生成智能体。

    接收所有上游分析结果，通过 LLM 生成自然语言报告。
    支持四种输出模式：
    - 球员风格分析报告
    - 球员对比分析报告
    - 反事实分析报告
    - 综合分析报告
    """

    name = "ReportGeneratorAgent"

    def generate_style_report(
        self,
        profile: PlayerProfile,
        model: PlayerBehaviorModel,
        question: str = "",
    ) -> str:
        """生成单个球员的风格分析报告。"""
        system_prompt = """## 角色
你是足球战术分析师 (Style Reporter)。职责是基于球员画像和行为模型，生成一份详细的球员风格分析报告。

## 输入
- 球员画像：战术角色、活动区域、速度统计、进攻风格、防守风格
- 行为模型：特征向量、风格标签、动作权重、传球偏好
- 用户问题（可选）：报告应侧重回答该问题

## 输出
纯文本报告，分段清晰，不要 markdown 格式。篇幅不设硬性上限：内容需要充分展开时不要吝啬篇幅，内容简单时保持精炼，根据事实的丰富程度自然决定长短，不要为凑字数而重复或为求短而丢失信息。

## 报告结构（必需包含以下段落）
1. **整体风格定位**：用 1-2 句概括该球员的风格标签和战术角色
2. **进攻特点**：传球倾向、盘带习惯、射门选择、无球跑动
3. **防守特点**：拦截倾向、防守站位、协防意识、回追速度
4. **战术价值**：该球员在体系中的作用、最佳使用方式

## 写作规则
0. **数值与区域输出规则（最高优先级）**：默认严禁输出任何具体数值（次数、占比、百分比、评分等），先理解数据再翻译为感受性语言（"频繁"、"偶尔"、"擅长"、"偏弱"、"绝大多数时间"、"几乎不"）。但当用户明确索要具体数据/数值/主要活动区域/战术区域时，必须如实引用画像中的精确数值或区域标签作答（如主要活动区域、速度、跑动距离等），不得含糊、不得拒绝、不得换成其他数值。
1. 语言专业但不晦涩，像资深球探报告
2. 如果是门将，报告结构调整为：门线技术、出击范围、出球能力、指挥防线"""

        user_msg = self._build_player_data_prompt(profile, model, question)

        result = call_llm(system_prompt, user_msg, stream=True)
        if result:
            return result.strip()

        # 回退
        return f"【{profile.jersey_label} 风格分析】\n{model.style_description or '基于数据推断的风格模型。'}"

    def generate_comparison_report(
        self,
        profiles: list[PlayerProfile],
        models: list[PlayerBehaviorModel],
        question: str = "",
    ) -> str:
        """生成球员对比分析报告。"""
        if len(profiles) < 2:
            return "需要至少两名球员才能进行对比分析。"

        system_prompt = """## 角色
你是足球战术分析师 (Comparison Reporter)。职责是对比分析两名或多名球员的风格特点和战术角色差异。

## 输入
多名球员的画像数据（球衣号、风格标签、战术角色、风格描述、主要区域、跑动距离(米)、速度统计、场地覆盖率、接近球时间占比）
用户问题（可选）：报告应侧重回答该问题

## 输出
纯文本报告，不要 markdown 格式。篇幅不设硬性上限：根据对比内容的丰富程度自然展开，该详则详、该简则简。

## 报告结构
1. **各自特点**：分别说明每名球员的独特之处
2. **差异分析**：他们最核心的差异是什么（如一个偏进攻一个偏防守）
3. **互补/冲突**：如果他们在同一体系中，会如何互补或产生冲突
4. **场景建议**：什么战术场景下适合使用哪名球员
5. **明确结论**：若用户问的是比较/胜负类问题（如"谁更积极""谁跑得最多"），必须以「跑动距离(米)」为第一依据、"速度统计"与"场地覆盖率"、"接近球时间占比"为辅助，逐项对比后点名哪名球员更优，并引用具体数据作为依据。**严禁以"画像相同/标签相同"为由拒绝判定**——风格标签/描述相同是正常现象，必须基于硬数据（跑动距离等）给出结论。

## 写作规则
- 语言专业，先理解数据再下判断，用自然语言表达差异（"明显更积极"、"略偏防守"、"几乎相同"）
- **数值输出规则（最高优先级）**：默认严禁输出任何具体数值（次数、占比、百分比、评分等），先理解数据再翻译为感受性语言（"频繁"、"偶尔"、"擅长"、"偏弱"、"绝大多数时间"、"几乎不"）。**但当用户明确索要具体数据/数值/跑动距离/速度时，必须如实引用画像中的精确数值作答**。**当用户问的是比较/胜负类问题（如"谁更积极"）时，必须引用跑动距离(米)、速度统计、场地覆盖率中的硬数据作为判断依据，并给出明确结论。**
- 公平公正：不要偏袒某一方，客观描述差异
- 如果两人风格相似，重点说明细微差异和各自的适用场景
- 风格标签/描述相同时，跑动距离(米)、速度统计（平均速度、峰值速度、冲刺次数）、场地覆盖率、接近球时间占比是区分两者的依据，必须仔细对比这些数据后给出结论"""

        players_data = []
        for profile, model in zip(profiles, models):
            players_data.append({
                "球员画像": profile.to_dict(),
                "行为模型": model.to_dict(),
            })

        user_msg = f"请对比分析以下球员：\n\n{json.dumps(players_data, ensure_ascii=False, indent=2)}"
        if question:
            user_msg = f"用户问题：{question}\n\n{user_msg}"

        result = call_llm(system_prompt, user_msg, stream=True)
        if result:
            return result.strip()

        names = "和".join(p.jersey_label for p in profiles)
        return f"【{names}对比分析】\n两名球员展现了不同的战术特点。"

    def generate_counterfactual_report(
        self,
        report: CounterfactualReport,
    ) -> str:
        """生成反事实分析报告。

        LLM 生成已经在 CounterfactualEngineAgent 中完成，
        此处做格式化和补充。
        """
        if report.analysis_summary:
            header = f"【反事实分析】{report.scenario.describe_altered()}"
            stats = (
                "\n\n模拟统计：" + _simulation_stats_note(
                    report.total_simulations, report.confidence,
                )
            )
            return header + "\n\n" + report.analysis_summary + stats
        return "无法生成反事实分析报告：数据不足。"

    def generate_team_report(
        self,
        tactical_facts: dict[str, Any],
        question: str = "",
        memory_context: str = "",
    ) -> str:
        """生成球队级战术分析报告（阵型/进攻方式/空间利用/攻防转换/核心球员）。"""
        system_prompt = """## 角色
你是资深足球战术分析师 (Team Tactics Reporter)。职责是基于球队级战术事实数据，生成一份专业的整体战术分析报告。

## 输入
- 战术事实：球队阵型与站位、进攻态势（推进方向、进攻空间、传球特征）、球员参与度与跑动、攻防转换线索
- 用户问题（可选）：用户原始提问，报告应侧重回答该问题
- 对话上下文（可选）

## 输出
纯文本报告，分段清晰，不要 markdown 格式。篇幅不设硬性上限：根据事实的丰富程度自然展开，信息多则充分论述，信息少则简洁收束，不要为凑字数而堆砌。

## 报告结构（按需组织，不必机械分段）
1. **阵型与站位**：双方阵型结构、整体站位倾向（高位/中位/低位）、阵型紧凑程度
2. **进攻方式与空间利用**：主要进攻通道（边路/中路）、推进方式（短传渗透/长传转移）、主要进攻空间区域
3. **核心球员**：依据参与球路次数和接近球时间占比识别比赛核心，并说明其战术作用
4. **攻防转换**：转换频率、由守转攻的威胁程度
5. **整体评价**：一句话概括该片段的比赛态势

## 写作规则
0. **数值输出规则（最高优先级，无条件遵守）**：默认严禁输出任何具体数值——次数、占比、百分比、帧号、人数统计等一律禁止。你必须先理解数据，再把数值翻译为战术判断和感受性语言：
   - 错误示范："44次球路中向前推进仅占18%，向后回传高达61%，短传占比89%"
   - 正确示范："球路绝大多数都在向后回传，向前推进很少，且几乎全是短传渗透，缺少快速冲击"
   - 其他翻译示例：绝大多数/几乎每次/极少/偶尔/明显更活跃/节奏偏慢/覆盖紧凑/近乎全部/零星
   - **唯一例外**：用户问题中明确索要具体数据（如"具体是几次""数据是多少""占比多少"）时，才直接引用事实中给出的数值
1. 先结论后依据，每个判断都要能回溯到战术事实中的数据
2. 禁止输出坐标数值、禁止编造数据
3. 语言专业，像资深球探/战术分析师，不做超出数据的推测"""

        payload: dict[str, Any] = {"战术事实": tactical_facts}
        if question:
            payload["用户问题"] = question
        if memory_context and memory_context != "（尚未建立对话上下文）":
            payload["对话上下文"] = memory_context

        # 球队报告内容较长，统一使用全局 max_tokens（81920）避免截断
        result = call_llm(
            system_prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            stream=True,
        )
        if result:
            return result.strip()
        return "球队战术分析暂时无法生成：分析服务不可用。"

    def generate_comprehensive_report(
        self,
        profiles: list[PlayerProfile],
        models: list[PlayerBehaviorModel],
        counterfactual: CounterfactualReport | None = None,
    ) -> str:
        """生成综合分析报告。"""
        parts: list[str] = []

        # 风格分析
        for profile, model in zip(profiles, models):
            parts.append(self.generate_style_report(profile, model))

        # 对比（多球员时）
        if len(profiles) >= 2:
            parts.append(self.generate_comparison_report(profiles, models))

        # 反事实分析
        if counterfactual:
            parts.append(self.generate_counterfactual_report(counterfactual))

        separator = "\n\n" + "=" * 50 + "\n\n"
        return separator.join(parts)

    def _build_player_data_prompt(
        self,
        profile: PlayerProfile,
        model: PlayerBehaviorModel,
        question: str = "",
    ) -> str:
        """构建球员数据的 LLM 提示词。"""
        data = profile.to_dict()
        model_data = model.to_dict()

        payload: dict[str, Any] = {
            "球员": data,
            "行为模型": model_data,
        }
        if question:
            payload["用户问题"] = question
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _simulation_stats_note(total_simulations: int, confidence: float) -> str:
    """把模拟统计数值翻译为自然语言描述（默认不向用户暴露原始数字）。"""
    if total_simulations >= 1000:
        scale = "推演规模充分"
    elif total_simulations >= 300:
        scale = "推演规模适中"
    else:
        scale = "推演规模有限"
    if confidence >= 0.8:
        trust = "结果可信度较高"
    elif confidence >= 0.6:
        trust = "结果可信度中等"
    else:
        trust = "结果可信度偏低"
    return f"{scale}，{trust}，结果仅供参考"
