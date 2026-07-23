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
    ) -> str:
        """生成单个球员的风格分析报告。"""
        system_prompt = """## 角色
你是足球战术分析师 (Style Reporter)。职责是基于球员画像和行为模型，生成一份详细的球员风格分析报告。

## 输入
- 球员画像：战术角色、活动区域、速度统计、进攻风格、防守风格
- 行为模型：特征向量、风格标签、动作权重、传球偏好

## 输出
纯文本报告，300-500 字，分段清晰。不要 markdown 格式。

## 报告结构（必需包含以下段落）
1. **整体风格定位**：用 1-2 句概括该球员的风格标签和战术角色
2. **进攻特点**：传球倾向、盘带习惯、射门选择、无球跑动
3. **防守特点**：拦截倾向、防守站位、协防意识、回追速度
4. **战术价值**：该球员在体系中的作用、最佳使用方式

## 写作规则
- 语言专业但不晦涩，像资深球探报告
- 不使用具体数值，用感受性语言描述（"频繁"、"偶尔"、"擅长"、"偏弱"）
- 如果是门将，报告结构调整为：门线技术、出击范围、出球能力、指挥防线"""

        user_msg = self._build_player_data_prompt(profile, model)

        result = call_llm(system_prompt, user_msg, stream=True)
        if result:
            return result.strip()

        # 回退
        return f"【{profile.jersey_label} 风格分析】\n{model.style_description or '基于数据推断的风格模型。'}"

    def generate_comparison_report(
        self,
        profiles: list[PlayerProfile],
        models: list[PlayerBehaviorModel],
    ) -> str:
        """生成球员对比分析报告。"""
        if len(profiles) < 2:
            return "需要至少两名球员才能进行对比分析。"

        system_prompt = """## 角色
你是足球战术分析师 (Comparison Reporter)。职责是对比分析两名或多名球员的风格特点和战术角色差异。

## 输入
多名球员的画像数据（球衣号、风格标签、战术角色、风格描述、主要区域）

## 输出
纯文本报告，200-400 字。不要 markdown 格式。

## 报告结构
1. **各自特点**：分别说明每名球员的独特之处
2. **差异分析**：他们最核心的差异是什么（如一个偏进攻一个偏防守）
3. **互补/冲突**：如果他们在同一体系中，会如何互补或产生冲突
4. **场景建议**：什么战术场景下适合使用哪名球员

## 写作规则
- 语言专业，不使用具体数值
- 公平公正：不要偏袒某一方，客观描述差异
- 如果两人风格相似，重点说明细微差异和各自的适用场景"""

        players_data = []
        for profile, model in zip(profiles, models):
            players_data.append({
                "球衣号": profile.jersey_label,
                "风格标签": model.style_label,
                "战术角色": profile.tactical_role,
                "风格描述": model.style_description,
                "主要区域": profile.dominant_zone,
            })

        user_msg = f"请对比分析以下球员：\n\n{json.dumps(players_data, ensure_ascii=False, indent=2)}"

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
                f"\n\n模拟统计：共{report.total_simulations}次推演，"
                f"置信度{report.confidence:.1%}"
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
纯文本报告，300-500 字，分段清晰。不要 markdown 格式。

## 报告结构（按需组织，不必机械分段）
1. **阵型与站位**：双方阵型结构、整体站位倾向（高位/中位/低位）、阵型紧凑程度
2. **进攻方式与空间利用**：主要进攻通道（边路/中路）、推进方式（短传渗透/长传转移）、主要进攻空间区域
3. **核心球员**：依据参与球路次数和接近球时间占比识别比赛核心，并说明其战术作用
4. **攻防转换**：转换频率、由守转攻的威胁程度
5. **整体评价**：一句话概括该片段的比赛态势

## 写作规则
- 先结论后依据，每个判断都要能回溯到战术事实中的数据
- 可以引用事实中给出的占比和次数，禁止输出坐标数值、禁止编造数据
- 语言专业，像资深球探/战术分析师，不做超出数据的推测"""

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
    ) -> str:
        """构建球员数据的 LLM 提示词。"""
        data = profile.to_dict()
        model_data = model.to_dict()

        return json.dumps(
            {
                "球员": data,
                "行为模型": model_data,
            },
            ensure_ascii=False,
            indent=2,
        )
