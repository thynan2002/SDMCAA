"""查询路由器。

**全部意图由 LLM 判断**，不硬编码关键词映射。
LLM 不可用时回退到 GENERAL_QA（综合问答兜底）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.llm_client import call_llm
from agents.professional.types import (
    RoutedQuery,
    CounterfactualScenario,
    IntentType,
    ScenarioType,
)


class QueryRouter:
    """自然语言意图路由器（LLM 驱动）。

    所有意图识别由 LLM 完成，
    无硬编码关键词 → 意图映射。
    """

    name = "QueryRouter"

    # ── 系统命令（快速通道，避免 LLM 调用） ──

    SYSTEM_COMMANDS: dict[str, IntentType] = {
        "exit": IntentType.EXIT,
        "quit": IntentType.EXIT,
        "q": IntentType.EXIT,
        "退出": IntentType.EXIT,
        "help": IntentType.HELP,
        "帮助": IntentType.HELP,
        "?": IntentType.HELP,
        "？": IntentType.HELP,
        "h": IntentType.HELP,
        "clear": IntentType.CLEAR,
        "清除": IntentType.CLEAR,
    }

    # ── LLM 系统提示词 ──

    SYSTEM_PROMPT = """## 角色
你是足球分析系统的**意图路由器 (Intent Router)**。职责是从用户的自然语言输入中准确识别其核心意图，并按固定 JSON 格式输出。

## 输入
用户输入的任意中文文本，可能包含球员号、动作描述、比赛时间等。

## 输出
纯 JSON 对象，包含以下字段：
- `intent`: 字符串，从下方 8 个意图中选择一个
- `target_players`: 字符串数组，提取到的球员号（如 ["6号", "7号"]），无则为 []
- `confidence`: 0-1 浮点数，识别确信度
- `scenario`: 仅当 intent 为 `counterfactual` 时需要，否则为 null

## 意图定义（按优先级排列）

### 1. focus_player
- **触发条件**: 用户表达"看/聚焦/关注/盯/分析"某球员，且**不含**反事实假设
- **示例**: "看7号"、"聚焦6号"、"关注10号球员"
- **边界**: 如果含分析类词（风格、特点）→ 归 style_analysis，不是 focus_player

### 2. style_analysis
- **触发条件**: 询问球员的风格/特点/能力/跑动模式/技术/类型/表现
- **示例**: "7号的跑动风格是什么"、"分析3号的特点"、"6号是什么类型的球员"
- **也包括**: 询问某支球队整体风格或队内风格之最（"A队谁的风格最激进"、"B队谁跑动最积极"），此时 target_players 可为空（由系统展开为该队球员）
- **边界**: 如果同时包含"如果/假如"→ 归 counterfactual；纯问球队整体战术（进攻方式/阵型/防守漏洞）→ 归 general_analysis

### 3. counterfactual
- **触发条件**: 反事实假设，含"如果/假如/假设/换成/改成/要是"等假设词
- **示例**: "如果7号不传给11号而是直接打门会怎样"、"假如6号选择突破而不是回传"
- **输出要求**: 必须填充 scenario 字段，包括 `scenario_type`, `subject_player`, `frame`, `time_second`, `original_action`, `original_target`, `altered_action`, `altered_target`
- **可选字段**:
  - `generate_data`: 布尔值。当用户要求"生成/导出/输出...数据/轨迹/文件/CSV"时设为 true，否则 false
  - `duration_seconds`: 数值。当用户指定"生成后面/未来X秒"时为 X，未指定时为 0（表示补全到原始数据结束）

### 4. comparison
- **触发条件**: 对比两名或多名球员
- **示例**: "对比7号和10号"、"6号和9号谁更积极"
- **特征词**: 对比/比较/谁更/哪名球员...更/区别/差异/不同

### 5. general_analysis
- **触发条件**: 整体比赛/球队级战术分析请求，不聚焦单一具体球员
- **示例**: "整体分析一下这场比赛"、"比赛节奏如何"、"攻防转换怎样"
- **也包括**: 球队进攻/防守态势（"当前球队进攻方式是什么"、"球队防守是否存在漏洞"）、空间利用（"哪个区域是主要进攻空间"、"阵型如何变化"）、球队核心（"谁是本场比赛核心球员"）
- **边界**: 查询具体事实（某时刻发生了什么、谁传给谁）→ 归 general_qa

### 6. verify_data
- **触发条件**: 数据核验/质疑/回查，包括：
  - 明确提到帧号或秒数（"第90帧"、"第3秒"）
  - 质疑或纠正之前的回答（"不对"、"不是这样"、"你错了"、"明明是..."）
  - 要求重新确认某球员信息（"再确认一下1号"）
- **边界**: 单纯问"1号是哪个队的"不包含质疑意图 → 归 general_qa

### 7. general_qa
- **触发条件**: 所有不匹配上述 1-6 的比赛相关问题
- **兜底规则**: 当不确定时，默认归为此类

### 8. 系统命令
- help/exit/clear — 由上层代码处理，不会到达此提示词

## 重要规则
1. **verify_data 优先级高于 general_qa**: 只要含"再/又/确认/核验/不对/纠正"等词，优先归 verify_data
2. **边界模糊时归 general_qa**: 如果无法确定意图，设为 general_qa，不要强行匹配
3. **confidence < 0.5 是允许的**: 低确信度比错误匹配更好

## 输出示例

### 示例 1: focus_player
输入: "聚焦7号"
输出: {"intent": "focus_player", "target_players": ["7号"], "confidence": 0.95, "scenario": null}

### 示例 2: counterfactual
输入: "如果6号不传给15号而是自己射门会怎样"
输出: {"intent": "counterfactual", "target_players": ["6号", "15号"], "confidence": 0.9, "scenario": {"scenario_type": "self_choice_change", "subject_player": "6号", "frame": 150, "time_second": 5.0, "original_action": "pass", "original_target": "15号", "altered_action": "shoot", "altered_target": "", "generate_data": false, "duration_seconds": 0}}

### 示例 2b: counterfactual + 生成数据
输入: "如果6号第3秒直接射门，生成后面5秒的轨迹数据"
输出: {"intent": "counterfactual", "target_players": ["6号"], "confidence": 0.9, "scenario": {"scenario_type": "self_choice_change", "subject_player": "6号", "frame": 90, "time_second": 3.0, "original_action": "", "original_target": "", "altered_action": "shoot", "altered_target": "", "generate_data": true, "duration_seconds": 5}}

### 示例 3: general_qa
输入: "13号是谁"
输出: {"intent": "general_qa", "target_players": ["13号"], "confidence": 0.7, "scenario": null}

### 示例 4: verify_data
输入: "不对吧，7号明明在左路"
输出: {"intent": "verify_data", "target_players": ["7号"], "confidence": 0.85, "scenario": null}"""

    # ── 公开接口 ──

    def parse(self, user_text: str) -> RoutedQuery:
        """LLM 驱动的意图解析。

        路由优先级：
        1. 系统命令（快速通道，避免 LLM 调用）
        2. LLM 解析（第一意图来源）
        3. 关键词规则兜底（LLM 不可用时）
        4. 回退 → GENERAL_QA（永不返回 UNKNOWN）
        """
        text = user_text.strip()

        # ── 快速通道：系统命令 ──
        lower = text.lower()
        if lower in self.SYSTEM_COMMANDS:
            return RoutedQuery(
                intent=self.SYSTEM_COMMANDS[lower],
                original_text=user_text,
                confidence=1.0,
            )

        # ── LLM 解析 ──
        try:
            result = self._parse_with_llm(user_text)
            if result:
                return result
        except Exception:
            pass

        # ── 关键词规则兜底（LLM 不可用或解析失败时） ──
        fallback = self._rule_based_parse(text)
        if fallback:
            return fallback

        # ── 回退：一切未识别走综合问答 ──
        return RoutedQuery(
            intent=IntentType.GENERAL_QA,
            original_text=user_text,
            confidence=0.3,
        )

    # ── LLM 解析 ──

    def _parse_with_llm(self, user_text: str) -> RoutedQuery | None:
        """调用 LLM 解析意图。"""
        llm_result = call_llm(self.SYSTEM_PROMPT, user_text)
        if not llm_result:
            return None

        # 清理 markdown 代码块
        cleaned = llm_result.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        return self._build_query_from_json(data, user_text)

    def _build_query_from_json(
        self,
        data: dict[str, Any],
        original_text: str,
    ) -> RoutedQuery:
        """将 LLM JSON 转为 RoutedQuery。"""
        # 归一化意图字符串：strip + lower，兼容 "General_QA" / " Style_Analysis " 等
        intent_str = str(data.get("intent", "general_qa")).strip().lower()
        try:
            intent = IntentType(intent_str)
        except ValueError:
            intent = IntentType.GENERAL_QA  # 未识别的走兜底

        # target_players 去重，保持顺序
        raw_players = [
            self._normalize_jersey(p)
            for p in data.get("target_players", [])
        ]
        target_players = list(dict.fromkeys(raw_players))

        scenario = None
        if intent == IntentType.COUNTERFACTUAL and "scenario" in data:
            sc = data["scenario"]
            try:
                scenario_type = ScenarioType(sc.get("scenario_type", "self_choice_change"))
            except ValueError:
                scenario_type = ScenarioType.SELF_CHOICE_CHANGE

            # 强制类型转换，避免 LLM 返回字符串导致下游格式化失败
            try:
                frame = int(sc.get("frame", 0))
            except (TypeError, ValueError):
                frame = 0
            try:
                time_second = float(sc.get("time_second", 0.0))
            except (TypeError, ValueError):
                time_second = 0.0

            # 生成数据标志与时长（类型保护）
            generate_data = bool(sc.get("generate_data", False))
            try:
                duration_seconds = float(sc.get("duration_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                duration_seconds = 0.0

            scenario = CounterfactualScenario(
                scenario_type=scenario_type,
                subject_player=self._normalize_jersey(sc.get("subject_player", "")),
                frame=frame,
                time_second=time_second,
                original_action=str(sc.get("original_action", "")),
                original_target=self._normalize_jersey(sc.get("original_target", "")),
                altered_action=str(sc.get("altered_action", "")),
                altered_target=self._normalize_jersey(sc.get("altered_target", "")),
                generate_data=generate_data,
                duration_seconds=duration_seconds,
                raw_query=original_text,
            )

        # 置信度也做类型保护
        try:
            confidence = float(data.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7

        return RoutedQuery(
            intent=intent,
            target_players=target_players,
            scenario=scenario,
            original_text=original_text,
            confidence=confidence,
        )

    def _rule_based_parse(self, text: str) -> RoutedQuery | None:
        """关键词规则兜底（LLM 不可用时）。"""
        import re

        # 球员号提取
        jerseys = list(dict.fromkeys(
            f"{m}号" for m in re.findall(r"(\d+)\s*号", text)
        ))

        # --- counterfactual ---
        if re.search(r"如果|假如|假设|要是|换成|改成", text):
            subject = jerseys[0] if jerseys else ""

            # 尝试从文本提取基本 scenario 信息
            scenario = None
            original_action = ""
            original_target = ""
            altered_action = ""
            altered_target = ""
            frame = 0
            time_second = 0.0

            m_orig = re.search(r"不传给\s*(\d+)\s*号", text)
            if m_orig:
                original_action = "pass"
                original_target = f"{m_orig.group(1)}号"
            elif re.search(r"不传|没有传", text):
                original_action = "pass"

            m_alt = re.search(r"而是传给\s*(\d+)\s*号", text)
            if m_alt:
                altered_action = "pass"
                altered_target = f"{m_alt.group(1)}号"
            elif re.search(r"而是.*射门|而是.*打门|自己射门|选择射门", text):
                altered_action = "shoot"
            elif re.search(r"而是.*突破|而是.*盘带|选择突破|选择盘带", text):
                altered_action = "dribble"
            elif re.search(r"而是传给|而是回传|而是横传", text):
                altered_action = "pass"
            # 兜底：如果文本中明确提到了"射门/打门"但不在"而是"从句中
            if not altered_action:
                if re.search(r"直接\s*射门|直接\s*打门|起脚\s*射门", text):
                    altered_action = "shoot"
                elif re.search(r"直接\s*突破|带球\s*突破", text):
                    altered_action = "dribble"

            m_time = re.search(r"第\s*(\d+(?:\.\d+)?)\s*秒", text)
            if m_time:
                time_second = float(m_time.group(1))
                frame = int(time_second * 30)

            # 生成数据文件检测（"生成/导出/输出" + "数据/轨迹/文件/CSV"）
            generate_data = bool(
                re.search(r"生成|导出|输出", text)
                and re.search(r"数据|轨迹|文件|csv", text, re.IGNORECASE)
            )
            duration_seconds = 0.0
            m_dur = re.search(r"(?:后面|未来|之后|接下来)\s*(\d+(?:\.\d+)?)\s*秒", text)
            if m_dur:
                duration_seconds = float(m_dur.group(1))

            # 只要提取到原始动作或替换动作之一就构造 scenario
            if original_action or altered_action:
                scenario = CounterfactualScenario(
                    scenario_type=ScenarioType.SELF_CHOICE_CHANGE,
                    subject_player=subject,
                    frame=frame,
                    time_second=time_second,
                    original_action=original_action,
                    original_target=original_target,
                    altered_action=altered_action,
                    altered_target=altered_target,
                    generate_data=generate_data,
                    duration_seconds=duration_seconds,
                    raw_query=text,
                )

            return RoutedQuery(
                intent=IntentType.COUNTERFACTUAL,
                target_players=jerseys,
                scenario=scenario,
                original_text=text,
                confidence=0.4,
            )

        # --- comparison ---
        if re.search(r"对比|比较|谁更|哪.*更|区别|差异", text) and len(jerseys) >= 2:
            return RoutedQuery(
                intent=IntentType.COMPARISON,
                target_players=jerseys,
                original_text=text,
                confidence=0.5,
            )

        # --- verify_data ---
        if re.search(r"不对|不是这样|不是吧|你搞错|搞错了|错了|你错|明明|再确认|重新查|核实|第\d+帧|第\d+秒", text):
            return RoutedQuery(
                intent=IntentType.VERIFY_DATA,
                target_players=jerseys,
                original_text=text,
                confidence=0.5,
            )

        # --- style_analysis ---
        if re.search(r"风格|特点|跑动|技术|类型|能力|擅长|弱项|表现", text) and (
            jerseys or re.search(r"[ABab]\s*队", text)
        ):
            return RoutedQuery(
                intent=IntentType.STYLE_ANALYSIS,
                target_players=jerseys,
                original_text=text,
                confidence=0.4,
            )

        # --- focus_player ---
        if re.search(r"^(看|聚焦|关注|盯)\s*\d+", text) or (
            re.search(r"看|聚焦|关注", text) and jerseys
        ):
            return RoutedQuery(
                intent=IntentType.FOCUS_PLAYER,
                target_players=jerseys,
                original_text=text,
                confidence=0.4,
            )

        # --- general_analysis ---
        if re.search(r"整体|全场|比赛.*分析|节奏|攻防|阵型|战术|进攻方式|进攻空间|空间利用|漏洞|核心球员|转换", text) and not jerseys:
            return RoutedQuery(
                intent=IntentType.GENERAL_ANALYSIS,
                target_players=jerseys,
                original_text=text,
                confidence=0.4,
            )

        return None

    @staticmethod
    def _normalize_jersey(raw: str) -> str:
        """标准化球衣号。"""
        match = re.search(r"(\d+)", str(raw))
        if match:
            return f"{match.group(1)}号"
        return str(raw).strip()
