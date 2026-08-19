"""查询路由器。

关键词规则优先（确定性、零 LLM 调用），规则未覆盖时 LLM 兜底
（submit_intent 工具契约），LLM 不可用时回退到 GENERAL_QA。
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
    """自然语言意图路由器（规则优先 + LLM 兜底）。

    常见意图由关键词规则确定性判定（零 LLM 调用），
    规则未覆盖时经 LLM submit_intent 工具契约判定。
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

    def parse(self, user_text: str, available_players: list[str] | None = None) -> RoutedQuery:
        """意图解析（精简后：关键词规则优先，LLM 兜底）。

        路由优先级：
        1. 系统命令（快速通道，避免 LLM 调用）
        2. 关键词规则（确定性、零 LLM 调用，覆盖常见意图）
        3. LLM 解析（规则未覆盖时，经 submit_intent 工具契约）
        4. 回退 → GENERAL_QA（永不返回 UNKNOWN）

        available_players: 当前数据中的球衣号列表（注入路由 prompt，供
        target_players 合法性判断，降低不存在球员的误路由）。
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

        # ── 关键词规则优先 ──
        ruled = self._rule_based_parse(text)
        if ruled:
            return ruled

        # ── LLM 解析（规则未覆盖时兜底） ──
        try:
            result = self._parse_with_llm(user_text, available_players)
            if result:
                return result
        except Exception:
            pass

        # ── 回退：一切未识别走综合问答 ──
        # 特例：同时含反事实假设词与球员号（如"7号"）且规则与 LLM 均未
        # 解析出场景 → 保持 COUNTERFACTUAL 意图（scenario=None），由 manager
        # 输出场景澄清话术；仅含假设词的普通问题（如"如果下雨呢"）不再
        # 拦截，避免 LLM 不可用时被误路由为反事实。
        if re.search(r"如果|假如|假设|要是|换成|改成", text) and re.search(r"\d+号", text):
            return RoutedQuery(
                intent=IntentType.COUNTERFACTUAL,
                original_text=user_text,
                confidence=0.3,
            )
        return RoutedQuery(
            intent=IntentType.GENERAL_QA,
            original_text=user_text,
            confidence=0.3,
        )

    # ── LLM 解析 ──

    def _parse_with_llm(self, user_text: str, available_players: list[str] | None = None) -> RoutedQuery | None:
        """调用 LLM 解析意图。"""
        message = user_text
        if available_players:
            message = (
                f"{user_text}\n\n[数据中存在的球衣号] {', '.join(available_players)}。"
                "target_players 只能从上述球衣号中选取；若问题提及的号码不在其中，"
                "请降低 confidence 并优先判断为 general_qa（由下游如实声明数据不足）。"
            )
        llm_result = call_llm(self.SYSTEM_PROMPT, message, temperature=0.2, max_tokens=4096)
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

    # ── 关键词规则 ↔ LLM 提示词意图映射表（单一事实源说明，防双轨漂移） ──
    # 下方规则分支与 SYSTEM_PROMPT 的"意图定义"章节是同一意图分类法的
    # 双轨编码（规则轨：确定性快速通道；提示词轨：LLM 兜底）。
    # 修改任一侧时必须同步另一侧，对照关系如下：
    #
    #   规则分支（_rule_based_parse 执行顺序）        SYSTEM_PROMPT 意图定义
    #   ────────────────────────────────────  ────────────────────────────
    #   SYSTEM_COMMANDS（快速通道，先于规则）    ### 8. 系统命令（help/exit/clear）
    #   counterfactual（假设词命中）            ### 3. counterfactual（如果/假如/假设/要是）
    #   comparison（对比词 + ≥2 球员）          ### 4. comparison（对比/谁更/区别）
    #   verify_data（质疑词/帧号秒数）          ### 6. verify_data（核验/质疑/回查）
    #   style_analysis（风格词 + 球员或队）     ### 2. style_analysis（风格/特点/类型）
    #   focus_player（看/聚焦/关注 + 球员）     ### 1. focus_player（不含分析类词）
    #   general_analysis（球队级词 + 无球员）   ### 5. general_analysis（不聚焦单一球员）
    #   纯事实疑问 → GENERAL_QA（直返，省一次 LLM） ### 7. general_qa（兜底）
    #   返回 None → LLM 解析 → 仍失败则兜底       ### 7. general_qa（不确定时默认）
    #
    # 注意：规则分支的执行顺序 ≠ 提示词的编号优先级，这是有意差异——
    # 规则按信号强度从强到弱匹配（假设词/质疑词优先），与提示词编号无关。
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

            # 兜底："改为/换成/改成…"句式（如"如果7号在第3秒改为传球给10号会怎样"）
            # 旧版只覆盖"不传…而是…"与"直接射门/突破"，该句式提取不到任何动作，
            # 导致 scenario=None 被 manager 退回澄清话术。排除"不传给X号"的误命中。
            if not altered_action and re.search(r"改为|改成|换成|变成", text):
                m_change_pass = re.search(r"(?<!不)传(?:球)?给\s*(\d+)\s*号", text)
                if m_change_pass:
                    altered_action = "pass"
                    altered_target = f"{m_change_pass.group(1)}号"
                elif re.search(r"传球|分球|转移", text):
                    altered_action = "pass"
                elif re.search(r"射门|打门", text):
                    altered_action = "shoot"
                elif re.search(r"突破|盘带|带球", text):
                    altered_action = "dribble"
                elif re.search(r"解围", text):
                    altered_action = "clear"
                elif re.search(r"控球|护球", text):
                    altered_action = "hold"

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

            # 提取到原始动作或替换动作之一 → 构造 scenario 直接返回；
            # 两者皆空 → 规则无法解析场景，返回 None 交由 LLM 兜底解析
            #（旧版此处返回 scenario=None 的 COUNTERFACTUAL，manager 只能退回
            # 澄清话术，LLM 兜底永远不会被触发 —— 反事实路由失效的结构性根因）
            if not (original_action or altered_action):
                return None

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
        if re.search(r"整体|全场|比赛.*分析|节奏|攻防|阵型|战术|进攻方式|进攻空间|漏洞|核心球员|转换", text) and not jerseys:
            return RoutedQuery(
                intent=IntentType.GENERAL_ANALYSIS,
                target_players=jerseys,
                original_text=text,
                confidence=0.4,
            )

        # --- 纯事实类疑问 → 直达 general_qa（省一次 LLM 兜底调用） ---
        if re.search(r"多少|几.*个|是什么|是谁|哪个|哪支|什么.*队|怎么|怎样|如何|是否|有没有", text) or text.endswith("?") or text.endswith("？"):
            return RoutedQuery(
                intent=IntentType.GENERAL_QA,
                target_players=jerseys,
                original_text=text,
                confidence=0.35,
            )

        return None

    @staticmethod
    def _normalize_jersey(raw: str) -> str:
        """标准化球衣号。"""
        match = re.search(r"(\d+)", str(raw))
        if match:
            return f"{match.group(1)}号"
        return str(raw).strip()


# ── 工具契约绑定：意图路由改为 function calling 交付结构化结果 ──
# 模型经 submit_intent 工具调用提交解析结果；工具参数即 JSON 契约，
# 与原有文本 JSON 契约字段一致 → _build_query_from_json 原样复用。
# 模型直接返回文本（mock/旧 golden/模型行为差异）时，既有文本解析
# 路径照常工作；LLM 失败时关键词规则兜底不变。
# tool_choice 使用 auto（DeepSeek 推理模式不接受 required 强制）；
# 传输层对 tool_choice 类 400 错误会自动降级重试。
from agents.llm_client import bind_prompt_tools  # noqa: E402
from agents.tools.schemas import build_intent_tool  # noqa: E402

bind_prompt_tools(
    QueryRouter.SYSTEM_PROMPT,
    [build_intent_tool()],
    tool_choice="auto",
)
