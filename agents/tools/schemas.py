"""结构化决策工具 schema（pydantic 定义 → OpenAI JSON Schema）。

全部为「终止型工具」（terminal=True）：模型通过发起工具调用直接交付
最终结构化结果，工具循环立即结束。调用方现有 JSON 解析路径（类型保护、
归一化、规则兜底）原样复用 —— 工具契约只改变「模型如何输出」，不改变
「系统如何解析与降级」。

校验策略：字段保持宽容（str/list/float，不设 Literal 枚举）——
非法值由下游既有的容忍式解析与规则兜底处理，避免模型一次输出越界
就触发额外重试轮（节省调用、保持等价性）。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.tools.base import (
    TOOL_FAILURE_INVALID_ARGS,
    ToolExecutionError,
    ToolSpec,
)


class _Base(BaseModel):
    """宽松模式：忽略模型多输出的字段，容忍类型强制。"""

    model_config = ConfigDict(extra="ignore")


# 意图枚举（与 agents/professional/types.IntentType 一致；写进 schema 供
# 模型参考，解析端仍走 _build_query_from_json 的容忍归一化 → 非法值兜底）
_INTENT_VALUES = [
    "focus_player", "style_analysis", "counterfactual", "comparison",
    "general_analysis", "verify_data", "general_qa", "help", "exit", "clear",
]

_ACTION_VALUES = ["pass", "shoot", "dribble", "hold", "clear"]
_DIRECTION_VALUES = ["forward", "left", "right"]
_PRESS_VALUES = ["high", "mid", "low"]
_FOCUS_VALUES = ["left", "center", "right"]
_TEMPO_VALUES = ["fast", "normal"]


def _with_enum(schema: dict[str, Any], field: str, values: list[str]) -> dict[str, Any]:
    """向 JSON Schema 注入字段枚举（仅作模型输出引导，校验仍宽松）。"""
    schema["properties"][field]["enum"] = values
    return schema


# ── 意图路由（session/router.py） ──────────────────────────────

class IntentScenario(_Base):
    """反事实场景（intent=counterfactual 时必填）。"""

    scenario_type: str = Field(default="self_choice_change", description="场景类型")
    subject_player: str = Field(default="", description="主体球员，如「6号」")
    frame: int = Field(default=0, description="关键帧号")
    time_second: float = Field(default=0.0, description="对应秒数")
    original_action: str = Field(default="", description="原始动作 pass/shoot/dribble/clear/hold")
    original_target: str = Field(default="", description="原传球目标球衣号")
    altered_action: str = Field(default="", description="替换动作 pass/shoot/dribble/clear/hold")
    altered_target: str = Field(default="", description="替换传球目标球衣号")
    generate_data: bool = Field(default=False, description="是否要求生成/导出轨迹数据文件")
    duration_seconds: float = Field(default=0.0, description="生成时长（秒），0=补全到数据结束")


class IntentParse(_Base):
    """意图路由的结构化输出。"""

    intent: str = Field(default="general_qa", description="意图（见提示词中的 8 类定义）")
    target_players: list[str] = Field(default_factory=list, description="提取到的球员号列表")
    confidence: float = Field(default=0.7, description="识别确信度 0-1")
    scenario: Optional[IntentScenario] = Field(default=None, description="仅 counterfactual 时填写")


def build_intent_tool() -> ToolSpec:
    """submit_intent：模型交付意图解析结果。"""

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        return _validated(IntentParse, args)

    return ToolSpec(
        name="submit_intent",
        description=(
            "提交用户输入的自然语言意图解析结果。你只能选择提示词中定义的意图之一；"
            "当用户输入包含「如果/假如/假设/换成/改成/要是」等假设词时，intent 必须为 "
            "counterfactual 并填写 scenario（含 subject_player、frame、time_second、"
            "original_action、original_target、altered_action、altered_target）。"
        ),
        parameters=_with_enum(
            IntentParse.model_json_schema(), "intent", _INTENT_VALUES,
        ),
        handler=handler,
        terminal=True,
    )


# ── 战术决策（llm_brain.decide） ───────────────────────────────

class TacticalDecision(_Base):
    """持球者的下一步动作决策。"""

    action: str = Field(default="hold", description="pass | shoot | dribble | hold | clear")
    target: str = Field(default="", description="传球目标球衣号（action=pass 时必填，只能来自队友列表）")
    direction: str = Field(default="forward", description="盘带方向 forward | left | right")
    press: str = Field(default="mid", description="全队防守压迫强度 high | mid | low")
    attack_focus: str = Field(default="center", description="进攻侧重 left | center | right")
    tempo: str = Field(default="normal", description="比赛节奏 fast | normal")
    reason: str = Field(default="", description="一句话决策理由")


def build_decision_tool() -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        return _validated(TacticalDecision, args)

    return ToolSpec(
        name="submit_decision",
        description=(
            "提交当前持球决策点的战术决策结果（直接驱动比赛模拟）。"
            "action=pass 时 target 只能从给定队友列表中选择，禁止编造球员。"
        ),
        parameters=_with_enum(
            _with_enum(
                _with_enum(
                    _with_enum(
                        _with_enum(
                            TacticalDecision.model_json_schema(),
                            "action", _ACTION_VALUES,
                        ),
                        "direction", _DIRECTION_VALUES,
                    ),
                    "press", _PRESS_VALUES,
                ),
                "attack_focus", _FOCUS_VALUES,
            ),
            "tempo", _TEMPO_VALUES,
        ),
        handler=handler,
        terminal=True,
    )


# ── 战术剧本（llm_brain.generate_script） ──────────────────────

class TacticalScript(_Base):
    """反事实模拟开始前的战术剧本框架。"""

    narrative: str = Field(default="", description="一句话总体战术思路")
    attacking_plan: str = Field(default="", description="控球方进攻打法")
    defending_plan: str = Field(default="", description="防守方应对策略")
    key_instructions: dict[str, str] = Field(default_factory=dict, description="球衣号→个体指令（最多3人）")
    expected_events: list[str] = Field(default_factory=list, description="预期事件序列（3-6条）")


def build_script_tool() -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        return _validated(TacticalScript, args)

    return ToolSpec(
        name="submit_script",
        description="提交模拟前生成的战术剧本框架（反事实设定必须作为第一条预期事件）。",
        parameters=TacticalScript.model_json_schema(),
        handler=handler,
        terminal=True,
    )


# ── MCTS 策略表（llm_brain.generate_strategy_table） ───────────

class StrategyTable(_Base):
    """球员行为策略概率分布（供 MCTS rollout 采样）。"""

    action_weights: dict[str, float] = Field(
        default_factory=dict,
        description="动作权重：pass/shoot/dribble/clear/hold 各 0-1",
    )
    pass_preferences: dict[str, float] = Field(
        default_factory=dict, description="队友球衣号→传球权重（总和约等于 1）",
    )
    reasoning: str = Field(default="", description="一句话说明")


def build_strategy_tool() -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        return _validated(StrategyTable, args)

    return ToolSpec(
        name="submit_strategy",
        description="提交该球员在当前情境下的行为策略概率分布。",
        parameters=StrategyTable.model_json_schema(),
        handler=handler,
        terminal=True,
    )


# ── 质疑判断（data_verifier._judge_challenge） ─────────────────

class ChallengeJudgment(_Base):
    """用户输入是否构成对上一回答的质疑。"""

    is_challenge: bool = Field(default=False, description="true=确实在质疑")
    challenge_target: str = Field(default="", description="被质疑的具体内容摘要")
    need_recheck: bool = Field(default=False, description="是否需要重新核验数据")
    recheck_scope: str = Field(default="", description="需要核验的范围")
    reasoning: str = Field(default="", description="判断理由")


def build_challenge_tool() -> ToolSpec:
    def handler(args: dict[str, Any]) -> dict[str, Any]:
        return _validated(ChallengeJudgment, args)

    return ToolSpec(
        name="submit_challenge_judgment",
        description=(
            "提交质疑判定结果：判断用户最新输入是否在质疑/纠正系统之前的回答。"
            "不确定时默认 is_challenge=false；is_challenge=true 时 need_recheck 默认 true。"
        ),
        parameters=ChallengeJudgment.model_json_schema(),
        handler=handler,
        terminal=True,
    )


def _validated(model: type[_Base], args: dict[str, Any]) -> dict[str, Any]:
    """pydantic 校验并返回规范化字典；失败按 invalid_arguments 分类。"""
    try:
        return model.model_validate(args).model_dump()
    except ValidationError as exc:
        raise ToolExecutionError(
            TOOL_FAILURE_INVALID_ARGS, f"参数校验失败: {exc}",
        ) from exc
