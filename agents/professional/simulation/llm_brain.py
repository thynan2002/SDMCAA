"""LLM 战术决策引擎。

以 LLM 为核心决策引擎，替代规则硬编码的战术判断：

- generate_script()        模拟开始前一次性生成战术剧本（全局先验）
- decide()                 每个持球决策点：比赛快照 → LLM → 决策指令
- generate_strategy_table() 为 MCTS rollout 一次性生成 LLM 策略表
                              （替代 action_weights 公式采样权重）

数值计算仅保留在物理执行与数据格式化层；所有"判断"均来自 LLM。
LLM 不可用 / 输出非法时，回退到内置启发式（保证流程不中断）。

语义档位描述（数值 → 自然语言标签）由 agents/prompts.py 的确定性
阈值函数承担（零 LLM 调用），最终自然语言化交给各回答 LLM 完成。
"""

from __future__ import annotations

import json
import re
from math import hypot
from typing import Any

from agents.constants import FIELD_WIDTH, FIELD_HEIGHT, PIXELS_PER_METER
from agents.llm_client import call_llm
from agents.logging_config import get_logger
from agents.player.tracker import _which_zone

logger = get_logger("llm_brain")

_ACTIONS = ("pass", "shoot", "dribble", "hold", "clear")

# ═══════════════════════════════════════════════════════════════════
# 提示词
# ═══════════════════════════════════════════════════════════════════

SCRIPT_SYSTEM_PROMPT = """## 角色
你是足球战术教练 (Tactical Scriptwriter)。在反事实模拟开始前，根据比赛局面与反事实设定，为后续模拟撰写战术剧本框架。

## 输出
直接输出纯 JSON 对象（不要用 markdown 代码块包裹，不要输出思考过程）：
{
  "narrative": "一句话总体战术思路",
  "attacking_plan": "控球方的进攻打法（如：左路渗透 / 快速横向转移 / 直塞身后）",
  "defending_plan": "防守方的应对策略（如：高位逼抢 / 中位压迫 / 低位收缩）",
  "key_instructions": {"球衣号": "给该球员的个体指令（最多3人，可空）"},
  "expected_events": ["预期发生的事件序列（3-6条，简练）"]
}

## 要求
- 基于给出的局面快照与球员风格，做符合足球常理的部署
- 反事实设定是剧本的起点，必须以其为第一条预期事件
- 不输出坐标数值，只用区域与方向语言"""

DECIDE_SYSTEM_PROMPT = """## 角色
你是足球战术决策引擎 (Tactical Decision Engine)。每个决策点，你根据比赛快照为持球者做出下一步动作决策，并给出全队移动意图。你的输出将直接驱动比赛模拟。

## 输出
直接输出纯 JSON 对象（不要用 markdown 代码块包裹，不要输出任何思考过程或多余文字）：
{
  "action": "pass | shoot | dribble | hold | clear 五选一",
  "target": "传球目标球衣号（action=pass 时必填，只能从队友列表中选）",
  "direction": "盘带方向 forward|left|right（action=dribble 时必填）",
  "press": "全队防守压迫强度 high|mid|low",
  "attack_focus": "进攻侧重 left|center|right",
  "tempo": "比赛节奏 fast|normal",
  "reason": "一句话决策理由"
}

## 决策原则
- 遵循战术剧本先验，但根据当前局面灵活应变
- 距球门近且身前开阔 → 优先射门
- 有队友处于明显空档且向前 → 优先传球
- 被贴身紧逼（压迫=贴身）→ 优先出球，不要强行盘带
- 避免与上一脚形成两人来回倒脚
- 在本方禁区附近受压 → 优先解围
- 只从给定队友列表中选择 target，禁止编造球员"""

STRATEGY_SYSTEM_PROMPT = """## 角色
你是足球球员行为策略建模师。根据球员风格模型与比赛情境，输出该球员的行为策略概率分布，供蒙特卡洛模拟采样使用。

## 输出
直接输出纯 JSON 对象（不要用 markdown 代码块包裹，不要输出思考过程）：
{
  "action_weights": {"pass": 0-1, "shoot": 0-1, "dribble": 0-1, "clear": 0-1, "hold": 0-1},
  "pass_preferences": {"队友球衣号": 0-1 权重},
  "reasoning": "一句话说明"
}

## 要求
- 权重反映该球员的风格标签与特征倾向（如组织型中场传球高、抢点前锋射门高）
- 结合当前比赛情境（位置、比分、反事实设定）微调
- pass_preferences 只包含给定队友，权重总和约等于 1"""


# ═══════════════════════════════════════════════════════════════════
# 决策引擎
# ═══════════════════════════════════════════════════════════════════

class LLMDecisionEngine:
    """LLM 战术决策引擎（剧本 + 逐决策点调用）。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.script: dict[str, Any] = {}
        self._decide_cache: dict[str, dict[str, Any]] = {}
        self.call_count = 0
        self._illegal_warned = 0  # 已输出 warning 的次数（超过阈值后降级为 debug，避免刷屏）

    # ── 战术剧本 ────────────────────────────────────────────────

    def generate_script(
        self,
        scenario_desc: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """模拟开始前一次性生成战术剧本。

        Args:
            scenario_desc: 反事实场景描述（如"6号一开始直接传球给10号"）
            context: 比赛上下文（队伍、球员风格、初始局面、剩余时长）

        Returns:
            剧本 dict；LLM 不可用时返回空 dict（后续决策不带先验）
        """
        self.script = {}
        if not self.enabled:
            return self.script

        user_msg = json.dumps(
            {"反事实设定": scenario_desc, "比赛上下文": context},
            ensure_ascii=False, indent=2,
        )
        # 结构化决策输出：低温 + 小预算（输出仅小 JSON，防推理链吃满预算）
        result = call_llm(SCRIPT_SYSTEM_PROMPT, user_msg, temperature=0.2, max_tokens=4096)
        parsed = _parse_json(result)
        if parsed:
            self.script = parsed
            self.call_count += 1
            logger.info("战术剧本已生成: %s", parsed.get("narrative", ""))
        return self.script

    # ── 逐决策点决策 ────────────────────────────────────────────

    def decide(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        """把一个比赛快照交给 LLM，获取决策指令。

        Returns:
            合法决策 dict（含 action/target/direction/press/attack_focus/tempo/reason），
            不可用或非法时返回 None（由调用方回退）。
        """
        if not self.enabled:
            return None

        # 局面签名缓存（同一模拟内极少重复，主要服务 MCTS 类反复查询）
        signature = _snapshot_signature(snapshot)
        if signature in self._decide_cache:
            return dict(self._decide_cache[signature])

        if self.script:
            snapshot = {**snapshot, "战术剧本": self.script}

        user_msg = json.dumps(snapshot, ensure_ascii=False, indent=2)
        for attempt in range(2):  # 非法输出重试一次
            result = call_llm(DECIDE_SYSTEM_PROMPT, user_msg, temperature=0.2, max_tokens=2048)
            decision = _validate_decision(_parse_json(result), snapshot)
            if decision:
                self.call_count += 1
                self._decide_cache[signature] = decision
                return decision
            # 首次失败用 warning（便于排查），后续重复失败降级为 debug（避免控制台刷屏）
            self._illegal_warned += 1
            if self._illegal_warned <= 3:
                logger.warning("LLM 决策输出非法（第%d次），重试", attempt + 1)
            else:
                logger.debug("LLM 决策输出非法（累计%d次），重试", self._illegal_warned)
        return None

    # ── MCTS 策略表 ─────────────────────────────────────────────

    def generate_strategy_table(
        self,
        subject_style: dict[str, Any],
        teammates: list[str],
        situation: str,
    ) -> dict[str, Any] | None:
        """为 MCTS rollout 一次性生成 LLM 策略表。

        Returns:
            {"action_weights": {...}, "pass_preferences": {...}} 或 None。
        """
        if not self.enabled:
            return None

        user_msg = json.dumps(
            {
                "球员风格": subject_style,
                "可选队友": teammates,
                "当前情境": situation,
            },
            ensure_ascii=False, indent=2,
        )
        result = call_llm(STRATEGY_SYSTEM_PROMPT, user_msg, temperature=0.2, max_tokens=4096)
        parsed = _parse_json(result)
        if not parsed:
            return None

        weights = parsed.get("action_weights")
        prefs = parsed.get("pass_preferences")
        if not isinstance(weights, dict) or not weights:
            return None
        cleaned_weights = {
            k: max(0.01, float(v)) for k, v in weights.items()
            if k in _ACTIONS and _is_number(v)
        }
        if not cleaned_weights:
            return None
        cleaned_prefs = {}
        if isinstance(prefs, dict):
            cleaned_prefs = {
                k: max(0.01, float(v)) for k, v in prefs.items()
                if k in teammates and _is_number(v)
            }
        self.call_count += 1
        return {
            "action_weights": cleaned_weights,
            "pass_preferences": cleaned_prefs,
            "reasoning": str(parsed.get("reasoning", "")),
        }


# ═══════════════════════════════════════════════════════════════════
# 快照构建（数值 → 语义，供 LLM 理解）
# ═══════════════════════════════════════════════════════════════════

def build_decision_snapshot(
    frame: int,
    holder_jersey: str,
    holder_team: str,
    holder_style: str,
    holder_pos: tuple[float, float],
    holder_pressure: float,
    teammates: list[dict[str, Any]],
    opponents: list[dict[str, Any]],
    attack_dir: int,
    recent_events: list[str],
    scenario_desc: str = "",
) -> dict[str, Any]:
    """把数值状态序列化为 LLM 可理解的语义快照。

    Args:
        holder_pressure: 最近防守者到持球者的距离（px）
        teammates/opponents: [{jersey, x, y, dist_to_holder, openness}]
        attack_dir: +1 攻 +y / -1 攻 -y
    """
    goal_y = FIELD_HEIGHT if attack_dir > 0 else 0.0
    goal_dist_m = hypot(holder_pos[0] - FIELD_WIDTH / 2, holder_pos[1] - goal_y) / PIXELS_PER_METER

    snapshot: dict[str, Any] = {
        "时间": f"{frame / 30:.1f}秒",
        "持球者": {
            "球衣号": holder_jersey,
            "队伍": holder_team,
            "风格": holder_style or "未知",
            "位置": _which_zone(holder_pos[0], holder_pos[1]),
            "受压": _pressure_label(holder_pressure),
        },
        "进攻方向": "向对方球门（纵向推进）",
        "距对方球门": f"约{goal_dist_m:.0f}米",
        "队友": [
            {
                "球衣号": t["jersey"],
                "位置": _which_zone(t["x"], t["y"]),
                "距离": _dist_label(t["dist_to_holder"]),
                "开阔度": "空档" if t["openness"] >= 120 else ("有人盯防" if t["openness"] >= 50 else "被贴身"),
                "纵向关系": _forward_label(t["y"] - holder_pos[1], attack_dir),
            }
            for t in teammates
        ],
        "附近对手": [
            {
                "球衣号": o["jersey"],
                "位置": _which_zone(o["x"], o["y"]),
                "距持球者": _dist_label(o["dist_to_holder"]),
            }
            for o in opponents[:5]
        ],
        "最近事件": recent_events[-3:],
    }
    if scenario_desc:
        snapshot["反事实设定"] = scenario_desc
    return snapshot


def _pressure_label(dist: float) -> str:
    if dist < 55:
        return "贴身紧逼"
    if dist < 120:
        return "有人盯防"
    return "空档"


def _dist_label(dist: float) -> str:
    m = dist / PIXELS_PER_METER
    if m < 10:
        return "很近"
    if m < 20:
        return "中距离"
    if m < 35:
        return "较远"
    return "远"


def _forward_label(dy: float, attack_dir: int) -> str:
    rel = dy * attack_dir
    if rel > 60:
        return "身前"
    if rel < -60:
        return "身后"
    return "平行"


# ═══════════════════════════════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════════════════════════════

def _parse_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    # 快速路径：直接 json.loads
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    # 兼容 markdown 代码块（```json ... ``` 或 ``` ... ```）
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(1))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    # 兜底：从文本中提取第一个 {...} 平衡花括号块（容忍前后说明文字）
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(cleaned[start : i + 1])
                        return data if isinstance(data, dict) else None
                    except json.JSONDecodeError:
                        break
    return None


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _match_jersey(target: str, valid: set[str]) -> str | None:
    """球衣号容错匹配：精确 → 去掉/补上"号"后缀 → 纯数字匹配。"""
    if not target:
        return None
    if target in valid:
        return target
    # 尝试去掉 "号" 后缀
    stripped = target.rstrip("号")
    if stripped in valid:
        return stripped
    # 尝试补上 "号" 后缀
    if target + "号" in valid:
        return target + "号"
    # 纯数字匹配：在 valid 中找数字部分相同的
    tgt_digits = "".join(c for c in target if c.isdigit())
    if tgt_digits:
        for v in valid:
            v_digits = "".join(c for c in str(v) if c.isdigit())
            if v_digits == tgt_digits:
                return v
    return None


def _snapshot_signature(snapshot: dict[str, Any]) -> str:
    holder = snapshot.get("持球者", {})
    mates = tuple(t.get("球衣号") + t.get("位置", "") for t in snapshot.get("队友", []))
    return f"{snapshot.get('时间')}|{holder.get('球衣号')}|{holder.get('位置')}|{mates}"


def _validate_decision(
    data: dict[str, Any] | None,
    snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    """校验并规范化 LLM 决策输出；非法返回 None。"""
    if not data:
        return None
    action = str(data.get("action", "")).strip().lower()
    if action not in _ACTIONS:
        return None

    decision: dict[str, Any] = {
        "action": action,
        "press": str(data.get("press", "mid")).lower(),
        "attack_focus": str(data.get("attack_focus", "center")).lower(),
        "tempo": str(data.get("tempo", "normal")).lower(),
        "reason": str(data.get("reason", "")),
    }

    valid_teammates = {t.get("球衣号") for t in snapshot.get("队友", [])}
    if action == "pass":
        target = str(data.get("target", "")).strip()
        # 精确匹配优先；容错：LLM 可能返回 "7" 而非 "7号"（或反之）
        resolved = _match_jersey(target, valid_teammates)
        if resolved is None:
            return None
        decision["target"] = resolved
    if action == "dribble":
        direction = str(data.get("direction", "forward")).lower()
        decision["direction"] = direction if direction in ("forward", "left", "right") else "forward"
    if decision["press"] not in ("high", "mid", "low"):
        decision["press"] = "mid"
    if decision["attack_focus"] not in ("left", "center", "right"):
        decision["attack_focus"] = "center"
    if decision["tempo"] not in ("fast", "normal"):
        decision["tempo"] = "normal"
    return decision


# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# 工具契约绑定：结构化决策输出改为 function calling 交付
# ═══════════════════════════════════════════════════════════════════
# 模型经 submit_* 终止型工具提交结构化结果（工具参数即 JSON 契约，
# 与原有文本契约字段一致）→ call_llm 返回 json.dumps(参数)，既有
# _parse_json / _validate_decision 原样
# 复用；模型直接返回文本（mock/旧 golden/模型行为差异）时文本解析
# 路径照常工作；LLM 失败时启发式兜底不变。
# 调用点签名未改（conftest 的 mock 以固定签名替换 call_llm）。
from agents.llm_client import bind_prompt_tools  # noqa: E402
from agents.tools.schemas import (  # noqa: E402
    build_decision_tool,
    build_script_tool,
    build_strategy_tool,
)

bind_prompt_tools(SCRIPT_SYSTEM_PROMPT, [build_script_tool()], tool_choice="auto")
bind_prompt_tools(DECIDE_SYSTEM_PROMPT, [build_decision_tool()], tool_choice="auto")
bind_prompt_tools(STRATEGY_SYSTEM_PROMPT, [build_strategy_tool()], tool_choice="auto")
