"""比赛数据工具 — 把确定性分析能力封装为 LLM 可调用的工具。

LLM 需要任何外部信息只能通过工具获取；工具结果不可用时必须如实返回
「数据不足」，不得编造（与「忠实于数据」提示词纪律一致）。

工具（全部为数据型，terminal=False）：
- get_tactical_facts         战术事实提取（阵型/进攻态势/参与度/攻防转换）
- get_ball_timeline          球路分析时间线（逐事件方向/速度/距离/高度）
- get_frame_snapshot         帧级核验（指定帧的球/球员/事件快照）
- get_player_raw_data        球员轨迹原始数据核验
- get_player_profile         球员画像（速度/覆盖/区域/冲刺等统计）
- run_counterfactual_simulation  反事实轨迹模拟（启发式，不触发 LLM 递归）

语料来源：SessionManager.load_data / main.run_once 注入 active corpus；
语料缺失时工具返回「数据不足」，由模型如实声明。
"""

from __future__ import annotations

from typing import Any

from agents.tools.base import (
    ToolExecutionError,
    ToolSpec,
    insufficient_data,
    register_tool,
)

# 当前会话语料（由 SessionManager.load_data / main.run_once 注入）
_active_corpus: Any = None
_registered = False


def set_active_corpus(corpus: Any) -> None:
    """注入当前会话语料（工具处理器的数据来源）。"""
    global _active_corpus
    _active_corpus = corpus


def get_active_corpus() -> Any:
    return _active_corpus


def _require_corpus() -> Any:
    """语料未加载时按「数据不足」分类失败。"""
    if _active_corpus is None:
        raise ToolExecutionError(
            "insufficient_data", "比赛数据未加载，无法提供该数据",
        )
    return _active_corpus


# ── 工具实现 ───────────────────────────────────────────────────

def _get_tactical_facts(args: dict[str, Any]) -> dict[str, Any]:
    """战术事实提取（阵型/进攻态势/参与度/攻防转换）。"""
    corpus = _require_corpus()
    from agents.professional.agents.tactical_facts import extract_tactical_facts

    facts = extract_tactical_facts(corpus)
    return insufficient_data("无法提取战术事实") if not facts else facts


def _get_ball_timeline(args: dict[str, Any]) -> dict[str, Any]:
    """球路分析时间线（确定性档位描述，不调用 LLM）。"""
    corpus = _require_corpus()
    from agents.constants import GROUND_Z_THRESHOLD, LOW_BALL_Z_MAX
    from agents.prompts import distance_tier, speed_tier

    ball = corpus.ball_analysis
    if not ball or not ball.events:
        return insufficient_data("球路数据不可用")

    timeline: list[dict[str, Any]] = []
    for e in ball.events:
        start_p = e.nearest_player if e.nearest_player != "?" else "未知"
        end_p = e.nearest_player_end if e.nearest_player_end != "?" else "未知"
        route = f"{start_p}控球移动" if start_p == end_p else f"{start_p}→{end_p}"
        end_z = e.end_pos[2] if len(e.end_pos) > 2 else 0
        if end_z <= GROUND_Z_THRESHOLD:
            height = "地面"
        elif end_z <= LOW_BALL_Z_MAX:
            height = "低平"
        else:
            height = "高空"
        timeline.append({
            "时间": f"{e.start_frame / 30:.1f}~{e.end_frame / 30:.1f}秒",
            "球路": route,
            "方向": e.direction,
            "速度": speed_tier(e.speed),
            "距离": distance_tier(e.distance, e.start_pos[1], e.end_pos[1]),
            "球高度": height,
        })
    return {"球路时间线": timeline}


def _get_frame_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    """帧级核验：指定帧号的球/球员/事件快照。"""
    corpus = _require_corpus()
    from math import hypot

    from agents.player.tracker import positions_at_frame

    raw = args.get("frame")
    try:
        frame = int(raw)
    except (TypeError, ValueError):
        raise ToolExecutionError(
            "invalid_arguments", f"frame 参数非法: {raw!r}",
        ) from None

    ball_pos = corpus.ball_frames.get(frame)
    if ball_pos is None:
        return insufficient_data(f"第{frame}帧的球数据不存在")

    bx, by, bz = ball_pos
    players_at = positions_at_frame(corpus.players, frame)
    jersey_of = {tid: p.jersey_label for tid, p in corpus.players.items()}
    near: list[dict[str, Any]] = []
    nearest = ""
    nearest_d = float("inf")
    for tid, (px, py) in players_at.items():
        d = hypot(bx - px, by - py)
        jl = jersey_of.get(tid, "?")
        if d < nearest_d:
            nearest_d, nearest = d, jl
        near.append({"球衣号": jl, "距球": round(d, 0)})
    near.sort(key=lambda x: x["距球"])

    events: list[dict[str, Any]] = []
    ball_analysis = corpus.ball_analysis
    if ball_analysis and ball_analysis.events:
        for e in ball_analysis.events:
            if e.start_frame <= frame <= e.end_frame:
                start_p = e.nearest_player if e.nearest_player != "?" else "未知"
                end_p = e.nearest_player_end if e.nearest_player_end != "?" else "未知"
                events.append({"球路": f"{start_p}→{end_p}", "方向": e.direction})

    return {
        "帧号": frame,
        "时间": f"约第{frame / 30:.1f}秒",
        "球位置": f"(x={bx:.0f}, y={by:.0f}, z={bz:.0f})",
        "离球最近球员": nearest,
        "附近球员": near[:8],
        "该帧球路事件": events,
    }


def _get_player_raw_data(args: dict[str, Any]) -> dict[str, Any]:
    """球员轨迹原始数据核验。"""
    corpus = _require_corpus()
    jersey = str(args.get("jersey") or "")
    for player in corpus.players.values():
        if player.jersey_label == jersey:
            return {
                "球员": jersey,
                "所属队伍": player.color,
                "颜色标识": player.color,
                "总帧数": len(player.frames),
                "轨迹范围": {
                    "x": [round(player.x_range[0], 0), round(player.x_range[1], 0)],
                    "y": [round(player.y_range[0], 0), round(player.y_range[1], 0)],
                },
                "主要区域": player.dominant_zone,
                "平均位置": {"x": round(player.avg_x(), 0), "y": round(player.avg_y(), 0)},
                "平均速度(px/s)": round(player.avg_speed, 1),
                "最大速度(px/s)": round(player.max_speed, 1),
                "冲刺次数": sum(1 for s in player.speed_curve if s > 300),
                "场地覆盖率(%)": round(player.field_coverage_pct, 1),
            }
    return insufficient_data(f"未找到球员 {jersey} 的轨迹数据")


def _get_player_profile(args: dict[str, Any]) -> dict[str, Any]:
    """球员画像（速度/覆盖/区域/参与度等统计）。"""
    corpus = _require_corpus()
    jersey = str(args.get("jersey") or "")
    ball = corpus.ball_analysis

    involvement = 0
    if ball and ball.events:
        for e in ball.events:
            if e.nearest_player == jersey or e.nearest_player_end == jersey:
                involvement += 1

    for player in corpus.players.values():
        if player.jersey_label == jersey:
            zones = sorted(
                player.zone_distribution.items(), key=lambda x: x[1], reverse=True,
            )
            return {
                "球员": jersey,
                "所属队伍": {"A": "A队", "B": "B队", "C": "门将"}.get(
                    player.color, f"{player.color}队"),
                "是否门将": player.is_goalkeeper,
                "战术角色": (
                    "门将" if player.is_goalkeeper else "由数据推断"
                ),
                "主要活动区域": player.dominant_zone,
                "区域分布Top3": [
                    f"{z}({v:.0f}%时间)" for z, v in zones[:3]
                ],
                "平均速度(px/s)": round(player.avg_speed, 1),
                "最大速度(px/s)": round(player.max_speed, 1),
                "冲刺次数": sum(1 for s in player.speed_curve if s > 300),
                "场地覆盖率(%)": round(player.field_coverage_pct, 1),
                "参与球路次数": involvement,
                "接近球时间占比(%)": round(player.ball_proximity_pct, 0),
            }
    return insufficient_data(f"未找到球员 {jersey} 的画像数据")


def _run_counterfactual_simulation(args: dict[str, Any]) -> dict[str, Any]:
    """反事实轨迹模拟（启发式引擎，不触发 LLM 递归）。"""
    corpus = _require_corpus()
    from agents.professional.mcts.node import Action, ActionType
    from agents.professional.simulation.engine import TrajectorySimulator

    subject = str(args.get("subject_player") or "")
    try:
        time_second = float(args.get("time_second") or 0)
    except (TypeError, ValueError):
        raise ToolExecutionError(
            "invalid_arguments", f"time_second 参数非法: {args.get('time_second')!r}",
        ) from None
    action_name = str(args.get("altered_action") or "dribble")
    target = str(args.get("altered_target") or "")
    try:
        duration = float(args.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        duration = 0.0

    if action_name == "pass":
        forced = Action(ActionType.PASS, target_player=target)
    elif action_name == "shoot":
        forced = Action(ActionType.SHOOT)
    elif action_name == "dribble":
        forced = Action(ActionType.DRIBBLE, direction=(0, 30))
    elif action_name == "clear":
        forced = Action(ActionType.CLEAR)
    else:
        forced = Action(ActionType.HOLD)

    sim = TrajectorySimulator(corpus)  # llm_engine=None → 纯启发式，快速且不递归
    result = sim.simulate(
        intervention_frame=int(time_second * 30),
        forced_action=forced,
        subject=subject,
        duration_seconds=duration,
        scenario_desc="反事实模拟（工具调用）",
    )
    outcome_cn = {
        "goal": "进球", "save_held": "射门被门将没收", "miss": "射门偏出",
        "out": "球出界", "completed": "自然推演至片段结束",
    }
    return {
        "模拟区间": f"帧 {result.start_frame} → {result.end_frame}",
        "结局": outcome_cn.get(result.outcome or "", result.outcome or "-"),
        "事件时间线": list(result.events),
    }


# ── 注册 ───────────────────────────────────────────────────────

_TOOL_SPECS: list[ToolSpec] | None = None


def get_football_tools() -> list[ToolSpec]:
    """返回数据工具规格列表（幂等构造，供提示词绑定使用）。"""
    global _TOOL_SPECS
    if _TOOL_SPECS is None:
        _TOOL_SPECS = [
            ToolSpec(
                name="get_tactical_facts",
                description=(
                    "获取球队级战术事实：阵型结构与站位、进攻态势（推进方向/进攻空间/传球特征）、"
                    "球员参与度与跑动、攻防转换线索。用户询问球队战术、阵型、进攻方式、核心球员、"
                    "攻防转换等问题时调用。"
                ),
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=_get_tactical_facts,
            ),
            ToolSpec(
                name="get_ball_timeline",
                description=(
                    "获取球路分析时间线：逐球路事件的起止时间、球路（如「6号→7号」）、方向、"
                    "速度档位、距离档位、球高度。用户询问球的流转、传球序列、球路等问题时调用。"
                ),
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=_get_ball_timeline,
            ),
            ToolSpec(
                name="get_frame_snapshot",
                description=(
                    "帧级核验：获取指定帧的球位置、离球最近球员、附近球员及该帧球路事件。"
                    "用户询问特定时刻/帧发生了什么、质疑某时刻场上位置时调用。"
                ),
                parameters={
                    "type": "object",
                    "properties": {"frame": {"type": "integer", "description": "帧号（帧率30，帧号≈秒数×30）"}},
                    "required": ["frame"],
                },
                handler=_get_frame_snapshot,
            ),
            ToolSpec(
                name="get_player_raw_data",
                description=(
                    "获取球员的原始轨迹数据（总帧数、轨迹范围、主要区域、平均位置、速度与冲刺统计）。"
                    "用户核实某球员的位置/轨迹/活动范围时调用。"
                ),
                parameters={
                    "type": "object",
                    "properties": {"jersey": {"type": "string", "description": "球衣号，如「7号」"}},
                    "required": ["jersey"],
                },
                handler=_get_player_raw_data,
            ),
            ToolSpec(
                name="get_player_profile",
                description=(
                    "获取球员画像统计：主要活动区域、区域分布、速度/冲刺、场地覆盖率、"
                    "参与球路次数、接近球时间占比。用户询问球员风格、活跃度、特点时调用。"
                ),
                parameters={
                    "type": "object",
                    "properties": {"jersey": {"type": "string", "description": "球衣号，如「7号」"}},
                    "required": ["jersey"],
                },
                handler=_get_player_profile,
            ),
            ToolSpec(
                name="run_counterfactual_simulation",
                description=(
                    "反事实轨迹模拟（启发式）：基于给定干预时间与替换动作，从真实数据出发推演"
                    "未来轨迹并返回关键事件与结局。仅在用户明确描述「如果/假如…会怎样」且需要"
                    "推演结果时调用；耗时数秒。altered_action 取值 pass/shoot/dribble/clear/hold，"
                    "pass 时可提供 altered_target。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "subject_player": {"type": "string", "description": "主体球员球衣号，如「7号」"},
                        "time_second": {"type": "number", "description": "干预时间（秒）"},
                        "altered_action": {
                            "type": "string", "enum": ["pass", "shoot", "dribble", "clear", "hold"],
                        },
                        "altered_target": {"type": "string", "description": "替换传球目标球衣号（pass 时）"},
                        "duration_seconds": {"type": "number", "description": "推演时长（秒），0=到数据结束"},
                    },
                    "required": ["subject_player", "time_second", "altered_action"],
                },
                handler=_run_counterfactual_simulation,
            ),
        ]
    return list(_TOOL_SPECS)


def ensure_football_tools_registered() -> None:
    """幂等注册数据工具到默认注册表（由使用方模块导入时调用）。"""
    global _registered
    if _registered:
        return
    _registered = True
    for spec in get_football_tools():
        register_tool(spec)
