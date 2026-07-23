"""球员轨迹分析智能体。

面向 PlayerTrajectory 做个体分析、球员对比、角色推断和决策融合。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from .tracker import (
    PlayerTrajectory, PrefixPlayerCorpus, BallTrajectoryAnalysis, BallEvent, ZONE_LABELS,
    classify_position, classify_side,
)
from .ball_relation import BallPositionRelationAgent, BallPositionReport, BallHeightSummary
from ..types import AgentReport
from ..prompts import (
    DECISION_FAST_TEMPO,
    DECISION_NORMAL_TEMPO,
    DECISION_SLOW_TEMPO,
    COMPARISON_SUMMARY_TEMPLATE,
    COMPARISON_BALL_ADDON,
)


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


# ── 角色推断辅助 ──────────────────────────────────────────────

def _infer_role(player: PlayerTrajectory) -> str:
    """根据活动热区和跑动特征推断场上角色。"""
    if player.is_goalkeeper:
        return "门将"

    total = player.total_distance
    avg_y = mean(player.ys) if player.ys else 0
    avg_x = mean(player.xs) if player.xs else 0

    roles: list[str] = [classify_position(avg_y)]

    side = classify_side(avg_x)
    side_label = {"左": "左路", "中": "中路", "右": "右路"}.get(side, "中路")
    roles.append(side_label)

    # 跑动量修正
    if total > 300:
        roles.append("高跑动")

    return "、".join(roles)


def _role_score(role: str) -> float:
    """角色战术重要性权重。"""
    weights = {
        "前锋、中路": 0.95,
        "前锋": 0.85,
        "中场、中路": 0.90,
        "中场": 0.80,
        "后卫、中路": 0.85,
        "后卫": 0.75,
    }
    for key, w in weights.items():
        if key in role:
            return w
    return 0.70


# ── 个体球员分析智能体 ────────────────────────────────────────

class PlayerMovementAgent:
    """分析单个球员的运动特征：跑动距离、速度、覆盖范围。"""

    name = "PlayerMovementAgent"

    def analyze(self, player: PlayerTrajectory) -> dict[str, Any]:
        # 缓存角色推断结果避免重复计算
        role = _infer_role(player)
        return {
            "track_id": player.track_id,
            "jersey": player.jersey_label,
            "color": player.color,
            "total_distance": player.total_distance,
            "avg_speed": player.avg_speed,
            "max_speed": player.max_speed,
            "field_coverage_pct": player.field_coverage_pct,
            "x_range": player.x_range,
            "y_range": player.y_range,
            "dominant_zone": player.dominant_zone,
            "zone_distribution": player.zone_distribution,
            "role": role,
            "role_weight": round(_role_score(role), 2),
        }

    def summarize_all(
        self, corpus: PrefixPlayerCorpus
    ) -> list[dict[str, Any]]:
        return [self.analyze(player) for player in corpus.sorted_players()]


class PlayerBallInteractionAgent:
    """分析球员与球的互动关系。"""

    name = "PlayerBallInteractionAgent"

    def analyze(self, player: PlayerTrajectory) -> dict[str, Any]:
        return {
            "track_id": player.track_id,
            "jersey": player.jersey_label,
            "avg_ball_distance": player.avg_ball_distance,
            "ball_proximity_pct": player.ball_proximity_pct,
            "min_ball_distance": round(min(player.ball_distances), 2) if player.ball_distances else None,
        }

    def summarize_all(
        self, corpus: PrefixPlayerCorpus
    ) -> list[dict[str, Any]]:
        return [self.analyze(player) for player in corpus.sorted_players()]


class PlayerComparisonAgent:
    """球员横向对比智能体。"""

    name = "PlayerComparisonAgent"

    def __init__(self) -> None:
        self.ball_relation = BallPositionRelationAgent()

    def summarize(
        self,
        corpus: PrefixPlayerCorpus,
        movement_results: list[dict[str, Any]],
        interaction_results: list[dict[str, Any]],
        ball_relation_results: list[dict[str, Any]] | None = None,
    ) -> AgentReport:
        players = list(corpus.players.values())
        if not players:
            return AgentReport(
                agent_name=self.name,
                summary="无球员可比。",
                score_0_to_100=0,
                confidence_0_to_1=0,
            )

        by_distance = sorted(players, key=lambda p: p.total_distance, reverse=True)
        by_ball_proximity = sorted(players, key=lambda p: p.ball_proximity_pct, reverse=True)

        # 分颜色统计
        color_groups: dict[str, list[PlayerTrajectory]] = {}
        for p in players:
            color_groups.setdefault(p.color or "未知", []).append(p)

        # 自然语言摘要
        active_label = by_distance[0].jersey_label
        ball_player = by_ball_proximity[0]
        summary = COMPARISON_SUMMARY_TEMPLATE.format(active=active_label)
        if ball_player.ball_proximity_pct > 30:
            summary += COMPARISON_BALL_ADDON.format(ball_player=ball_player.jersey_label)

        # 空中争顶王
        if ball_relation_results:
            by_aerial = sorted(ball_relation_results, key=lambda r: r.get("aerial_contest_count", 0), reverse=True)
            if by_aerial and by_aerial[0].get("aerial_contest_count", 0) > 0:
                aerial_king = by_aerial[0]["jersey"]
                summary += f"；{aerial_king}在空中球的争夺中最为积极"

        return AgentReport(
            agent_name=self.name,
            summary=summary,
            score_0_to_100=min(100, len(players) * 5),
            confidence_0_to_1=0.85,
            key_points=[],
        )


class PlayerDecisionAgent:
    """球员分析决策融合智能体。"""

    name = "PlayerDecisionAgent"

    def decide(
        self,
        corpus: PrefixPlayerCorpus,
        movement_results: list[dict[str, Any]],
        interaction_results: list[dict[str, Any]],
        comparison_report: AgentReport,
        ball_relation_results: list[dict[str, Any]] | None = None,
    ) -> tuple[str, float, str, list[str], list[str]]:
        players = list(corpus.players.values())
        if not players:
            return ("无球员数据", 0.0, "", [], [])

        avg_distance = mean(p.total_distance for p in players)
        avg_coverage = mean(p.field_coverage_pct for p in players)
        effective_frames = max(
            (max((p.frames[-1] for p in players if p.frames), default=0) - min((p.frames[0] for p in players if p.frames), default=0) + 1),
            1,
        )
        distance_rate = avg_distance / max(1, effective_frames / 30)

        # 基础节奏判定
        if distance_rate > 800:
            action = DECISION_FAST_TEMPO
        elif distance_rate > 450:
            action = DECISION_NORMAL_TEMPO
        else:
            action = DECISION_SLOW_TEMPO

        # 球高度风格修正：高空球占比高时调整节奏描述
        if ball_relation_results:
            # 从球关系分析中提取高空球占比
            high_ball_count = sum(r.get("aerial_contest_count", 0) for r in ball_relation_results)
            total_high_frames = sum(r.get("nearest_high_ball_total", 0) for r in ball_relation_results)
            if total_high_frames > 0 and high_ball_count / max(1, total_high_frames) > 0.3:
                # 高空球多 → 节奏偏慢，以长传为主
                if action == DECISION_FAST_TEMPO:
                    action = "比赛节奏较快，但高空球频繁，攻防转换以长传冲吊为主。"

        # 评分：按片段时长归一化后再映射，避免短片段和长片段不可比
        # 每秒 0-30 单位映射到 0-60，覆盖率 0-20% 映射到 0-40
        distance_score = min(60, distance_rate / 30 * 60)
        coverage_score = min(40, avg_coverage / 20 * 40)
        overall = round(distance_score + coverage_score, 2)

        rationale = comparison_report.summary

        # 下一步关注建议：跑动最多和最贴球的球员
        by_dist = sorted(corpus.players.values(), key=lambda p: p.total_distance, reverse=True)
        next_focus = [p.jersey_label for p in by_dist[:2] if p.total_distance > 0]

        # 风险提示
        risks: list[str] = []
        if avg_coverage < 3.0:
            risks.append("全队活动范围偏窄，可能存在站位过于集中的问题。")
        if ball_relation_results:
            low_proximity = sum(
                1 for r in ball_relation_results
                if r.get("ground_proximity_pct", 0) < 10
            )
            if low_proximity > len(ball_relation_results) * 0.5:
                risks.append("过半球员与球的距离较远，控球衔接可能存在问题。")

        return (
            action,
            overall,
            rationale,
            next_focus,
            risks,
        )


def _team_names(players: list[PlayerTrajectory]) -> dict[str, str]:
    """队名映射，C → '门将'。"""
    color_groups: dict[str, list[PlayerTrajectory]] = {}
    for p in players:
        color_groups.setdefault(p.color, []).append(p)
    names: dict[str, str] = {}
    for c in sorted(color_groups):
        if c == "C":
            names[c] = "门将"
        else:
            names[c] = f"{c}队"
    return names


class PlayerCompositionAgent:
    """基于球路事件的精细化足球解说——文字由 LLM 生成。"""

    name = "PlayerCompositionAgent"

    def compose(
        self,
        corpus: PrefixPlayerCorpus,
        movement_results: list[dict[str, Any]],
        interaction_results: list[dict[str, Any]],
        comparison_report: AgentReport,
        decision: tuple[str, float, str, list[str], list[str]],
        focus_jerseys: list[str] | None = None,
        ball_relation_results: list[dict[str, Any]] | None = None,
    ) -> str:
        players = list(corpus.players.values())
        if not players:
            return f"【{corpus.prefix}】本段没有捕捉到可用的球员轨迹数据。"

        field = corpus.field_players()
        goalkeepers = corpus.goalkeepers()
        ball = corpus.ball_analysis
        tn = _team_names(players)

        # 聚合事件 episode（分析逻辑保留）
        episodes: list[dict[str, Any]] = []
        if ball and ball.events:
            episodes = self._group_into_episodes(ball.events)

        # 构造提示词
        from .commentary import SYSTEM_PROMPT, build_user_message
        from ..llm_client import call_llm

        user_msg = build_user_message(
            corpus.prefix, field, goalkeepers, ball, tn, episodes,
            focus_jerseys=focus_jerseys,
            ball_relation_results=ball_relation_results,
        )
        result = call_llm(SYSTEM_PROMPT, user_msg, stream=True)

        if result and result.strip():
            return result.strip()
        # 回退到规则生成
        return self._fallback_commentary(field, goalkeepers, ball, tn, episodes)

    # ── 事件聚合（分析逻辑，不变）───────────────────────────────

    def _group_into_episodes(self, events: list[BallEvent]) -> list[dict[str, Any]]:
        if not events:
            return []

        episodes: list[dict[str, Any]] = []
        current_type = events[0].event_type
        current_events: list[BallEvent] = [events[0]]

        for e in events[1:]:
            same_group = (
                e.event_type == current_type
                or (current_type in ("传球", "长传") and e.event_type in ("传球", "长传"))
                or (current_type == "带球" and e.event_type in ("带球", "移动"))
            )
            if same_group:
                current_events.append(e)
                if e.event_type == "长传" and current_type == "传球":
                    current_type = "长传"
            else:
                episodes.append(self._build_episode(current_type, current_events))
                current_type = e.event_type
                current_events = [e]

        episodes.append(self._build_episode(current_type, current_events))
        return episodes

    def _build_episode(self, etype: str, evts: list[BallEvent]) -> dict[str, Any]:
        # 注意：BallEvent 是非 frozen dataclass，不可哈希，不能用 set() 去重
        # 改用 dict.fromkeys 保持顺序去重
        unique_start_players = list(dict.fromkeys(
            e.nearest_player for e in evts if e.nearest_player != "?"
        ))
        unique_end_players = list(dict.fromkeys(
            e.nearest_player_end for e in evts if e.nearest_player_end != "?"
        ))
        return {
            "type": etype,
            "events": evts,
            "players": unique_start_players,
            "players_end": unique_end_players,
            "total_dist": sum(e.distance for e in evts),
            "max_spd": max(e.speed for e in evts) if evts else 0,
        }

    # ── 回退解说（LLM 不可用时）──────────────────────────────────

    def _fallback_commentary(
        self,
        field: list[PlayerTrajectory],
        goalkeepers: list[PlayerTrajectory],
        ball: BallTrajectoryAnalysis | None,
        tn: dict[str, str],
        episodes: list[dict[str, Any]],
    ) -> str:
        from .commentary import _frame_to_sec, _frame_range_to_sec

        parts: list[str] = []

        # 门将
        for gk in goalkeepers:
            parts.append(f"0~{_frame_to_sec(max(gk.frames)) if gk.frames else 1}s：{gk.jersey_label}守在门前。")

        # 按 episode 分段
        if episodes:
            for ep in episodes:
                ep_events: list = ep.get("events", [])
                if ep_events:
                    time_range = _frame_range_to_sec(
                        min(e.start_frame for e in ep_events),
                        max(e.end_frame for e in ep_events),
                    )
                else:
                    time_range = ""

                etype = ep["type"]
                players = ep["players"]
                who = "、".join(players[:4]) if players else "球员"

                segment_parts: list[str] = []
                if etype == "射门":
                    segment_parts.append(f"{who}完成了一次射门。")
                elif etype in ("长传", "传球"):
                    segment_parts.append(f"{who}进行了传递配合。")
                elif etype == "带球":
                    segment_parts.append(f"{who}持球推进。")
                elif etype == "争顶":
                    segment_parts.append(f"{who}参与了空中争顶。")
                else:
                    segment_parts.append(f"{who}参与了{etype}。")

                if time_range and segment_parts:
                    parts.append(f"{time_range}s：{' '.join(segment_parts)}")

        # 收尾
        if field and not episodes:
            by_ball = sorted(field, key=lambda p: p.ball_proximity_pct, reverse=True)
            parts.append(f"{by_ball[0].jersey_label}是本段触球最多的球员。")

        return "\n\n".join(parts)
