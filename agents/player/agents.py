"""球员解说智能体。

精简后的球员追踪路径只保留一个 LLM 智能体：基于球路事件生成解说稿。
原 PlayerMovementAgent / PlayerBallInteractionAgent / PlayerComparisonAgent /
PlayerDecisionAgent 的中间统计结果未被解说链路消费（compose 只使用
corpus 与球关系结果），已在中间层精简中移除 —— 运动/互动/对比/决策
信息由 commentary.build_user_message 直接从 corpus 组装。
"""

from __future__ import annotations

from typing import Any

from .tracker import (
    PlayerTrajectory, PrefixPlayerCorpus, BallTrajectoryAnalysis, BallEvent,
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
