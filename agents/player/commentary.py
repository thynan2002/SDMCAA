"""足球比赛解说提示词模板。

负责将结构化赛场数据组装成 LLM 可读的 JSON 消息。
所有提示词定义在 prompts.py，此文件只做数据注入。
"""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Any

from .tracker import (
    BallEvent,
    BallTrajectoryAnalysis,
    PlayerTrajectory,
)
from ..constants import GROUND_Z_THRESHOLD, LOW_BALL_Z_MAX
from ..prompts import (
    COMPOSITION_SYSTEM_PROMPT,
    speed_tier,
    distance_tier,
    coverage_tier,
    activity_tier,
    sprint_tier,
)
from ..professional.simulation.llm_brain import SemanticTierBatcher

# 向后兼容的别名
SYSTEM_PROMPT = COMPOSITION_SYSTEM_PROMPT

# ── 帧 → 秒 转换 ──────────────────────────────────────────────
FPS = 30


def _frame_to_sec(frame: int) -> int:
    """帧号 → 秒数（向下取整）。"""
    return frame // FPS


def _frame_range_to_sec(start_frame: int, end_frame: int) -> str:
    """帧范围 → 秒数范围字符串，如 "3~8"。"""
    s = _frame_to_sec(start_frame)
    e = _frame_to_sec(end_frame)
    return f"{s}~{e}"


def build_user_message(
    prefix: str,
    field: list[PlayerTrajectory],
    goalkeepers: list[PlayerTrajectory],
    ball: BallTrajectoryAnalysis | None,
    tn: dict[str, str],
    episodes: list[dict[str, Any]],
    focus_jerseys: list[str] | None = None,
    ball_relation_results: list[dict[str, Any]] | None = None,
) -> str:
    """将结构化赛场数据组装成 JSON 用户消息。

    采用两遍模式：
    1. 第一遍收集中间值入队（batcher.tier）；
    2. 批量 LLM 调用（batcher.flush）；
    3. 第二遍用已缓存的语义描述构建数据。
    """
    # ── 整体时间范围 ──
    total_frames = 0
    if ball and ball.total_frames > 0:
        max_frame = max(
            max((p.frames[-1] for p in field + goalkeepers if p.frames), default=ball.total_frames - 1),
            ball.total_frames - 1,
        )
        total_frames = max_frame
    total_duration_sec = total_frames / FPS
    total_duration = f"{_frame_to_sec(total_frames)}秒（{total_frames}帧，30fps）" if total_frames else ""

    # ── 语义推断批量器 ──
    batcher = SemanticTierBatcher()
    batcher.set_context(
        num_players=len(field) + len(goalkeepers),
        duration_seconds=total_duration_sec,
    )
    # 第一遍：入队所有待推断值
    _queue_commentary_tiers(
        batcher, field, goalkeepers, ball, episodes,
        focus_jerseys, ball_relation_results,
    )
    # 批量 LLM 调用
    batcher.flush()

    # ═══════════════════════════════════════════════════════════════
    # 第二遍：构建数据（batcher 缓存已填充，tier 函数返回 LLM 描述）
    # ═══════════════════════════════════════════════════════════════

    # ── 阵型概况 ──
    formation: dict[str, Any] = {}
    if field:
        color_data: dict[str, dict[str, Any]] = {}
        for p in field:
            c = p.color
            if c not in color_data:
                color_data[c] = {"count": 0, "avg_y": [], "avg_x": [], "total_dist": []}
            color_data[c]["count"] += 1
            color_data[c]["avg_y"].append(p.avg_y())
            color_data[c]["avg_x"].append(p.avg_x())
            color_data[c]["total_dist"].append(p.total_distance)

        for c, d in color_data.items():
            avg_y_val = round(mean(d["avg_y"]), 0)
            if avg_y_val > 420:
                position_label = "压上至前场"
            elif avg_y_val > 320:
                position_label = "中场区域"
            else:
                position_label = "后场防线"
            avg_coverage_pct = round(mean(d["total_dist"]), 0)
            formation[tn.get(c, c)] = {
                "人数": d["count"],
                "阵型位置": position_label,
                "攻击侧重": "右路" if round(mean(d["avg_x"]), 0) > 800 else "左路" if round(mean(d["avg_x"]), 0) < 400 else "中路均衡",
                "跑动量": coverage_tier(avg_coverage_pct, batcher),
            }

    gk_list: list[str] = [gk.jersey_label for gk in goalkeepers]

    # ── 球路事件（去类型化：不提供硬编码事件类型，由 LLM 自行判断）──
    event_list: list[dict[str, Any]] = []
    if ball and ball.events:
        for e in ball.events:
            # 球路描述：球员A→球员B（距离过远或跟踪丢失标记为未知）
            start_p = e.nearest_player if e.nearest_player != "?" else "未知"
            end_p = e.nearest_player_end if e.nearest_player_end != "?" else "未知"
            if start_p == end_p:
                ball_route = f"{start_p}控球移动"
            else:
                ball_route = f"{start_p}→{end_p}"

            # 球高度上下文（取终点 z 坐标，单位：米，阈值见 agents.constants）
            end_z = e.end_pos[2] if len(e.end_pos) > 2 else 0
            if end_z <= GROUND_Z_THRESHOLD:
                height_ctx = "地面球"
            elif end_z <= LOW_BALL_Z_MAX:
                height_ctx = "低平球"
            else:
                height_ctx = "高空球"

            event_list.append({
                "球路": ball_route,
                "方向": e.direction,
                "速度(px/s)": round(e.speed, 0),
                "速度描述": speed_tier(e.speed, batcher),
                "距离(px)": round(e.distance, 0),
                "距离描述": distance_tier(e.distance, e.start_pos[1], e.end_pos[1], batcher),
                "球高度": height_ctx,
                "帧范围": [e.start_frame, e.end_frame],
            })

    # ── 聚合 episode（去类型化）──
    episode_list: list[dict[str, Any]] = []
    for ep in episodes:
        ep_events: list = ep.get("events", [])
        if ep_events:
            start_frame = min(e.start_frame for e in ep_events)
            end_frame = max(e.end_frame for e in ep_events)
            time_range = _frame_range_to_sec(start_frame, end_frame)
        else:
            time_range = "未知"
        # 描述该阶段涉及哪些球员之间的球权流转
        start_players = ep.get("players", [])
        end_players = ep.get("players_end", [])
        if start_players and end_players:
            route_desc = " → ".join(start_players[:3]) + " → " + " → ".join(end_players[:3])
        elif start_players:
            route_desc = "、".join(start_players[:4])
        else:
            route_desc = "球员"
        episode_list.append({
            "赛段时间": time_range,
            "球权流转": route_desc,
            "移动距离(px)": round(ep["total_dist"], 0),
            "移动距离描述": distance_tier(ep["total_dist"], batcher=batcher),
            "峰值速度(px/s)": round(ep["max_spd"], 0),
            "峰值速度描述": speed_tier(ep["max_spd"], batcher),
            "事件数": len(ep["events"]),
        })

    # ── 球员亮点 ──
    if field:
        by_dist = sorted(field, key=lambda p: p.total_distance, reverse=True)
        by_ball = sorted(field, key=lambda p: p.ball_proximity_pct, reverse=True)
        highlights = {
            "跑动最多": [f"{p.jersey_label}({tn.get(p.color,'')})" for p in by_dist[:3]],
            "离球最近": [f"{p.jersey_label}({tn.get(p.color,'')})" for p in by_ball[:3]],
        }
    else:
        highlights = {}

    # ── 球整体特征 ──
    ball_features: dict[str, Any] = {}
    if ball and ball.total_frames > 0:
        ball_features = {
            "空中球占比": "高" if ball.aerial_pct > 40 else "中" if ball.aerial_pct > 10 else "低",
            "整体球速(px/s)": round(ball.avg_speed, 0),
            "整体球速描述": speed_tier(ball.avg_speed, batcher),
            "峰值球速(px/s)": round(ball.max_speed, 0),
            "峰值球速描述": speed_tier(ball.max_speed, batcher),
            "主要方向": _top_directions(ball.direction_counts),
        }

    # ── 球高度与球员关系（BallPositionRelationAgent 输出） ──
    ball_relation_summary: dict[str, Any] = {}
    if ball_relation_results:
        # 聚合争顶统计
        top_aerial = sorted(ball_relation_results, key=lambda r: r.get("aerial_contest_count", 0), reverse=True)
        top_3d = sorted(ball_relation_results, key=lambda r: r.get("avg_3d_distance", float("inf")))
        top_ground = sorted(ball_relation_results, key=lambda r: r.get("ground_proximity_pct", 0), reverse=True)

        ball_relation_summary = {
            "争顶最积极": [
                {"球员": r["jersey"], "空中争顶次数": r.get("aerial_contest_count", 0)}
                for r in top_aerial[:3] if r.get("aerial_contest_count", 0) > 0
            ],
            "综合球距最近(含高度)": [
                {"球员": r["jersey"], "3D平均距离": r.get("avg_3d_distance", 0)}
                for r in top_3d[:3]
            ],
            "地面球最紧密": [
                {"球员": r["jersey"], "地面接近率": f"{r.get('ground_proximity_pct', 0)}%"}
                for r in top_ground[:3]
            ],
        }

    data: dict[str, Any] = {
        "片段": prefix,
        "帧率": "30fps（30帧=1秒）",
        "数据总时长": total_duration,
        "阵型概况": formation,
        "门将": gk_list,
        "球整体特征": ball_features,
        "球高度与球员关系": ball_relation_summary,
        "比赛阶段": episode_list,
        "球员亮点": highlights,
    }

    # ── 聚焦球员 ──
    if focus_jerseys:
        focus_players: list[dict[str, Any]] = []
        for fj in focus_jerseys:
            for p in field + goalkeepers:
                if p.track_id == fj or p.jersey_label == f"{fj}号":
                    speed_curve = p.speed_curve
                    sprint_count = sum(1 for s in speed_curve if s > 300)
                    player_info: dict[str, Any] = {
                        "球衣号": p.jersey_label,
                        "所属队伍": tn.get(p.color, ""),
                        "跑动量": coverage_tier(p.field_coverage_pct, batcher),
                        "活跃纵深度": "前场" if p.avg_y() > 400 else "后场" if p.avg_y() < 280 else "中场",
                        "活跃宽度": "右路" if p.avg_x() > 800 else "左路" if p.avg_x() < 400 else "中路",
                        "主要区域": p.dominant_zone,
                        "触球频率": "高" if p.ball_proximity_pct > 50 else "中" if p.ball_proximity_pct > 20 else "低",
                        "球距远近": "贴身的" if p.avg_ball_distance < 30 else "较近的" if p.avg_ball_distance < 100 else "较远的",
                        "峰值速度(px/s)": round(p.max_speed, 0),
                        "速度描述": speed_tier(p.max_speed, batcher),
                        "冲刺次数": sprint_count,
                        "冲刺描述": sprint_tier(sprint_count, batcher),
                    }
                    # 注入球高度关系数据
                    if ball_relation_results:
                        for br in ball_relation_results:
                            if br.get("track_id") == p.track_id:
                                player_info["空中争顶次数"] = br.get("aerial_contest_count", 0)
                                player_info["地面接近率"] = f"{br.get('ground_proximity_pct', 0)}%"
                                player_info["相对运动趋势"] = (
                                    "向球靠拢" if br.get("approaching_pct", 0) > 55
                                    else "保持距离" if br.get("approaching_pct", 0) > 40
                                    else "远离球路"
                                )
                                player_info["高空球参与度"] = f"{br.get('high_ball_proximity_pct', 0)}%"
                                break
                    focus_players.append(player_info)
                    break
        if focus_players:
            data["重点关注球员"] = focus_players
            labels = "、".join(p["球衣号"] for p in focus_players)
            instruction = f"请围绕{labels}这几名球员的表现展开解说，深入分析他们的个人特点和彼此之间的配合联动："
        else:
            instruction = f"请根据以下赛场数据，生成一段{prefix}的足球比赛解说："
    else:
        instruction = f"请根据以下赛场数据，生成一段{prefix}的足球比赛解说："

    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    return f"{instruction}\n\n{json_text}"


def _queue_commentary_tiers(
    batcher: SemanticTierBatcher,
    field: list[PlayerTrajectory],
    goalkeepers: list[PlayerTrajectory],
    ball: BallTrajectoryAnalysis | None,
    episodes: list[dict[str, Any]],
    focus_jerseys: list[str] | None,
    ball_relation_results: list[dict[str, Any]] | None,
) -> None:
    """第一遍：收集所有需要语义推断的原始数值到 batcher 队列。"""
    # ── 阵型概况：队伍平均跑动量 ──
    if field:
        color_data: dict[str, dict[str, Any]] = {}
        for p in field:
            c = p.color
            if c not in color_data:
                color_data[c] = {"total_dist": []}
            color_data[c]["total_dist"].append(p.total_distance)
        for d in color_data.values():
            batcher.tier(round(mean(d["total_dist"]), 0), "coverage")

    # ── 球路事件：速度 / 距离 ──
    if ball and ball.events:
        for e in ball.events:
            batcher.tier(e.speed, "speed")
            extra = {"起点y": round(e.start_pos[1], 0), "终点y": round(e.end_pos[1], 0)}
            batcher.tier(e.distance, "distance", extra)

    # ── episode：移动距离 / 峰值速度 ──
    for ep in episodes:
        batcher.tier(ep["total_dist"], "distance")
        batcher.tier(ep["max_spd"], "speed")

    # ── 球整体特征：球速 ──
    if ball and ball.total_frames > 0:
        batcher.tier(ball.avg_speed, "speed")
        batcher.tier(ball.max_speed, "speed")

    # ── 聚焦球员：跑动量 / 峰值速度 / 冲刺次数 ──
    if focus_jerseys:
        all_players = list(field) + list(goalkeepers)
        for fj in focus_jerseys:
            for p in all_players:
                if p.track_id == fj or p.jersey_label == f"{fj}号":
                    batcher.tier(p.field_coverage_pct, "coverage")
                    batcher.tier(p.max_speed, "speed")
                    sprint_count = float(sum(1 for s in p.speed_curve if s > 300))
                    batcher.tier(sprint_count, "sprint")
                    break


def _top_directions(dir_counts: dict[str, int]) -> str:
    if not dir_counts:
        return "无"
    sorted_dirs = sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)
    return "、".join(f"{d}({c}次)" for d, c in sorted_dirs[:3])
