"""球队战术事实提取器。

从 PrefixPlayerCorpus 中计算球队级战术事实（纯数据计算，不依赖 LLM）：
- 球队阵型与整体站位
- 进攻态势（推进方向、进攻空间、传球特征）
- 球员参与度与跑动特征（核心球员识别）
- 攻防转换线索

供 GeneralQAAgent 与球队战术报告共用，保证球队级问题有足够数据支撑。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from agents.player.tracker import (
    BallTrajectoryAnalysis,
    PlayerTrajectory,
    PrefixPlayerCorpus,
    classify_position,
    classify_side,
    _which_zone,
)
from agents.prompts import distance_tier, speed_tier, sprint_tier
from agents.professional.simulation.llm_brain import SemanticTierBatcher


# 颜色 → 中文队名（与 data_collector / general_qa 保持一致）
_TEAM_NAMES = {"A": "A队", "B": "B队", "C": "门将"}

# 前场（进攻三区）y 阈值，与 tracker.Y_ZONES 一致
_ATTACKING_THIRD_Y = 466

# 冲刺速度阈值，与 DataCollectorAgent._build_speed_stats 一致
_SPRINT_SPEED = 300


def extract_tactical_facts(
    corpus: PrefixPlayerCorpus,
    batcher: SemanticTierBatcher | None = None,
) -> dict[str, Any]:
    """提取球队级战术事实，供 LLM 进行足球领域推理。

    Args:
        corpus: 比赛语料
        batcher: 可选语义推断批量器。提供时，档位字段将由 LLM 生成自然语言描述。
    """
    players = list(corpus.players.values())
    teams = _split_teams(players)

    return {
        "球队阵型与站位": _team_formations(teams),
        "进攻态势": _attacking_facts(corpus.ball_analysis, batcher),
        "球员参与度与跑动": _player_involvement(corpus, players, batcher),
        "攻防转换线索": _transition_facts(corpus.ball_analysis),
    }


# ── 球队划分 ──


def _team_name(color: str) -> str:
    return _TEAM_NAMES.get(color, f"{color}队")


def _split_teams(players: list[PlayerTrajectory]) -> dict[str, list[PlayerTrajectory]]:
    """按颜色划分球队（不含门将组）。"""
    teams: dict[str, list[PlayerTrajectory]] = defaultdict(list)
    for p in players:
        if p.is_goalkeeper or p.color == "C":
            continue
        teams[_team_name(p.color)].append(p)
    return dict(teams)


# ── 阵型与站位 ──


def _team_formations(teams: dict[str, list[PlayerTrajectory]]) -> dict[str, Any]:
    """为每支球队推断阵型结构与整体站位倾向。"""
    result: dict[str, Any] = {}
    for team_name, squad in teams.items():
        if not squad:
            continue

        # 按平均纵坐标分三线
        lines: dict[str, list[str]] = {"后卫": [], "中场": [], "前锋": []}
        for p in squad:
            lines[classify_position(p.avg_y())].append(p.jersey_label)

        formation = "-".join(str(len(lines[line])) for line in ("后卫", "中场", "前锋"))

        avg_y = sum(p.avg_y() for p in squad) / len(squad)
        avg_x = sum(p.avg_x() for p in squad) / len(squad)

        if avg_y > 420:
            block = "整体压上至前场"
        elif avg_y > 320:
            block = "主要在中场区域"
        else:
            block = "收缩在后场防线"

        side = classify_side(avg_x)
        side_desc = {"左": "偏向左路", "右": "偏向右路"}.get(side, "中路均衡")

        # 场地覆盖（衡量阵型紧凑/拉开程度）
        coverage = sum(p.field_coverage_pct for p in squad) / len(squad)

        result[team_name] = {
            "阵型结构": f"{formation}（按平均站位推断）",
            "后卫线球员": lines["后卫"],
            "中场球员": lines["中场"],
            "前锋球员": lines["前锋"],
            "整体站位": block,
            "横向侧重": side_desc,
            "平均覆盖面积档位": "拉开" if coverage > 8 else ("适中" if coverage > 3 else "紧凑"),
        }
    return result


# ── 进攻态势 ──


def _attacking_facts(
    ball: BallTrajectoryAnalysis | None,
    batcher: SemanticTierBatcher | None = None,
) -> dict[str, Any]:
    """从球路事件提取进攻态势事实。"""
    if not ball or not ball.events:
        return {"说明": "球路数据不可用"}

    events = ball.events
    total = len(events)

    # 1. 推进方向统计（"前"=向对方球门方向）
    forward = sum(1 for e in events if "前" in e.direction)
    backward = sum(1 for e in events if "后" in e.direction)
    lateral = total - forward - backward

    # 2. 进攻空间：球路终点区域热区 + 进入进攻三区次数
    end_zones: dict[str, int] = defaultdict(int)
    third_entries = 0
    for e in events:
        end_zones[_which_zone(e.end_pos[0], e.end_pos[1])] += 1
        if e.end_pos[1] >= _ATTACKING_THIRD_Y:
            third_entries += 1
    top_zones = sorted(end_zones.items(), key=lambda x: x[1], reverse=True)[:3]

    # 3. 传球距离/速度特征
    long_balls = sum(1 for e in events if e.distance >= 150)
    short_balls = sum(1 for e in events if e.distance < 50)
    fast_balls = sum(1 for e in events if e.speed >= 400)
    avg_dist = _avg_distance(events)

    return {
        "球路事件总数": total,
        "推进方向分布": {
            "向前推进占比": f"{forward / total:.0%}",
            "向后回传占比": f"{backward / total:.0%}",
            "横向转移占比": f"{lateral / total:.0%}",
        },
        "进攻空间": {
            "球路落点热区Top3": [f"{zone}（{cnt}次）" for zone, cnt in top_zones],
            "进入进攻三区次数": third_entries,
            "进攻三区到达率": f"{third_entries / total:.0%}",
        },
        "传球特征": {
            "长传占比": f"{long_balls / total:.0%}",
            "短传占比": f"{short_balls / total:.0%}",
            "快速球占比": f"{fast_balls / total:.0%}",
            "空中球占比": f"{ball.aerial_pct:.0f}%",
            "主导球速(px/s)": round(ball.avg_speed, 0),
            "主导球速描述": speed_tier(ball.avg_speed, batcher),
            "主导传球距离(px)": round(avg_dist, 0),
            "主导传球距离描述": distance_tier(avg_dist, batcher=batcher),
        },
    }


def _avg_distance(events: list) -> float:
    return sum(e.distance for e in events) / len(events) if events else 0.0


def _distance_level(total_distance: float) -> str:
    """跑动量档位（阈值与 DataCollectorAgent 战术角色推断一致）。"""
    if total_distance > 250:
        return "大"
    if total_distance > 100:
        return "中"
    return "小"


# ── 球员参与度与跑动 ──


def _player_involvement(
    corpus: PrefixPlayerCorpus,
    players: list[PlayerTrajectory],
    batcher: SemanticTierBatcher | None = None,
) -> list[dict[str, Any]]:
    """统计每名球员参与球路次数与跑动特征（核心球员识别依据）。"""
    involvement: dict[str, int] = defaultdict(int)
    ball = corpus.ball_analysis
    if ball and ball.events:
        for e in ball.events:
            for jersey in (e.nearest_player, e.nearest_player_end):
                if jersey and jersey != "?":
                    involvement[jersey] += 1

    result: list[dict[str, Any]] = []
    for p in players:
        sprint_count = sum(1 for s in p.speed_curve if s > _SPRINT_SPEED)
        result.append({
            "球衣号": p.jersey_label,
            "所属队伍": _team_name(p.color),
            "参与球路次数": involvement.get(p.jersey_label, 0),
            "接近球时间占比": f"{p.ball_proximity_pct:.0f}%",
            "跑动量": _distance_level(p.total_distance),
            "冲刺次数": sprint_count,
            "冲刺描述": sprint_tier(sprint_count, batcher),
            "主要活动区域": p.dominant_zone,
            "位置分类": f"{classify_side(p.avg_x())}路{classify_position(p.avg_y())}",
        })

    # 按参与度降序，便于 LLM 识别核心球员
    result.sort(key=lambda x: x["参与球路次数"], reverse=True)
    return result


# ── 攻防转换 ──


def _transition_facts(ball: BallTrajectoryAnalysis | None) -> dict[str, Any]:
    """检测球路方向前后反转（攻防转换的近似信号）。"""
    if not ball or not ball.events or len(ball.events) < 2:
        return {"说明": "球路事件不足，无法分析攻防转换"}

    transitions: list[str] = []
    prev_forward: bool | None = None
    for e in ball.events:
        if "前" in e.direction:
            cur_forward = True
        elif "后" in e.direction:
            cur_forward = False
        else:
            continue  # 横向球不构成转换信号

        if prev_forward is not None and cur_forward != prev_forward:
            time_s = e.start_frame / 30
            kind = "由守转攻（球路转为向前）" if cur_forward else "由攻转守（球路转为向后）"
            transitions.append(f"第{time_s:.1f}秒：{kind}")
        prev_forward = cur_forward

    return {
        "方向反转次数": len(transitions),
        "转换时刻": transitions[:8] if transitions else ["片段内未出现明显攻防转换"],
    }
