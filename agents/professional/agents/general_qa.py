"""综合问答智能体。

处理所有非特定意图的比赛相关问题。
从比赛数据中提取结构化事实，
通过 LLM 生成自然语言回答。
所有输出文本由 LLM 生成。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents.player.tracker import (
    PrefixPlayerCorpus,
    PlayerTrajectory,
    BallTrajectoryAnalysis,
    BallEvent,
)
from agents.llm_client import call_llm
from agents.prompts import speed_tier, distance_tier
from agents.constants import GROUND_Z_THRESHOLD, LOW_BALL_Z_MAX, FIELD_WIDTH, FIELD_HEIGHT
from agents.professional.simulation.llm_brain import SemanticTierBatcher
from .tactical_facts import extract_tactical_facts


# ── LLM 提示词 ──

SYSTEM_PROMPT = """## 角色
你是资深足球战术分析师 (General Q&A Agent)。职责是基于提供的比赛追踪数据，结合足球专业知识，回答用户关于比赛的各种自然语言问题——不仅描述数据，更要从数据中推理出战术含义。

## 输入
- `比赛事实`: 结构化数据，包含：
  - 比赛信息 / 球员清单 / 球员位置分布
  - 球路线索（逐事件的球路、方向、速度/距离档位、球高度）
  - 战术事实（球队阵型与站位、进攻态势、球员参与度与跑动、攻防转换线索）
- `对话上下文`: 之前轮次讨论过的内容（可选）
- `用户问题`: 当前用户的自然语言提问

## 输出
纯文本回答，格式要求：
- 先给结论，再给数据依据（"因为……所以……"的推理结构）
- 用简洁专业的语言，像资深球探或战术分析师
- 纯文本输出，不要 markdown 格式
- **数值输出规则（最高优先级，无条件遵守）**：默认严禁罗列统计数值（次数、占比、百分比等）。必须先把数据理解，再翻译为自然语言判断（"绝大多数时间"、"极少"、"几乎每次"、"明显更活跃"）。**唯一例外**：用户问题中明确索要具体数据（如"具体几次""数据是多少""占比多少"）时，才引用"比赛事实"中已给出的统计数值
- 严禁输出坐标数值，严禁编造数据中没有的数字
- **多子问题处理**：如果用户一次提出多个问题（如"从谁开始传出，随后路径如何"、"谁最积极？谁最长？"），必须逐一作答，用"首先/其次/随后"等衔接词组织，不得遗漏任何子问题，也不要因为问题有多个就声称数据不足

## 足球分析推理框架（回答时主动运用）

### 1. 阵型与站位
- 由"阵型结构"和三线人数判断阵型（如 4-3-3、4-4-2），结合"整体站位"判断球队是高位压迫、中位组织还是低位收缩
- 三线人数为 0 的线（如 0-6-4）说明该片段球队整体压上、没有拖后球员

### 2. 进攻方式识别
- 短传占比高 + 地面球为主 → 控球渗透打法
- 长传/空中球占比高 → 长传冲吊或直接打法
- 向前推进占比高 + 进入进攻三区频繁 → 积极进攻态势；向后回传多 → 控球倒脚或进攻受阻
- 球路落点热区集中在一侧 → 该侧是主要进攻通道

### 3. 空间利用
- 结合球路落点热区与球员主要活动区域：边路热区 → 边路进攻；中路热区 → 中路渗透
- 球员活动区域与球路落点重合度高 → 该球员是进攻支点

### 4. 球员风格推理（示例，照此逻辑推理而非照抄）
- 活动区域集中在边路 + 频繁冲刺 + 向前推进多 → 边路突破型球员
- 中路活动 + 参与球路次数最高 + 短传为主 → 组织核心/节拍器
- 参与球路次数 + 接近球时间占比最高者 → 本场比赛核心球员
- 很少参与球路 + 活动区域固定 → 角色型/站位型球员

### 5. 攻防转换
- "方向反转"频繁 → 比赛开放、转换节奏快；很少反转 → 一方持续压制
- 由守转攻后快速进入进攻三区 → 反击威胁大

### 6. 事件类型判断
球路线索中的「球路」「速度档位」「距离档位」「方向」「球高度」字段，请自行判断事件类型：
- 同一球员控球 + 慢速 + 短距 → 盘带/控球
- 不同球员间 + 快速 + 中距 → 传球；长距 → 长传
- 朝球门方向 + 快速 → 射门
- 「未知」表示该端球员因跟踪缺失无法确定，可根据上下文推断

## 回答规则（按优先级排序）

### 规则 1: 忠实于数据
只基于提供的比赛事实和对话上下文回答。推理必须能回溯到具体事实字段。数据确实不足时，明确说"根据当前数据无法确定"，不要编造。

### 规则 2: 推理优于罗列
不要只罗列数据。每个结论都要说明战术含义。
错误示范："6号参与球路次数最多。"
正确示范："6号是本场比赛的进攻枢纽——他的触球和接球次数全队最高，且球路多数经他脚下流转，说明球队进攻围绕他展开。"

### 规则 3: 利用对话上下文
如果对话上下文中已经讨论过某球员或事件，应结合历史信息回答，避免自相矛盾。

### 规则 4: 诚实面对不确定性
- 数据不足时明确说明，不要编造
- 问题超出数据范围时（如战术建议、历史类比），说明"这是基于数据的客观描述"

## 边界（不做事项）
- 不要编造数据中不存在的球员、事件或时间
- 不要输出坐标数值（如"坐标(844, 320)"）
- 不要做超出数据范围的推测（如对手心理状态、教练意图）"""


class GeneralQAAgent:
    """综合问答智能体。

    提取比赛结构化事实作为 LLM 的知识上下文，
    回答任意与当前比赛相关的自然语言问题。
    """

    name = "GeneralQAAgent"

    def answer(self, question: str, corpus: PrefixPlayerCorpus, memory_context: str = "") -> str:
        """回答用户关于比赛的问题。

        Args:
            question: 用户原始问题
            corpus: 比赛语料
            memory_context: 对话记忆上下文
        """
        # ── 语义推断批量器 ──
        batcher = SemanticTierBatcher()
        batcher.set_context(
            num_players=len(corpus.players),
            duration_seconds=self._total_duration_sec(corpus),
        )
        # 第一遍：入队待推断值，随后 batch LLM 调用
        self._queue_qa_tiers(batcher, corpus)
        batcher.flush()

        facts = self._extract_facts(corpus, batcher)
        user_msg = self._build_prompt(facts, question, memory_context)
        result = call_llm(SYSTEM_PROMPT, user_msg, stream=True)
        if result:
            return result.strip()
        # 注意：此处仅当 LLM 调用失败（网络/超时/空响应）时触发，
        # 与"数据不足"无关，文案需如实区分，避免误导用户。
        return (
            f"关于「{question}」，分析服务暂时未能生成回答"
            "（模型调用失败，并非数据不足）。请稍后重试，"
            "或将复合问题拆分为多个单问题分别提问。"
        )

    # ── 事实提取 ──

    def _extract_facts(
        self, corpus: PrefixPlayerCorpus, batcher: SemanticTierBatcher,
    ) -> dict[str, Any]:
        """从 corpus 提取比赛结构化事实。"""
        players = list(corpus.players.values())
        field = corpus.field_players()
        goalkeepers = corpus.goalkeepers()
        ball = corpus.ball_analysis

        return {
            "比赛信息": {
                "片段标识": corpus.prefix,
                "总帧数": self._total_frames(corpus),
                "总时长": self._total_duration(corpus),
                "球员总数": len(players),
                "场上球员数": len(field),
                "门将数": len(goalkeepers),
            },
            "球员清单": self._extract_player_list(players),
            "球队概况": self._extract_team_summary(players),
            "球员位置分布": self._extract_position_summary(players),
            "球路线索": self._extract_ball_timeline(ball, batcher),
            "球特征": self._extract_ball_features(ball),
            "战术事实": extract_tactical_facts(corpus, batcher),
        }

    def _total_duration_sec(self, corpus: PrefixPlayerCorpus) -> float:
        """计算总时长（秒）。"""
        return self._total_frames(corpus) / 30

    def _queue_qa_tiers(
        self, batcher: SemanticTierBatcher, corpus: PrefixPlayerCorpus,
    ) -> None:
        """第一遍：收集所有需要语义推断的原始数值到 batcher 队列。"""
        ball = corpus.ball_analysis
        if ball and ball.events:
            # 球路事件：速度 / 距离
            for e in ball.events:
                batcher.tier(e.speed, "speed")
                extra = {"起点y": round(e.start_pos[1], 0), "终点y": round(e.end_pos[1], 0)}
                batcher.tier(e.distance, "distance", extra)

            # 球整体特征（供 tactical_facts 使用）
            batcher.tier(ball.avg_speed, "speed")
            avg_dist = sum(e.distance for e in ball.events) / len(ball.events)
            batcher.tier(avg_dist, "distance")

        # 球员冲刺次数（供 tactical_facts 使用）
        for p in corpus.players.values():
            sprint_count = sum(1 for s in p.speed_curve if s > 300)
            batcher.tier(float(sprint_count), "sprint")

    def _total_frames(self, corpus: PrefixPlayerCorpus) -> int:
        """计算总帧数。"""
        all_frames = set()
        for p in corpus.players.values():
            all_frames.update(p.frames)
        all_frames.update(corpus.ball_frames.keys())
        return max(all_frames) if all_frames else 0

    def _total_duration(self, corpus: PrefixPlayerCorpus) -> str:
        """计算总时长（秒）。"""
        total = self._total_frames(corpus)
        seconds = total / 30
        return f"{seconds:.1f}秒（{total}帧）"

    def _extract_player_list(
        self, players: list[PlayerTrajectory],
    ) -> list[dict[str, Any]]:
        """提取球员清单。"""
        result = []
        for p in sorted(players, key=lambda x: _safe_sort_key(x.track_id)):
            info = {
                "球衣号": p.jersey_label,
                "所属队伍": self._team_name(p),
                "颜色标识": p.color,
                "是否门将": p.is_goalkeeper,
                "主要活动区域": p.dominant_zone,
                "位置角色": self._infer_simple_role(p),
            }
            result.append(info)
        return result

    def _extract_team_summary(
        self, players: list[PlayerTrajectory],
    ) -> dict[str, Any]:
        """提取球队概况。"""
        from collections import defaultdict

        teams: dict[str, list[PlayerTrajectory]] = defaultdict(list)
        for p in players:
            teams[self._team_name(p)].append(p)

        result = {}
        for team_name, squad in teams.items():
            if not squad:
                continue
            avg_y = sum(p.avg_y() for p in squad) / len(squad)
            avg_x = sum(p.avg_x() for p in squad) / len(squad)

            if avg_y > 420:
                position = "整体压上至前场"
            elif avg_y > 320:
                position = "主要在中场区域"
            else:
                position = "收缩在后场防线"

            if avg_x > 800:
                side = "偏向右路"
            elif avg_x < 400:
                side = "偏向左路"
            else:
                side = "中路均衡"

            result[team_name] = {
                "人数": len(squad),
                "阵型位置": position,
                "攻击侧重": side,
                "球员球衣号": [p.jersey_label for p in squad],
            }
        return result

    def _extract_position_summary(
        self, players: list[PlayerTrajectory],
    ) -> list[dict[str, Any]]:
        """提取球员位置分布摘要。"""
        result = []
        for p in players:
            zones = sorted(p.zone_distribution.items(), key=lambda x: x[1], reverse=True)
            top_zones = [f"{z}({v:.0f}%时间)" for z, v in zones[:3]]
            result.append({
                "球衣号": p.jersey_label,
                "主要区域": p.dominant_zone,
                "区域分布Top3": top_zones,
            })
        return result

    def _extract_ball_timeline(
        self, ball: BallTrajectoryAnalysis | None, batcher: SemanticTierBatcher,
    ) -> list[dict[str, Any]]:
        """提取球路时间线（去类型化：不提供硬编码事件类型，由 LLM 自行判断）。"""
        if not ball or not ball.events:
            return [{"说明": "球路数据不可用"}]

        timeline = []
        for e in ball.events:
            time_start = e.start_frame / 30
            time_end = e.end_frame / 30

            # 球路描述（距离过远或跟踪丢失标记为未知）
            start_p = e.nearest_player if e.nearest_player != "?" else "未知"
            end_p = e.nearest_player_end if e.nearest_player_end != "?" else "未知"
            if start_p == end_p:
                ball_route = f"{start_p}控球移动"
            else:
                ball_route = f"{start_p}→{end_p}"

            # 球高度上下文（z 单位：米，阈值见 agents.constants）
            end_z = e.end_pos[2] if len(e.end_pos) > 2 else 0
            if end_z <= GROUND_Z_THRESHOLD:
                height_ctx = "地面"
            elif end_z <= LOW_BALL_Z_MAX:
                height_ctx = "低平"
            else:
                height_ctx = "高空"

            timeline.append({
                "时间": f"{time_start:.1f}~{time_end:.1f}秒",
                "球路": ball_route,
                "方向": e.direction,
                "距离(px)": round(e.distance, 0),
                "距离描述": distance_tier(e.distance, e.start_pos[1], e.end_pos[1], batcher),
                "速度(px/s)": round(e.speed, 0),
                "速度描述": speed_tier(e.speed, batcher),
                "球高度": height_ctx,
                "描述": self._describe_event(e, batcher),
            })

        # 合并相邻同类事件，保持时间线精简
        return self._merge_timeline(timeline)

    def _extract_ball_features(
        self, ball: BallTrajectoryAnalysis | None,
    ) -> dict[str, Any]:
        """提取球特征。"""
        if not ball:
            return {"说明": "球路数据不可用"}

        # 球速档位
        if ball.avg_speed > 300:
            speed_label = "球速很快"
        elif ball.avg_speed > 150:
            speed_label = "球速中等"
        else:
            speed_label = "球速较慢"

        # 空中球
        if ball.aerial_pct > 30:
            aerial_label = "高空球比例高，长传较多"
        elif ball.aerial_pct > 10:
            aerial_label = "高低结合，偶有空中球"
        else:
            aerial_label = "以地面球为主"

        return {
            "整体球速": speed_label,
            "空中球特征": aerial_label,
            "事件总数": len(ball.events) if ball.events else 0,
        }

    # ── 辅助 ──

    def _team_name(self, player: PlayerTrajectory) -> str:
        """颜色 → 中文队名。"""
        mapping = {"A": "A队", "B": "B队", "C": "门将"}
        return mapping.get(player.color, f"{player.color}队")

    def _infer_simple_role(self, player: PlayerTrajectory) -> str:
        """简单推断位置角色。"""
        if player.is_goalkeeper:
            return "门将"
        avg_y = player.avg_y()
        avg_x = player.avg_x()

        if avg_y < 220:
            pos = "后卫"
        elif avg_y < 380:
            pos = "中场"
        else:
            pos = "前锋"

        if avg_x < 400:
            side = "左"
        elif avg_x > 800:
            side = "右"
        else:
            side = "中"

        if side == "中":
            return f"{pos}（{side}路）"
        return f"{pos}（{side}路）"

    def _describe_event(self, event: BallEvent, batcher: SemanticTierBatcher) -> str:
        """为球路事件生成简短文本描述（去类型化，纯描述球从哪到哪）。"""
        start_player = event.nearest_player if event.nearest_player != "?" else "未知"
        end_player = event.nearest_player_end if event.nearest_player_end != "?" else "未知"
        dist_label = distance_tier(event.distance, event.start_pos[1], event.end_pos[1], batcher)
        spd_label = speed_tier(event.speed, batcher)
        return f"{start_player}→{end_player}（{dist_label}，{spd_label}，{event.direction}）"

    def _merge_timeline(
        self, timeline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """合并相邻同类球路事件，精简时间线（按球路模式而非硬编码类型合并）。"""
        if len(timeline) <= 1:
            return timeline

        merged = [timeline[0]]
        for item in timeline[1:]:
            prev = merged[-1]
            # 按球路模式合并：同一球员组合的连续球路
            same_route = item.get("球路") == prev.get("球路")
            if same_route:
                prev["时间"] = prev["时间"].split("~")[0] + "~" + item["时间"].split("~")[-1]
                prev["描述"] = prev["描述"] + " ; " + item["描述"]
            else:
                merged.append(item)
        return merged

    def _build_prompt(self, facts: dict[str, Any], question: str, memory_context: str = "") -> str:
        """构建 LLM 用户消息。"""
        payload: dict[str, Any] = {
            "比赛事实": facts,
            "用户问题": question,
        }
        if memory_context and memory_context != "（尚未建立对话上下文）":
            payload["对话上下文（之前分析过的内容）"] = memory_context
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _safe_sort_key(track_id: str) -> tuple:
    """安全的 track_id 排序键：数字按数值排，非数字按字符串排。"""
    try:
        return (0, int(track_id), track_id)
    except (ValueError, TypeError):
        return (1, track_id, track_id)


# ── 工具契约绑定：综合问答获得比赛数据工具 ─────────────────────
# 比赛事实已内嵌在首轮用户消息中（payload 与重构前一致）；当回答需要
# 提示词中未提供的额外信息（特定帧/球员数据、战术事实、球路明细、
# 反事实推演）时，模型**只能**通过工具获取，不得编造。工具不可用时
# 按「忠实于数据」纪律如实声明数据不足。
# 正文仍为纯文本流式生成；模型直接返回文本时（mock/旧 golden）行为不变。
from agents.llm_client import bind_prompt_tools  # noqa: E402
from agents.tools.football import (  # noqa: E402
    ensure_football_tools_registered,
    get_football_tools,
)

ensure_football_tools_registered()
bind_prompt_tools(SYSTEM_PROMPT, get_football_tools(), tool_choice="auto")
