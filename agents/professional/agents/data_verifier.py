"""数据核验与回查智能体。

功能：
1. 精确帧级/球员级数据查询 — 从 corpus 提取指定时间点的详细状态
2. 数据缺口检测 — 判断用户问题是否需要回溯源数据
3. LLM 质疑判断 — 判断用户是否在质疑已有回答，决定是否触发核验
4. 核验报告生成 — 将核验结果以自然语言呈现，标注核验过程
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents.player.tracker import (
    PrefixPlayerCorpus,
    PlayerTrajectory,
    BallEvent,
)
from agents.llm_client import call_llm


# ═══════════════════════════════════════════════════════════════════
# LLM 提示词
# ═══════════════════════════════════════════════════════════════════

VERIFY_SYSTEM_PROMPT = """## 角色
你是足球比赛数据核验分析师 (Data Verifier)。职责是基于原始追踪数据进行精确的帧级或球员级核验，并生成核验回答。

## 输入
- 原始帧级数据：帧号、时间、球位置、战术区域、全队站位、离球最近球员、事件列表
- 用户原始问题

## 输出
纯文本回答：先给出该帧的场上情况描述（用足球专业语言的形势判断，含双方站位概况），再给数据依据（帧号/时间锚点）。系统会在回答末尾自动追加 [核验过程] 标注，你无需自行输出该标注，也不要重复描述核验步骤。

## 核心规则
- 只基于提供的数据描述，不编造任何信息
- 说明数据来源（帧号、时间等核验依据），回答必须包含帧号/时间锚点
- 默认把数值翻译为自然语言形势判断，不堆砌数字；威胁值是基于球所在场地位置的静态评估，只能描述为"该位置威胁潜力较高/较低"，严禁解读为"已形成进攻敏感性/实际威胁机会"；用户明确索要具体数值/坐标/距离/威胁值/战术区域时，必须如实引用核验数据中的精确数值作答，不得含糊、不得拒绝、不得换成其他数值
- 球员站位描述必须基于"全队站位"中的坐标与区域，覆盖双方阵型分布；严禁仅凭离球距离推断球员的前后站位、层次或接应关系；数据已提供全部球员位置时，严禁声称"数据未提供全部球员的位置"
- 用户问题含场景预设（如"开球"）而核验数据的球位置与该预设不符（如球不在中圈）时，必须先基于数据客观指出差异（如"第0帧球并不在中圈"），再描述该帧实际局面，不得迎合错误预设
- 如果某些数据缺失，诚实说明
- 当核验数据不足以支持确定结论时（如缺少所询时刻的精确数据，只有统计概况），必须如实声明不确定性（如"该时刻无精确数据，仅能基于现有数据推断"），不得以强势断言代替
- 自行判断球路事件类型：不要依赖预标注类型，根据「球路」字段和方向判断（传球/长传/射门/盘带/解围等）
- 「未知」表示该端球员因跟踪缺失无法确定"""

CHALLENGE_SYSTEM_PROMPT = """## 角色
你是足球分析系统的**质疑判断模块 (Challenge Detector)**。职责是判断用户的最新输入是否在质疑/纠正系统之前的回答。

## 输入
- 对话历史：最近 2 轮对话（用户 + 系统回答）
- 用户最新输入

## 输出
纯 JSON 对象，包含：
- `is_challenge`: bool — true 表示用户确实在质疑
- `challenge_target`: string — 被质疑的具体内容摘要
- `need_recheck`: bool — 是否需要重新核验数据
- `recheck_scope`: string — 需要核验的范围
- `reasoning`: string — 判断理由

## 质疑判定条件（满足任一即 true）

1. **明确否定**: 用户使用了"不对"、"不是"、"错了"、"你错"、"明明"、"应该是"、"搞错"、"你搞错了"等否定词
2. **纠正数据**: 用户提供了与系统回答不同的数据描述（如"7号当时在左路不是在右路"）
3. **要求重新核验**: 用户使用了"再确认"、"重新查"、"再算一遍"、"核实一下"等要求复查的措辞

## 非质疑判定条件（以下情况 `is_challenge: false`）

1. **纯追问**: 用户问了新问题，没有否定之前的回答（如"那后来发生了什么"）
2. **确认性疑问**: 用户只是表达不确定（"真的吗"、"你确定吗"）——这不是质疑，是追问
3. **补充信息**: 用户补充了自己观察到的新信息但没有否定系统回答

## 兜底规则
- 不确定时，默认为非质疑（`is_challenge: false`）
- `need_recheck`: 当 `is_challenge: true` 时默认 true"""


# ═══════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FrameSnapshot:
    """单帧快照。"""
    frame: int
    time_second: float
    ball_x: float = 0.0
    ball_y: float = 0.0
    ball_z: float = 0.0
    player_positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    nearest_to_ball: str = ""  # 离球最近球员 jersey_label
    events_at_frame: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VerifyResult:
    """核验结果。"""
    query_type: str = ""               # "frame_detail" | "player_check" | "challenge_response"
    is_challenge: bool = False
    challenge_judgment: str = ""       # LLM 判断理由
    data_found: bool = False
    snapshot: FrameSnapshot | None = None
    verification_process: str = ""     # 核验过程说明
    verified_answer: str = ""          # LLM 生成的核验回答
    corrections: list[str] = field(default_factory=list)  # 修正点列表


# ═══════════════════════════════════════════════════════════════════
# 智能体
# ═══════════════════════════════════════════════════════════════════

class DataVerifierAgent:
    """数据核验与回查智能体。

    具备三大核心能力：
    1. 精确查询：帧级/球员级数据提取
    2. 质疑判断：LLM 判断用户是否在质疑
    3. 核验报告：生成包含核验过程的自然语言回答
    """

    name = "DataVerifierAgent"

    def verify(
        self,
        question: str,
        corpus: PrefixPlayerCorpus,
        conversation_history: list[dict[str, str]] | None = None,
        memory_context: str = "",
    ) -> str:
        """主入口：数据核验与回查。

        自动判断查询类型（帧查询/球员核验/质疑响应），
        执行对应核验流程，返回 LLM 生成的核验报告。
        """
        # Step 1: 提取帧号/时间点。帧/秒明确的精确查询直接走帧级核验，
        # 无需质疑判断 LLM（省一次串行调用）；含质疑关键词的帧级质疑
        # 仍走质疑判断（回溯上下文核验）。
        frame_info = self._extract_frame_info(question)
        if frame_info["frame"] is not None and not self._has_challenge_keywords(question):
            return self._verify_frame_detail(question, frame_info, corpus, memory_context)

        # Step 2: 判断是否为质疑
        challenge_result = self._judge_challenge(question, conversation_history)

        if challenge_result.get("is_challenge"):
            return self._handle_challenge(
                question, challenge_result, corpus, memory_context,
                conversation_history,
            )

        # Step 3: 帧级查询（含帧但被判定为非质疑时）
        if frame_info["frame"] is not None:
            return self._verify_frame_detail(question, frame_info, corpus, memory_context)

        # Step 4: 球员级核验
        return self._verify_player_detail(question, corpus, memory_context)

    # ── 质疑判断 ──

    # 与 _judge_challenge LLM 回退口径一致的质疑关键词
    _CHALLENGE_KEYWORDS = ("不对", "不是", "错了", "你错", "明明", "应该是", "搞错")

    @staticmethod
    def _has_challenge_keywords(text: str) -> bool:
        """是否包含明确的质疑/纠正关键词（与 LLM 回退判定一致）。"""
        return any(kw in text for kw in DataVerifierAgent._CHALLENGE_KEYWORDS)

    def _judge_challenge(
        self,
        text: str,
        history: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        """LLM 判断用户输入是否为质疑。"""
        # 构建上下文
        context = ""
        if history:
            recent = history[-4:]  # 最近两轮
            context = "\n".join(
                f"[{h['role']}]: {h['content'][:200]}"
                for h in recent
            )

        user_msg = f"对话历史：\n{context}\n\n用户最新输入：{text}"

        result = call_llm(CHALLENGE_SYSTEM_PROMPT, user_msg)
        if result:
            try:
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    lines = [l for l in lines if not l.startswith("```")]
                    cleaned = "\n".join(lines)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # 回退：关键词判断
        is_challenge = self._has_challenge_keywords(text)
        return {
            "is_challenge": is_challenge,
            "challenge_target": text,
            "need_recheck": is_challenge,
            "recheck_scope": "用户指出的内容",
            "reasoning": "关键词判断",
        }

    def _handle_challenge(
        self,
        question: str,
        challenge_result: dict,
        corpus: PrefixPlayerCorpus,
        memory_context: str = "",
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """处理用户质疑：重新核验数据并给出纠错/确认回答。"""
        recheck_scope = challenge_result.get("recheck_scope", question)

        # 从质疑中提取核查目标（全部球衣号，而非仅第一个）
        frame_info = self._extract_frame_info(question)
        jerseys = self._extract_all_jerseys(question)

        # 收集相关数据（多路证据并列收集，不再互斥）
        evidence_parts: list[str] = []
        corrections: list[str] = []

        # 帧级数据
        if frame_info["frame"] is not None:
            snapshot = self._capture_frame(frame_info["frame"], corpus)
            if snapshot:
                evidence_parts.append(self._format_snapshot(snapshot))
                corrections.append(
                    f"已重新核验第{snapshot.time_second:.1f}秒（帧{snapshot.frame}）的原始数据"
                )

        # 争议时刻的精确球员位置（位置类质疑的直接证据）
        # 旧版证据只有"距球距离 + 全场轨迹统计"，LLM 拿不到争议时刻的
        # 具体坐标，只能凭全场主区域强势断言（verify_7s_pos 缺陷）；
        # 此处补齐插值位置，无数据时显式标注"无法精确核验"供 LLM 如实声明。
        if frame_info["frame"] is not None and jerseys:
            pos_evidence = self._extract_players_at_frame(
                frame_info["frame"], jerseys, corpus,
            )
            if pos_evidence:
                evidence_parts.append(pos_evidence)
                corrections.append("已提取争议时刻相关球员的精确位置（插值）")

        # 球路链条物理证据（高度轨迹/方向/最近球员）
        # —— 核验"高空球/头顶经过/是否传球"类质疑的关键证据
        context_text = ""
        if conversation_history:
            assistant_msgs = [
                h["content"] for h in conversation_history
                if h.get("role") == "assistant"
            ]
            if assistant_msgs:
                context_text = assistant_msgs[-1]
        route_evidence = self._extract_route_evidence(
            question, corpus, memory_context, context_text,
        )
        if route_evidence:
            evidence_parts.append(route_evidence)
            corrections.append("已重新核验争议球路的完整轨迹与高度数据")

        # 球员数据（全部提及球员，不再只取第一个）
        player_hits = 0
        for jersey in jerseys[:4]:
            player_data = self._extract_player_raw_data(jersey, corpus)
            if player_data:
                evidence_parts.append(player_data)
                player_hits += 1
        if player_hits:
            corrections.append(
                f"已重新提取{', '.join(jerseys[:player_hits])}的原始轨迹数据"
            )

        # 事件级证据：质疑"未知球员"等事件标注时，回溯球路事件的判定依据
        if not evidence_parts:
            event_evidence = self._extract_event_evidence(question, memory_context, corpus)
            if event_evidence:
                evidence_parts.append(event_evidence)
                corrections.append("已重新核查球路事件中「未知」标注的判定依据")

        # LLM 生成质疑回复
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "未能提取到相关原始数据"

        system_prompt = """你是足球数据分析师。用户对你的上一个回答提出了质疑。
请基于重新核验的原始数据，诚实、客观、语气克制地回答。

【输出纪律（必须遵守）】
- 严禁使用"您的质疑不成立"、"您错了"、"您的说法不对"等强势否定或居高临下的断言；结论一律用客观中性表述呈现（如"核验数据显示……"、"与您的描述存在差异：……"）。
- 证据层级与结论口径（倾向性结论、口径说明、不确定性声明三者不得互相冲突）：
  1) 证据含争议时刻相关球员的精确位置（包括"相邻样本插值估计"）时：插值坐标就是该时刻的直接核验证据，必须基于该坐标与所在区域给出倾向性结论（与用户说法一致或不一致），并注明口径为"相邻样本插值估计"；不确定性声明仅限"插值估计与直接观测可能存在偏差"一句，不得反过来否定刚给出的结论（严禁出现"无法确认该时刻位置"、"无法排除用户说法的可能性"等与结论冲突的表述）。
  2) 仅当证据明确标注"该帧无位置数据/无法精确核验/仅有最后已知位置"时，才声明"现有数据不足以精确核验"，说明缺少什么数据，仅给出基于其他证据（全场轨迹、球路时间线等）的倾向性说明，不得断言用户质疑错误。
- 全文只允许一个结论方向，最终结论必须与所引用证据一致；严禁引用坐标与区域后又推翻结论，严禁对同一位置给出两个互相矛盾的区域判定。
- 区域判定只使用证据中提供的"所在区域"字段；严禁自行推断中线、禁区线等场地参考线的位置，严禁拿坐标与臆造的参考线比较得出新判定。
- 引用坐标时优先引用证据中提供的场地坐标（米），可附像素坐标。
- 不得用全场平均位置/主要活动区域冒充特定时刻的状态。

格式：
【核验结果】
- 如果用户质疑正确：明确承认并给出修正后的信息
- 如果核验数据与用户说法不一致：客观指出差异并引用具体核验数据
- 如果数据不足以判断：明确声明数据不足及原因

【核验过程】
列出本次核验做了哪些数据回溯操作

要求：专业、诚实、简洁、语气中性克制

当证据包含「球路链条物理证据」时，按以下物理规律判断"触球/传球"与"从头顶经过"：
- 高度轨迹平滑（单次上升-下降抛物线）且方向连续一致 → 倾向一次高空飞行，途中离球最近的球员未必触球
- 高度下降后重新上升、方向/速度在某球员附近突变、或水平距离极近（约1米内）→ 倾向该球员确实触球（头球/停球/摆渡）
- 证据无法完全区分时，如实给出倾向性结论并说明两种可能性及各自依据，不得直接断言「数据不足」"""

        user_msg = f"""用户质疑：{question}

质疑判断：{json.dumps(challenge_result, ensure_ascii=False)}

对话上下文：
{memory_context or '（无）'}

重新核验的原始数据：
{evidence_text}"""

        result = call_llm(system_prompt, user_msg, stream=True)
        if result:
            return result.strip()

        return (
            f"针对您的质疑「{question}」，系统已重新核查原始数据。\n"
            + "\n".join(corrections) if corrections else "未找到可核验的相关数据。"
        )

    def _out_of_range_refusal(
        self,
        frame: int,
        frame_info: dict,
        corpus: PrefixPlayerCorpus,
    ) -> str:
        """越界帧/时刻的拒答文案（带完整证据链）。

        拒答纪律：不只说"不存在"，还要展示数据实际覆盖范围、越界推理
        过程（30fps 换算）与最近合法时刻的可查证指引，让用户能复核拒答理由。

        口径：触发拒答的判据是帧落在"球数据"首末帧之外（_capture_frame），
        故拒答理由严格以球数据覆盖区间为准；球员轨迹覆盖范围仅作补充信息，
        避免"并集口径"与拒答判据自相矛盾。
        """
        ball_ids = sorted(corpus.ball_frames.keys())
        asked = (
            f"所问的\"第{frame_info['time_second']}秒\"按 30帧/秒 换算对应第{frame}帧"
            if frame_info.get("source") == "秒数"
            else f"所问的第{frame}帧对应约第{frame / 30:.1f}秒"
        )

        # 球员轨迹覆盖范围（仅作补充信息分别给出）
        player_frame_ids: set[int] = set()
        for p in corpus.players.values():
            player_frame_ids.update(p.frames)
        if player_frame_ids:
            pmin, pmax = min(player_frame_ids), max(player_frame_ids)
            player_note = (
                f"补充信息：球员轨迹覆盖第{pmin}~{pmax}帧"
                f"（{len(corpus.players)}名球员）；但帧级态势核验以球数据为锚点，"
                f"球数据缺失时无法构建该帧的攻防快照。"
            )
        else:
            player_note = "补充信息：当前语料也没有任何球员轨迹数据。"

        if not ball_ids:
            return (
                f"无法回答该问题：当前语料没有任何球数据，无法核验任意时刻的球状态。\n\n"
                f"推理过程：{asked}，而语料中球数据为空，第{frame}帧自然不存在球数据记录；"
                f"若强行描述该时刻只能是编造，故如实拒答。\n\n"
                f"{player_note}"
            )

        bmin, bmax = ball_ids[0], ball_ids[-1]
        ball_span = f"球数据采样{len(ball_ids)}帧，仅覆盖第{bmin}~{bmax}帧（约第{bmin / 30:.1f}~{bmax / 30:.1f}秒）"

        if frame < bmin:
            direction = (
                f"而第{frame}帧早于球数据首帧（第{bmin}帧，约第{bmin / 30:.1f}秒），"
                f"该时刻尚未进入球数据的观测范围，第{frame}帧无球数据；"
                f"若强行描述该时刻的攻防态势只能是编造，故如实拒答。\n\n"
                f"可查证的最近合法时刻：球数据的最早时刻为约第{bmin / 30:.1f}秒（第{bmin}帧），"
                f"如需了解片段开段的攻防态势，可改问\"第{max(1, int(bmin / 30) + 1)}秒发生了什么\"，"
                f"我会基于真实数据回溯核验后回答。"
            )
        else:
            last_sec = int(bmax / 30)
            direction = (
                f"而数据的末帧仅为第{bmax}帧（约第{bmax / 30:.1f}秒），"
                f"第{frame}帧已超出球数据覆盖范围，第{frame}帧无球数据；"
                f"若强行描述该时刻的攻防态势只能是编造，故如实拒答。\n\n"
                f"可查证的最近合法时刻：球数据的最后时刻为约第{bmax / 30:.1f}秒（第{bmax}帧），"
                f"如需了解片段末段的攻防态势，可改问\"第{last_sec}秒发生了什么\"，"
                f"我会基于真实数据回溯核验后回答。"
            )

        return (
            f"无法回答该问题：球数据仅覆盖第{bmin}~{bmax}帧，第{frame}帧无球数据。\n\n"
            f"数据实际覆盖范围：{ball_span}，按 30fps 换算总时长约{bmax / 30:.1f}秒。\n\n"
            f"推理过程：{asked}，{direction}\n\n"
            f"{player_note}"
        )

    # ── 帧级查询 ──

    def _extract_frame_info(self, question: str) -> dict:
        """从问题中提取帧号/时间点信息。"""
        import re

        # 第N秒 → 帧号
        sec_match = re.search(r"第\s*(\d+(?:\.\d+)?)\s*秒", question)
        if sec_match:
            seconds = float(sec_match.group(1))
            return {"frame": int(seconds * 30), "time_second": seconds, "source": "秒数"}

        # 第N帧 → 帧号
        frame_match = re.search(r"第\s*(\d+)\s*帧", question)
        if frame_match:
            frame = int(frame_match.group(1))
            return {"frame": frame, "time_second": frame / 30, "source": "帧号"}

        return {"frame": None, "time_second": 0, "source": "未提取"}

    def _capture_frame(
        self, frame: int, corpus: PrefixPlayerCorpus,
    ) -> FrameSnapshot | None:
        """捕获指定帧的完整状态快照。

        球位置：原始采样帧命中或相邻采样帧线性插值（与独立金标准
        goldref.ball_at_frame 同口径）；插值区间外（早于首帧/晚于末帧）
        返回 None，由上层如实声明数据不存在。
        """
        ball_frames = corpus.ball_frames
        ball_pos = ball_frames.get(frame)

        # 未命中采样帧 → 相邻原始采样帧线性插值
        if ball_pos is None:
            lo: int | None = None
            hi: int | None = None
            for f in sorted(ball_frames):
                if f <= frame:
                    lo = f
                elif hi is None:
                    hi = f
                    break
            if lo is None or hi is None:
                return None
            p0, p1 = ball_frames[lo], ball_frames[hi]
            t = (frame - lo) / (hi - lo)
            ball_pos = tuple(p0[k] + t * (p1[k] - p0[k]) for k in range(3))

        bx, by, bz = ball_pos

        # 所有球员在该帧的位置
        player_positions: dict[str, dict] = {}
        nearest_player = ""
        nearest_dist = float("inf")

        for pid, player in corpus.players.items():
            pos = self._find_position(player, frame)
            if pos:
                px, py = pos
                dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
                player_positions[player.jersey_label] = {
                    "x": round(px, 1),
                    "y": round(py, 1),
                    "color": player.color,
                    "distance_to_ball": round(dist, 1),
                }
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_player = player.jersey_label

        # 该帧附近的球路事件（去类型化）
        events = []
        ball_analysis = corpus.ball_analysis
        if ball_analysis and ball_analysis.events:
            for e in ball_analysis.events:
                if e.start_frame <= frame <= e.end_frame:
                    start_p = e.nearest_player if e.nearest_player != "?" else "未知"
                    end_p = e.nearest_player_end if e.nearest_player_end != "?" else "未知"
                    events.append({
                        "球路": f"{start_p}→{end_p}",
                        "方向": e.direction,
                    })

        return FrameSnapshot(
            frame=frame,
            time_second=round(frame / 30, 1),
            ball_x=bx,
            ball_y=by,
            ball_z=bz,
            player_positions=player_positions,
            nearest_to_ball=nearest_player,
            events_at_frame=events,
        )

    def _format_snapshot(self, snap: FrameSnapshot) -> str:
        """格式化帧快照为 LLM 可读文本。"""
        lines = [
            f"帧号: {snap.frame}（约第{snap.time_second}秒）",
            f"球位置: (x={snap.ball_x:.0f}, y={snap.ball_y:.0f}, z={snap.ball_z:.0f})",
            f"离球最近球员: {snap.nearest_to_ball}",
        ]

        if snap.events_at_frame:
            lines.append("该帧相关球路：")
            for e in snap.events_at_frame:
                route = e.get("球路", "?→?")
                direction = e.get("方向", "")
                lines.append(f"  - {route}（{direction}）")

        # 球员位置摘要（Top 5 距离最近）
        by_dist = sorted(
            snap.player_positions.items(),
            key=lambda x: x[1]["distance_to_ball"],
        )
        lines.append("附近球员：")
        for jlabel, info in by_dist[:8]:
            lines.append(f"  {jlabel}: 距球{info['distance_to_ball']:.0f}单位")

        return "\n".join(lines)

    def _verify_frame_detail(
        self,
        question: str,
        frame_info: dict,
        corpus: PrefixPlayerCorpus,
        memory_context: str = "",
    ) -> str:
        """帧级详细核验。"""
        frame = frame_info["frame"]
        snap = self._capture_frame(frame, corpus)

        if not snap:
            return self._out_of_range_refusal(frame, frame_info, corpus)

        # LLM 生成帧级描述
        from math import hypot

        from agents.constants import PX_PER_M_X, PX_PER_M_Y
        from agents.player.tracker import _which_zone
        from .tactical_facts import threat_value, tactical_zone

        # 问题中明确索要的球员与该帧球的距离（米，x/y 分轴标定）
        queried_dists: dict[str, str] = {}
        for j in self._extract_all_jerseys(question):
            info = snap.player_positions.get(j)
            if info is not None:
                dx = (info["x"] - snap.ball_x) / PX_PER_M_X
                dy = (info["y"] - snap.ball_y) / PX_PER_M_Y
                queried_dists[j] = f"{hypot(dx, dy):.1f}米"

        payload_data: dict[str, Any] = {
            "帧号": snap.frame,
            "时间": f"第{snap.time_second}秒",
            "球位置": f"(x={snap.ball_x:.0f}, y={snap.ball_y:.0f}, z={snap.ball_z:.0f})",
            "战术区域": tactical_zone(snap.ball_x, snap.ball_y),
            "威胁值": threat_value(snap.ball_x, snap.ball_y),
            "离球最近": snap.nearest_to_ball,
            "该帧事件": snap.events_at_frame,
            "球员位置概要": {
                j: f"距球{d['distance_to_ball']:.0f}"
                for j, d in sorted(
                    snap.player_positions.items(),
                    key=lambda x: x[1]["distance_to_ball"],
                )[:5]
            },
        }
        # 全队站位（按队伍分组，含区域标注）：帧类开放描述题需要覆盖双方
        # 阵型分布，仅提供离球 Top5 会让 LLM 误判"数据未提供全部球员"
        def _jersey_key(label: str) -> int:
            digits = "".join(c for c in label if c.isdigit())
            return int(digits) if digits else 9999

        team_view: dict[str, list[str]] = {}
        for j, d in snap.player_positions.items():
            team_view.setdefault(f"{d.get('color') or '?'}队", []).append(
                f"{j}（{_which_zone(d['x'], d['y'])}，距球{d['distance_to_ball']:.0f}）"
            )
        for members in team_view.values():
            members.sort(key=_jersey_key)
        payload_data["全队站位"] = team_view
        payload_data["该帧可见球员数"] = len(snap.player_positions)
        all_labels = {p.jersey_label for p in corpus.players.values()}
        missing = sorted(
            all_labels - set(snap.player_positions.keys()), key=_jersey_key,
        )
        if missing:
            payload_data["该帧未出现的球员"] = missing
        if queried_dists:
            payload_data["所询球员与球距离(米)"] = queried_dists

        user_msg = json.dumps(
            {
                "原始问题": question,
                "核验数据": payload_data,
            },
            ensure_ascii=False,
            indent=2,
        )

        result = call_llm(VERIFY_SYSTEM_PROMPT, user_msg, stream=True)
        if result:
            process_note = (
                f"\n\n[核验过程：回溯原始数据 → 定位第{snap.frame}帧"
                f"（{snap.time_second}秒）→ 提取{len(snap.player_positions)}名球员位置 + "
                f"球坐标 + {len(snap.events_at_frame)}条事件]"
            )
            return result.strip() + process_note

        return self._format_snapshot(snap)

    # ── 球员核验 ──

    def _extract_jersey_from_text(self, text: str) -> str:
        """从文本提取球衣号。
        注意：避免将"第10秒"/"第10帧"中的数字误识别为球衣号。
        """
        import re
        # 先排除含"秒"或"帧"的数字，再匹配 "X号"（后面不跟"秒"或"帧"）
        match = re.search(r"(\d+)\s*号(?!\d|秒|帧)", text)
        if match:
            return f"{match.group(1)}号"
        # 兜底：匹配裸 "X号"（无后缀）
        match = re.search(r"(\d+)号", text)
        return f"{match.group(1)}号" if match else ""

    def _extract_all_jerseys(self, text: str) -> list[str]:
        """从文本提取全部球衣号（去重、保持出现顺序）。

        质疑中常涉及多名球员（如"6到14号中间"），只取第一个会丢失
        球路的起点/终点，导致证据收集不完整。
        支持写法："6号"/"6到14号"/"6-14号"/"6→14号"/"6->7->10->14"。
        """
        import re
        if not text:
            return []
        jerseys: list[str] = []
        # 区间写法："6到14号"/"6-14号"/"6→14号" —— 首尾号码都保留
        for m in re.finditer(r"(\d+)\s*(?:到|至|→|->|-|—|~)\s*(\d+)\s*号", text):
            jerseys.extend([f"{m.group(1)}号", f"{m.group(2)}号"])
        # 箭头链条写法（可不带"号"）："6->7->10->14"
        for m in re.finditer(r"(\d+)\s*(?:→|->)\s*(\d+)(?!\d)", text):
            jerseys.extend([f"{m.group(1)}号", f"{m.group(2)}号"])
        # 常规"X号"写法
        jerseys.extend(f"{m}号" for m in re.findall(r"(\d+)\s*号", text))
        return list(dict.fromkeys(jerseys))

    # ── 球路链条物理证据 ──

    def _extract_route_evidence(
        self,
        question: str,
        corpus: PrefixPlayerCorpus,
        memory_context: str = "",
        context_text: str = "",
    ) -> str:
        """提取球路链条的物理证据（高度轨迹 + 运动方向 + 逐采样点最近球员）。

        用于核验"X到Y中间是不是高空球/从头顶经过/是否传球"类质疑：
        - 定位争议球路的时间窗口（起点球员最后触球 → 终点球员首次接球）
        - 汇总窗口内逐段球路、逐采样点高度、离球最近球员及水平距离
        - 给出中性的物理判定提示，最终结论交由 LLM 判断
        """
        import re
        from math import hypot
        from agents.constants import (
            GROUND_Z_THRESHOLD,
            LOW_BALL_Z_MAX,
            PIXELS_PER_METER,
        )
        from agents.player.tracker import positions_at_frame

        analysis = corpus.ball_analysis
        if not analysis or not analysis.events or not corpus.ball_frames:
            return ""
        events = analysis.events

        # 提及球员：优先取问题中的全部球衣号；不足 2 人时从最近回答补充
        jerseys = self._extract_all_jerseys(question)
        if len(jerseys) < 2 and context_text:
            for j in self._extract_all_jerseys(context_text):
                if j not in jerseys:
                    jerseys.append(j)
                if len(jerseys) >= 4:
                    break

        route_kw = re.search(
            r"高空|头顶|传球|球路|路径|经过|中间|争顶|头球|挑传|长传", question,
        )
        if len(jerseys) < 2 and not route_kw:
            return ""

        # ── 确定争议球路的时间窗口 ──
        def _involves(e: BallEvent, j: str) -> bool:
            return e.nearest_player == j or e.nearest_player_end == j

        lo: int | None = None
        hi: int | None = None
        if len(jerseys) >= 2:
            first, last = jerseys[0], jerseys[-1]
            ev_first = [e for e in events if _involves(e, first)]
            ev_last = [e for e in events if _involves(e, last)]
            if ev_first and ev_last:
                # 起点球员在终点球员首次出现前的最后一次触球 → 终点球员首次接球
                last_min_start = min(e.start_frame for e in ev_last)
                before = [e.end_frame for e in ev_first if e.end_frame <= last_min_start]
                lo = max(before) if before else min(e.start_frame for e in ev_first)
                after = [e.start_frame for e in ev_last if e.start_frame >= lo]
                hi = min(after) if after else max(e.end_frame for e in ev_first)
        if lo is None or hi is None:
            related = [e for e in events if any(_involves(e, j) for j in jerseys)]
            if not related:
                return ""
            lo = min(e.start_frame for e in related)
            hi = max(e.end_frame for e in related)

        # 窗口向后延伸约 0.5 秒，覆盖球落地/接球瞬间
        hi_ext = hi + 15
        chain = [e for e in events if e.end_frame >= lo and e.start_frame <= hi_ext]
        samples = [
            (f, p) for f, p in sorted(corpus.ball_frames.items())
            if lo <= f <= hi_ext
        ]
        if not chain or not samples:
            return ""

        def _height_label(z: float) -> str:
            if z <= GROUND_Z_THRESHOLD:
                return "地面"
            if z <= LOW_BALL_Z_MAX:
                return "低平"
            return "高空"

        jersey_of = {tid: p.jersey_label for tid, p in corpus.players.items()}
        team_of = {p.jersey_label: p.color for p in corpus.players.values()}

        lines: list[str] = [
            f"球路链条物理证据（第{lo / 30:.1f}秒 ~ 第{hi_ext / 30:.1f}秒，"
            f"窗口内共{len(chain)}段球路）：",
            "【逐段球路】",
        ]
        for e in chain[:20]:
            zs = [p[2] for f, p in samples if e.start_frame <= f <= e.end_frame]
            z_max = max(zs) if zs else e.end_pos[2]
            lines.append(
                f"  - {e.start_frame / 30:.1f}秒 {e.nearest_player}→{e.nearest_player_end}"
                f"（{e.direction}，终点高度{e.end_pos[2]:.1f}米"
                f"[{_height_label(e.end_pos[2])}]，段内峰值{z_max:.1f}米）"
            )

        # ── 逐采样点高度 + 离球最近球员（触球/头顶经过判定的核心物理证据）──
        lines.append("【逐采样点高度与最近球员】（球轨迹采样，水平距离已换算为米）")
        z_series: list[float] = []
        nearest_series: list[str] = []
        sample_lines: list[str] = []
        for f, (bx, by, bz) in samples:
            z_series.append(bz)
            players_at = positions_at_frame(corpus.players, f)
            nearest, ndist = "?", float("inf")
            for pid, (px, py) in players_at.items():
                d = hypot(bx - px, by - py)
                if d < ndist:
                    ndist, nearest = d, jersey_of.get(pid, "?")
            nearest_series.append(nearest)
            sample_lines.append(
                f"  - {f / 30:.1f}秒 高度{bz:.1f}米[{_height_label(bz)}] "
                f"离球最近={nearest}({team_of.get(nearest, '?')}队) "
                f"水平距离约{ndist / PIXELS_PER_METER:.1f}米"
            )
        # 采样点过多时优先保留高空段与首尾（含落地证据）
        if len(sample_lines) > 24:
            head = sample_lines[:3]
            aerial = [
                line for line, (_, (_, _, z)) in zip(sample_lines, samples) if z > 1.0
            ]
            tail = sample_lines[-2:]
            merged = list(dict.fromkeys(head + aerial + tail))
            lines.extend(merged[:24])
            lines.append(f"  - ……（共{len(sample_lines)}个采样点，仅展示低空首尾与高空段）")
        else:
            lines.extend(sample_lines)

        # ── 物理事实汇总 ──
        if z_series:
            peak_z = max(z_series)
            peak_t = samples[z_series.index(peak_z)][0] / 30
            # 重新上升次数：下降后再上升超过 0.5 米（空中多次发力触球的信号）
            re_rises = sum(
                1 for i in range(2, len(z_series))
                if z_series[i] - z_series[i - 1] > 0.5
                and z_series[i - 1] < z_series[i - 2]
            )
            dirs = [e.direction for e in chain if e.direction != "原地"]
            dir_changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
            lines.append("【物理事实汇总】")
            lines.append(
                f"  - 高度轨迹（米）：{' → '.join(f'{z:.1f}' for z in z_series)}"
            )
            lines.append(f"  - 峰值高度 {peak_z:.1f} 米，出现在第 {peak_t:.1f} 秒")
            lines.append(f"  - 下降后再上升（>0.5米）次数：{re_rises}")
            lines.append(
                f"  - 方向序列：{' → '.join(dirs) if dirs else '无'}"
                f"（方向改变 {dir_changes} 次）"
            )
            lines.append(f"  - 离球最近球员序列：{' → '.join(nearest_series)}")

        # ── 中性判定提示（物理规律，供 LLM 参考，不在此下结论）──
        lines += [
            "【判定提示（物理规律，供参考）】",
            "  - 高度轨迹平滑（单次上升-下降）且方向连续一致 → 倾向一次高空飞行，"
            "途中离球最近的球员未必触球，可能只是球从其头顶附近经过",
            "  - 高度下降后重新上升、球在某球员附近方向/速度突变、或水平距离极近"
            "（约1米内） → 倾向该球员确实触球（头球/停球/摆渡）",
            "  - 球员位置为地面投影、球为三维轨迹，无法仅凭数据100%确认触球，"
            "需给出倾向性结论并说明依据",
        ]
        return "\n".join(lines)


    def _extract_event_evidence(
        self,
        question: str,
        memory_context: str,
        corpus: PrefixPlayerCorpus,
    ) -> str:
        """提取球路事件级核查证据。

        当用户质疑"未知球员"等事件标注时，从问题与对话上下文中提取
        时间/球员线索，定位相关事件，给出"未知"标注的判定依据
        （端点最近球员实际距离 vs 控球判定半径）。
        """
        import re
        from agents.constants import MAX_POSSESSION_DIST

        analysis = corpus.ball_analysis
        if not analysis or not analysis.events:
            return ""

        unknown_events = [
            e for e in analysis.events
            if e.nearest_player == "?" or e.nearest_player_end == "?"
        ]
        if not unknown_events:
            return "全部球路事件的端点均有明确球员，不存在「未知」标注。"

        # 从问题 + 对话上下文提取时间（秒）与球员线索
        text = f"{question}\n{memory_context or ''}"
        times = [float(t) for t in re.findall(r"(\d+(?:\.\d+)?)\s*秒", text)]
        jerseys = {f"{m}号" for m in re.findall(r"(\d+)\s*号", text)}

        matched = unknown_events
        if times:
            lo, hi = min(times) - 0.6, max(times) + 0.6
            matched = [
                e for e in unknown_events
                if lo <= e.start_frame / 30 <= hi or lo <= e.end_frame / 30 <= hi
            ] or unknown_events
        if jerseys:
            by_jersey = [
                e for e in matched
                if e.nearest_player in jerseys or e.nearest_player_end in jerseys
            ]
            matched = by_jersey or matched

        lines = [
            f"球路事件中共 {len(unknown_events)} 条含「未知」端点。",
            f"判定规则：事件端点的最近球员距离超过 {MAX_POSSESSION_DIST:.0f} 单位"
            f"（控球判定半径）时，该端被标记为「未知」。",
            "相关事件核查明细：",
        ]
        def _fmt_dist(d: float) -> str:
            # tracker 以 -1 表示该帧无任何球员位置数据
            if d < 0:
                return "该帧无球员位置数据（跟踪缺失）"
            return f"最近球员实际距离{d:.0f}单位(超阈值)"

        for e in matched[:6]:
            seg = (
                f"第{e.start_frame / 30:.1f}秒（帧{e.start_frame}-{e.end_frame}）"
                f"{e.event_type}：{e.nearest_player}→{e.nearest_player_end}（{e.direction}）；"
            )
            if e.nearest_player == "?":
                seg += f"起点{_fmt_dist(e.nearest_dist)}；"
            if e.nearest_player_end == "?":
                seg += f"终点{_fmt_dist(e.end_nearest_dist)}；"
            lines.append("  - " + seg.rstrip("；"))
        return "\n".join(lines)

    def _extract_players_at_frame(
        self, frame: int, jerseys: list[str], corpus: PrefixPlayerCorpus,
    ) -> str:
        """提取争议时刻相关球员的精确位置（相邻样本线性插值）。

        位置类质疑（"X号第N秒明明在Y区域"）的直接证据；球员在该帧
        无位置数据时显式标注"无法精确核验"，供 LLM 如实声明数据不足。
        坐标同时给出像素与场地米制两套口径（评测金标准以米为单位），
        插值标注采用"可作核验依据 + 与直接观测可能存在偏差"的中性口径，
        避免 LLM 把插值证据当作"不可用"而自相矛盾地否定自己的结论。
        """
        from agents.constants import PX_PER_M_X, PX_PER_M_Y
        from agents.player.tracker import position_at_frame, _which_zone

        lines = [f"争议时刻（帧{frame}，约第{frame / 30:.1f}秒）相关球员精确位置："]
        hit = 0
        for jersey in jerseys[:4]:
            player = next(
                (p for p in corpus.players.values() if p.jersey_label == jersey),
                None,
            )
            if player is None or not player.frames:
                continue
            pos = position_at_frame(player, frame)
            if pos is None:
                lines.append(
                    f"  - {jersey}: 该帧无位置数据（首个样本为帧{player.frames[0]}），"
                    "无法精确核验该时刻位置"
                )
            else:
                x, y = pos
                if frame > player.frames[-1]:
                    # last-known 语义：position_at_frame 在 frame 晚于末帧时
                    # 返回最后已知位置而非插值，须如实标注，避免误导 LLM
                    # 把陈旧位置当作争议时刻的插值证据。
                    lines.append(
                        f"  - {jersey}: 坐标(x={x:.0f}, y={y:.0f})，"
                        f"所在区域：{_which_zone(x, y)}（该帧晚于其最后观测帧{player.frames[-1]}，"
                        "仅有最后已知位置，无法核验争议时刻位置）"
                    )
                else:
                    lines.append(
                        f"  - {jersey}: 场地坐标约(x={x / PX_PER_M_X:.1f}米, y={y / PX_PER_M_Y:.1f}米)，"
                        f"像素坐标(x={x:.0f}, y={y:.0f})，"
                        f"所在区域：{_which_zone(x, y)}（相邻样本插值估计，可作为该时刻的核验依据；"
                        "插值与直接观测可能存在偏差）"
                    )
            hit += 1
        return "\n".join(lines) if hit else ""

    def _extract_player_raw_data(
        self, jersey: str, corpus: PrefixPlayerCorpus,
    ) -> str:
        """提取球员的原始轨迹摘要。"""
        for pid, player in corpus.players.items():
            if player.jersey_label == jersey:
                lines = [
                    f"球员: {jersey}",
                    f"颜色: {player.color}",
                    f"总帧数: {len(player.frames)}",
                    f"轨迹范围: x=[{player.x_range[0]:.0f}, {player.x_range[1]:.0f}], y=[{player.y_range[0]:.0f}, {player.y_range[1]:.0f}]",
                    f"主要区域: {player.dominant_zone}",
                    f"平均位置: x={player.avg_x():.0f}, y={player.avg_y():.0f}",
                ]
                return "\n".join(lines)
        return ""

    def _verify_player_detail(
        self, question: str, corpus: PrefixPlayerCorpus, memory_context: str = "",
    ) -> str:
        """球员信息核验（无特定帧时）。"""
        jersey = self._extract_jersey_from_text(question)
        if not jersey:
            # 没有球员号也没有帧号 → 退回 GeneralQA
            from .general_qa import GeneralQAAgent
            return GeneralQAAgent().answer(question, corpus)

        raw = self._extract_player_raw_data(jersey, corpus)
        if not raw:
            return f"未找到{jersey}的相关数据。"

        system_prompt = """你是足球数据核验分析师。
请基于原始轨迹数据核验并回答用户关于球员的问题。
说明你核验了哪些数据。
纯文本输出。"""

        result = call_llm(system_prompt, f"原始数据：\n{raw}\n\n用户问题：{question}", stream=True)
        if result:
            return result.strip() + f"\n\n[核验过程：回溯原始轨迹 → 提取{jersey}的完整轨迹数据]"

        return f"【{jersey}数据核验】\n{raw}"

    # ── 辅助 ──

    def _find_position(
        self, player: PlayerTrajectory, target_frame: int,
    ) -> tuple[float, float] | None:
        """球员在任意帧的估计位置（线性插值，跟踪数据稀疏时不再失真）。"""
        from agents.player.tracker import position_at_frame
        return position_at_frame(player, target_frame)


# ── 工具契约绑定：质疑判断改为 function calling 交付结构化结果 ──
# 模型经 submit_challenge_judgment 工具调用提交判定（字段与原有文本
# JSON 契约一致）→ 既有 json.loads 路径原样复用；模型直接返回文本
# （mock/旧 golden/模型行为差异）时文本解析路径照常工作；LLM 失败时
# 关键词判断兜底不变。
from agents.llm_client import bind_prompt_tools  # noqa: E402
from agents.tools.schemas import build_challenge_tool  # noqa: E402

bind_prompt_tools(
    CHALLENGE_SYSTEM_PROMPT,
    [build_challenge_tool()],
    tool_choice="auto",
)
