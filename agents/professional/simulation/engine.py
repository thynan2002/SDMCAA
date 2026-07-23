"""反事实轨迹模拟引擎。

工作流程：
1. 从干预帧的真实数据快照初始化比赛状态（全员位置 + 球 + 球权）
2. 第一步强制执行用户指定的反事实宏观决策（传球 / 射门 / 盘带 / 解围）
3. 后续每个决策点使用轻量确定性策略（启发式评分 argmax）选择最优动作
4. 全体球员由战术质点模型同步驱动，球由物理模型驱动
5. 输出单一最可能未来轨迹的关键帧序列（直线段只记端点）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from agents.constants import FIELD_WIDTH, FIELD_HEIGHT, PIXELS_PER_METER
from agents.player.tracker import PrefixPlayerCorpus, _which_zone
from agents.progress import push_stage
from ..mcts.node import Action, ActionType
from ..types import PlayerBehaviorModel
from . import physics
from .llm_brain import LLMDecisionEngine, build_decision_snapshot
from .tactical import SimPlayer, TacticalModel

DT_FRAMES = 3                       # 模拟步长（帧），0.1 秒
GOAL_HALF_WIDTH = 40.0              # 球门半宽（px ≈ 3.6m）
GOAL_X = FIELD_WIDTH / 2            # 球门中心 x


@dataclass
class SimulationResult:
    """一次反事实模拟的输出。"""
    start_frame: int = 0
    end_frame: int = 0                # 实际模拟终止帧
    requested_end_frame: int = 0
    player_keyframes: dict[str, list[tuple[int, float, float]]] = field(default_factory=dict)
    ball_keyframes: list[tuple[int, float, float, float]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    outcome: str = ""                 # goal / save_held / miss / out / completed
    scorer: str = ""


class TrajectorySimulator:
    """反事实轨迹模拟器。"""

    def __init__(
        self,
        corpus: PrefixPlayerCorpus,
        behavior_models: dict[str, PlayerBehaviorModel] | None = None,
        llm_engine: LLMDecisionEngine | None = None,
    ) -> None:
        self.corpus = corpus
        # 行为模型以 jersey_label 为键；模拟内部统一用 track_id，按需映射
        self.models = behavior_models or {}
        # LLM 决策引擎（核心决策）；None 时回退启发式
        self.llm_engine = llm_engine

    # ═══════════════════════════════════════════════════════════
    # 公开接口
    # ═══════════════════════════════════════════════════════════

    def simulate(
        self,
        intervention_frame: int,
        forced_action: Action | None = None,
        subject: str = "",
        duration_seconds: float = 0.0,
        scenario_desc: str = "",
    ) -> SimulationResult:
        """从干预帧开始生成单一最可能未来轨迹。

        Args:
            intervention_frame: 反事实干预帧（该帧及之后的数据被生成内容替换）
            forced_action: 第一步强制执行的反事实动作（None = 纯自然推演）
            subject: 反事实主体球员 jersey_label（强制其为第一持球者）
            duration_seconds: 指定生成时长（秒）；<=0 时补全到原始数据结束帧
            scenario_desc: 反事实场景自然语言描述（供 LLM 剧本与决策先验）
        """
        # ── 终止帧 ──
        data_end = self._data_end_frame()
        if duration_seconds > 0:
            end_frame = min(data_end, intervention_frame + int(duration_seconds * physics.FPS))
        else:
            end_frame = data_end
        if end_frame <= intervention_frame:
            end_frame = intervention_frame + DT_FRAMES

        # ── 初始状态 ──
        # players: 干预帧在场球员；pending: 干预后进入视野的球员（首帧时动态加入）
        players, pending = self._build_players(intervention_frame)
        if not players:
            return SimulationResult(
                start_frame=intervention_frame, end_frame=intervention_frame,
                requested_end_frame=end_frame, events=["干预帧无可用球员数据，模拟终止"],
                outcome="completed",
            )
        ball = self._build_ball(intervention_frame)
        attack_direction = self._attack_directions(players)
        tactical = TacticalModel(players, attack_direction, (ball.x, ball.y))

        jersey_to_tid = {p.jersey_label: tid for tid, p in self.corpus.players.items()}
        tid_to_jersey = {tid: jl for jl, tid in jersey_to_tid.items()}

        holder = self._detect_holder(players, ball)
        subject_tid: str | None = None
        if subject and subject in jersey_to_tid:
            subject_tid = jersey_to_tid[subject]
            holder = subject_tid  # 反事实假设隐含主体持球
        elif holder is not None:
            subject_tid = holder  # 未指定主体时以干预帧控球者为主体

        # ── LLM 战术剧本（模拟开始前一次性生成，作为全局决策先验） ──
        if self.llm_engine and self.llm_engine.enabled:
            push_stage("traj_script", "正在生成战术剧本…")
            script_ctx = self._build_script_context(
                players, ball, holder, tid_to_jersey,
                end_frame - intervention_frame,
            )
            self.llm_engine.generate_script(scenario_desc or "自然推演", script_ctx)
            push_stage("traj_script", "战术剧本生成完成", status="done")

        # 初始关键帧
        self._protect_until = -10**9  # 新控球者免抢断保护截止帧（防乒乓式球权转换）
        for p in players.values():
            p.record_keyframe(intervention_frame, force=True)
        ball_kfs: list[tuple[int, float, float, float]] = [
            (intervention_frame, round(ball.x, 1), round(ball.y, 1), round(ball.z, 1))
        ]

        events: list[str] = []
        if forced_action is not None and holder is not None:
            events.append(
                f"帧{intervention_frame}: 【反事实干预】"
                f"{tid_to_jersey.get(holder, holder)} 被迫改变原决策"
            )

        # ── 主循环状态 ──
        ball_status = "controlled" if holder else "loose"
        flight: dict = {"kind": "", "target": None, "goal_dir": 0}
        forced_pending = forced_action
        dribble_left = 0
        dribble_dir = (0.0, 0.0)
        outcome = "completed"
        scorer = ""
        goal_celebrate_left = 0
        wait_logged = False  # "持球等待目标进场"事件只记录一次
        wait_start_frame = intervention_frame
        pass_streak = 0      # 同队连续传球计数（驱动"厌倦因子"防止无限倒脚）
        last_passer: str | None = None   # 上一传球者（抑制立即回传的乒乓球）
        touch_until = -1       # 最小控球时间截止帧（接球后需稳定控球再决策）

        f = intervention_frame
        # 轨迹推演进度推送间隔（约 8 次更新）
        total_sim_frames = max(1, end_frame - intervention_frame)
        traj_report_interval = max(DT_FRAMES, total_sim_frames // 8)
        while f < end_frame and ball_status != "dead":
            dt = DT_FRAMES / physics.FPS

            # 阶段性进度推送（复用 "trajectory" key）
            if (f - intervention_frame) % traj_report_interval == 0:
                pct = round((f - intervention_frame) / total_sim_frames * 100)
                push_stage(
                    "trajectory", "正在生成反事实轨迹",
                    detail=f"{pct}%", percent=pct,
                )

            # 干预后进入视野的球员：到达其原始首帧时动态加入反事实世界
            for tid in list(pending):
                first_frame, px, py = pending[tid]
                if f >= first_frame:
                    players[tid] = self._make_sim_player(tid, px, py)
                    players[tid].record_keyframe(f, force=True)
                    events.append(f"帧{f}: {tid_to_jersey.get(tid, tid)} 进入视野")
                    del pending[tid]

            if goal_celebrate_left > 0:
                # 进球后：全员减速停下，记录静止帧后结束
                goal_celebrate_left -= DT_FRAMES
                tactical.update_targets(None, ball.x, ball.y)
                for p in tactical.players.values():
                    p.tx, p.ty, p.speed = p.x, p.y, physics.WALK_SPEED
                tactical.step(f, dt)
                f += DT_FRAMES
                continue

            if ball_status == "flying":
                ball.step(dt)
                ball_status, outcome, scorer = self._flight_arrival_check(
                    f, ball, flight, players, tactical, attack_direction,
                    tid_to_jersey, events, ball_kfs,
                )
                if ball_status == "controlled":
                    holder = flight.get("new_holder")
                    touch_until = f + 9  # 接球后稳定控球再决策
                    # 记录传球者供回传抑制；拦截得球则无上一传球者
                    last_passer = flight.get("passer") if flight.get("kind") == "pass" else None
                elif ball_status == "dead" and outcome == "goal":
                    goal_celebrate_left = 15
                # 空中球每 2 步记录一次以表现抛物线
                if ball_status == "flying" and (f - intervention_frame) % (DT_FRAMES * 2) == 0 and ball.z > 1:
                    ball_kfs.append((f, round(ball.x, 1), round(ball.y, 1), round(ball.z, 1)))
            elif ball_status == "controlled" and holder is not None:
                hp = players[holder]
                # 反事实强制动作仅能由主体本人执行；球权易主则作废
                if forced_pending is not None and subject_tid is not None and holder != subject_tid:
                    events.append(
                        f"帧{f}: 球权在反事实动作执行前易主，强制动作取消"
                    )
                    forced_pending = None
                if forced_pending is not None:
                    action = forced_pending
                    waiting = False
                    if action.action_type == ActionType.PASS:
                        target_tid = self._resolve_target(
                            action.target_player, holder, players, jersey_to_tid)
                        if target_tid is None:
                            ptid = self._pending_target(action.target_player, pending, jersey_to_tid)
                            if ptid is not None and f - wait_start_frame <= 90:
                                # 传球目标即将进入视野：带球向其接应点推进并保持球权，
                                # 目标进场后立即执行传球（剧本保护：等待期间豁免抢断）
                                waiting = True
                                if not wait_logged:
                                    events.append(
                                        f"帧{f}: {tid_to_jersey.get(holder, holder)} "
                                        f"带球推进，等待 {action.target_player} 进入视野"
                                    )
                                    wait_logged = True
                                self._protect_until = f + 12
                                _, jx, jy = pending[ptid]
                                dx, dy = jx - hp.x, jy - hp.y
                                norm = hypot(dx, dy) or 1.0
                                hp.tx, hp.ty = physics.clip_to_field(
                                    hp.x + dx / norm * 60, hp.y + dy / norm * 60, margin=10)
                                hp.speed = physics.DRIBBLE_SPEED
                            else:
                                reason = "等待超时" if ptid is not None else "不在比赛数据中"
                                events.append(
                                    f"帧{f}: 反事实目标 {action.target_player or '未指定'}"
                                    f"{reason}，改为自然最优决策"
                                )
                                forced_pending = None
                    if not waiting and forced_pending is not None:
                        forced_pending = None
                        ball_status, flight, dribble_left, dribble_dir, events = self._execute_action(
                            f, holder, action, players, ball, attack_direction,
                            jersey_to_tid, tid_to_jersey, events, forced=True,
                        )
                elif dribble_left > 0:
                    dribble_left -= DT_FRAMES
                    hp.tx, hp.ty = physics.clip_to_field(
                        hp.x + dribble_dir[0] * 80, hp.y + dribble_dir[1] * 80, margin=10)
                    hp.speed = physics.DRIBBLE_SPEED
                elif f < touch_until:
                    # 接球后稳定控球：原地护球观察，暂不决策
                    hp.tx, hp.ty, hp.speed = hp.x, hp.y, physics.WALK_SPEED
                else:
                    # ── 决策：LLM 决策引擎优先，不可用时回退启发式 ──
                    action = None
                    decision: dict | None = None
                    if self.llm_engine and self.llm_engine.enabled:
                        decision = self._llm_decide(
                            f, holder, players, ball, attack_direction,
                            tid_to_jersey, scenario_desc, events,
                        )
                        if decision:
                            tactical.set_intent(decision)
                            action = self._action_from_llm(decision, holder, players, attack_direction)
                    if action is None:
                        action = self._deterministic_action(
                            holder, players, ball, attack_direction, pass_streak, last_passer)
                    prev_events = len(events)
                    ball_status, flight, dribble_left, dribble_dir, events = self._execute_action(
                        f, holder, action, players, ball, attack_direction,
                        jersey_to_tid, tid_to_jersey, events, forced=False,
                    )
                    if decision and len(events) > prev_events and decision.get("reason"):
                        events[-1] += f"（{decision['reason']}）"

                # 厌倦因子计数：短传倒脚 +1，盘带/射门/解围清零，控球帧保持不变
                if ball_status == "flying" and flight.get("kind") == "pass":
                    pass_streak += 1
                elif dribble_left > 0 or (ball_status == "flying" and flight.get("kind") in ("shot", "clear")):
                    pass_streak = 0

                # 地面抢断判定（仅球在脚下且持球者不在保护期时）
                if ball_status == "controlled" and f >= self._protect_until:
                    new_holder = self._tackle_check(holder, players, ball, attack_direction)
                    if new_holder and new_holder != holder:
                        events.append(
                            f"帧{f}: {tid_to_jersey.get(new_holder, new_holder)} 完成抢断，球权转换"
                        )
                        holder = new_holder
                        pass_streak = 0
                        last_passer = None
                        touch_until = f + 9
                        self._protect_until = f + 12
                        players[holder].record_keyframe(f, force=True)
            else:  # loose
                chaser = self._nearest_player(players, ball.x, ball.y)
                if chaser and hypot(players[chaser].x - ball.x, players[chaser].y - ball.y) < physics.CONTROL_RADIUS:
                    holder = chaser
                    ball_status = "controlled"
                    ball.stop()
                    last_passer = None
                    touch_until = f + 9
                    self._protect_until = f + 12
                    events.append(f"帧{f}: {tid_to_jersey.get(chaser, chaser)} 拿到失控球")

            # ── 战术模型驱动无球球员，引擎直接驱动持球者与球 ──
            chase_ids: set[str] = set()
            if ball_status == "flying" and flight.get("target"):
                chase_ids = {flight["target"]}
            elif ball_status == "loose":
                c = self._nearest_player(players, ball.x, ball.y)
                if c:
                    chase_ids = {c}
            tactical.update_targets(
                holder if ball_status == "controlled" else None,
                ball.x, ball.y, chase_ids,
            )
            skip = {holder} if ball_status == "controlled" and holder else set()
            tactical.step(f, dt, skip_ids=skip)

            if ball_status == "controlled" and holder is not None:
                hp = players[holder]
                old_x, old_y = hp.x, hp.y
                nx, ny, _ = physics.clamp_step(hp.x, hp.y, hp.tx, hp.ty, hp.speed, dt)
                step = hypot(nx - old_x, ny - old_y)
                hp.moved = step > 0.3
                if hp.moved:
                    hp._cur_dir = ((nx - old_x) / step, (ny - old_y) / step)
                hp.x, hp.y = nx, ny
                hp.record_keyframe(f)
                # 球被带在脚下：以带球速度追向持球者（避免瞬移造成的表观超速）
                bx, by, _ = physics.clamp_step(
                    ball.x, ball.y, hp.x, hp.y,
                    physics.DRIBBLE_SPEED + hp.speed, dt,
                )
                ball.x, ball.y, ball.z = bx, by, 0.0

            f += DT_FRAMES

        # ── 收尾：记录最终帧（球位置压回场地，保证导出坐标合法） ──
        final_frame = min(f, end_frame)
        for p in players.values():
            p.record_keyframe(final_frame, force=True)
        cx, cy = physics.clip_to_field(ball.x, ball.y, margin=0.5)
        ball_kfs.append((final_frame, round(cx, 1), round(cy, 1), round(ball.z, 1)))
        if outcome == "goal":
            events.append(f"帧{final_frame}: 模拟结束（进球后比赛暂停）")

        return SimulationResult(
            start_frame=intervention_frame,
            end_frame=final_frame,
            requested_end_frame=end_frame,
            player_keyframes={tid: p.keyframes for tid, p in players.items()},
            ball_keyframes=_dedup_kfs(ball_kfs),
            events=events,
            outcome=outcome,
            scorer=scorer,
        )

    # ═══════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════

    def _data_end_frame(self) -> int:
        frames: list[int] = list(self.corpus.ball_frames.keys())
        for p in self.corpus.players.values():
            frames.extend(p.frames)
        return max(frames) if frames else 0

    def _build_players(
        self, frame: int,
    ) -> tuple[dict[str, SimPlayer], dict[str, tuple[int, float, float]]]:
        """构建初始球员集合。

        Returns:
            (players, pending):
            players — 干预帧已在场球员 {track_id: SimPlayer}
            pending — 干预帧尚未进入视野、但之后会进场的球员
                      {track_id: (原始首帧, x, y)}，模拟时钟到达其首帧时动态加入
        """
        players: dict[str, SimPlayer] = {}
        pending: dict[str, tuple[int, float, float]] = {}
        for tid, traj in self.corpus.players.items():
            pos = self._pos_at_frame(traj, frame)
            if pos is None:
                if traj.frames and traj.frames[0] > frame:
                    pending[tid] = (traj.frames[0], traj.xs[0], traj.ys[0])
                continue
            players[tid] = self._make_sim_player(tid, pos[0], pos[1])
        return players, pending

    def _make_sim_player(self, tid: str, x: float, y: float) -> SimPlayer:
        traj = self.corpus.players[tid]
        model = self.models.get(traj.jersey_label)
        fwd = model.feature_vector[7] if model and len(model.feature_vector) >= 8 else 0.5
        itc = model.feature_vector[3] if model and len(model.feature_vector) >= 8 else 0.5
        return SimPlayer(
            track_id=tid, color=traj.color or "unknown",
            x=x, y=y, home_x=x, home_y=y, tx=x, ty=y,
            forward_bias=fwd, intercept_tendency=itc,
            is_goalkeeper=traj.is_goalkeeper,
        )

    @staticmethod
    def _pending_target(
        target: str,
        pending: dict[str, tuple[int, float, float]],
        jersey_to_tid: dict[str, str],
    ) -> str | None:
        """判断传球目标是否是即将进场的球员，是则返回其 track_id。"""
        tid = target if target in pending else jersey_to_tid.get(target)
        return tid if tid in pending else None

    def _build_ball(self, frame: int) -> physics.BallPhysics:
        best = -1
        for bf in self.corpus.ball_frames:
            if bf <= frame and bf > best:
                best = bf
        if best < 0 and self.corpus.ball_frames:
            best = min(self.corpus.ball_frames)
        x, y, z = self.corpus.ball_frames.get(best, (FIELD_WIDTH / 2, FIELD_HEIGHT / 2, 0.0))
        # 数据 z 轴为米，模拟内部统一为像素（导出时再换算回米，保持闭环）
        return physics.BallPhysics(x=x, y=y, z=z * PIXELS_PER_METER)

    @staticmethod
    def _pos_at_frame(traj, frame: int) -> tuple[float, float] | None:
        # 任意帧线性插值（跟踪数据稀疏，last-known 会让干预帧快照严重失真）
        from agents.player.tracker import position_at_frame
        return position_at_frame(traj, frame)

    def _attack_directions(self, players: dict[str, SimPlayer]) -> dict[str, int]:
        """判定各队进攻方向。

        硬约束：不同队伍必须攻相反方向。按各队干预帧平均 y 排序交替分配
        （平均 y 偏小的队守 y=0 球门、攻 +y）。
        """
        sums: dict[str, list[float]] = {}
        for p in players.values():
            sums.setdefault(p.color, []).append(p.home_y)
        avgs = {c: sum(ys) / len(ys) for c, ys in sums.items()}
        directions: dict[str, int] = {}
        sign = 1
        for color in sorted(avgs, key=lambda c: avgs[c]):
            directions[color] = sign
            sign = -sign
        return directions

    def _detect_holder(self, players: dict[str, SimPlayer], ball: physics.BallPhysics) -> str | None:
        best_tid, best_d = None, float("inf")
        for tid, p in players.items():
            d = hypot(p.x - ball.x, p.y - ball.y)
            if d < best_d:
                best_tid, best_d = tid, d
        return best_tid if best_d <= 60 else None

    @staticmethod
    def _nearest_player(players: dict[str, SimPlayer], x: float, y: float,
                        exclude: str | None = None) -> str | None:
        best_tid, best_d = None, float("inf")
        for tid, p in players.items():
            if tid == exclude:
                continue
            d = hypot(p.x - x, p.y - y)
            if d < best_d:
                best_tid, best_d = tid, d
        return best_tid

    # ═══════════════════════════════════════════════════════════
    # LLM 决策（核心决策引擎；快照 → LLM → 决策指令）
    # ═══════════════════════════════════════════════════════════

    def _llm_decide(
        self,
        frame: int,
        holder: str,
        players: dict[str, SimPlayer],
        ball: physics.BallPhysics,
        attack_direction: dict[str, int],
        tid_to_jersey: dict[str, str],
        scenario_desc: str,
        events: list[str],
    ) -> dict | None:
        """构建语义快照并调用 LLM 决策引擎。"""
        hp = players[holder]
        teammates: list[dict] = []
        opponents: list[dict] = []
        for tid, p in players.items():
            if tid == holder:
                continue
            d = hypot(p.x - hp.x, p.y - hp.y)
            if p.color == hp.color:
                opp_near = min(
                    (hypot(q.x - p.x, q.y - p.y) for q in players.values() if q.color != hp.color),
                    default=150.0,
                )
                teammates.append({
                    "jersey": tid_to_jersey.get(tid, tid),
                    "x": p.x, "y": p.y,
                    "dist_to_holder": d, "openness": opp_near,
                })
            else:
                opponents.append({
                    "jersey": tid_to_jersey.get(tid, tid),
                    "x": p.x, "y": p.y, "dist_to_holder": d,
                })
        opponents.sort(key=lambda o: o["dist_to_holder"])
        pressure = opponents[0]["dist_to_holder"] if opponents else 999.0
        model = self.models.get(tid_to_jersey.get(holder, holder))
        snapshot = build_decision_snapshot(
            frame,
            tid_to_jersey.get(holder, holder),
            hp.color,
            model.style_label if model else "",
            (hp.x, hp.y),
            pressure,
            teammates,
            opponents,
            attack_direction.get(hp.color, 1),
            events,
            scenario_desc,
        )
        return self.llm_engine.decide(snapshot)

    def _action_from_llm(
        self,
        decision: dict,
        holder: str,
        players: dict[str, SimPlayer],
        attack_direction: dict[str, int],
    ) -> Action:
        """把 LLM 决策指令映射为引擎 Action。"""
        atk_dir = attack_direction.get(players[holder].color, 1)
        act = decision["action"]
        if act == "pass":
            return Action(ActionType.PASS, target_player=decision["target"])
        if act == "shoot":
            return Action(ActionType.SHOOT)
        if act == "dribble":
            # 方向以球员视角（面向进攻方向）：左 = 攻+y 时 -x / 攻-y 时 +x
            side = {"left": -0.45, "right": 0.45}.get(decision.get("direction", "forward"), 0.0)
            dx, dy = side * atk_dir, 1.0 * atk_dir
            norm = hypot(dx, dy)
            return Action(ActionType.DRIBBLE, direction=(dx / norm, dy / norm))
        if act == "clear":
            return Action(ActionType.CLEAR)
        return Action(ActionType.HOLD)

    def _build_script_context(
        self,
        players: dict[str, SimPlayer],
        ball: physics.BallPhysics,
        holder: str | None,
        tid_to_jersey: dict[str, str],
        total_frames: int,
    ) -> dict:
        """组装 LLM 剧本生成的比赛上下文。"""
        teams: dict[str, list[str]] = {}
        for tid, p in players.items():
            teams.setdefault(p.color, []).append(tid_to_jersey.get(tid, tid))
        styles = {
            jl: m.style_label for jl, m in self.models.items() if m.style_label
        }
        return {
            "参赛队伍": teams,
            "球员风格": styles,
            "初始持球者": tid_to_jersey.get(holder, holder) if holder else "无",
            "初始局面": f"球位于{_which_zone(ball.x, ball.y)}",
            "模拟时长": f"{total_frames / 30:.1f}秒",
        }

    # ═══════════════════════════════════════════════════════════
    # 决策（轻量确定性策略：评分 argmax）—— LLM 不可用时的回退
    # ═══════════════════════════════════════════════════════════

    def _deterministic_action(
        self,
        holder: str,
        players: dict[str, SimPlayer],
        ball: physics.BallPhysics,
        attack_direction: dict[str, int],
        pass_streak: int = 0,
        last_passer: str | None = None,
    ) -> Action:
        hp = players[holder]
        atk_dir = attack_direction.get(hp.color, 1)
        goal_y = FIELD_HEIGHT if atk_dir > 0 else 0.0
        goal_dist = hypot(hp.x - GOAL_X, hp.y - goal_y)

        model = self.models.get(self.corpus.players[holder].jersey_label)
        w = model.action_weights if model and model.action_weights else {}
        pref = model.pass_target_preference if model else {}

        # 压迫度：最近防守者距离
        opp_dist = min(
            (hypot(p.x - hp.x, p.y - hp.y) for p in players.values() if p.color != hp.color),
            default=999.0,
        )
        pressure_boost = 1.25 if opp_dist < 60 else 1.0

        candidates: list[tuple[float, int, Action]] = []  # (score, tiebreak, action)

        # 射门
        if goal_dist < 380 and ((atk_dir > 0 and hp.y > 350) or (atk_dir < 0 and hp.y < 350)):
            score = (0.55 * (1 - goal_dist / 400) + 0.9 * w.get("shoot", 0.15) + 0.1) * pressure_boost
            candidates.append((score, 3, Action(ActionType.SHOOT)))

        # 传球（逐队友评分）
        for tid, tp in players.items():
            if tid == holder or tp.color != hp.color:
                continue
            dist = hypot(tp.x - hp.x, tp.y - hp.y)
            if dist < 20:
                continue
            fwd_gain = max(-1.0, min(1.0, (tp.y - hp.y) * atk_dir / 300))
            opp_near = min(
                (hypot(p.x - tp.x, p.y - tp.y) for p in players.values() if p.color != hp.color),
                default=150.0,
            )
            openness = min(1.0, opp_near / 150)
            # 传球路径通畅度：路径上有防守者站位则大幅降分（避免无脑送截）
            lane = _lane_openness(hp.x, hp.y, tp.x, tp.y, players, hp.color)
            score = (
                0.7 * w.get("pass", 0.4)
                + 0.5 * pref.get(self.corpus.players[tid].jersey_label, 0.1)
                + 0.2 * fwd_gain + 0.2 * openness
            ) * pressure_boost
            score *= 0.35 + 0.65 * lane
            if dist > 500:
                score *= 0.6
            # 向前长传转移奖励：倒脚局中打破平衡、推进比赛节奏
            if dist > 300 and fwd_gain > 0.2:
                score *= 1.5
            # 回传抑制：立即回传给上一传球者形成"乒乓球"，大幅降分
            if tid == last_passer:
                score *= 0.3
            # 明显向后回传（非战术需要时）降分
            if fwd_gain < -0.2:
                score *= 0.7
            # 厌倦因子：同队连续安全倒脚越多，短传评分指数衰减
            if pass_streak >= 4:
                score *= 0.6 ** (pass_streak - 3)
            candidates.append((score, 2, Action(ActionType.PASS, target_player=tid)))

        # 盘带：攻方前方空间
        space_ahead = all(
            not (p.color != hp.color
                 and (p.y - hp.y) * atk_dir > 0 and (p.y - hp.y) * atk_dir < 110
                 and abs(p.x - hp.x) < 60)
            for p in players.values()
        )
        drb_score = 0.8 * w.get("dribble", 0.25) + (0.35 if space_ahead else -0.1) + 0.1 * (1 - goal_dist / 700)
        if opp_dist < 40:
            drb_score *= 0.4  # 被贴身压迫时不宜强行盘带（避免抢断循环）
        if pass_streak >= 4:
            drb_score *= 1.5  # 倒脚过多时鼓励向前推进
        candidates.append((drb_score, 1, Action(ActionType.DRIBBLE)))

        # 解围：在本方禁区附近且受压
        own_goal_y = 0.0 if atk_dir > 0 else FIELD_HEIGHT
        own_goal_dist = hypot(hp.x - GOAL_X, hp.y - own_goal_y)
        if own_goal_dist < 220 and opp_dist < 70:
            candidates.append((0.5 + 0.3 * w.get("clear", 0.1), 4, Action(ActionType.CLEAR)))

        # 等待（兜底）
        candidates.append((0.08 + 0.2 * w.get("hold", 0.1), 0, Action(ActionType.HOLD)))

        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        return candidates[0][2]

    # ═══════════════════════════════════════════════════════════
    # 动作执行
    # ═══════════════════════════════════════════════════════════

    def _execute_action(
        self,
        frame: int,
        holder: str,
        action: Action,
        players: dict[str, SimPlayer],
        ball: physics.BallPhysics,
        attack_direction: dict[str, int],
        jersey_to_tid: dict[str, str],
        tid_to_jersey: dict[str, str],
        events: list[str],
        forced: bool,
    ) -> tuple[str, dict, int, tuple[float, float], list[str]]:
        """执行宏观决策动作，返回 (ball_status, flight, dribble_left, dribble_dir, events)。"""
        hp = players[holder]
        atk_dir = attack_direction.get(hp.color, 1)
        tag = "【反事实】" if forced else ""
        hj = tid_to_jersey.get(holder, holder)
        flight: dict = {"kind": "", "target": None, "goal_dir": 0}
        dribble_left = 0
        dribble_dir = (0.0, 0.0)

        if action.action_type == ActionType.PASS:
            target_tid = self._resolve_target(action.target_player, holder, players, jersey_to_tid)
            if target_tid is None:
                # 无合法目标 → 退化为盘带
                action = Action(ActionType.DRIBBLE)
            else:
                tp = players[target_tid]
                # 提前量：目标当前位置沿攻方向提前
                aim_x, aim_y = tp.x, tp.y + atk_dir * 15
                dist = hypot(aim_x - ball.x, aim_y - ball.y)
                aerial = dist > 250
                speed = physics.PASS_SPEED_LONG if aerial else physics.PASS_SPEED_SHORT
                ball.launch(aim_x, aim_y, speed, aerial=aerial)
                tj = tid_to_jersey.get(target_tid, target_tid)
                events.append(f"帧{frame}: {tag}{hj} {'长传' if aerial else '短传'}给 {tj}")
                flight = {"kind": "pass", "target": target_tid, "goal_dir": 0,
                          "aim": (aim_x, aim_y), "passer": holder}
                hp.record_keyframe(frame, force=True)
                return "flying", flight, dribble_left, dribble_dir, events

        if action.action_type == ActionType.SHOOT:
            goal_y = float(FIELD_HEIGHT if atk_dir > 0 else 0)
            # 瞄准球门中心略偏（避开正中门将），确定性选择远角
            aim_x = GOAL_X + (GOAL_HALF_WIDTH * 0.6 if hp.x <= GOAL_X else -GOAL_HALF_WIDTH * 0.6)
            dist = hypot(aim_x - ball.x, goal_y - ball.y)
            if dist > 380:
                # 超远距离强行射门：确定性打飞（偏出立柱）
                aim_x = GOAL_X + (GOAL_HALF_WIDTH + 30) * (1 if hp.x <= GOAL_X else -1)
            ball.launch(aim_x, goal_y, physics.SHOT_SPEED, aerial=dist > 220)
            events.append(f"帧{frame}: {tag}{hj} 起脚射门（距门约 {dist / PIXELS_PER_METER:.0f} 米）")
            flight = {"kind": "shot", "target": None, "goal_dir": atk_dir}
            hp.record_keyframe(frame, force=True)
            return "flying", flight, dribble_left, dribble_dir, events

        if action.action_type == ActionType.CLEAR:
            aim_x = min(max(ball.x + (200 if ball.x < GOAL_X else -200), 100), FIELD_WIDTH - 100)
            aim_y = ball.y + atk_dir * 400
            ball.launch(aim_x, min(max(aim_y, 50), FIELD_HEIGHT - 50),
                        physics.PASS_SPEED_LONG, aerial=True)
            events.append(f"帧{frame}: {tag}{hj} 大脚解围")
            flight = {"kind": "clear", "target": None, "goal_dir": 0}
            hp.record_keyframe(frame, force=True)
            return "flying", flight, dribble_left, dribble_dir, events

        if action.action_type == ActionType.DRIBBLE:
            # 优先采用决策指定方向（LLM），零向量时自动选空档侧
            dx, dy = action.direction
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                dribble_dir = self._dribble_direction(hp, players, atk_dir)
            else:
                norm = hypot(dx, dy) or 1.0
                dribble_dir = (dx / norm, dy / norm)
            dribble_left = 15
            hp.tx, hp.ty = physics.clip_to_field(
                hp.x + dribble_dir[0] * 80, hp.y + dribble_dir[1] * 80, margin=10)
            hp.speed = physics.DRIBBLE_SPEED
            events.append(f"帧{frame}: {tag}{hj} 盘带推进")
            hp.record_keyframe(frame, force=True)
            return "controlled", flight, dribble_left, dribble_dir, events

        # HOLD：原地观察
        hp.tx, hp.ty, hp.speed = hp.x, hp.y, physics.WALK_SPEED
        return "controlled", flight, dribble_left, dribble_dir, events

    def _resolve_target(
        self,
        target: str,
        holder: str,
        players: dict[str, SimPlayer],
        jersey_to_tid: dict[str, str],
    ) -> str | None:
        """把动作目标（jersey_label 或 track_id）解析为合法队友 track_id。"""
        if target in players and target != holder:
            return target
        if target in jersey_to_tid and jersey_to_tid[target] in players:
            tid = jersey_to_tid[target]
            if tid != holder:
                return tid
        return None

    def _dribble_direction(
        self, hp: SimPlayer, players: dict[str, SimPlayer], atk_dir: int,
    ) -> tuple[float, float]:
        """确定性选择盘带方向：攻方向 + 防守较空的一侧。"""
        left_block = sum(
            1 for p in players.values()
            if p.color != hp.color and 0 < (p.y - hp.y) * atk_dir < 120 and -80 < p.x - hp.x < 0
        )
        right_block = sum(
            1 for p in players.values()
            if p.color != hp.color and 0 < (p.y - hp.y) * atk_dir < 120 and 0 < p.x - hp.x < 80
        )
        side = -0.45 if left_block < right_block else 0.45
        dx, dy = side, 1.0 * atk_dir
        norm = hypot(dx, dy)
        return dx / norm, dy / norm

    # ═══════════════════════════════════════════════════════════
    # 飞行到达 / 对抗判定
    # ═══════════════════════════════════════════════════════════

    def _flight_arrival_check(
        self,
        frame: int,
        ball: physics.BallPhysics,
        flight: dict,
        players: dict[str, SimPlayer],
        tactical: TacticalModel,
        attack_direction: dict[str, int],
        tid_to_jersey: dict[str, str],
        events: list[str],
        ball_kfs: list[tuple[int, float, float, float]],
    ) -> tuple[str, str, str]:
        """检查飞行球的到达/被截/进球/出界，返回 (ball_status, outcome, scorer)。"""
        kind = flight.get("kind", "pass")

        # 传球/解围飞行途中：防守者低空拦截（保护期内不触发）
        if kind in ("pass", "clear") and ball.z < 30 and frame >= self._protect_until:
            for tid, p in players.items():
                if flight.get("target") == tid or p.is_goalkeeper:
                    continue
                if hypot(p.x - ball.x, p.y - ball.y) < physics.INTERCEPT_RADIUS:
                    target_tid = flight.get("target")
                    same_team = target_tid and players[target_tid].color == p.color
                    if not same_team:
                        events.append(f"帧{frame}: {tid_to_jersey.get(tid, tid)} 空中拦截成功")
                        ball.stop()  # 球停在当前位置，不瞬移吸附（保持速度连续）
                        self._protect_until = frame + 12
                        p.record_keyframe(frame, force=True)
                        ball_kfs.append((frame, round(ball.x, 1), round(ball.y, 1), 0.0))
                        flight["new_holder"] = tid
                        return "controlled", "", ""

        # 射门：先判门线（避免越线一步后被误判为普通出界）
        if kind == "shot":
            goal_dir = flight.get("goal_dir", 1)
            goal_line = FIELD_HEIGHT if goal_dir > 0 else 0
            crossed = (goal_dir > 0 and ball.y >= goal_line - 2) or (goal_dir < 0 and ball.y <= 2)
            if crossed:
                cx, cy = physics.clip_to_field(ball.x, ball.y, margin=0.5)
                ball_kfs.append((frame, round(cx, 1), round(cy, 1), round(ball.z, 1)))
                if abs(ball.x - GOAL_X) < GOAL_HALF_WIDTH and ball.z < 26:
                    events.append(f"帧{frame}: 球进了！GOAL！")
                    return "dead", "goal", ""
                events.append(f"帧{frame}: 射门偏出球门")
                return "dead", "miss", ""

        # 出界判定（记录点压回边线，保证导出坐标合法）
        if not physics.in_field(ball.x, ball.y, margin=0):
            events.append(f"帧{frame}: 球出界，模拟停止")
            cx, cy = physics.clip_to_field(ball.x, ball.y, margin=0.5)
            ball_kfs.append((frame, round(cx, 1), round(cy, 1), round(ball.z, 1)))
            return "dead", "out", ""

        if kind == "shot":
            # 门将扑救
            for tid, p in players.items():
                if p.is_goalkeeper and hypot(p.x - ball.x, p.y - ball.y) < 45 and ball.z < 40:
                    events.append(f"帧{frame}: {tid_to_jersey.get(tid, tid)}（门将）将球没收")
                    ball.stop()
                    self._protect_until = frame + 12
                    p.record_keyframe(frame, force=True)
                    ball_kfs.append((frame, round(ball.x, 1), round(ball.y, 1), 0.0))
                    flight["new_holder"] = tid
                    return "controlled", "save_held", ""

        # 传球/解围到达
        if kind in ("pass", "clear"):
            target_tid = flight.get("target")
            if target_tid and target_tid in players:
                tp = players[target_tid]
                if hypot(tp.x - ball.x, tp.y - ball.y) < physics.RECEIVE_RADIUS and ball.z < 25:
                    events.append(f"帧{frame}: {tid_to_jersey.get(target_tid, target_tid)} 接球")
                    ball.stop()  # 停球在当前位置（距接球者 <34px），后续由带球跟随自然过渡
                    self._protect_until = frame + 12
                    tp.record_keyframe(frame, force=True)
                    ball_kfs.append((frame, round(ball.x, 1), round(ball.y, 1), 0.0))
                    flight["new_holder"] = target_tid
                    return "controlled", "", ""
            # 球减速到接近停止 → 失控
            if ball.z == 0 and ball.speed() < 60:
                return "loose", "", ""

        return "flying", "", ""

    def _tackle_check(
        self,
        holder: str,
        players: dict[str, SimPlayer],
        ball: physics.BallPhysics,
        attack_direction: dict[str, int],
    ) -> str | None:
        """地面抢断：防守者进入持球者控球半径则完成抢断。"""
        hp = players[holder]
        for tid, p in players.items():
            if p.color == hp.color or p.is_goalkeeper:
                continue
            if hypot(p.x - ball.x, p.y - ball.y) < physics.TACKLE_RADIUS:
                ball.x, ball.y = p.x, p.y
                return tid
        return None


def _lane_openness(
    x1: float, y1: float, x2: float, y2: float,
    players: dict[str, SimPlayer], holder_color: str,
) -> float:
    """传球路径通畅度 0-1：传球线段与防守者的最近距离归一化（45px 为完全通畅）。

    忽略距接球点 60px 内的防守者（贴身盯防由接球开阔度因子另行处理）。
    """
    seg_len = hypot(x2 - x1, y2 - y1)
    if seg_len < 1e-6:
        return 1.0
    best = float("inf")
    for p in players.values():
        if p.color == holder_color:
            continue
        if hypot(p.x - x2, p.y - y2) < 60:
            continue
        # 点到线段的最短距离
        t = ((p.x - x1) * (x2 - x1) + (p.y - y1) * (y2 - y1)) / (seg_len * seg_len)
        t = max(0.0, min(1.0, t))
        d = hypot(p.x - (x1 + t * (x2 - x1)), p.y - (y1 + t * (y2 - y1)))
        best = min(best, d)
    return min(1.0, best / 45) if best != float("inf") else 1.0


def _dedup_kfs(kfs: list[tuple[int, float, float, float]]) -> list[tuple[int, float, float, float]]:
    """去除连续重复帧（保留每帧最后一次记录）。"""
    out: list[tuple[int, float, float, float]] = []
    for kf in kfs:
        if out and out[-1][0] == kf[0]:
            out[-1] = kf
        else:
            out.append(kf)
    return out
