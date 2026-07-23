"""全体球员战术质点运动模型。

将球员视为遵循战术指令的"质点"，模拟球队整体攻防策略风格：

- 控球方（进攻）：无球队员整体随球压上、保持接应三角、前插型球员前压
- 防守方：最近两人上抢持球者，其余收缩防线并随球横移
- 失控球：双方最近球员追球
- 门将：沿门线随球横移

所有移动经 physics.clamp_step 限速，保证运动学合理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from agents.constants import FIELD_WIDTH, FIELD_HEIGHT
from . import physics


@dataclass
class SimPlayer:
    """模拟中的球员质点。"""
    track_id: str
    color: str                       # 队伍标识（A/B/C...）
    x: float
    y: float
    home_x: float                    # 干预帧基准位置（阵型参考点）
    home_y: float
    tx: float = 0.0                  # 当前战术目标点
    ty: float = 0.0
    speed: float = physics.JOG_SPEED
    forward_bias: float = 0.5        # 前插倾向 0-1（来自行为模型）
    intercept_tendency: float = 0.5  # 上抢倾向 0-1
    is_goalkeeper: bool = False

    # 关键帧记录状态
    keyframes: list[tuple[int, float, float]] = field(default_factory=list)
    moved: bool = False              # 本步是否发生位移
    _cur_dir: tuple[float, float] = (0.0, 0.0)
    _last_dir: tuple[float, float] = (0.0, 0.0)
    _last_moving: bool = False
    _last_kf_frame: int = -10**9

    def record_keyframe(self, frame: int, force: bool = False) -> None:
        """在状态变化点记录关键帧（直线段只记端点，中间插值交给下游）。

        触发条件（迟滞设计，避免关键帧爆炸）：
        - force：事件边界（动作开始/结束、球权变化等）
        - 运动方向变化超过约 45°
        - 动 / 停状态切换
        - 距上次记录超过 45 帧（1.5 秒）兜底
        """
        moving = self.moved

        # 同帧已有记录：无条件刷新为该帧最终位置，
        # 否则下一关键帧会累积两步位移造成表观超速
        if self.keyframes and self.keyframes[-1][0] == frame:
            self.keyframes[-1] = (frame, round(self.x, 1), round(self.y, 1))
            self._last_kf_frame = frame
            self._last_moving = moving
            if moving:
                self._last_dir = self._cur_dir
            return

        changed_dir = (
            moving and self._last_moving
            and (self._last_dir[0] * self._cur_dir[0]
                 + self._last_dir[1] * self._cur_dir[1]) < 0.7
        )
        stale = frame - self._last_kf_frame > 45

        if force or changed_dir or moving != self._last_moving or stale:
            self.keyframes.append((frame, round(self.x, 1), round(self.y, 1)))
            self._last_kf_frame = frame

        self._last_moving = moving
        if moving:
            self._last_dir = self._cur_dir


class TacticalModel:
    """全体球员战术运动模型。"""

    def __init__(
        self,
        players: dict[str, SimPlayer],
        attack_direction: dict[str, int],
        ball_home: tuple[float, float],
    ) -> None:
        """
        Args:
            players: {track_id: SimPlayer}
            attack_direction: {color: +1/-1}，+1 表示攻 +y 方向球门
            ball_home: 干预帧球位置（阵型随球平移的参考原点）
        """
        self.players = players
        self.attack_direction = attack_direction
        self.ball_home = ball_home
        # 全队移动意图（由 LLM 决策引擎下发；缺省为中性）
        self.press_level = "mid"      # 防守压迫强度 high/mid/low
        self.attack_focus = "center"  # 进攻侧重 left/center/right
        self.tempo = "normal"         # 比赛节奏 fast/normal

    def set_intent(self, intent: dict | None) -> None:
        """应用 LLM 输出的全队移动意图（每个决策点刷新）。"""
        intent = intent or {}
        press = str(intent.get("press", self.press_level)).lower()
        focus = str(intent.get("attack_focus", self.attack_focus)).lower()
        tempo = str(intent.get("tempo", self.tempo)).lower()
        if press in ("high", "mid", "low"):
            self.press_level = press
        if focus in ("left", "center", "right"):
            self.attack_focus = focus
        if tempo in ("fast", "normal"):
            self.tempo = tempo

    # ── 目标点计算 ─────────────────────────────────────────────

    def update_targets(
        self,
        holder_id: str | None,
        ball_x: float,
        ball_y: float,
        chase_ids: set[str] | None = None,
    ) -> None:
        """根据当前球权状态为全员计算战术目标点与速度档位。

        Args:
            holder_id: 当前控球球员 track_id（None 表示球失控/飞行中）
            ball_x, ball_y: 球当前位置
            chase_ids: 需要直接追球的球员（如传球接球者、失控球最近者）
        """
        chase_ids = chase_ids or set()
        delta_x = ball_x - self.ball_home[0]
        delta_y = ball_y - self.ball_home[1]
        holder_color = (
            self.players[holder_id].color
            if holder_id is not None and holder_id in self.players
            else None
        )

        # 防守方上抢者：人数由 LLM 压迫强度意图决定（high=3/mid=2/low=1）
        pressers: set[str] = set()
        if holder_color is not None:
            press_count = {"high": 3, "mid": 2, "low": 1}.get(self.press_level, 2)
            defenders = [
                (hypot(p.x - ball_x, p.y - ball_y), tid)
                for tid, p in self.players.items()
                if p.color != holder_color and not p.is_goalkeeper
            ]
            defenders.sort()
            pressers = {tid for _, tid in defenders[:press_count]}

        for tid, p in self.players.items():
            if p.is_goalkeeper:
                self._goalkeeper_target(p, ball_x, ball_y)
                continue
            if tid == holder_id:
                # 持球者的目标点由引擎的决策动作控制，此处不覆盖
                continue
            if tid in chase_ids:
                p.tx, p.ty = _clip_x(ball_x), _clip_y(ball_y)
                p.speed = physics.SPRINT_SPEED
                continue

            atk_dir = self.attack_direction.get(p.color, 1)
            if holder_color is not None and p.color == holder_color:
                self._attacker_target(p, atk_dir, delta_x, delta_y, ball_x, ball_y)
            elif holder_color is not None:
                if tid in pressers:
                    p.tx, p.ty = _clip_x(ball_x), _clip_y(ball_y)
                    p.speed = physics.SPRINT_SPEED
                else:
                    self._defender_target(p, atk_dir, delta_x, delta_y, ball_x)
            else:
                # 球失控：非追球者保持阵型随球微调
                p.tx = _clip_x(p.home_x + delta_x * 0.3)
                p.ty = _clip_y(p.home_y + delta_y * 0.3)
                p.speed = physics.JOG_SPEED

    def _attacker_target(
        self, p: SimPlayer, atk_dir: int,
        dx: float, dy: float, ball_x: float, ball_y: float,
    ) -> None:
        """进攻方无球者：整体随球压上 + 前插 + 与球保持接应距离。

        进攻侧重（LLM 意图 attack_focus）驱动横向偏移：left/right 按进攻方向镜像。
        """
        base_x = p.home_x + dx * 0.5
        base_y = p.home_y + dy * 0.5 + atk_dir * p.forward_bias * 90
        # 进攻侧重横向偏移（攻 -y 时左右镜像）
        focus_sign = {"left": -1, "center": 0, "right": 1}.get(self.attack_focus, 0)
        base_x += focus_sign * atk_dir * 90
        # 与球保持最小间距，拉开传球角度（横向错开）
        dist_to_ball = hypot(base_x - ball_x, base_y - ball_y)
        if dist_to_ball < 90:
            side = 1 if base_x >= ball_x else -1
            base_x = ball_x + side * 110
        p.tx, p.ty = _clip_x(base_x), _clip_y(base_y)
        dist_target = hypot(p.tx - p.x, p.ty - p.y)
        p.speed = physics.SPRINT_SPEED if dist_target > 150 else physics.JOG_SPEED
        if self.tempo == "fast":
            p.speed = min(physics.SPRINT_SPEED, p.speed * 1.2)

    def _defender_target(
        self, p: SimPlayer, atk_dir: int,
        dx: float, dy: float, ball_x: float,
    ) -> None:
        """防守方非上抢者：收缩防线（向己方球门退）+ 横向随球。"""
        base_x = p.home_x + dx * 0.3 + (ball_x - FIELD_WIDTH / 2) * 0.1
        base_y = p.home_y + dy * 0.3 - atk_dir * 70  # atk_dir 为攻方方向，退反方向
        p.tx, p.ty = _clip_x(base_x), _clip_y(base_y)
        p.speed = physics.JOG_SPEED

    def _goalkeeper_target(self, p: SimPlayer, ball_x: float, ball_y: float) -> None:
        """门将：锚定其真实把守位置（干预帧 home 点），随球小幅横移。

        不使用理论球门坐标——片段视野可能不含球门（如 21 号把守画面右边缘），
        强行拉向理论门线会导致门将狂奔穿越全场。活动范围限制在 home 附近。
        """
        offset_x = max(-80.0, min(80.0, (ball_x - self.ball_home[0]) * 0.15))
        offset_y = max(-50.0, min(50.0, (ball_y - self.ball_home[1]) * 0.08))
        p.tx = _clip_x(p.home_x + offset_x)
        p.ty = _clip_y(p.home_y + offset_y)
        p.speed = physics.JOG_SPEED

    # ── 步进 ───────────────────────────────────────────────────

    def step(self, frame: int, dt: float, skip_ids: set[str] | None = None) -> None:
        """全员朝各自目标点移动 dt 秒（限速），并按需记录关键帧。"""
        skip_ids = skip_ids or set()
        for tid, p in self.players.items():
            if tid in skip_ids:
                continue
            old_x, old_y = p.x, p.y
            nx, ny, _ = physics.clamp_step(p.x, p.y, p.tx, p.ty, p.speed, dt)
            step_dist = hypot(nx - old_x, ny - old_y)
            p.moved = step_dist > 0.3
            if p.moved:
                p._cur_dir = ((nx - old_x) / step_dist, (ny - old_y) / step_dist)
            p.x, p.y = nx, ny
            p.record_keyframe(frame)


def _clip_x(x: float) -> float:
    return min(max(x, 10.0), FIELD_WIDTH - 10.0)


def _clip_y(y: float) -> float:
    return min(max(y, 10.0), FIELD_HEIGHT - 10.0)
