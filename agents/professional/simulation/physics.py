"""物理校验与运动学约束模块。

为反事实轨迹生成提供物理合理性保障：

- 球员运动学限速：移动速度不超过人类极限（防瞬移）
- 足球飞行物理：z 轴重力、落地反弹、地面滚动摩擦
- 场地边界约束

坐标系：1200 × 700 像素，30fps，PIXELS_PER_METER ≈ 11。
所有速度内部统一为 px/s，位移步进按 dt（秒）计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from agents.constants import FIELD_WIDTH, FIELD_HEIGHT, PIXELS_PER_METER

FPS = 30.0

# ── 球员运动学约束（px/s） ─────────────────────────────────────
MAX_PLAYER_SPEED = 12.0 * PIXELS_PER_METER   # 132：人类极限（校验上限）
SPRINT_SPEED = 9.0 * PIXELS_PER_METER        #  99：冲刺
JOG_SPEED = 6.0 * PIXELS_PER_METER           #  66：慢跑
WALK_SPEED = 3.0 * PIXELS_PER_METER          #  33：走动
DRIBBLE_SPEED = 7.6 * PIXELS_PER_METER       #  ~84：带球推进（略慢于冲刺）

# ── 球体物理（px/s, px/s²） ────────────────────────────────────
GRAVITY = 9.8 * PIXELS_PER_METER             # 107.8：z 轴重力加速度
GROUND_FRICTION = 5.0 * PIXELS_PER_METER     # 55：草地滚动摩擦减速度
BOUNCE_DAMPING = 0.45                        # 落地反弹垂直速度保留比
GROUND_SPIN_RETAIN = 0.8                     # 落地时水平速度保留比
BALL_STOP_SPEED = 8.0                        # 低于此速度视为静止

# ── 球速档位（px/s） ───────────────────────────────────────────
PASS_SPEED_SHORT = 20.0 * PIXELS_PER_METER   # 220：短传地滚球
PASS_SPEED_LONG = 28.0 * PIXELS_PER_METER    # 308：长传/转移
SHOT_SPEED = 32.0 * PIXELS_PER_METER         # 352：射门

# ── 交互判定半径（px） ─────────────────────────────────────────
CONTROL_RADIUS = 22.0        # 可控球距离
RECEIVE_RADIUS = 34.0        # 接球判定距离
TACKLE_RADIUS = 14.0         # 抢断判定距离
INTERCEPT_RADIUS = 26.0      # 空中拦截判定距离


def clamp_step(
    x: float, y: float,
    tx: float, ty: float,
    max_speed: float, dt: float,
) -> tuple[float, float, bool]:
    """从 (x, y) 朝 (tx, ty) 以不超过 max_speed 的速度移动 dt 秒。

    Returns:
        (new_x, new_y, reached) — reached 表示本步到达目标点。
    """
    dx, dy = tx - x, ty - y
    dist = hypot(dx, dy)
    step = max_speed * dt
    if dist <= step or dist < 1e-6:
        return tx, ty, True
    ratio = step / dist
    return x + dx * ratio, y + dy * ratio, False


def validate_player_segment(
    x1: float, y1: float, f1: int,
    x2: float, y2: float, f2: int,
    max_speed: float = MAX_PLAYER_SPEED,
) -> bool:
    """校验球员两关键帧之间的移动是否符合运动学约束（防瞬移）。"""
    if f2 <= f1:
        return False
    dist = hypot(x2 - x1, y2 - y1)
    speed = dist * FPS / (f2 - f1)
    return speed <= max_speed + 1e-6


def in_field(x: float, y: float, margin: float = 0.0) -> bool:
    """坐标是否在场地范围内。"""
    return (
        -margin <= x <= FIELD_WIDTH + margin
        and -margin <= y <= FIELD_HEIGHT + margin
    )


def clip_to_field(x: float, y: float, margin: float = 2.0) -> tuple[float, float]:
    """把坐标裁剪回场地范围。"""
    return (
        min(max(x, margin), FIELD_WIDTH - margin),
        min(max(y, margin), FIELD_HEIGHT - margin),
    )


@dataclass
class BallPhysics:
    """足球运动状态与物理步进。

    速度单位 px/s；z 为离地高度（z=0 表示贴地滚动）。
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    def speed(self) -> float:
        return hypot(self.vx, self.vy)

    def is_stopped(self) -> bool:
        return self.z <= 0 and self.speed() < BALL_STOP_SPEED

    def launch(self, tx: float, ty: float, speed: float, aerial: bool = False) -> None:
        """朝目标点 (tx, ty) 以指定水平速度出球。

        aerial=True 时给 z 初速度形成抛物线，使球大致在目标点附近落地。
        """
        dx, dy = tx - self.x, ty - self.y
        dist = hypot(dx, dy)
        if dist < 1e-6:
            return
        self.vx = dx / dist * speed
        self.vy = dy / dist * speed
        if aerial:
            flight_time = dist / speed
            # 略小于对称抛物线初速 → 提前落地后滚向目标
            self.vz = 0.45 * GRAVITY * flight_time
        else:
            self.vz = 0.0

    def step(self, dt: float) -> None:
        """推进 dt 秒的球体物理（重力 / 反弹 / 滚动摩擦）。"""
        if self.z > 0 or self.vz != 0:
            # 空中飞行
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.z += self.vz * dt
            self.vz -= GRAVITY * dt
            if self.z <= 0:
                self.z = 0.0
                if self.vz < -BALL_STOP_SPEED:
                    # 落地反弹
                    self.vz = -self.vz * BOUNCE_DAMPING
                    self.vx *= GROUND_SPIN_RETAIN
                    self.vy *= GROUND_SPIN_RETAIN
                else:
                    self.vz = 0.0
        else:
            # 地面滚动：摩擦减速
            speed = self.speed()
            if speed > 0:
                new_speed = max(0.0, speed - GROUND_FRICTION * dt)
                if new_speed < BALL_STOP_SPEED:
                    new_speed = 0.0
                ratio = new_speed / speed
                self.vx *= ratio
                self.vy *= ratio
            self.x += self.vx * dt
            self.y += self.vy * dt

    def stop(self) -> None:
        self.vx = self.vy = self.vz = 0.0
        self.z = 0.0
