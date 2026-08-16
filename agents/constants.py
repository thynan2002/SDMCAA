"""共享常量定义。

所有与距离、速度、高度相关的阈值集中于此，
避免散落在各模块中造成不一致。
"""

# ── 球场尺寸 ──
FIELD_WIDTH = 1200
FIELD_HEIGHT = 700

# ── 球高度分层阈值（单位：米，与输入 CSV 的 z 列一致）──
# 注意：输入数据的 z 轴是米（实测范围 0~5.1，为球心离地高度），
# 不是像素。历史上曾按像素阈值（1 / 50）判定，导致高空球永远不会被识别。
GROUND_Z_THRESHOLD = 0.2        # z ≤ 0.2m → 地面球
LOW_BALL_Z_MAX = 2.5            # 0.2m < z ≤ 2.5m → 低平球
# z > 2.5m → 高空球
AERIAL_Z_THRESHOLD = 1.0        # z > 1m 记为空中球（aerial_pct 统计口径）

# ── 球距接近判定半径（水平距离） ──
GROUND_PROXIMITY_RADIUS = 30.0
LOW_BALL_PROXIMITY_RADIUS = 50.0
HIGH_BALL_PROXIMITY_RADIUS = 60.0

# 通用"可能触球"判定距离
MAX_POSSESSION_DIST = 60.0

# 运动判定阈值
APPROACH_DELTA = 5.0            # 帧间距离减少 > 5 → 靠近
RETREAT_DELTA = 5.0             # 帧间距离增加 > 5 → 远离

# 速度档位阈值（px/s）
SPEED_SPRINT = 300.0            # 冲刺（用于 sprint_count）
SPEED_HIGH = 500.0              # 高速（射门/长传）
SPEED_MEDIUM = 150.0            # 中速（传球）
SPEED_LOW = 30.0                # 低速（带球）

# ── 物理单位转换 ──
# 标准球场 ~105m × 68m，映射到 1200 × 700 像素
PIXELS_PER_METER = 11           # 约 11 px/m，用于粗略换算

# ── 精确米制标定（1200×700 px → 标准 105×68 m） ──
# x 轴（1200px）= 长度（门线到门线，105m）
# y 轴（700px） = 宽度（边线到边线，68m）
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PX_PER_M_X = FIELD_WIDTH / PITCH_LENGTH_M   # ≈ 11.43 px/m
PX_PER_M_Y = FIELD_HEIGHT / PITCH_WIDTH_M   # ≈ 10.29 px/m


def px_to_m(pixels: float) -> float:
    """像素 → 米（粗略单标量换算）"""
    return round(pixels / PIXELS_PER_METER, 2)


def px_x_to_m(px: float) -> float:
    """x 轴像素 → 米（长度方向，PX_PER_M_X）"""
    return round(px / PX_PER_M_X, 2)


def px_y_to_m(px: float) -> float:
    """y 轴像素 → 米（宽度方向，PX_PER_M_Y）"""
    return round(px / PX_PER_M_Y, 2)


def px_s_to_kmh(px_s: float) -> float:
    """像素/秒 → 公里/小时"""
    m_s = px_s / PIXELS_PER_METER
    return round(m_s * 3.6, 2)


def px_s_to_ms(px_s: float) -> float:
    """像素/秒 → 米/秒（按进攻轴 x 的比例折算）"""
    return round(px_s / PX_PER_M_X, 2)

