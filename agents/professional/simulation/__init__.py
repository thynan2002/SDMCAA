"""反事实轨迹模拟子系统。

在现有 MCTS 概率分析之上，提供"单一最可能未来轨迹"的生成能力：

- physics   物理校验与运动学约束（限速 / 防瞬移 / 球体重力与摩擦）
- tactical  全体球员战术质点运动模型（接应 / 前插 / 上抢 / 收缩）
- engine    轨迹模拟引擎（强制反事实首步 + 轻量确定性策略）
- exporter  数据拼接导出（干预前原始数据 + 干预后生成关键帧 → CSV）
"""

from .engine import SimulationResult, TrajectorySimulator
from .exporter import export_simulation

__all__ = [
    "SimulationResult",
    "TrajectorySimulator",
    "export_simulation",
]
