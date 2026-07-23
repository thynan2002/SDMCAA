"""球员行为模型。

包含特征向量构建、进攻/防守风格量化。
"""

from .feature_vector import build_feature_vector, FEATURE_DIM, FEATURE_NAMES
from .attacking_style import build_attacking_profile
from .defending_style import build_defending_profile

__all__ = [
    "build_feature_vector",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "build_attacking_profile",
    "build_defending_profile",
]
