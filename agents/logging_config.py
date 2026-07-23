"""日志配置模块。

提供统一的 logging 配置，替换各模块中的 print()。
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """初始化全局日志配置。"""
    logger = logging.getLogger("football_agent")
    logger.setLevel(level)

    if logger.handlers:
        return  # 避免重复配置

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(name)s] %(levelname)s: %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。"""
    return logging.getLogger(f"football_agent.{name}")
