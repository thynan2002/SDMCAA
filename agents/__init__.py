"""agents 包导出入口。

Multi-Agent 足球战术分析系统的外部接口。
"""

from .player.tracker import PlayerTrajectory, PrefixPlayerCorpus
from .player.orchestrator import PlayerTrackingOrchestrator
from .session.manager import SessionManager
from .session.router import QueryRouter

__all__ = [
    "PlayerTrajectory",
    "PrefixPlayerCorpus",
    "PlayerTrackingOrchestrator",
    "SessionManager",
    "QueryRouter",
]
