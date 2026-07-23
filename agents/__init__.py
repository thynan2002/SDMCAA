"""agents 包导出入口。

Multi-Agent 足球战术分析系统的外部接口。
"""

from .player.tracker import build_player_corpus, PlayerTrajectory, PrefixPlayerCorpus
from .player.orchestrator import PlayerTrackingOrchestrator
from .session.manager import SessionManager
from .session.router import QueryRouter
from .output_writer import write_player_trajectories

__all__ = [
    "build_player_corpus",
    "PlayerTrajectory",
    "PrefixPlayerCorpus",
    "PlayerTrackingOrchestrator",
    "SessionManager",
    "QueryRouter",
    "write_player_trajectories",
]
