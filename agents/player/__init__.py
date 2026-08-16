"""球员追踪解说系统（精简后：球关系分析 + LLM 解说）。"""

from .tracker import (
    PlayerTrajectory,
    PrefixPlayerCorpus,
    BallEvent,
    BallTrajectoryAnalysis,
    build_corpus_from_paths,
)
from .agents import PlayerCompositionAgent
from .orchestrator import PlayerTrackingOrchestrator
from .commentary import build_user_message, SYSTEM_PROMPT
