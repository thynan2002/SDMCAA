"""球员追踪多智能体系统。"""

from .tracker import (
    PlayerTrajectory,
    PrefixPlayerCorpus,
    BallEvent,
    BallTrajectoryAnalysis,
    build_player_corpus,
    build_corpus_from_paths,
)
from .agents import (
    PlayerMovementAgent,
    PlayerBallInteractionAgent,
    PlayerComparisonAgent,
    PlayerDecisionAgent,
    PlayerCompositionAgent,
)
from .orchestrator import PlayerTrackingOrchestrator
from .commentary import build_user_message, SYSTEM_PROMPT
