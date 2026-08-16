"""球员追踪编排器（精简版）。

聚焦解说路径精简为两步：确定性球关系分析 + 单次 LLM 解说生成。
原 LangGraph 六节点流水线（运动/互动/球关系/对比/决策/解说）中，
运动/互动/对比/决策的中间结果未被解说链路消费，已随中间层精简移除；
解说所需数据由 commentary.build_user_message 直接从 corpus 组装。
"""

from __future__ import annotations

from .tracker import PrefixPlayerCorpus
from .agents import PlayerCompositionAgent
from .ball_relation import BallPositionRelationAgent


class PlayerTrackingOrchestrator:
    """按球员追踪轨迹的解说编排器（球关系分析 + LLM 解说）。"""

    def __init__(self) -> None:
        self.ball_relation_agent = BallPositionRelationAgent()
        self.composer_agent = PlayerCompositionAgent()

    def run_single(
        self,
        corpus: PrefixPlayerCorpus,
        focus_jerseys: list[str] | None = None,
    ) -> str:
        """对单场比赛运行解说，可选聚焦球员。"""
        ball_relation_results = self.ball_relation_agent.summarize_all(corpus)
        return self.composer_agent.compose(
            corpus,
            focus_jerseys=list(focus_jerseys) if focus_jerseys else None,
            ball_relation_results=ball_relation_results,
        )
