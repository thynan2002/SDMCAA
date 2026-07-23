"""球员轨迹追踪编排器。

按前缀遍历所有球员，对每位球员做运动分析+球互动分析，
再对比排名，最后给出融合决策和解说稿。
"""

from __future__ import annotations

from operator import add
from pathlib import Path
from typing import Any, Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:
    MemorySaver = None

from .tracker import PrefixPlayerCorpus, build_player_corpus
from .agents import (
    PlayerMovementAgent,
    PlayerBallInteractionAgent,
    PlayerComparisonAgent,
    PlayerDecisionAgent,
    PlayerCompositionAgent,
)
from .ball_relation import BallPositionRelationAgent
from ..types import AgentReport, FinalDecision


class PlayerState(TypedDict, total=False):
    corpus: PrefixPlayerCorpus
    context: dict[str, Any]
    movement_results: list[dict[str, Any]]
    interaction_results: list[dict[str, Any]]
    ball_relation_results: list[dict[str, Any]]
    comparison_report: AgentReport
    decision_payload: tuple[str, float, str, list[str], list[str]]
    composed_summary: str
    final_decision: FinalDecision


class PlayerTrackingOrchestrator:
    """按球员追踪轨迹的多智能体编排器。"""

    def __init__(self, enable_checkpoint: bool = True) -> None:
        self.movement_agent = PlayerMovementAgent()
        self.interaction_agent = PlayerBallInteractionAgent()
        self.ball_relation_agent = BallPositionRelationAgent()
        self.comparison_agent = PlayerComparisonAgent()
        self.decision_agent = PlayerDecisionAgent()
        self.composer_agent = PlayerCompositionAgent()
        self.graph = self._build_graph(enable_checkpoint=enable_checkpoint)

    def _build_graph(self, enable_checkpoint: bool):
        builder = StateGraph(PlayerState)

        builder.add_node("analyze_movement", self._analyze_movement)
        builder.add_node("analyze_interaction", self._analyze_interaction)
        builder.add_node("analyze_ball_relation", self._analyze_ball_relation)
        builder.add_node("compare", self._compare)
        builder.add_node("decide", self._decide)
        builder.add_node("compose", self._compose)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "analyze_movement")
        builder.add_edge(START, "analyze_interaction")
        builder.add_edge(START, "analyze_ball_relation")
        builder.add_edge(["analyze_movement", "analyze_interaction", "analyze_ball_relation"], "compare")
        builder.add_edge("compare", "decide")
        builder.add_edge("decide", "compose")
        builder.add_edge("compose", "finalize")
        builder.add_edge("finalize", END)

        if enable_checkpoint and MemorySaver is not None:
            return builder.compile(checkpointer=MemorySaver())
        return builder.compile()

    # ── 节点 ──────────────────────────────────────────────────

    def _analyze_movement(self, state: PlayerState) -> dict:
        corpus = state["corpus"]
        return {"movement_results": self.movement_agent.summarize_all(corpus)}

    def _analyze_interaction(self, state: PlayerState) -> dict:
        corpus = state["corpus"]
        return {"interaction_results": self.interaction_agent.summarize_all(corpus)}

    def _analyze_ball_relation(self, state: PlayerState) -> dict:
        corpus = state["corpus"]
        return {"ball_relation_results": self.ball_relation_agent.summarize_all(corpus)}

    def _compare(self, state: PlayerState) -> dict:
        corpus = state["corpus"]
        report = self.comparison_agent.summarize(
            corpus,
            state.get("movement_results", []),
            state.get("interaction_results", []),
            state.get("ball_relation_results", []),
        )
        return {"comparison_report": report}

    def _decide(self, state: PlayerState) -> dict:
        corpus = state["corpus"]
        payload = self.decision_agent.decide(
            corpus,
            state.get("movement_results", []),
            state.get("interaction_results", []),
            state.get("comparison_report", AgentReport(
                agent_name="", summary="", score_0_to_100=0, confidence_0_to_1=0,
            )),
            state.get("ball_relation_results", []),
        )
        return {"decision_payload": payload}

    def _compose(self, state: PlayerState) -> dict:
        corpus = state["corpus"]
        summary = self.composer_agent.compose(
            corpus,
            state.get("movement_results", []),
            state.get("interaction_results", []),
            state.get("comparison_report", AgentReport(
                agent_name="", summary="", score_0_to_100=0, confidence_0_to_1=0,
            )),
            state.get("decision_payload", ("", 0.0, "", [], [])),
            ball_relation_results=state.get("ball_relation_results", []),
        )
        return {"composed_summary": summary}

    def _finalize(self, state: PlayerState) -> dict:
        action, overall, rationale, next_focus, risks = state.get(
            "decision_payload", ("", 0.0, "", [], [])
        )
        final = FinalDecision(
            recommended_action=action,
            overall_score_0_to_100=overall,
            rationale=rationale,
            next_focus=next_focus,
            merged_risks=risks,
            reports=[state.get("comparison_report", AgentReport(
                agent_name="", summary="", score_0_to_100=0, confidence_0_to_1=0,
            ))],
            composed_summary=state.get("composed_summary", ""),
        )
        return {"final_decision": final}

    def run_for_prefix(
        self,
        corpus: PrefixPlayerCorpus,
        context: dict | None = None,
        thread_id: str = "default-thread",
    ) -> FinalDecision:
        result = self.graph.invoke(
            {"corpus": corpus, "context": dict(context or {})},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["final_decision"]

    def run_all(
        self,
        test_input_dir: str | Path,
        context: dict | None = None,
    ) -> dict[str, tuple[PrefixPlayerCorpus, FinalDecision]]:
        """对所有前缀执行球员追踪分析。

        返回 {prefix: (corpus, decision)}。
        """
        all_corpora = build_player_corpus(test_input_dir)
        results: dict[str, tuple[PrefixPlayerCorpus, FinalDecision]] = {}

        for prefix, corpus in sorted(all_corpora.items()):
            thread_id = f"player-{prefix}"
            decision = self.run_for_prefix(corpus, context, thread_id=thread_id)
            results[prefix] = (corpus, decision)

        return results

    def run_single(
        self,
        corpus: PrefixPlayerCorpus,
        context: dict | None = None,
        focus_jerseys: list[str] | None = None,
    ) -> FinalDecision:
        """对单场比赛运行分析，可选聚焦球员。

        focus_jerseys 非空时，LLM 解说围绕这些球衣号展开。
        """
        result = self.graph.invoke(
            {"corpus": corpus, "context": dict(context or {})},
            config={"configurable": {"thread_id": f"single-{corpus.prefix}"}},
        )

        # 重跑 compose 以传递 focus_jerseys
        ball_relation = self.ball_relation_agent.summarize_all(corpus)
        summary = self.composer_agent.compose(
            corpus,
            result.get("movement_results", []),
            result.get("interaction_results", []),
            result.get("comparison_report", AgentReport(
                agent_name="", summary="", score_0_to_100=0, confidence_0_to_1=0,
            )),
            result.get("decision_payload", ("", 0.0, "", [], [])),
            focus_jerseys=focus_jerseys,
            ball_relation_results=ball_relation,
        )
        final: FinalDecision = result["final_decision"]
        final.composed_summary = summary
        return final
