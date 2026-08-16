"""专业分析编排器。

通过 LangGraph 编排 DataCollector → StyleModeler → [CounterfactualEngine] → ReportGenerator 流水线。
支持：风格分析、球员对比、反事实分析、综合问答。
"""

from __future__ import annotations

from typing import Any, TypedDict
import hashlib

from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:
    MemorySaver = None

from agents.player.tracker import PrefixPlayerCorpus
from agents.progress import push_stage
from .types import (
    PlayerProfile,
    PlayerBehaviorModel,
    CounterfactualScenario,
    CounterfactualReport,
    RoutedQuery,
    IntentType,
)
from .agents.data_collector import DataCollectorAgent
from .agents.style_modeler import StyleModelingAgent
from .agents.counterfactual_engine import CounterfactualEngineAgent
from .agents.report_generator import ReportGeneratorAgent
from .agents.general_qa import GeneralQAAgent
from .agents.data_verifier import DataVerifierAgent


class ProfessionalState(TypedDict, total=False):
    """专业分析子系统状态。"""
    corpus: PrefixPlayerCorpus
    query: RoutedQuery
    profiles: list[PlayerProfile]
    behavior_models: dict[str, PlayerBehaviorModel]
    counterfactual_report: CounterfactualReport | None
    output_text: str
    memory: Any  # MemoryStore 实例，供节点读取缓存画像和对话上下文


class ProfessionalAnalysisOrchestrator:
    """专业分析编排器。

    下辖五个子智能体：
    - DataCollectorAgent      数据采集
    - StyleModelingAgent      风格建模
    - CounterfactualEngineAgent MCTS反事实分析
    - ReportGeneratorAgent    LLM报告生成
    - GeneralQAAgent          综合问答（独立路径）
    - DataVerifierAgent       数据核验与回查（独立路径）
    """

    def __init__(self, enable_checkpoint: bool = False) -> None:
        from .simulation.llm_brain import LLMDecisionEngine

        self.data_collector = DataCollectorAgent()
        self.style_modeler = StyleModelingAgent()
        # 共享 LLM 决策引擎：驱动 MCTS 策略与轨迹模拟的战术判断
        self.llm_engine = LLMDecisionEngine()
        self.counterfactual_engine = CounterfactualEngineAgent(
            num_simulations=1000, llm_engine=self.llm_engine,
        )
        self.report_generator = ReportGeneratorAgent()
        self.general_qa_agent = GeneralQAAgent()
        self.data_verifier = DataVerifierAgent()
        self.graph = self._build_graph(enable_checkpoint=enable_checkpoint)

    def _build_graph(self, enable_checkpoint: bool):
        builder = StateGraph(ProfessionalState)

        builder.add_node("collect_data", self._collect_data)
        builder.add_node("model_styles", self._model_styles)
        builder.add_node("run_counterfactual", self._run_counterfactual)
        builder.add_node("generate_report", self._generate_report)
        builder.add_node("generate_style_report", self._generate_style_report)

        # 条件路由：根据意图选择跳过 MCTS
        builder.add_conditional_edges(
            "model_styles",
            self._route_after_modeling,
            {
                "counterfactual": "run_counterfactual",
                "direct_report": "generate_style_report",
            },
        )
        builder.add_edge(START, "collect_data")
        builder.add_edge("collect_data", "model_styles")
        builder.add_edge("run_counterfactual", "generate_report")
        builder.add_edge("generate_report", END)
        builder.add_edge("generate_style_report", END)

        if enable_checkpoint and MemorySaver is not None:
            return builder.compile(checkpointer=MemorySaver())
        return builder.compile()

    # ── 节点函数 ──

    def _collect_data(self, state: ProfessionalState) -> dict:
        push_stage("collect_data", "正在采集球员数据…")
        corpus = state["corpus"]
        query = state.get("query")
        memory = state.get("memory")
        target_players = query.target_players if query else None
        profiles = self.data_collector.collect(corpus, target_players)
        push_stage("collect_data", "球员数据采集完成", status="done")
        # 从记忆缓存中补充已有画像，避免重复计算
        if memory and target_players:
            for idx, profile in enumerate(profiles):
                cached = memory.get_cached_profile(profile.jersey_label)
                if cached:
                    # 用缓存替换新计算的画像（保留已验证的数据）
                    profiles[idx] = cached
        return {"profiles": profiles}

    def _model_styles(self, state: ProfessionalState) -> dict:
        push_stage("model_styles", "正在构建球员行为模型…")
        profiles = state.get("profiles", [])
        memory = state.get("memory")
        models: list[Any] = []
        # 复用记忆缓存中已建好的行为模型（避免逐人重复 LLM 风格标注）
        uncached_profiles: list[Any] = []
        for profile in profiles:
            cached = memory.get_cached_model(profile.jersey_label) if memory else None
            if cached is not None:
                models.append(cached)
            else:
                uncached_profiles.append(profile)
        # 仅为未缓存的球员构建模型；传球偏好仍以全量画像为参照
        if uncached_profiles:
            models.extend(self.style_modeler.build_models_batch(uncached_profiles, all_profiles=profiles))
        model_map = {m.jersey_label: m for m in models}
        push_stage("model_styles", "行为模型构建完成", status="done")
        # 更新记忆缓存
        if memory:
            for profile in profiles:
                memory.cache_profile(profile.jersey_label, profile)
            for model in models:
                memory.cache_model(model.jersey_label, model)
        return {"behavior_models": model_map}

    def _run_counterfactual(self, state: ProfessionalState) -> dict:
        query = state.get("query")
        if not query or not query.scenario:
            return {"counterfactual_report": None}

        corpus = state["corpus"]
        models = state.get("behavior_models", {})
        report = self.counterfactual_engine.analyze(
            query.scenario, models, corpus,
        )
        # 用户要求生成轨迹数据文件时：模拟单一最可能未来轨迹并拼接导出 CSV
        if query.scenario.generate_data and report is not None:
            self._generate_counterfactual_data(corpus, models, query.scenario, report)
        return {"counterfactual_report": report}

    # ── 反事实轨迹数据生成 ──

    def _generate_counterfactual_data(
        self,
        corpus: PrefixPlayerCorpus,
        models: dict[str, PlayerBehaviorModel],
        scenario: CounterfactualScenario,
        report: CounterfactualReport,
    ) -> None:
        """运行轨迹模拟引擎并导出拼接 CSV，结果写回 report。"""
        from pathlib import Path
        from .simulation.engine import TrajectorySimulator
        from .simulation.exporter import export_simulation

        push_stage("trajectory", "正在生成反事实轨迹…")
        forced = self._action_from_scenario(scenario)
        simulator = TrajectorySimulator(corpus, models, llm_engine=self.llm_engine)
        result = simulator.simulate(
            intervention_frame=scenario.frame,
            forced_action=forced,
            subject=scenario.subject_player,
            duration_seconds=scenario.duration_seconds,
            scenario_desc=scenario.raw_query or scenario.describe_altered(),
        )

        output_dir = Path(__file__).resolve().parent.parent.parent / "Output"
        try:
            person_path, ball_path, warnings = export_simulation(corpus, result, output_dir)
            report.generated_files = [str(person_path), str(ball_path)]
            report.trajectory_events = result.events + [
                f"[物理校验警告] {w}" for w in warnings
            ]
            report.outcome = result.outcome
            push_stage("trajectory", "反事实轨迹生成完成", status="done")
        except Exception as exc:  # 导出失败不影响主报告
            report.trajectory_events = result.events
            report.analysis_summary += f"\n\n[轨迹数据生成失败] {exc}"
            push_stage("trajectory", "轨迹数据导出失败", status="done")

    @staticmethod
    def _action_from_scenario(scenario: CounterfactualScenario):
        """从场景描述构建第一步强制执行的反事实动作。"""
        from .mcts.node import Action, ActionType

        name = scenario.altered_action
        if name == "pass":
            return Action(ActionType.PASS, target_player=scenario.altered_target)
        if name == "shoot":
            return Action(ActionType.SHOOT)
        if name == "dribble":
            return Action(ActionType.DRIBBLE, direction=(0, 30))
        if name == "clear":
            return Action(ActionType.CLEAR)
        if name == "hold":
            return Action(ActionType.HOLD)
        return None  # 未指定替换动作 → 纯自然推演

    @staticmethod
    def _format_generated_section(report: CounterfactualReport) -> str:
        """格式化生成数据段落（程序拼接，不依赖 LLM，保证路径与事件准确）。"""
        lines = ["─" * 46, "【反事实轨迹数据已生成】"]
        if report.trajectory_events:
            lines.append("关键事件时间线：")
            lines.extend(f"  · {e}" for e in report.trajectory_events[:12])
            if len(report.trajectory_events) > 12:
                lines.append(f"  · ……共 {len(report.trajectory_events)} 个事件")
        outcome_cn = {
            "goal": "进球", "save_held": "射门被门将没收", "miss": "射门偏出",
            "out": "球出界", "completed": "自然推演至片段结束",
        }
        if report.outcome:
            lines.append(f"模拟结局：{outcome_cn.get(report.outcome, report.outcome)}")
        lines.append("数据文件（干预帧前为原始真实数据，之后为生成关键帧）：")
        lines.extend(f"  → {p}" for p in report.generated_files)
        return "\n".join(lines)

    def _generate_report(self, state: ProfessionalState) -> dict:
        push_stage("report", "正在整合分析报告…")
        profiles = state.get("profiles", [])
        models = state.get("behavior_models", {})
        cf_report = state.get("counterfactual_report")
        model_list = [models[p.jersey_label] for p in profiles if p.jersey_label in models]

        if cf_report:
            text = self.report_generator.generate_counterfactual_report(cf_report)
            if cf_report.generated_files:
                text += "\n\n" + self._format_generated_section(cf_report)
        else:
            text = self.report_generator.generate_comprehensive_report(
                profiles, model_list, cf_report,
            )
        push_stage("report", "报告整合完成", status="done")
        return {"output_text": text}

    def _generate_style_report(self, state: ProfessionalState) -> dict:
        push_stage("report", "正在生成分析报告…")
        profiles = state.get("profiles", [])
        models = state.get("behavior_models", {})
        query = state.get("query")
        model_list = [models[p.jersey_label] for p in profiles if p.jersey_label in models]

        # 空画像兜底：目标球员不在数据中时如实声明，不返回空报告
        if not profiles:
            wanted = "、".join(query.target_players) if query and query.target_players else "指定球员"
            push_stage("report", "无匹配球员画像", status="done")
            return {"output_text": (
                f"数据不足：未找到 {wanted} 的任何轨迹数据，"
                "无法进行风格分析。请确认球员在当前数据中存在。"
            )}

        texts: list[str] = []
        question = query.original_text if query else ""
        if query and query.intent == IntentType.COMPARISON and len(profiles) >= 2:
            texts.append(self.report_generator.generate_comparison_report(profiles, model_list, question))
        else:
            for profile, model in zip(profiles, model_list):
                texts.append(self.report_generator.generate_style_report(profile, model, question))

        push_stage("report", "报告生成完成", status="done")
        return {"output_text": "\n\n".join(texts)}

    # ── 路由逻辑 ──

    def _route_after_modeling(self, state: ProfessionalState) -> str:
        query = state.get("query")
        if query and query.intent == IntentType.COUNTERFACTUAL and query.scenario:
            return "counterfactual"
        return "direct_report"

    # ── 公开接口 ──

    def run(
        self,
        corpus: PrefixPlayerCorpus,
        query: RoutedQuery,
        memory: Any | None = None,
    ) -> str:
        """执行专业分析（风格/对比/反事实）。"""
        # 注入记忆到状态
        extra = {}
        if memory:
            extra["memory"] = memory
        # 使用稳定哈希（md5）生成 thread_id，避免 Python hash() 跨进程不一致
        thread_hash = hashlib.md5(query.original_text.encode("utf-8")).hexdigest()[:16]
        result = self.graph.invoke(
            {"corpus": corpus, "query": query, **extra},
            config={"configurable": {"thread_id": f"pro-{thread_hash}"}},
        )
        return result.get("output_text", "分析未能完成。")

    def run_team_analysis(
        self,
        corpus: PrefixPlayerCorpus,
        question: str = "",
        memory_context: str = "",
    ) -> str:
        """球队级战术分析（阵型/进攻/空间/转换/核心球员），不走 LangGraph。"""
        from .agents.tactical_facts import extract_tactical_facts

        facts = extract_tactical_facts(corpus)
        return self.report_generator.generate_team_report(facts, question, memory_context)

    def run_general_qa(
        self,
        corpus: PrefixPlayerCorpus,
        question: str,
        memory_context: str = "",
    ) -> str:
        """综合问答（独立路径，不走 LangGraph）。"""
        return self.general_qa_agent.answer(question, corpus, memory_context)

    def run_verify_data(
        self,
        corpus: PrefixPlayerCorpus,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        memory_context: str = "",
    ) -> str:
        """数据核验与回查（独立路径）。"""
        return self.data_verifier.verify(
            question, corpus, conversation_history, memory_context,
        )
