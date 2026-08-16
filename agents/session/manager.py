"""交互式会话管理器。

管理数据加载、命令分发和 REPL 交互循环。
**所有意图路由由 LLM 驱动，无硬编码规则。**
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.player.tracker import (
    PrefixPlayerCorpus,
    build_corpus_from_paths,
)
from agents.player.orchestrator import PlayerTrackingOrchestrator
from agents.professional.orchestrator import ProfessionalAnalysisOrchestrator
from agents.professional.types import IntentType, RoutedQuery, CounterfactualScenario, ScenarioType
from agents.progress import push_stage
from .router import QueryRouter
from .memory import MemoryStore


class SessionManager:
    """交互式多智能体会话管理器。

    加载一次数据，支持多轮交互：
    - 聚焦球员 → PlayerTrackingOrchestrator
    - 风格分析 → ProfessionalAnalysisOrchestrator
    - 反事实分析 → ProfessionalAnalysisOrchestrator
    - 球员对比 → ProfessionalAnalysisOrchestrator
    - 综合问答 → ProfessionalAnalysisOrchestrator (GeneralQAAgent)
    """

    def __init__(self) -> None:
        self.corpus: PrefixPlayerCorpus | None = None
        self.focus_jerseys: list[str] = []
        self.conversation_history: list[dict[str, str]] = []
        self._data_files: tuple[str, str] = ("", "")

        # 子系统
        self.tracking_orchestrator = PlayerTrackingOrchestrator()
        self.professional_orchestrator = ProfessionalAnalysisOrchestrator()

        # 路由器 + 记忆
        self.router = QueryRouter()
        self.memory = MemoryStore()

        # 最近一次反事实生成产出的文件路径（供前端拉取轨迹对比）
        self.last_counterfactual_files: list[str] = []

    def load_data(self, person_csv: str | Path, ball_csv: str | Path) -> bool:
        """加载比赛数据。"""
        try:
            self.corpus = build_corpus_from_paths(person_csv, ball_csv)
            self._data_files = (str(person_csv), str(ball_csv))
            # 数据工具（get_frame_snapshot 等）的语料来源
            from agents.tools.football import set_active_corpus
            set_active_corpus(self.corpus)
            return True
        except Exception as e:
            print(f"[错误] 数据加载失败：{e}")
            traceback.print_exc()
            return False

    def handle_input(self, user_text: str) -> str:
        """处理用户输入，返回响应文本。"""
        if not user_text.strip():
            return ""

        # 路由解析（LLM 驱动）— 这是首个耗时步骤，必须推送状态
        push_stage("routing", "正在理解问题意图…")
        available = (
            sorted({p.jersey_label for p in self.corpus.players.values()})
            if self.corpus is not None else None
        )
        query = self.router.parse(user_text, available_players=available)
        push_stage("routing", "意图识别完成", status="done")

        # clear 命令特殊处理：在记录对话之前清除，避免记忆被重新污染
        if query.intent == IntentType.CLEAR:
            self.focus_jerseys = []
            self.conversation_history = []
            self.memory.clear()
            return "上下文已清除。"

        # 记录对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        try:
            response = self._dispatch(query)
        except Exception as e:
            response = f"[错误] 处理请求时发生异常：{e}"
            traceback.print_exc()

        # __EXIT__ 是内部控制信号，不记录到对话历史
        if response != "__EXIT__":
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
            })
            # 更新记忆（每轮对话后）
            self.memory.update_after_turn(user_text, response)

        return response

    # ── 反事实表单（前端结构化输入，绕过路由直接构造场景）──

    _ACTION_CN = {
        "shoot": "直接射门",
        "dribble": "盘带突破",
        "pass": "传球",
        "clear": "解围",
        "hold": "控球",
    }

    def handle_counterfactual_form(
        self,
        subject_player: str,
        time_second: float,
        altered_action: str,
        altered_target: str = "",
        duration_seconds: float = 0.0,
    ) -> str:
        """处理前端反事实表单提交，直接构造结构化场景并触发反事实分析。

        Args:
            subject_player: 主体球员 jersey_label，如 "7号"
            time_second: 干预时间点（秒）
            altered_action: 替换动作 pass/shoot/dribble/clear/hold
            altered_target: 替换传球目标 jersey_label（仅 pass 时有效）
            duration_seconds: 生成轨迹时长（秒），<=0 表示补全到原始数据结束
        """
        if self.corpus is None:
            return "请先加载比赛数据。"

        push_stage("routing", "正在构建反事实场景…")

        # 标准化球衣号
        subject_player = self.router._normalize_jersey(subject_player)
        if altered_target:
            altered_target = self.router._normalize_jersey(altered_target)

        # 构造自然语言用户消息（写入对话历史，保持可追溯）
        action_cn = self._ACTION_CN.get(altered_action, altered_action)
        if altered_action == "pass" and altered_target:
            user_text = f"如果{subject_player}在第{time_second:.1f}秒改为传球给{altered_target}，生成后面{duration_seconds:.0f}秒的轨迹数据"
        else:
            user_text = f"如果{subject_player}在第{time_second:.1f}秒改为{action_cn}，生成后面{duration_seconds:.0f}秒的轨迹数据"

        frame = int(time_second * 30)
        scenario = CounterfactualScenario(
            scenario_type=ScenarioType.SELF_CHOICE_CHANGE,
            subject_player=subject_player,
            frame=frame,
            time_second=float(time_second),
            original_action="",
            original_target="",
            altered_action=altered_action,
            altered_target=altered_target,
            generate_data=True,
            duration_seconds=float(duration_seconds),
            raw_query=user_text,
        )
        query = RoutedQuery(
            intent=IntentType.COUNTERFACTUAL,
            target_players=[subject_player] + ([altered_target] if altered_target else []),
            scenario=scenario,
            original_text=user_text,
            confidence=1.0,
        )

        self.conversation_history.append({"role": "user", "content": user_text})
        push_stage("routing", "场景构建完成", status="done")
        try:
            response = self._handle_counterfactual(query)
        except Exception as e:
            response = f"[错误] 反事实分析发生异常：{e}"
            traceback.print_exc()

        self.conversation_history.append({"role": "assistant", "content": response})
        self.memory.update_after_turn(user_text, response)
        return response

    def _dispatch(self, query) -> str:
        """根据 LLM 判断的意图分发到对应处理器。"""
        intent = query.intent

        if intent == IntentType.EXIT:
            return "__EXIT__"

        if intent == IntentType.HELP:
            return self._help_text()

        if intent == IntentType.CLEAR:
            # clear 已在 handle_input 中提前处理，此处不会再到达
            return "上下文已清除。"

        if intent == IntentType.FOCUS_PLAYER:
            return self._handle_focus(query)

        if intent == IntentType.STYLE_ANALYSIS:
            return self._handle_style_analysis(query)

        if intent == IntentType.COUNTERFACTUAL:
            return self._handle_counterfactual(query)

        if intent == IntentType.COMPARISON:
            return self._handle_comparison(query)

        if intent == IntentType.GENERAL_ANALYSIS:
            return self._handle_general_analysis(query)

        if intent == IntentType.GENERAL_QA:
            return self._handle_general_qa(query)

        if intent == IntentType.VERIFY_DATA:
            return self._handle_verify_data(query)

        # 最终兜底
        return self._handle_general_qa(query)

    # ── 处理器 ──

    # ── 球员存在性校验（数据不足如实声明，不得静默编造/返回空） ──

    def _validate_players(self, jerseys: list[str]) -> str | None:
        """校验球衣号在数据中存在；返回 None 表示通过，否则返回错误消息。"""
        if self.corpus is None:
            return "请先加载比赛数据。"
        if not jerseys:
            return None
        existing = {p.jersey_label for p in self.corpus.players.values()}
        missing = [j for j in jerseys if j not in existing]
        if missing:
            order = sorted(existing, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 9999)
            return (f"数据中未找到以下球员：{'、'.join(missing)}。"
                    f"当前数据中的可用球员：{'、'.join(order)}。"
                    "请确认球衣号后重试。")
        return None

    def _handle_focus(self, query) -> str:
        """处理聚焦球员请求。"""
        # 先校验数据是否已加载，再写入 focus_jerseys，避免脏状态
        if self.corpus is None:
            return "请先加载比赛数据。"

        if not query.target_players:
            return "请指定要聚焦的球员，如：'聚焦7号'"

        invalid = self._validate_players(query.target_players)
        if invalid:
            return invalid

        self.focus_jerseys = query.target_players

        push_stage("tracking", "正在生成聚焦解说…")
        print(f"\n[系统] 正在分析 {', '.join(query.target_players)} ...")
        decision = self.tracking_orchestrator.run_single(
            self.corpus,
            focus_jerseys=query.target_players,
        )
        push_stage("tracking", "聚焦解说完成", status="done")
        return decision

    def _handle_style_analysis(self, query) -> str:
        """处理风格分析请求。"""
        if self.corpus is None:
            return "请先加载比赛数据。"

        if not query.target_players:
            if self.focus_jerseys:
                query.target_players = self.focus_jerseys
            else:
                # 未指名球员但提到球队（如"A队谁的风格最激进"）→ 展开为该队球员对比
                team_jerseys = self._resolve_team_players(query.original_text)
                if team_jerseys:
                    query.target_players = team_jerseys
                    query.intent = IntentType.COMPARISON
                    return self._handle_comparison(query)
                # 未指名球员也未提球队（如"哪个球员跑动距离最长"）→ 展开为全体球员对比
                all_jerseys = self._all_field_jerseys()
                if all_jerseys:
                    query.target_players = all_jerseys
                    query.intent = IntentType.COMPARISON
                    return self._handle_comparison(query)
                return "请指定要分析的球员，如：'7号的跑动风格是什么'"

        # 检查缓存
        cached = []
        for jersey in query.target_players:
            if self.memory.is_cached(jersey):
                cached.append(jersey)
        if cached:
            print(f"  [记忆] {', '.join(cached)} 已有缓存，跳过重复分析")

        invalid = self._validate_players(query.target_players)
        if invalid:
            return invalid

        print(f"\n[系统] 正在分析 {', '.join(query.target_players)} 的风格特征 ...")
        push_stage("style", "正在分析球员风格…")
        result = self.professional_orchestrator.run(
            self.corpus, query, self.memory,
        )
        push_stage("style", "风格分析完成", status="done")
        return result

    def _handle_counterfactual(self, query) -> str:
        """处理反事实分析请求。"""
        if self.corpus is None:
            return "请先加载比赛数据。"

        if not query.scenario or not query.scenario.subject_player:
            return (
                "请描述反事实场景，例如：\n"
                "  '如果7号在第3秒时不传给11号而是直接打门会怎样'\n"
                "  '假如6号选择盘带突破而不是回传'\n"
                "  '如果球传给了另一侧的9号'"
            )

        # 主体（及传球目标）必须存在于数据，避免 MCTS 对虚构球员推演
        involved = [query.scenario.subject_player]
        if query.scenario.altered_target:
            involved.append(query.scenario.altered_target)
        invalid = self._validate_players(involved)
        if invalid:
            return invalid

        print(
            f"\n[系统] 正在进行反事实模拟"
            f"（MCTS {self.professional_orchestrator.counterfactual_engine.num_simulations}次推演）..."
        )
        push_stage("counterfactual", "反事实推演启动…")
        response = self.professional_orchestrator.run(self.corpus, query)
        # 记录本次生成的反事实轨迹文件（orchestrator 内部已写入 report.generated_files，
        # 通过扫描 Output 目录中匹配前缀的最新 cf 文件对来获取，供前端对比展示）
        self.last_counterfactual_files = self._collect_counterfactual_files(query.scenario)
        push_stage("counterfactual", "反事实推演完成", status="done")
        return response

    def _collect_counterfactual_files(self, scenario) -> list[str]:
        """扫描 Output 目录，收集本次反事实生成的轨迹文件对。"""
        import re
        output_dir = Path(__file__).resolve().parent.parent.parent / "Output"
        if not output_dir.is_dir() or self.corpus is None:
            return []
        prefix = self.corpus.prefix
        # 匹配 {prefix}_cf_f{frame}_person.csv / _ball.csv
        pat = re.compile(rf"^{re.escape(prefix)}_cf_f(\d+)_(person|ball)\.csv$")
        matches = [(p, int(pat.match(p.name).group(1)))
                   for p in output_dir.iterdir() if pat.match(p.name)]
        if not matches:
            return []
        # 取与 scenario.frame 最接近（或最新）的文件对
        target = scenario.frame
        matches.sort(key=lambda x: abs(x[1] - target))
        frame_chosen = matches[0][1]
        pair = sorted([str(p) for p, f in matches if f == frame_chosen])
        return pair

    def _handle_comparison(self, query) -> str:
        """处理球员对比请求。"""
        if self.corpus is None:
            return "请先加载比赛数据。"

        if len(query.target_players) < 2:
            # 未足两名球员：若为"谁最X"类无指名提问 → 展开为全体球员对比
            all_jerseys = self._all_field_jerseys()
            if all_jerseys:
                query.target_players = all_jerseys
            else:
                return "对比分析需要至少两名球员，如：'对比7号和11号'"

        invalid = self._validate_players(query.target_players)
        if invalid:
            return invalid

        print(f"\n[系统] 正在对比 {', '.join(query.target_players)} ...")
        push_stage("comparison", "正在对比球员风格…")
        result = self.professional_orchestrator.run(self.corpus, query)
        push_stage("comparison", "对比分析完成", status="done")
        return result

    def _handle_general_analysis(self, query) -> str:
        """处理整体比赛分析请求（球队级战术报告）。"""
        if self.corpus is None:
            return "请先加载比赛数据。"

        print("\n[系统] 正在执行整体战术分析 ...")
        push_stage("team_analysis", "正在执行整体战术分析…")
        result = self.professional_orchestrator.run_team_analysis(
            self.corpus,
            query.original_text,
            self.memory.get_full_context(),
        )
        push_stage("team_analysis", "战术分析完成", status="done")
        return result

    # ── 球队解析 ──

    # 球队别名 → 数据中的颜色标识
    _TEAM_ALIASES: dict[str, str] = {"a": "A", "b": "B"}

    def _all_field_jerseys(self) -> list[str]:
        """返回全部非门将球员的球衣号（"谁最X"类无指名提问的展开目标）。"""
        if self.corpus is None:
            return []
        return [
            p.jersey_label
            for p in self.corpus.players.values()
            if not p.is_goalkeeper
        ]

    def _resolve_team_players(self, text: str) -> list[str]:
        """从文本中识别球队提及（如"A队"/"B队"），返回该队全部球员球衣号。"""
        import re

        if self.corpus is None:
            return []
        match = re.search(r"([ABab])\s*队", text)
        if not match:
            return []
        color = self._TEAM_ALIASES.get(match.group(1).lower())
        if not color:
            return []
        return [
            p.jersey_label
            for p in self.corpus.players.values()
            if p.color == color and not p.is_goalkeeper
        ]

    def _handle_general_qa(self, query) -> str:
        """处理综合问答请求（兜底）。

        调用 GeneralQAAgent，提取比赛事实，
        通过 LLM 回答用户的任意比赛相关问题。
        """
        if self.corpus is None:
            return "请先加载比赛数据。"

        # 目标球员全部不存在时提前拒答（避免注定幻觉的完整事实提取+LLM 调用）
        if query.target_players:
            invalid = self._validate_players(query.target_players)
            if invalid:
                return invalid

        print(f"\n[系统] 正在查询...")
        push_stage("qa", "正在查询比赛信息…")
        result = self.professional_orchestrator.run_general_qa(
            self.corpus,
            query.original_text,
            self.memory.get_full_context(),
        )
        push_stage("qa", "查询完成", status="done")
        return result

    def _handle_verify_data(self, query) -> str:
        """处理数据核验/质疑/回查请求。

        调用 DataVerifierAgent：
        - 识别是否为用户质疑 → LLM 判断 → 回溯原始数据核验
        - 识别帧级/球员级精确查询 → 从 corpus 提取详细快照
        """
        if self.corpus is None:
            return "请先加载比赛数据。"

        # 核验对象全部不存在时提前拒答（如实声明，而非回溯空数据）
        if query.target_players:
            invalid = self._validate_players(query.target_players)
            if invalid:
                return invalid

        print(f"\n[系统] 正在进行数据核验...")
        push_stage("verify", "正在核验原始数据…")
        result = self.professional_orchestrator.run_verify_data(
            self.corpus,
            query.original_text,
            self.conversation_history,
            self.memory.get_full_context(),
        )
        push_stage("verify", "数据核验完成", status="done")
        return result

    # ── REPL ──

    def run_repl(self) -> None:
        """启动交互式 REPL 循环。"""
        if self.corpus is None:
            print("[错误] 未加载数据，无法启动交互模式。")
            return

        print(f"\n{'=' * 50}")
        print(f"  足球战术分析 Multi-Agent 交互系统")
        print(f"  数据: {self.corpus.prefix} ({self.corpus.player_count}名球员)")
        print(f"  输入 'help' 查看可用命令，'exit' 退出")
        print(f"{'=' * 50}\n")

        while True:
            try:
                user_input = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                self._save_conversation()
                break

            response = self.handle_input(user_input)
            if response == "__EXIT__":
                print("再见！")
                self._save_conversation()
                break

            if response:
                print(f"\n{response}\n")

    def _save_conversation(self) -> None:
        """保存对话历史到 Output 目录。"""
        if not self.conversation_history:
            return

        output_dir = Path(__file__).resolve().parent.parent.parent / "Output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 同时保存记忆（供下次启动恢复）
        try:
            memory_path = output_dir / f"session_memory.json"
            self.memory.save_to_disk(memory_path)
        except Exception:
            pass

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.corpus.prefix if self.corpus else "unknown"
        filename = f"{prefix}_conversation_{timestamp}.txt"
        filepath = output_dir / filename

        lines: list[str] = []
        lines.append(f"足球战术分析 Multi-Agent 对话记录")
        lines.append(f"数据文件: {self._data_files[0]}, {self._data_files[1]}")
        lines.append(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"对话轮数: {len(self.conversation_history) // 2}")
        lines.append("=" * 60)

        for entry in self.conversation_history:
            # 过滤内部控制信号，不写入用户可见的存档
            if entry["content"] == "__EXIT__":
                continue
            role_label = "用户" if entry["role"] == "user" else "系统"
            lines.append(f"\n[{role_label}]")
            lines.append(entry["content"])

        filepath.write_text("\n".join(lines), encoding="utf-8")
        print(f"对话已保存至: {filepath}")

    # ── 帮助文本 ──

    def _help_text(self) -> str:
        return """可用命令：

【聚焦解说】
  聚焦<球员号>        以指定球员为中心生成解说
  看<球员号>          同上

【风格分析】
  <球员号>的跑动风格   分析球员技术特点和战术角色
  分析<球员号>         深入分析攻防风格
  对比<球员A>和<球员B>  比较两名球员的风格差异

【反事实推演】
  如果<球员号>不传而是打门    模拟不同决策的战术结果
  假如<球员号>选择突破        模拟盘带替代方案
  如果传给<球员号>而非<球员号>  模拟不同传球目标

【综合问答】
  1号是哪个队伍的      查询球员基本信息
  第几秒发生了什么      查询比赛中特定时刻
  13号是谁 / 叫什么    查询球员详情
  <任何与比赛相关的问题>  自动分析回答

【数据核验】
  第90帧发生了什么      精确查询指定帧的详细状态
  不对吧，7号明明在左路   质疑系统回答，触发原始数据回溯核验
  再确认一下14号的位置    要求重新核验特定球员数据

【系统命令】
  help / 帮助          显示此帮助
  clear / 清除         清除对话上下文
  exit / 退出          退出系统"""
