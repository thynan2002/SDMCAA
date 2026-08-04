"""LLM 决策引擎接线测试（pytest 用例 + 命令行驱动）。

1. mock LLM：验证剧本生成 → 逐决策点调用 → Action 映射 → 事件理由 全链路接线
2. 回退验证：LLM 返回非法内容时自动回退启发式，模拟不中断
3. 无 LLM 引擎：纯启发式路径
4. 真实 API 冒烟（环境有 DEEPSEEK_API_KEY 时执行，pytest 中标记为 smoke 默认跳过）

`run_standalone()` 保留旧命令行契约（tools/test_llm_decision.py 调用），
输出用例明细与 [OK]/[FAIL]，退出码 0=全部通过 / 1=存在失败项。
"""

from __future__ import annotations

import os

import pytest

from agents.player.tracker import build_corpus_from_paths
from agents.professional.mcts.node import Action, ActionType
from agents.professional.simulation import llm_brain
from agents.professional.simulation.engine import TrajectorySimulator
from agents.professional.simulation.exporter import _validate_result
from agents.professional.simulation.llm_brain import LLMDecisionEngine

from tests._common import BALL_CSV, make_bad_llm, make_llm_mock, PERSON_CSV

FRAME = 102

# ═══════════════════════════════════════════════════════════════
# pytest 用例
# ═══════════════════════════════════════════════════════════════

def test_mock_full_chain(corpus, llm_mock) -> None:
    """用例 1：mock LLM 驱动模拟全链路接线。"""
    calls, _ = llm_mock

    engine = LLMDecisionEngine()
    sim = TrajectorySimulator(corpus, llm_engine=engine)
    result = sim.simulate(FRAME, None, scenario_desc="自然推演")

    assert calls.count("script") == 1, "剧本应恰好调用 1 次"
    assert calls.count("decide") >= 3, "决策点调用过少"
    assert engine.script.get("narrative"), "剧本未注入引擎"
    assert any("mock理由" in e for e in result.events), "事件未附带 LLM 决策理由"
    assert _validate_result(result) == [], "物理校验存在警告"


def test_invalid_llm_fallback(corpus, bad_llm) -> None:
    """用例 2：LLM 全部返回非法内容 → 回退启发式，模拟不中断。"""
    engine2 = LLMDecisionEngine()
    sim2 = TrajectorySimulator(corpus, llm_engine=engine2)
    result2 = sim2.simulate(FRAME, Action(ActionType.SHOOT), subject="10号")
    assert result2.events, "回退后模拟无事件"
    assert _validate_result(result2) == [], "回退路径物理校验异常"


def test_no_engine_heuristic(corpus) -> None:
    """用例 3：无 LLM 引擎（None）→ 纯启发式正常工作。"""
    sim3 = TrajectorySimulator(corpus)
    result3 = sim3.simulate(FRAME, None)
    assert result3.events, "无引擎时启发式路径异常"


@pytest.mark.smoke
def test_real_api_smoke(corpus) -> None:
    """用例 4：真实 LLM API 冒烟（短时段控制调用次数）。"""
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("未配置 DEEPSEEK_API_KEY，真实 API 冒烟跳过")

    engine4 = LLMDecisionEngine()
    sim4 = TrajectorySimulator(corpus, llm_engine=engine4)
    result4 = sim4.simulate(FRAME, Action(ActionType.PASS, target_player="7号"),
                            subject="10号", duration_seconds=4.0,
                            scenario_desc="如果10号直接短传给7号")
    assert engine4.call_count >= 2, f"真实 LLM 调用次数过少: {engine4.call_count}"
    assert _validate_result(result4) == [], "真实 API 冒烟物理校验异常"


# ═══════════════════════════════════════════════════════════════
# 命令行驱动（保留旧 tools/test_llm_decision.py 契约）
# ═══════════════════════════════════════════════════════════════

def run_standalone(corpus=None) -> int:
    """完整接线套件（打印 [OK]/[FAIL]），返回退出码 0/1。"""
    if corpus is None:
        corpus = build_corpus_from_paths(PERSON_CSV, BALL_CSV)
    ok = True

    # ── 用例 1：mock LLM 全链路 ──
    print("=" * 60)
    print("用例 1: mock LLM 驱动模拟（帧102 起，自然推演）")
    print("=" * 60)
    calls, mock = make_llm_mock()
    real_call = llm_brain.call_llm
    llm_brain.call_llm = mock
    try:
        engine = LLMDecisionEngine()
        sim = TrajectorySimulator(corpus, llm_engine=engine)
        result = sim.simulate(FRAME, None, scenario_desc="自然推演")
    finally:
        llm_brain.call_llm = real_call

    n_script = calls.count("script")
    n_decide = calls.count("decide")
    print(f"LLM 调用: 剧本 {n_script} 次, 决策 {n_decide} 次")
    if n_script != 1:
        print("[FAIL] 剧本应恰好调用 1 次")
        ok = False
    if n_decide < 3:
        print("[FAIL] 决策点调用过少")
        ok = False
    if not engine.script.get("narrative"):
        print("[FAIL] 剧本未注入引擎")
        ok = False
    has_reason = any("mock理由" in e for e in result.events)
    print(f"事件示例: {result.events[:4]}")
    if not has_reason:
        print("[FAIL] 事件未附带 LLM 决策理由")
        ok = False
    warnings = _validate_result(result)
    if warnings:
        print(f"[FAIL] 物理警告: {warnings[:3]}")
        ok = False
    else:
        print("[OK] 物理校验通过")
    print("[OK] mock 全链路接线正确" if ok else "[FAIL] 用例 1 未通过")

    # ── 用例 2：LLM 非法输出 → 回退启发式 ──
    print("\n" + "=" * 60)
    print("用例 2: LLM 全部返回非法内容 → 回退启发式")
    print("=" * 60)
    llm_brain.call_llm = make_bad_llm()
    try:
        engine2 = LLMDecisionEngine()
        sim2 = TrajectorySimulator(corpus, llm_engine=engine2)
        result2 = sim2.simulate(FRAME, Action(ActionType.SHOOT), subject="10号")
    finally:
        llm_brain.call_llm = real_call
    if result2.events and _validate_result(result2) == []:
        print(f"[OK] 回退后模拟正常完成（{len(result2.events)} 事件，结局 {result2.outcome}）")
    else:
        print("[FAIL] 回退路径异常")
        ok = False

    # ── 用例 3：无 LLM 引擎（None）→ 纯启发式 ──
    sim3 = TrajectorySimulator(corpus)
    result3 = sim3.simulate(FRAME, None)
    if result3.events:
        print(f"[OK] 无引擎时启发式正常工作（{len(result3.events)} 事件）")
    else:
        print("[FAIL] 无引擎路径异常")
        ok = False

    # ── 用例 4：真实 API 冒烟（可选） ──
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv("DEEPSEEK_API_KEY"):
        print("\n" + "=" * 60)
        print("用例 4: 真实 LLM API 冒烟（剧本 + 前若干决策）")
        print("=" * 60)
        engine4 = LLMDecisionEngine()
        sim4 = TrajectorySimulator(corpus, llm_engine=engine4)
        result4 = sim4.simulate(FRAME, Action(ActionType.PASS, target_player="7号"),
                                subject="10号", duration_seconds=4.0,
                                scenario_desc="如果10号直接短传给7号")
        print(f"真实 LLM 调用次数: {engine4.call_count}")
        print(f"剧本: {engine4.script.get('narrative', '(无)')}")
        print("事件链:")
        for e in result4.events[:10]:
            print(f"  · {e}")
        if engine4.call_count >= 2 and _validate_result(result4) == []:
            print("[OK] 真实 API 冒烟通过")
        else:
            print("[FAIL] 真实 API 冒烟异常")
            ok = False
    else:
        print("\n[跳过] 未配置 DEEPSEEK_API_KEY，真实 API 冒烟跳过")

    print("\n" + ("[PASS] 全部用例通过" if ok else "[FAIL] 存在失败用例"))
    return 0 if ok else 1
