"""LLM 决策引擎接线测试。

1. mock LLM：验证剧本生成 → 逐决策点调用 → Action 映射 → 事件理由 全链路接线
2. 回退验证：LLM 返回非法内容时自动回退启发式，模拟不中断
3. 真实 API 冒烟（环境有 DEEPSEEK_API_KEY 时执行一次真实剧本+决策）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.player.tracker import build_corpus_from_paths
from agents.professional.mcts.node import Action, ActionType
from agents.professional.simulation import llm_brain
from agents.professional.simulation.engine import TrajectorySimulator
from agents.professional.simulation.exporter import _validate_result
from agents.professional.simulation.llm_brain import LLMDecisionEngine

BASE = Path(__file__).resolve().parent.parent
PERSON = BASE / "TestInput/Files/12s_person2d.csv"
BALL = BASE / "TestInput/Files/12s_soccer3d.csv"

FAKE_SCRIPT = json.dumps({
    "narrative": "A队左路渗透，B队中位压迫",
    "attacking_plan": "左路渗透后内切",
    "defending_plan": "中位压迫",
    "key_instructions": {"10号": "多接应"},
    "expected_events": ["6号分边给10号", "10号内切"],
}, ensure_ascii=False)

CALL_LOG: list[str] = []


def mock_call_llm(system_prompt, user_message, config=None, max_tokens=None, retries=1):
    """模拟 LLM：剧本请求返回固定剧本；决策请求返回"传给第一个队友"。"""
    CALL_LOG.append("script" if "Scriptwriter" in system_prompt or "战术教练" in system_prompt else "decide")
    if CALL_LOG[-1] == "script":
        return FAKE_SCRIPT
    try:
        snapshot = json.loads(user_message)
    except json.JSONDecodeError:
        return None
    mates = snapshot.get("队友", [])
    target = mates[0]["球衣号"] if mates else ""
    return json.dumps({
        "action": "pass" if target else "dribble",
        "target": target,
        "direction": "forward",
        "press": "high",
        "attack_focus": "left",
        "tempo": "fast",
        "reason": "mock理由",
    }, ensure_ascii=False)


def bad_call_llm(system_prompt, user_message, config=None, max_tokens=None, retries=1):
    """始终返回非法内容，验证回退。"""
    return "这不是JSON"


def main() -> int:
    corpus = build_corpus_from_paths(PERSON, BALL)
    ok = True

    # ── 用例 1：mock LLM 全链路 ──
    print("=" * 60)
    print("用例 1: mock LLM 驱动模拟（帧102 起，自然推演）")
    print("=" * 60)
    real_call = llm_brain.call_llm
    llm_brain.call_llm = mock_call_llm
    try:
        engine = LLMDecisionEngine()
        sim = TrajectorySimulator(corpus, llm_engine=engine)
        result = sim.simulate(102, None, scenario_desc="自然推演")
    finally:
        llm_brain.call_llm = real_call

    n_script = CALL_LOG.count("script")
    n_decide = CALL_LOG.count("decide")
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
    llm_brain.call_llm = bad_call_llm
    try:
        engine2 = LLMDecisionEngine()
        sim2 = TrajectorySimulator(corpus, llm_engine=engine2)
        result2 = sim2.simulate(102, Action(ActionType.SHOOT), subject="10号")
    finally:
        llm_brain.call_llm = real_call
    if result2.events and _validate_result(result2) == []:
        print(f"[OK] 回退后模拟正常完成（{len(result2.events)} 事件，结局 {result2.outcome}）")
    else:
        print("[FAIL] 回退路径异常")
        ok = False

    # ── 用例 3：无 LLM 引擎（None）→ 纯启发式 ──
    sim3 = TrajectorySimulator(corpus)  # 不注入
    result3 = sim3.simulate(102, None)
    if result3.events:
        print(f"[OK] 无引擎时启发式正常工作（{len(result3.events)} 事件）")
    else:
        print("[FAIL] 无引擎路径异常")
        ok = False

    # ── 用例 4：真实 API 冒烟（可选） ──
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv("DEEPSEEK_API_KEY"):
        print("\n" + "=" * 60)
        print("用例 4: 真实 LLM API 冒烟（剧本 + 前若干决策）")
        print("=" * 60)
        engine4 = LLMDecisionEngine()
        sim4 = TrajectorySimulator(corpus, llm_engine=engine4)
        # 短时段：只生成 4 秒，控制调用次数
        result4 = sim4.simulate(102, Action(ActionType.PASS, target_player="7号"),
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


if __name__ == "__main__":
    sys.exit(main())
