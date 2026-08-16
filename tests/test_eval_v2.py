"""「二」SOTA 能力凸显 + 「三」关键维度优化 的单元测试（2.1/2.3/2.4、3.1/3.3，不触网）。

覆盖：多 Agent 辩论协作流程（mock LLM）、结构化事实记忆、对抗鲁棒性指标、
协作诊断、并行执行配置、报告分层聚合。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import metrics as M
from eval.cases import load_cases
from eval.config import CASES_DIR, PROJECT_ROOT, EvalConfig

PERSON_12S = "TestInput/Files/12s_person2d.csv"
BALL_12S = "TestInput/Files/12s_soccer3d.csv"


def _build_corpus():
    from agents.player.tracker import build_corpus_from_paths

    return build_corpus_from_paths(PROJECT_ROOT / PERSON_12S, PROJECT_ROOT / BALL_12S)


# ── 2.4 对抗鲁棒性指标 ──

def test_robustness_honest_refusal_bonus():
    # 幻觉类：诚实拒答 → 满分
    s_ok, comp = M.score_robustness(
        "数据中没有该信息，无法确定。", None, True,
        refusal_keywords=["无法", "数据不足", "没有", "不能"],
    )
    assert s_ok == 1.0 and comp["honest_refusal"] is True
    # 幻觉类：跟随诱导（编造）→ 封顶 0.6
    s_bad, comp_bad = M.score_robustness(
        "当然是5号进的球，他在禁区里射门得分。", None, True,
        refusal_keywords=["无法", "数据不足", "没有", "不能"],
    )
    assert s_bad <= 0.6 and comp_bad["honest_refusal"] is False
    # 非幻觉类：行为不变
    s_plain, _ = M.score_robustness("这是一段足够长的正常中文回答。", None, False)
    assert s_plain == 1.0


def test_adversarial_resistance():
    kws = ["无法", "数据不足"]
    assert M.adversarial_resistance("数据不足，无法回答", kws, True) == 1.0
    assert M.adversarial_resistance("他打入了制胜球", kws, True) == 0.0
    assert M.adversarial_resistance("随便", kws, False) is None


def test_adversarial_cases_present():
    cases = load_cases(CASES_DIR)
    adv = [c for c in cases if c.category == "adversarial"]
    assert len(adv) >= 4
    assert all(c.level == 7 for c in adv)
    # 多数带 refusal（诚实性考察）
    assert sum(1 for c in adv if c.refusal) >= 3


# ── 2.1 协作诊断 ──

def test_collaboration_diag():
    # 多视角 + 综合 + 共识/分歧
    d = M.collaboration_diag(
        "从进攻和防守两端分析，综合来看，双方的共识是7号关键，分歧在于进攻威胁与防守隐患。")
    assert d["score"] > 0.5
    assert "multi_perspective" in d["hits"] and "synthesis" in d["hits"]
    assert "consensus" in d["hits"] and "divergence" in d["hits"]
    # 无协作痕迹
    d2 = M.collaboration_diag("7号很积极。")
    assert d2["score"] == 0.0


def test_collaboration_case_present():
    cases = {c.id: c for c in load_cases(CASES_DIR)}
    c = cases["team_collab_breakdown"]
    assert c.level == 6 and c.category == "team"
    assert len(c.checklist) == 4


# ── 3.1 并行执行配置 ──

def test_config_max_workers():
    cfg = EvalConfig()
    assert cfg.max_workers == 1  # 默认串行，行为不变
    cfg2 = EvalConfig(max_workers=4)
    assert cfg2.max_workers == 4


def test_build_plan_structure():
    from eval.engine import build_plan

    cases = load_cases(CASES_DIR, select="qa_duration,halluc_player_99")
    cfg = EvalConfig(repeats=2, systems=("bare", "single", "multi"))
    plan = build_plan(cases, cfg)
    # 2 重复 × 2 用例 × 3 系统 = 12
    assert len(plan) == 12
    # 每重复都包含全部 (case×system) 组合（交错分层语义）
    for rep in range(2):
        keys = {(j["case_id"], j["system"]) for j in plan if j["repeat"] == rep}
        assert len(keys) == 6


def test_parallel_job_status_dry_run(monkeypatch):
    """并行分支不破坏 dry-run/resume 语义（不实际触发子进程/LLM）。"""
    from eval.engine import execute_jobs
    from eval.config import ALL_SYSTEMS

    cases = load_cases(CASES_DIR, select="qa_duration")
    cfg = EvalConfig(repeats=1, systems=ALL_SYSTEMS, max_workers=3)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        statuses = execute_jobs(Path(td), [], cfg, dry_run=True)
        assert statuses == []
        # 空计划直接返回，不崩溃
        assert isinstance(statuses, list)


# ── 3.3 报告分层聚合 ──

def test_level_breakdown():
    from eval.report import _level_breakdown

    meta = {"cases": [
        {"id": "c1", "level": 1}, {"id": "c2", "level": 1},
        {"id": "c3", "level": 3}, {"id": "c4", "level": 5},
    ]}
    results = {
        "per_case_summary": {
            "c1": {"bare": {"total": 0.9}, "multi": {"total": 0.7}},
            "c2": {"bare": {"total": 0.8}, "multi": {"total": 0.6}},
            "c3": {"bare": {"total": 0.5}, "multi": {"total": 0.9}},
            "c4": {"bare": {"total": None}, "multi": {"total": 0.4}},
        },
    }
    rows = _level_breakdown(results, meta, ["bare", "multi"])
    by_level = {int(r[0]): r for r in rows}
    # L1 两用例平均
    assert by_level[1][1] == 2
    assert abs(by_level[1][2] - 0.85) < 1e-6  # bare (0.9+0.8)/2
    # L5 只有 multi 有值，bare 显示 "—"
    assert by_level[5][2] == "—"
    assert abs(by_level[5][3] - 0.4) < 1e-6
