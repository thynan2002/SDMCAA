"""「一」领域真实性与可量化性重构的单元测试（1.1/1.2/1.3/1.4，不触网）。

覆盖：goldref 战术区域/威胁值/球速/事件检测/verify 数值表达式、
metrics 的区域分类与事件 F1、scoring 的 tactical_accuracy 融合、Judge rubric v2。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import goldref, metrics as M
from eval.cases import _parse_case, lint_cases, load_cases
from eval.config import CASES_DIR, PROJECT_ROOT, EvalConfig
from eval.scoring import score_run

PERSON_12S = "TestInput/Files/12s_person2d.csv"
BALL_12S = "TestInput/Files/12s_soccer3d.csv"


@pytest.fixture(scope="module")
def stats_12s():
    return goldref.compute_stats(str(PROJECT_ROOT / PERSON_12S), str(PROJECT_ROOT / BALL_12S))


# ── 1.1 领域量化指标 ──

def test_pitch_and_speeds(stats_12s):
    xmin, xmax, ymin, ymax = stats_12s.pitch
    assert xmin < xmax and ymin < ymax
    assert stats_12s.ball_speed_max > 0
    assert 0 < stats_12s.ball_speed_avg <= stats_12s.ball_speed_max
    assert stats_12s.ball_speed_segments


def test_zone_of_bounds(stats_12s):
    # 进攻轴为 x（门线到门线，固定 1200px）；x=0 门线 → 禁区，x=600 中线 → 中场
    z = goldref.zone_of(stats_12s, 0, 350)
    assert z == "禁区"
    mid = goldref.zone_of(stats_12s, 600, 350)
    assert mid.startswith("中场")
    # 边界外夹紧不崩溃
    assert goldref.zone_of(stats_12s, 12000, -7000)


def test_threat_of_monotonic(stats_12s):
    # 越靠近球门线（x 轴）威胁越高（中路）
    near = goldref.threat_of(stats_12s, 0, 350)
    mid = goldref.threat_of(stats_12s, 600, 350)
    assert near > mid
    assert 0.0 <= near <= 1.0 and 0.0 <= mid <= 1.0


def test_ball_speed_at_frame_segment(stats_12s):
    assert goldref.ball_speed_at_frame(stats_12s, stats_12s.max_frame) > 0
    # 越界取最近段，不崩溃
    assert goldref.ball_speed_at_frame(stats_12s, stats_12s.max_frame + 1000) > 0


def test_ball_event_types(stats_12s):
    types = goldref.ball_event_types(stats_12s)
    assert isinstance(types, list) and types == sorted(types)
    assert any("带球" in t or "控球" in t or "长传" in t or "快速推进" in t for t in types)


def test_nearest_player_and_distance(stats_12s):
    nearest = goldref.nearest_player_at_second(stats_12s, 3.0)
    assert nearest is not None and nearest.endswith("号")
    d = goldref.ball_player_distance_at_second(stats_12s, int(nearest[:-1]), 3.0)
    assert d is not None and d >= 0


def test_resolve_domain_exprs(stats_12s):
    assert goldref.resolve_gold("ball_speed_max", stats_12s) == stats_12s.ball_speed_max_ms
    assert goldref.resolve_gold("ball_speed_avg", stats_12s) == stats_12s.ball_speed_avg_ms
    assert isinstance(goldref.resolve_gold("ball_zone@90", stats_12s), str)
    assert isinstance(goldref.resolve_gold("ball_side@90", stats_12s), str)
    assert isinstance(goldref.resolve_gold("threat_at_frame@90", stats_12s), (int, float))
    assert isinstance(goldref.resolve_gold("threat_max", stats_12s), (int, float))
    assert isinstance(goldref.resolve_gold("event_types", stats_12s), str)
    assert isinstance(goldref.resolve_gold("player_primary_zone:7", stats_12s), str)
    assert isinstance(goldref.resolve_gold("player_x_at_second:7,3", stats_12s), (int, float))
    assert isinstance(goldref.resolve_gold("nearest_player_at_second:3", stats_12s), str)
    # 米制化：距离/坐标金标准为米，速度金标准为米/秒
    assert goldref.resolve_gold("distance:6", stats_12s) == round(stats_12s.player_distance_m.get(6, 0.0), 1)


# ── 1.3 指标：区域分类 / 事件 F1 ──

def test_score_category():
    golds = [{"expr": "z", "label": "前场中路", "aliases": ["前场中路", "前场"]}]
    assert M.score_category("球在前场中路", golds)[0] == 1.0
    assert M.score_category("球在边路", golds)[0] == 0.0
    assert M.score_category("x", [])[0] is None
    # 别名宽松：只提"前场"也算
    assert M.score_category("球已推进到前场", golds)[0] == 1.0


def test_score_event_f1():
    gold = "控球,带球,长传,快速推进,射门"
    # 全对
    s, d = M.score_event_f1("这段球路包括控球、带球、长传、快速推进和一次射门。", gold)
    assert s == 1.0
    # 部分
    s2, _ = M.score_event_f1("主要是传球和射门。", gold)
    assert 0.0 < s2 < 1.0
    # 无金标准
    assert M.score_event_f1("随便说说", None)[0] is None
    # precision 惩罚假事件：答案提到词汇表内但金标准不存在的事件
    partial_gold = "控球,带球"
    s3, d3 = M.score_event_f1("有控球、带球和长传。", partial_gold)
    assert d3["precision"] < 1.0
    assert d3["recall"] == 1.0


# ── 1.2 用例分层 ──

def test_cases_lint_clean_with_new_fields():
    assert lint_cases(CASES_DIR) == []


def test_cases_have_levels_and_domain():
    cases = load_cases(CASES_DIR)
    assert all(1 <= c.level <= 7 for c in cases)
    assert len(cases) >= 28
    domain_ids = {"qa_ball_speed_max", "frame_ball_zone_90", "team_threat_peak",
                  "qa_player_zone_7", "timeline_event_types"}
    loaded = {c.id for c in cases}
    assert domain_ids <= loaded
    c = next(x for x in cases if x.id == "frame_ball_zone_90")
    assert c.level == 3 and c.category_gold
    v = next(x for x in cases if x.id == "verify_7s_pos")
    assert len(v.numeric) == 2  # 数值金标准已补


def test_case_default_level_from_category():
    raw = {"id": "t", "category": "hallucination",
           "dataset": {"person": PERSON_12S, "ball": BALL_12S}, "input": "q"}
    case = _parse_case(raw, Path("t.json"))
    assert case.level == 7


# ── 1.3 融合：tactical_accuracy ──

def _score(case, answer, judge=None, **kw):
    cfg = EvalConfig()
    stats = goldref.compute_stats(str(PROJECT_ROOT / PERSON_12S), str(PROJECT_ROOT / BALL_12S))
    run = {"system": "bare", "answer": answer, "error": None,
           "latency_ms": 1000, "llm_calls": 1,
           "tokens": {"total_tokens": 500}, "repeat": 0, **kw}
    return score_run(case, run, stats, cfg, judge)


def _judge(**kw):
    base = {"accuracy": 1.0, "completeness": 1.0, "reasoning": 5,
            "fabrication": 0.0, "tactical": 1.0, "failure_mode": "无"}
    base.update(kw)
    return base


def test_tactical_accuracy_judge_driven():
    cases = {c.id: c for c in load_cases(CASES_DIR)}
    c = cases["frame_ball_zone_90"]
    row = _score(c, "第90帧球在前场边路，威胁约0.45，接近球门。", judge=_judge())
    assert row["scores"]["tactical_accuracy"] == 1.0
    assert row["scores"]["accuracy"] == 1.0
    row_bad = _score(c, "球在中场边路，没有威胁。", judge=_judge(accuracy=0.2, tactical=0.2))
    assert row_bad["scores"]["tactical_accuracy"] < 0.6
    assert row_bad["scores"]["accuracy"] < 0.6


def test_tactical_accuracy_event_f1():
    cases = {c.id: c for c in load_cases(CASES_DIR)}
    c = cases["timeline_event_types"]
    full = _score(c, "有控球、带球、长传、快速推进和一次射门。", judge=_judge())
    assert full["scores"]["tactical_accuracy"] >= 0.99
    partial = _score(c, "主要是传球。", judge=_judge(tactical=0.4))
    assert partial["scores"]["tactical_accuracy"] < 0.9


def test_verify_numeric_gold_semantic():
    cases = {c.id: c for c in load_cases(CASES_DIR)}
    c = cases["verify_7s_pos"]
    # 无 Judge 时 accuracy 为 None（语义对齐，数值金标准交由 Judge 比对）
    ok = _score(c, "核验后确认7号第3秒位于(1070, 615)附近。")
    assert ok["scores"]["accuracy"] is None
    # Judge 语义比对打分
    ok_j = _score(c, "核验后确认7号第3秒位于(1070, 615)附近。", judge=_judge())
    assert ok_j["scores"]["accuracy"] == 1.0
    wrong_j = _score(c, "7号第3秒在(900, 300)（核验确认）。", judge=_judge(accuracy=0.2))
    assert wrong_j["scores"]["accuracy"] < 1.0


def test_scoring_includes_tactical_in_total():
    cfg = EvalConfig()
    assert "tactical_accuracy" in cfg.weights
    assert abs(sum(cfg.weights.values()) - 1.0) < 1e-6


# ── 1.4 Judge rubric v3 ──

def test_judge_rubric_v3():
    from eval.judge import RUBRIC_VERSION, _reference_block, JUDGE_SYSTEM_PROMPT
    assert RUBRIC_VERSION == "v3"
    assert "rubric v3" in JUDGE_SYSTEM_PROMPT
    assert "tactical" in JUDGE_SYSTEM_PROMPT  # 战术判断维度
    assert "0.8" in JUDGE_SYSTEM_PROMPT  # 细化锚点
    # 参考事实增强：category_gold 进入参考块（标签与独立重算一致）
    cases = {c.id: c for c in load_cases(CASES_DIR)}
    block = _reference_block(cases["frame_ball_zone_90"])
    assert "参考分类标签" in block
    assert "前场边路" in block
