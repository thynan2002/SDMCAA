"""评测框架离线单元测试（不触网，不依赖 API Key）。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval import goldref, metrics as M
from eval.cases import lint_cases, load_cases
from eval.config import CASES_DIR, PROJECT_ROOT, EvalConfig
from eval import stats as ST

PERSON_12S = "TestInput/Files/12s_person2d.csv"
BALL_12S = "TestInput/Files/12s_soccer3d.csv"


@pytest.fixture(scope="module")
def stats_12s():
    return goldref.compute_stats(str(PROJECT_ROOT / PERSON_12S), str(PROJECT_ROOT / BALL_12S))


# ── 配置 ──

def test_weights_sum_to_one_default():
    cfg = EvalConfig()
    assert math.isclose(sum(cfg.weights.values()), 1.0)


def test_config_rejects_bad_weights():
    with pytest.raises(ValueError):
        EvalConfig(weights={"accuracy": 0.5})


# ── 用例集 ──

def test_cases_lint_clean():
    assert lint_cases(CASES_DIR) == []


def test_cases_load_and_gold_resolve():
    cases = load_cases(CASES_DIR)
    assert len(cases) >= 20
    numeric_cases = [c for c in cases if c.numeric]
    assert all(g.value is not None for c in numeric_cases for g in c.numeric)
    # 幻觉用例存在且带 refusal 金标准
    halluc = [c for c in cases if c.category == "hallucination"]
    assert len(halluc) >= 3
    assert all(c.refusal for c in halluc)


def test_case_selection_filter():
    cases = load_cases(CASES_DIR, select="hallucination")
    assert cases and all(c.category == "hallucination" for c in cases)
    with pytest.raises(ValueError):
        load_cases(CASES_DIR, select="nonexistent_case")


# ── 金标准独立重算 ──

def test_goldref_independent_computation(stats_12s):
    # 与被测系统工具层无关的基本事实
    import pandas as pd

    dfp = pd.read_csv(PROJECT_ROOT / PERSON_12S)
    dfb = pd.read_csv(PROJECT_ROOT / BALL_12S)
    assert stats_12s.player_count == dfp["track_id"].nunique()
    assert stats_12s.max_frame == int(max(dfp["frame_num"].max(), dfb["frame_num"].max()))
    assert math.isclose(stats_12s.ball_z_max, float(dfb["z"].max()), rel_tol=1e-9)
    assert stats_12s.player_teams  # 每个球员有队伍


def test_goldref_ball_interpolation(stats_12s):
    # 已知帧直接命中，越界返回 None
    known = stats_12s.ball_frames[0]
    assert goldref.ball_at_frame(stats_12s, known) is not None
    assert goldref.ball_at_frame(stats_12s, stats_12s.max_frame + 100) is None


def test_goldref_expr_resolver(stats_12s):
    assert goldref.resolve_gold("player_count", stats_12s) == stats_12s.player_count
    assert goldref.resolve_gold("ball_zmax", stats_12s) == round(stats_12s.ball_z_max, 1)
    winner = goldref.resolve_gold("distance_winner:6,7", stats_12s)
    assert winner in ("6号", "7号")
    with pytest.raises(ValueError):
        goldref.resolve_gold("bad_expr", stats_12s)


def test_bare_summary_deterministic(stats_12s):
    s1 = goldref.build_bare_summary(stats_12s)
    s2 = goldref.build_bare_summary(stats_12s)
    assert s1 == s2
    assert "【比赛数据总览】" in s1
    assert "【球员轨迹原始数据】" in s1  # 信息给足：原始数据全量入 prompt


# ── 程序化指标 ──

def test_score_numeric_tolerance():
    golds = [{"expr": "x", "value": 100.0, "tolerance": 5}]
    assert M.score_numeric("答案约为 101", golds)[0] == 1.0
    assert M.score_numeric("答案约为 108", golds)[0] == 0.5   # 2*tol 内半分
    assert M.score_numeric("答案是 500", golds)[0] == 0.0
    assert M.score_numeric("没有数字", golds)[0] == 0.0
    assert M.score_numeric("无关", [])[0] is None


def test_score_winner_mentions():
    golds = [{"expr": "distance_winner:6,7", "value": "6号"}]
    s, d = M.score_winner("6号更积极，6号跑动更多", golds)
    assert s == 1.0
    s2, _ = M.score_winner("7号更积极", golds)
    assert s2 == 0.0


def test_score_direction():
    golds = [{"expr": "e", "keywords_pos": ["上升"], "keywords_neg": ["下降"]}]
    assert M.score_direction("胜率上升", golds)[0] == 1.0
    assert M.score_direction("胜率下降", golds)[0] == 0.0
    assert M.score_direction("既上升又下降", golds)[0] == 0.7
    assert M.score_direction("无关键词", golds)[0] == 0.2


def test_score_refusal():
    refusal = {"keywords": ["不存在", "数据不足"]}
    assert M.score_refusal("数据中没有99号该球员不存在", refusal)[0] == 1.0
    assert M.score_refusal("99号表现非常好", refusal)[0] == 0.0
    assert M.score_refusal("x", None)[0] is None


def test_entity_grounding_excludes_question_entities(stats_12s):
    # 问题中提到的实体不计入（幻觉用例的 99号 引用不算编造）
    s, d = M.entity_grounding("数据中没有99号", stats_12s, "99号球员表现怎么样？")
    assert s is None or d["mentions"] == 0
    # 答案中编造的实体计入
    s2, d2 = M.entity_grounding("77号在第500帧完成了射门", stats_12s, "发生了什么")
    assert s2 == 0.0 and "77号" in d2["invalid"] and "第500帧" in d2["invalid"]
    # 合法实体
    s3, _ = M.entity_grounding("第90帧球被传出", stats_12s, "x")
    assert s3 == 1.0


def test_score_checklist():
    items = [{"text": "高度", "keywords": ["米"]}]
    assert M.score_checklist("高度约3米", items)[0] == 1.0
    assert M.score_checklist("不知道", items)[0] == 0.0


def test_score_efficiency_clamped():
    from eval.config import EfficiencyRefs

    refs = EfficiencyRefs(latency_ms=1000, llm_calls=10, total_tokens=1000)
    s, _ = M.score_efficiency(500, 5, 500, refs)
    assert s == 1.0  # 全部优于参考 → 封顶 1
    s2, _ = M.score_efficiency(2000, 20, 2000, refs)
    assert math.isclose(s2, 0.5)
    s3, _ = M.score_efficiency(None, None, None, refs)
    assert s3 == 0.0


def test_robustness():
    s_ok, _ = M.score_robustness("这是一段足够长的正常中文回答。", None, False)
    assert s_ok == 1.0
    s_empty, _ = M.score_robustness("", None, False)
    assert s_empty < 0.5


def test_tool_use_diag():
    records = [
        {"name": "a", "args": {}, "ok": True},
        {"name": "a", "args": {}, "ok": True},   # 冗余
        {"name": "b", "args": {}, "ok": False},  # 失败
    ]
    d = M.tool_use_diag(records)
    assert d["calls"] == 3
    assert d["error_rate"] == pytest.approx(1 / 3, abs=0.01)
    assert d["redundancy"] == pytest.approx(1 / 3, abs=0.01)
    assert M.tool_use_diag(None)["score"] is None


def test_stability():
    assert M.stability_score(["abc"]) is None
    s = M.stability_score(["abc", "abc", "abc"])
    assert s == 1.0
    s2 = M.stability_score(["abc", "xyz"])
    assert 0.0 <= s2 < 0.5


# ── 统计 ──

def test_bootstrap_ci_covers_mean():
    vals = [0.5, 0.6, 0.55, 0.52, 0.58, 0.61, 0.49]
    lo, hi = ST.bootstrap_ci(vals, seed=1)
    mean = sum(vals) / len(vals)
    assert lo < mean < hi


def test_paired_tests_and_holm():
    x = [0.9, 0.8, 0.85, 0.7, 0.95]
    y = [0.45, 0.25, 0.3, 0.2, 0.55]
    out = ST.paired_tests(x, y)
    assert out["n"] == 5
    assert out["wilcoxon_p"] is not None and out["wilcoxon_p"] < 0.05
    assert out["cohens_d_paired"] > 1
    adj = ST.holm_correction({"a": 0.01, "b": 0.04, "c": 0.03})
    # Holm step-down: a=3*0.01=0.03; c=2*0.03=0.06; b=max(1*0.04, 0.06)=0.06
    assert adj["a"] == 0.03 and adj["b"] == 0.06 and adj["c"] == 0.06


def test_aggregate_structure():
    per_case = {
        "c1": {
            "bare": {"total": [0.5, 0.6], "extended_total": None,
                     "metrics": {"accuracy": [0.5, 0.6], "grounding": [1, 1],
                                 "completeness": [0.5, 0.5], "reasoning": [0.6, 0.6],
                                 "efficiency": [1, 1], "robustness": [1, 1]},
                     "stability": 0.8, "tool_use": [None, None]},
            "multi": {"total": [0.8, 0.9], "extended_total": [0.8, 0.9],
                      "metrics": {"accuracy": [0.9, 0.95], "grounding": [1, 1],
                                  "completeness": [0.8, 0.8], "reasoning": [0.9, 0.9],
                                  "efficiency": [0.5, 0.5], "robustness": [1, 1]},
                      "stability": 0.9, "tool_use": [0.9, 1.0]},
        },
    }
    agg = ST.aggregate(per_case)
    assert set(agg["systems"]) == {"bare", "multi"}
    assert agg["systems"]["multi"]["total_mean"] > agg["systems"]["bare"]["total_mean"]
    comp = agg["comparisons"]["bare_vs_multi"]
    assert "total" in comp
    # 单用例配对样本不足 → 如实标注（不少于 3 个用例才能检验）
    assert comp["total"].get("note") or "wilcoxon_p" in comp["total"]


# ── trace 对齐 ──

def _write_trace(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines),
                    encoding="utf-8")


def test_trace_parse_and_normalize(tmp_path: Path):
    from eval.tracealign import normalize, parse_trace

    trace = tmp_path / "trace_x.jsonl"
    _write_trace(trace, [
        {"type": "run_start", "run_id": "r", "ts": 1.0},
        {"type": "span_start", "run_id": "r", "span_id": "r-00001", "parent_id": None,
         "kind": "turn", "name": "turn[0]", "ts": 1.0, "attrs": {"input": "q"}},
        {"type": "span_start", "run_id": "r", "span_id": "r-00002", "parent_id": "r-00001",
         "kind": "llm", "name": "llm_call[0]", "ts": 1.1,
         "attrs": {"system_len": 10, "user_len": 20, "tools": ["get_frame_snapshot"]}},
        {"type": "span_end", "run_id": "r", "span_id": "r-00002", "parent_id": "r-00001",
         "kind": "llm", "name": "llm_call[0]", "ts": 1.2, "dur_ms": 100.0, "status": "ok"},
        {"type": "event", "run_id": "r", "span_id": "r-00001", "kind": "tool",
         "name": "tool[0]", "ts": 1.25, "attrs": {"name": "get_frame_snapshot", "ok": True}},
        {"type": "span_start", "run_id": "r", "span_id": "r-00003", "parent_id": "r-00001",
         "kind": "llm", "name": "llm_call[1]", "ts": 1.3,
         "attrs": {"system_len": 10, "user_len": 20}},
        {"type": "span_end", "run_id": "r", "span_id": "r-00003", "parent_id": "r-00001",
         "kind": "llm", "name": "llm_call[1]", "ts": 1.4, "dur_ms": 200.0, "status": "ok"},
        {"type": "span_end", "run_id": "r", "span_id": "r-00001", "parent_id": None,
         "kind": "turn", "name": "turn[0]", "ts": 1.5, "dur_ms": 500.0, "status": "ok"},
    ])
    spans = parse_trace(trace)
    steps = normalize("single", spans)
    assert [s["kind"] for s in steps] == ["llm", "tool", "llm"]
    # single：中间轮（带工具请求）→ retrieval，最终轮 → synthesis
    assert steps[0]["phase"] == "retrieval"
    assert steps[2]["phase"] == "synthesis"
    assert steps[0]["context_chars"] == 30
    assert steps[2]["context_chars"] == 30 + 300 + 30  # 工具结果回填计入上下文
    assert steps[0]["dur_ms"] == 100.0


def test_normalize_bare_single_step_synthesis(tmp_path: Path):
    from eval.tracealign import normalize, parse_trace

    trace = tmp_path / "t2.jsonl"
    _write_trace(trace, [
        {"type": "span_start", "run_id": "r", "span_id": "r-00001", "parent_id": None,
         "kind": "llm", "name": "llm_call[0]", "ts": 1.0,
         "attrs": {"system_len": 5, "user_len": 5}},
        {"type": "span_end", "run_id": "r", "span_id": "r-00001", "parent_id": None,
         "kind": "llm", "name": "llm_call[0]", "ts": 2.0, "dur_ms": 1000.0, "status": "ok"},
    ])
    steps = normalize("bare", parse_trace(trace))
    assert len(steps) == 1 and steps[0]["phase"] == "synthesis"


# ── 评分融合 ──

def test_scoring_blends_missing_judge():
    from eval.cases import _parse_case
    from eval.scoring import score_run

    raw = {
        "id": "t", "category": "general_qa",
        "dataset": {"person": PERSON_12S, "ball": BALL_12S},
        "input": "球员数?",
        "numeric": [{"expr": "player_count", "value": 21, "tolerance": 0}],
    }
    import dataclasses

    case = _parse_case(raw, Path("t.json"))
    run = {"system": "bare", "answer": "共21名球员", "error": None,
           "latency_ms": 1000, "llm_calls": 1,
           "tokens": {"total_tokens": 500}, "repeat": 0}
    cfg = EvalConfig()
    stats = goldref.compute_stats(str(PROJECT_ROOT / PERSON_12S), str(PROJECT_ROOT / BALL_12S))
    row = score_run(case, run, stats, cfg, judge_scores=None)
    assert row["scores"]["accuracy"] is None        # 语义对齐：无 Judge 时 accuracy 缺失
    assert row["scores"]["reasoning"] is None       # Judge 缺失
    assert row["total"] is not None
    assert row["total"] <= 1.0                       # 缺失分量重归一化不得放大总分
    assert row["judge_note"]  # 标注重归一化


def test_judge_json_extraction():
    from eval.judge import _extract_json

    text = '前置说明 {"甲": {"accuracy": 0.8}, "乙": {"accuracy": 0.5}} 后置'
    obj = _extract_json(text)
    assert obj["甲"]["accuracy"] == 0.8
    assert _extract_json("完全不是 JSON") is None
    assert _extract_json("```json\n{\"甲\": {\"accuracy\": 1}}\n```") is not None


def test_usage_totals_shape(monkeypatch):
    """usage 汇总必须与 result schema 对齐（llm_calls 标量 + tokens 嵌套字典）。"""
    import agents.llm_client as llm_mod
    from eval import runners

    monkeypatch.setattr(
        llm_mod, "take_usage_events",
        lambda: [{"prompt_tokens": 10, "completion_tokens": 5,
                  "total_tokens": 15, "ts": 0}],
    )
    totals = runners._usage_totals()
    assert totals["llm_calls"] == 1
    assert totals["tokens"]["total_tokens"] == 15   # 嵌套字典，可直接 result.update
    # 模拟 _base_result 默认值被正确覆盖
    result = {"llm_calls": 0,
              "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    result.update(totals)
    assert result["tokens"]["total_tokens"] == 15 and result["llm_calls"] == 1
