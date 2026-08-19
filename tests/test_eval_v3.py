"""eval 附加性改动（可信度增量）的单元测试（不触网）。

覆盖：
- cases.lint_warnings：拒答关键词过短过泛 / checklist 与 refusal 关键词重复
  的卫生检查警告（独立于 lint_cases，仅 warning 不阻断）；
- scoring.judge_program_agreement：程序化 accuracy 与 Judge accuracy 的
  分歧诊断（ok / disagreement / n/a，阈值 0.3），及 score_run 集成输出；
- engine.load_judge_meta：从 judge/{case}.json 缓存读回 meta
  （consistency/orders/judge_model/rubric_version），缺失时优雅返回空；
- engine.judge_disagreements：分歧汇总的 counts 统计 / 按 diff 降序 /
  top_n 截断 / 缺失键容错；
- cases._calibration_problems：坐标越界 >±5% 报错、合规（含边界）放行；
- report Judge 一致性小节最小回归：diff >0.15 标 ⚠，judge_meta 缺失时
  优雅占位不抛异常（函数级入口，不整页渲染）。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import eval.cases as cases_mod
from eval.cases import (
    CALIB_FIELD_PX,
    CALIB_SLACK,
    NumericGold,
    TestCase as CaseSpec,
    _calibration_problems,
    lint_cases,
    lint_warnings,
)
from eval.config import EvalConfig
from eval.engine import judge_disagreements, load_judge_meta
from eval.report import (
    JUDGE_CONSISTENCY_WARN,
    _judge_consistency_html,
    _judge_consistency_rows,
)
from eval.scoring import (
    JUDGE_PROGRAM_DISAGREE_THRESHOLD,
    judge_program_agreement,
    score_run,
)

PERSON_12S = "TestInput/Files/12s_person2d.csv"
BALL_12S = "TestInput/Files/12s_soccer3d.csv"

_DATASET = {"person": PERSON_12S, "ball": BALL_12S}


def _write_case(cases_dir: Path, raw: dict) -> None:
    path = cases_dir / f"{raw['id']}.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_case(case_id: str, numeric=None) -> CaseSpec:
    """直接构造内存 TestCase（不经 JSON 加载，避免触碰真实用例目录）。"""
    return CaseSpec(
        id=case_id, category="general_qa",
        dataset_person=PERSON_12S, dataset_ball=BALL_12S,
        input="场上有多少名球员？",
        numeric=list(numeric or []),
    )


def _stub_stats():
    """entity_grounding 所需的最小 stats 桩（不读任何文件）。"""
    return SimpleNamespace(player_ids={5, 7, 10}, max_frame=300, duration_s=12.0)


def _run(answer: str) -> dict:
    return {
        "system": "multi", "repeat": 0, "answer": answer,
        "latency_ms": 100, "llm_calls": 1, "tokens": {"total_tokens": 10},
        "tool_records": [],
    }


# ── lint_warnings：刷分点卫生检查 ──

def test_lint_warnings_flags_short_generic_keywords(tmp_path):
    # 单字关键词 + 双字泛词 → 各触发一条警告
    _write_case(tmp_path, {
        "id": "warn_short", "category": "hallucination",
        "dataset": _DATASET,
        "input": "99号球员表现如何？",
        "refusal": {"keywords": ["无", "无法"]},
    })
    warns = lint_warnings(tmp_path)
    assert isinstance(warns, list)
    assert len(warns) == 2
    assert any("warn_short" in w and "单字" in w for w in warns)
    assert any("warn_short" in w and "双字泛词" in w for w in warns)


def test_lint_warnings_flags_checklist_refusal_overlap(tmp_path):
    # checklist 关键词与 refusal 关键词重复 → 自相矛盾警告
    _write_case(tmp_path, {
        "id": "warn_dup", "category": "hallucination",
        "dataset": _DATASET,
        "input": "99号球员表现如何？",
        "refusal": {"keywords": ["不存在", "数据不足"]},
        "checklist": [{"text": "明确指出数据中没有99号",
                       "keywords": ["99", "数据不足"]}],
    })
    warns = lint_warnings(tmp_path)
    assert len(warns) == 1
    assert "warn_dup" in warns[0] and "重复" in warns[0]
    assert "数据不足" in warns[0]


def test_lint_warnings_clean_case_no_warning(tmp_path):
    # 具体、足够长的拒答措辞 + 无重复 → 无相关警告
    _write_case(tmp_path, {
        "id": "clean_case", "category": "hallucination",
        "dataset": _DATASET,
        "input": "99号球员表现如何？",
        "refusal": {"keywords": ["未包含该球员", "查不到相关记录"]},
        "checklist": [{"text": "明确指出数据中没有99号", "keywords": ["99"]}],
    })
    warns = lint_warnings(tmp_path)
    assert warns == []


def test_lint_warnings_independent_of_lint_cases(tmp_path):
    # lint_cases 校验通过（errors 为空），lint_warnings 仍可独立产出警告
    _write_case(tmp_path, {
        "id": "warn_only", "category": "hallucination",
        "dataset": _DATASET,
        "input": "99号球员表现如何？",
        "refusal": {"keywords": ["无法"]},
    })
    assert lint_cases(tmp_path) == []
    warns = lint_warnings(tmp_path)
    assert isinstance(warns, list) and len(warns) == 1


# ── judge_program_agreement：分歧诊断纯函数 ──

def test_agreement_threshold_constant():
    assert JUDGE_PROGRAM_DISAGREE_THRESHOLD == 0.3


def test_agreement_disagreement_when_diff_exceeds_threshold():
    ag = judge_program_agreement(1.0, 0.5)
    assert ag["flag"] == "disagreement"
    assert abs(ag["diff"] - 0.5) < 1e-9
    assert ag["program_accuracy"] == 1.0 and ag["judge_accuracy"] == 0.5


def test_agreement_ok_when_diff_within_threshold():
    # 差 < 阈值
    ag = judge_program_agreement(1.0, 0.9)
    assert ag["flag"] == "ok" and abs(ag["diff"] - 0.1) < 1e-9
    # 边界：差恰等于阈值 → 不算分歧（仅 >0.3 才 disagreement）
    ag_edge = judge_program_agreement(1.0, 0.7)
    assert ag_edge["flag"] == "ok" and abs(ag_edge["diff"] - 0.3) < 1e-9


def test_agreement_na_when_component_missing():
    for prog, judge in ((None, 0.8), (0.8, None), (None, None)):
        ag = judge_program_agreement(prog, judge)
        assert ag["flag"] == "n/a"
        assert ag["diff"] is None
        assert ag["program_accuracy"] == prog
        assert ag["judge_accuracy"] == judge


# ── judge_program_agreement：score_run 集成 ──

def test_score_run_emits_agreement_disagreement():
    # 程序金标准命中（prog_acc=1.0）+ Judge accuracy=0.5 → 差 0.5 → disagreement
    case = _make_case("agree_dis", [NumericGold(expr="player_count", value=5.0,
                                                 tolerance=0.1)])
    row = score_run(case, _run("场上共有5名球员。"), _stub_stats(),
                    EvalConfig(), {"accuracy": 0.5})
    ag = row["judge_program_agreement"]
    assert ag["flag"] == "disagreement"
    assert abs(ag["program_accuracy"] - 1.0) < 1e-9
    assert abs(ag["judge_accuracy"] - 0.5) < 1e-9


def test_score_run_emits_agreement_ok():
    case = _make_case("agree_ok", [NumericGold(expr="player_count", value=5.0,
                                                tolerance=0.1)])
    row = score_run(case, _run("场上共有5名球员。"), _stub_stats(),
                    EvalConfig(), {"accuracy": 0.9})
    assert row["judge_program_agreement"]["flag"] == "ok"


def test_score_run_emits_agreement_na():
    # 无程序金标准 → prog_acc=None → n/a（Judge 已评分也不行）
    case_no_gold = _make_case("agree_na_nogold")
    row = score_run(case_no_gold, _run("场上共有5名球员。"), _stub_stats(),
                    EvalConfig(), {"accuracy": 0.9})
    assert row["judge_program_agreement"]["flag"] == "n/a"
    # Judge 未评分（judge_scores=None）→ judge_accuracy=None → n/a
    case = _make_case("agree_na_nojudge", [NumericGold(expr="player_count",
                                                       value=5.0, tolerance=0.1)])
    row2 = score_run(case, _run("场上共有5名球员。"), _stub_stats(),
                     EvalConfig(), None)
    assert row2["judge_program_agreement"]["flag"] == "n/a"


# ── load_judge_meta：Judge 缓存 meta 读回 ──

def test_load_judge_meta_reads_cached_fields(tmp_path):
    meta = {
        "consistency": 0.75,
        "orders": ["accuracy", "completeness"],
        "judge_model": "mock-judge-v1",
        "rubric_version": "v2",
    }
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    (judge_dir / "case_a.json").write_text(
        json.dumps({"meta": meta, "scores": {"accuracy": 0.8}},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    result = load_judge_meta(tmp_path, [_make_case("case_a")])
    assert set(result.keys()) == {"case_a"}
    got = result["case_a"]
    assert got["consistency"] == 0.75
    assert got["orders"] == ["accuracy", "completeness"]
    assert got["judge_model"] == "mock-judge-v1"
    assert got["rubric_version"] == "v2"


def test_load_judge_meta_missing_file_returns_empty(tmp_path):
    # judge 目录/文件均不存在 → 优雅返回空 dict，不抛异常
    assert load_judge_meta(tmp_path, [_make_case("no_cache")]) == {}
    # event_dir 存在但无该用例缓存
    (tmp_path / "judge").mkdir()
    assert load_judge_meta(tmp_path, [_make_case("no_cache")]) == {}


def test_load_judge_meta_skips_invalid_or_metaless_cache(tmp_path):
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    # 损坏 JSON → 跳过
    (judge_dir / "bad_json.json").write_text("{not json", encoding="utf-8")
    # 历史缓存无 meta 字段 → 跳过
    (judge_dir / "no_meta.json").write_text(
        json.dumps({"scores": {"accuracy": 0.6}}), encoding="utf-8")
    cases = [_make_case("bad_json"), _make_case("no_meta"), _make_case("absent")]
    assert load_judge_meta(tmp_path, cases) == {}


# ── judge_disagreements：分歧汇总（counts / 排序 / 截断 / 容错） ──

def _agree_row(case_id: str, flag: str | None, diff=None,
               system="multi", repeat=0) -> dict:
    """构造带 judge_program_agreement 的 metrics 行（flag=None 时缺 flag 键）。"""
    row = {"system": system, "case_id": case_id, "repeat": repeat}
    ag: dict = {}
    if flag is not None:
        ag["flag"] = flag
    if flag == "disagreement":
        ag.update(program_accuracy=1.0, judge_accuracy=1.0 - (diff or 0.0),
                  diff=diff)
    row["judge_program_agreement"] = ag
    return row


def test_judge_disagreements_counts_and_missing_tolerance():
    rows = [
        _agree_row("c_ok", "ok"),
        _agree_row("c_na", "n/a", system="single"),
        _agree_row("c_dis", "disagreement", diff=0.6),
        {"system": "bare", "case_id": "c_key_missing", "repeat": 0},  # 无该键
        _agree_row("c_flag_missing", None, system="bare"),           # 缺 flag
        {"system": "bare", "case_id": "c_no_ag", "repeat": 0,
         "judge_program_agreement": None},                            # 值为 None
    ]
    out = judge_disagreements(rows)
    assert out["counts"] == {"ok": 1, "disagreement": 1, "n/a": 1}
    # 仅 disagreement 进入 top，字段回传完整
    assert len(out["top"]) == 1
    item = out["top"][0]
    assert item["case_id"] == "c_dis" and item["system"] == "multi"
    assert item["program_accuracy"] == 1.0
    assert abs(item["judge_accuracy"] - 0.4) < 1e-9
    assert abs(item["diff"] - 0.6) < 1e-9


def test_judge_disagreements_sorted_by_diff_desc_and_top_n():
    diffs = [0.35, 0.9, 0.5, 0.4, 0.7]
    rows = [_agree_row(f"case_{i}", "disagreement", diff=d)
            for i, d in enumerate(diffs)]
    out = judge_disagreements(rows, top_n=3)
    assert out["counts"]["disagreement"] == 5
    assert [it["diff"] for it in out["top"]] == [0.9, 0.7, 0.5]
    # diff 缺失（None）不抛异常，排序按 0.0 处理沉底
    rows.append({"system": "single", "case_id": "no_diff", "repeat": 0,
                 "judge_program_agreement": {"flag": "disagreement",
                                             "diff": None}})
    out2 = judge_disagreements(rows, top_n=10)
    assert len(out2["top"]) == 6
    assert out2["top"][-1]["case_id"] == "no_diff"


def test_judge_disagreements_empty_input():
    out = judge_disagreements([])
    assert out == {"counts": {"ok": 0, "disagreement": 0, "n/a": 0}, "top": []}


# ── _calibration_problems：场地标定防呆 ──

def _write_xy_csv(path: Path, xs: list[float], ys: list[float]) -> None:
    lines = ["frame_num,x,y"]
    lines += [f"{i},{x},{y}" for i, (x, y) in enumerate(zip(xs, ys))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _calib_case(case_id: str, person: str, ball: str) -> CaseSpec:
    return CaseSpec(id=case_id, category="general_qa",
                    dataset_person=person, dataset_ball=ball, input="标定校验")


def test_calibration_flags_out_of_range_and_passes_compliant(tmp_path, monkeypatch):
    monkeypatch.setattr(cases_mod, "PROJECT_ROOT", tmp_path)
    w, h = CALIB_FIELD_PX
    slack_x, slack_y = w * CALIB_SLACK, h * CALIB_SLACK
    # 合规组：坐标在场内与 ±5% 容差边界上（边界允许）
    _write_xy_csv(tmp_path / "ok_person.csv",
                  [-slack_x, w / 2, w + slack_x], [0.0, h / 2, h])
    _write_xy_csv(tmp_path / "ok_ball.csv", [0.0, w], [0.0, h])
    assert _calibration_problems(
        [_calib_case("calib_ok", "ok_person.csv", "ok_ball.csv")]) == []
    # 越界组：x 超出 +5% 容差 → 报错并点名用例与坐标轴
    _write_xy_csv(tmp_path / "oob_person.csv",
                  [0.0, w + slack_x + 1], [0.0, h / 2])
    probs = _calibration_problems(
        [_calib_case("calib_bad", "oob_person.csv", "ok_ball.csv")])
    assert len(probs) == 1
    assert "calib_bad" in probs[0] and " x " in probs[0] and "标定" in probs[0]


def test_calibration_dedup_per_dataset_and_missing_file_tolerant(tmp_path, monkeypatch):
    monkeypatch.setattr(cases_mod, "PROJECT_ROOT", tmp_path)
    w, h = CALIB_FIELD_PX
    _write_xy_csv(tmp_path / "oob2.csv", [0.0, w * 2], [0.0, h])
    _write_xy_csv(tmp_path / "ok2.csv", [0.0, w], [0.0, h])
    # 同一数据集被多用例引用 → 只校验一次，报错中合并列出使用者
    probs = _calibration_problems([
        _calib_case("dup_a", "oob2.csv", "ok2.csv"),
        _calib_case("dup_b", "oob2.csv", "ok2.csv"),
    ])
    assert len(probs) == 1 and "dup_a/dup_b" in probs[0]
    # 文件缺失 → 跳过（由 lint 的文件存在性检查负责报告），不抛异常
    assert _calibration_problems(
        [_calib_case("missing", "no_such.csv", "ok2.csv")]) == []


# ── report：Judge 一致性小节最小回归（函数级入口） ──

def test_report_consistency_warn_mark_above_threshold():
    assert JUDGE_CONSISTENCY_WARN == 0.15
    results = {"judge_meta": {
        "case_warn": {"consistency": {"single": 0.4, "multi": 0.05}},
        "case_missing_cons": {},                       # 无 consistency → 跳过
        "case_none_meta": None,                        # meta 为 None → 跳过
    }}
    rows = _judge_consistency_rows(results)
    assert ("case_warn", "single", 0.4) in rows
    assert ("case_warn", "multi", 0.05) in rows
    assert len(rows) == 2
    html = _judge_consistency_html(results)
    # 超阈值行：标黄 class + ⚠；未超阈值行：无 ⚠ 且无 warn class
    assert 'class="warn"' in html
    assert "<td>0.400</td><td>⚠</td>" in html
    assert "<td>0.050</td><td></td>" in html


def test_report_consistency_graceful_when_judge_meta_missing():
    # judge_meta 缺失 / 为 None / 空 dict → 空行提取 + 优雅占位，不抛异常
    for results in ({}, {"judge_meta": None}, {"judge_meta": {}}):
        assert _judge_consistency_rows(results) == []
        html = _judge_consistency_html(results)
        assert "无 Judge 位置互换一致性数据" in html
        assert "⚠" not in html


def test_report_consistency_diff_none_placeholder():
    # diff 为 None → 占位符 “—” 且不告警
    results = {"judge_meta": {"c": {"consistency": {"multi": None}}}}
    rows = _judge_consistency_rows(results)
    assert rows == [("c", "multi", None)]
    html = _judge_consistency_html(results)
    assert "<td>—</td><td></td>" in html
    assert 'class="warn"' not in html
