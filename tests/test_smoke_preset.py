"""--smoke 分层评测预设的离线验证（不发 LLM、不真实执行评测）。

覆盖：
- sample_smoke_cases 确定性抽样（每 category 1 例，按 case_id 排序取首个）
- smoke 计划生成：用例数=类别数、repeats=1、systems 仅 single+multi
- _apply_overrides 对 smoke 的 cfg 覆写
- create_event 在 meta.json 中附加 preset 标记且保留 usage_fix_version
"""

from __future__ import annotations

import json


def _fake_case(case_id: str, category: str):
    from eval.cases import TestCase

    return TestCase(
        id=case_id, category=category,
        dataset_person="TestInput/Files/7s-person.csv",
        dataset_ball="TestInput/Files/7s-ball.csv",
        input="q",
    )


def test_sample_smoke_cases_real_corpus():
    """真实用例集：每 category 恰好 1 例，且取 id 排序首个。"""
    from eval.cases import load_cases
    from eval.config import CASES_DIR
    from eval.engine import sample_smoke_cases

    cases = load_cases(CASES_DIR)
    picked = sample_smoke_cases(cases)
    cats = [c.category for c in picked]
    assert len(cats) == len(set(cats)), "每个 category 只能出现一次"
    assert set(cats) == {c.category for c in cases}, "不应遗漏任何 category"
    # 确定性：按 id 排序后每类首个
    by_case = {}
    for c in sorted(cases, key=lambda x: x.id):
        by_case.setdefault(c.category, c.id)
    assert {c.category: c.id for c in picked} == by_case
    # 复现性：多次调用结果一致
    assert [c.id for c in picked] == [c.id for c in sample_smoke_cases(cases)]


def test_sample_smoke_cases_synthetic_order_independent():
    """输入顺序不影响抽样结果（确定性）。"""
    from eval.engine import sample_smoke_cases

    a = [_fake_case("z_case", "cat_a"), _fake_case("a_case", "cat_a"),
         _fake_case("m_case", "cat_b")]
    b = list(reversed(a))
    assert [c.id for c in sample_smoke_cases(a)] == ["a_case", "m_case"]
    assert [c.id for c in sample_smoke_cases(b)] == ["a_case", "m_case"]


def test_smoke_plan_offline():
    """离线构造 smoke cfg：作业数 = 类别数 × 2 系统，repeats=1。"""
    from eval.cases import load_cases
    from eval.config import CASES_DIR, EvalConfig
    from eval.engine import build_plan, sample_smoke_cases

    cases = load_cases(CASES_DIR)
    picked = sample_smoke_cases(cases)
    cfg = EvalConfig(repeats=1, systems=("single", "multi"))
    plan = build_plan(picked, cfg)
    assert len(plan) == len(picked) * 2
    assert {j["system"] for j in plan} == {"single", "multi"}
    assert {j["repeat"] for j in plan} == {0}
    assert {j["case_id"] for j in plan} == {c.id for c in picked}


def test_apply_overrides_smoke():
    """_apply_overrides：smoke 预设覆写 repeats 与 systems。"""
    import argparse

    from eval.cli import _apply_overrides
    from eval.config import EvalConfig

    args = argparse.Namespace(
        systems=None, repeats=None, no_judge=False, timeout=None,
        workers=None, quick=False, smoke=True,
    )
    cfg = _apply_overrides(EvalConfig(), args)
    assert cfg.repeats == 1
    assert cfg.systems == ("single", "multi")

    # 不带 smoke：默认路径保持不变
    args_full = argparse.Namespace(
        systems=None, repeats=None, no_judge=False, timeout=None,
        workers=None, quick=False, smoke=False,
    )
    cfg_full = _apply_overrides(EvalConfig(), args_full)
    assert cfg_full.repeats == 5
    assert cfg_full.systems == ("bare", "single", "multi")


def test_create_event_meta_preset(tmp_path):
    """meta.json 附加 preset 标记，且不动 usage_fix_version。"""
    from eval.cases import load_cases
    from eval.config import CASES_DIR, EvalConfig
    from eval.engine import create_event

    cfg = EvalConfig(repeats=1, systems=("single", "multi"), score_root=tmp_path)
    cases = load_cases(CASES_DIR)
    event_dir = create_event("smoke_test", cfg, cases[:2], preset="smoke")
    meta = json.loads((event_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["preset"] == "smoke"
    assert meta["usage_fix_version"] == 1
    assert meta["config"]["systems"] == ["single", "multi"]
    assert meta["config"]["repeats"] == 1

    # 无 preset 时不写入该键（full/quick 路径不受影响）
    cfg2 = EvalConfig(score_root=tmp_path)
    event_dir2 = create_event("full_test", cfg2, cases[:2])
    meta2 = json.loads((event_dir2 / "meta.json").read_text(encoding="utf-8"))
    assert "preset" not in meta2
    assert meta2["usage_fix_version"] == 1


def test_cli_smoke_dry_run_end_to_end(tmp_path, monkeypatch):
    """CLI 端到端（dry-run）：--smoke 抽样 + 计划落盘，不发 LLM。"""
    from eval import cli
    from eval.cases import load_cases
    from eval.config import CASES_DIR, EvalConfig

    def fake_load(cls, override_path=None):
        return EvalConfig(score_root=tmp_path)

    monkeypatch.setattr(EvalConfig, "load", classmethod(fake_load))
    rc = cli.main(["run", "--event", "smokecheck", "--smoke", "--dry-run"])
    assert rc == 0

    event_dirs = sorted(tmp_path.glob("smokecheck_*"))
    assert len(event_dirs) == 1
    event_dir = event_dirs[0]
    meta = json.loads((event_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["preset"] == "smoke"
    assert meta["usage_fix_version"] == 1

    plan = json.loads((event_dir / "plan.json").read_text(encoding="utf-8"))
    n_cats = len({c.category for c in load_cases(CASES_DIR)})
    assert len(plan) == n_cats * 2                     # 每类 1 例 × 2 系统 × 1 重复
    assert {j["system"] for j in plan} == {"single", "multi"}
    assert {j["repeat"] for j in plan} == {0}
    # 抽样结果与确定性函数一致
    from eval.engine import sample_smoke_cases

    expect_ids = {c.id for c in sample_smoke_cases(load_cases(CASES_DIR))}
    assert {j["case_id"] for j in plan} == expect_ids
