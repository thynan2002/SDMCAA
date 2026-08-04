"""反事实轨迹生成端到端测试（pytest 用例 + 命令行驱动）。

测试场景（基于 12s 测试数据，干预帧 102）：
  A. 强制直接射门 → 生成到原始数据结束
  B. 强制短传给队友 → 仅生成后面 5 秒
  C. 无强制动作（纯自然推演）
  D. 帧 0 干预：6号 传给尚未进场的 10号

校验：
  1. 生成段物理校验（运动学限速 / 场地边界 / 球速上限）
  2. 导出 CSV 格式与输入严格一致（列名、列序、小数位）
  3. 干预帧之前的行与原始文件逐字节一致
  4. 生成的 CSV 能被 build_corpus_from_paths 重新加载（格式闭环）
  5. 攻守方向约束 / 未知球员证据 / 距离档位分类

`run_standalone()` 保留旧命令行契约（tools/test_counterfactual_sim.py 调用），
输出场景明细与 [OK]/[FAIL]，退出码 0=全部通过 / 1=存在失败项。
"""

from __future__ import annotations

from math import hypot
from pathlib import Path

from agents.player.tracker import build_corpus_from_paths
from agents.professional.mcts.node import Action, ActionType
from agents.professional.simulation.engine import TrajectorySimulator
from agents.professional.simulation.exporter import export_simulation

from tests._common import (
    BALL_CSV,
    FRAME,
    PERSON_CSV,
    PROJECT_ROOT,
    compute_holder,
    count_pass_pairs,
    run_scenario,
    verify_generated_speeds,
    verify_prefix_identical,
    verify_roundtrip,
)

# ═══════════════════════════════════════════════════════════════
# pytest 用例
# ═══════════════════════════════════════════════════════════════

def _check_export(corpus, person_path: Path, ball_path: Path, cut: int) -> None:
    assert verify_prefix_identical(PERSON_CSV, person_path, cut, "球员文件")
    assert verify_prefix_identical(BALL_CSV, ball_path, cut, "足球文件")
    assert verify_generated_speeds(person_path, cut)
    assert verify_roundtrip(person_path, ball_path)


def test_scenario_a_forced_shot(corpus, tmp_path) -> None:
    """强制直接射门 → 生成到原始数据结束。"""
    pa, ba, ra, wa = run_scenario(corpus, tmp_path, "A: 强制直接射门 → 补全到原始结束",
                                  Action(ActionType.SHOOT), tag="shoot")
    assert ra.start_frame == FRAME
    assert ra.events, "事件链为空"
    assert not wa, f"物理校验存在警告: {wa[:5]}"
    _check_export(corpus, pa, ba, FRAME)


def test_scenario_b_short_pass(corpus, tmp_path) -> None:
    """强制短传给队友 → 仅生成后面 5 秒。"""
    holder, _ = compute_holder(corpus)
    assert holder is not None, "干预帧无法判定控球者"
    teammate = next((t for t in corpus.players
                     if t != holder and corpus.players[t].color == corpus.players[holder].color),
                    None)
    pb, bb, rb, wb = run_scenario(corpus, tmp_path, f"B: 强制短传给 {teammate}号 → 仅生成后 5 秒",
                                  Action(ActionType.PASS, target_player=f"{teammate}号"),
                                  duration=5.0, tag="pass")
    assert rb.end_frame - rb.start_frame <= 5 * 30 + 3
    assert not wb, f"物理校验存在警告: {wb[:5]}"
    _check_export(corpus, pb, bb, FRAME)


def test_scenario_c_natural(corpus, tmp_path) -> None:
    """无强制动作（纯自然推演）。"""
    pc, bc, rc, wc = run_scenario(corpus, tmp_path, "C: 无强制动作（自然推演）",
                                  None, tag="natural")
    assert rc.events, "自然推演事件链为空"
    assert not wc, f"物理校验存在警告: {wc[:5]}"
    _check_export(corpus, pc, bc, FRAME)


def test_scenario_d_frame0_intervention(corpus, tmp_path) -> None:
    """帧 0 干预：6号 传给 10号（10号 首帧=73，干预帧不在场）。"""
    sim_d = TrajectorySimulator(corpus)
    rd = sim_d.simulate(intervention_frame=0,
                        forced_action=Action(ActionType.PASS, target_player="10号"),
                        subject="6号")

    # D-1: 10号必须出现在生成数据中（不消失）
    ten_kfs = rd.player_keyframes.get("10", [])
    assert ten_kfs, "10号 未出现在生成数据中"
    assert ten_kfs[0][0] >= 73, "10号 应于原首帧动态加入"

    # D-2: 必须存在 6号→10号 的传球事件（等待后执行）
    assert any("传给 10号" in e and "6号" in e for e in rd.events), "事件链中没有 6号→10号 传球"

    # D-2b: 不得出现两人反复互传（"乒乓球"）：同一对互传组合 ≤2 次
    pingpong = {p: c for p, c in count_pass_pairs(rd.events).items() if c > 2}
    assert not pingpong, f"存在乒乓互传: {pingpong}"

    # D-3: 守门员全程不得远离其把守位置（home 点）
    gk_kfs = rd.player_keyframes.get("21", [])
    if gk_kfs:
        hx, hy = gk_kfs[0][1], gk_kfs[0][2]
        max_dev = max(hypot(x - hx, y - hy) for _, x, y in gk_kfs)
        assert max_dev <= 150, f"门将偏离把守位置 {max_dev:.0f}px"

    # D-4: 攻方球员位置均需物理合法（通用校验）
    _, _, wd = export_simulation(corpus, rd, tmp_path, tag="d")
    assert not wd, f"场景D物理警告: {wd[:5]}"


def test_attack_directions_opposite(corpus) -> None:
    """专项：攻守方向约束 — A/B 两队必须反向。"""
    sim_chk = TrajectorySimulator(corpus)
    players0, _ = sim_chk._build_players(0)
    dirs = sim_chk._attack_directions(players0)
    team_dirs = {c: d for c, d in dirs.items()}
    assert {team_dirs.get("A"), team_dirs.get("B")} != {None}, f"队伍标记缺失: {team_dirs}"
    assert len({team_dirs.get("A"), team_dirs.get("B")}) == 2, f"A/B 两队进攻方向相同: {team_dirs}"


def test_unknown_player_evidence(corpus) -> None:
    """专项：'未知球员'事件证据提取。"""
    from agents.professional.agents.data_verifier import DataVerifierAgent

    verifier = DataVerifierAgent()
    evidence = verifier._extract_event_evidence(
        "为啥会有未知球员？核查一下",
        "5.9～6.5秒 6号 传向一名未知球员",
        corpus,
    )
    n_unknown = sum(
        1 for e in corpus.ball_analysis.events
        if e.nearest_player == "?" or e.nearest_player_end == "?"
    )
    assert ("判定规则" in evidence and "实际距离" in evidence) \
        or "不存在「未知」标注" in evidence, f"证据不完整（未知端点数 {n_unknown}）"


def test_distance_tiers() -> None:
    """专项：距离档位分类（单位换算 / 阈值 / 跨中场语义）。"""
    from agents.prompts import distance_tier

    cases = [
        ("7号→10号 约28米同侧", distance_tier(313, 681.0, 505.0), "中距离"),
        ("短传 9米", distance_tier(100), "短距离"),
        ("跨中线 41米", distance_tier(450, 600.0, 100.0), "跨越半场的长传"),
        ("同侧转移 32米", distance_tier(355, 600.0, 420.0), "长距离转移"),
    ]
    for name, got, want in cases:
        assert got == want, f"{name}: 期望[{want}] 实得[{got}]"


# ═══════════════════════════════════════════════════════════════
# 命令行驱动（保留旧 tools/test_counterfactual_sim.py 契约）
# ═══════════════════════════════════════════════════════════════

def run_standalone(out_dir: Path | None = None, corpus=None) -> int:
    """完整端到端套件（打印 [OK]/[FAIL]），返回退出码 0/1。"""
    out_dir = out_dir or (PROJECT_ROOT / "Output")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if corpus is None:
        corpus = build_corpus_from_paths(PERSON_CSV, BALL_CSV)

    print(f"数据: {corpus.player_count} 名球员（门将 {len(corpus.goalkeepers())} 人）,"
          f"球帧 {min(corpus.ball_frames)}-{max(corpus.ball_frames)}")
    ball_at = corpus.ball_frames.get(FRAME)
    print(f"干预帧 {FRAME}: 球位于 {ball_at}")

    holder, dist = compute_holder(corpus, FRAME)
    if holder:
        print(f"干预帧控球者: {holder}号（距球 {dist:.0f}px）")

    results = []

    # 场景 A：直接射门，补全到结束
    pa, ba, ra, wa = run_scenario(corpus, out_dir, "A: 强制直接射门 → 补全到原始结束",
                                  Action(ActionType.SHOOT), tag="shoot")
    results.append(("A", pa, ba, ra, wa, FRAME))

    # 场景 B：短传给队友，仅生成后面 5 秒
    teammate = None
    if holder and holder in corpus.players:
        hc = corpus.players[holder].color
        teammate = next((t for t in corpus.players
                         if t != holder and corpus.players[t].color == hc), None)
    pb, bb, rb, wb = run_scenario(corpus, out_dir, f"B: 强制短传给 {teammate}号 → 仅生成后 5 秒",
                                  Action(ActionType.PASS, target_player=f"{teammate}号"),
                                  duration=5.0, tag="pass")
    results.append(("B", pb, bb, rb, wb, FRAME))

    # 场景 C：纯自然推演
    pc, bc, rc, wc = run_scenario(corpus, out_dir, "C: 无强制动作（自然推演）", None, tag="natural")
    results.append(("C", pc, bc, rc, wc, FRAME))

    # 场景 D：帧 0 干预，6号 传给 10号（10号 首帧=73，干预帧不在场）
    print("\n" + "=" * 66)
    print("场景 D: 帧0干预 6号→10号（10号第73帧才进入视野）")
    print("=" * 66)
    sim_d = TrajectorySimulator(corpus)
    rd = sim_d.simulate(intervention_frame=0,
                        forced_action=Action(ActionType.PASS, target_player="10号"),
                        subject="6号")
    print("事件时间线（前 12 条）:")
    for e in rd.events[:12]:
        print(f"  · {e}")
    d_ok = True

    # D-1: 10号必须出现在生成数据中（不消失）
    ten_kfs = rd.player_keyframes.get("10", [])
    if ten_kfs and ten_kfs[0][0] >= 73:
        print(f"[OK] 10号 于帧 {ten_kfs[0][0]} 动态加入，共 {len(ten_kfs)} 个关键帧")
    else:
        print("[FAIL] 10号 未出现在生成数据中")
        d_ok = False

    # D-2: 必须存在 6号→10号 的传球事件（等待后执行）
    if any("传给 10号" in e and "6号" in e for e in rd.events):
        print("[OK] 事件链中存在 6号→10号 传球")
    else:
        print("[FAIL] 事件链中没有 6号→10号 传球")
        d_ok = False

    # D-2b: 不得出现两人反复互传（"乒乓球"）：同一对互传组合 ≤2 次
    pair_count = count_pass_pairs(rd.events)
    pingpong = {p: c for p, c in pair_count.items() if c > 2}
    if not pingpong:
        print(f"[OK] 无乒乓互传（最多互传对: {max(pair_count.values(), default=0)} 次）")
    else:
        print(f"[FAIL] 存在乒乓互传: {pingpong}")
        d_ok = False

    # D-3: 守门员全程不得远离其把守位置（home 点）
    gk_kfs = rd.player_keyframes.get("21", [])
    if gk_kfs:
        hx, hy = gk_kfs[0][1], gk_kfs[0][2]
        max_dev = max(hypot(x - hx, y - hy) for _, x, y in gk_kfs)
        if max_dev <= 150:
            print(f"[OK] 门将最大偏离把守位置 {max_dev:.0f}px（≤150）")
        else:
            print(f"[FAIL] 门将偏离把守位置 {max_dev:.0f}px")
            d_ok = False

    # D-4: 攻方球员位置均需物理合法（通用校验）
    _, _, wd = export_simulation(corpus, rd, out_dir, tag="d")
    if wd:
        print(f"[FAIL] 场景D物理警告: {wd[:5]}")
        d_ok = False
    else:
        print("[OK] 场景D物理校验全部通过")
    results.append(("D", *export_simulation(corpus, rd, out_dir, tag="d")[:2], rd, wd, 0))

    # 专项验证 1：攻守方向两队必须反向
    print("\n[专项] 攻守方向约束:")
    sim_chk = TrajectorySimulator(corpus)
    players0, _ = sim_chk._build_players(0)
    dirs = sim_chk._attack_directions(players0)
    team_dirs = {c: d for c, d in dirs.items()}
    print(f"  各队方向: {team_dirs}")
    if len({team_dirs.get('A'), team_dirs.get('B')}) == 2:
        print("[OK] A/B 两队进攻方向相反")
    else:
        print("[FAIL] A/B 两队进攻方向相同")
        d_ok = False

    # 专项验证 2：未知球员事件证据提取
    print("\n[专项] '未知球员'事件证据提取:")
    from agents.professional.agents.data_verifier import DataVerifierAgent
    verifier = DataVerifierAgent()
    evidence = verifier._extract_event_evidence(
        "为啥会有未知球员？核查一下",
        "5.9～6.5秒 6号 传向一名未知球员",
        corpus,
    )
    print("\n".join("  " + l for l in evidence.split("\n")[:8]))
    n_unknown = sum(
        1 for e in corpus.ball_analysis.events
        if e.nearest_player == "?" or e.nearest_player_end == "?"
    )
    print(f"  （当前数据中「未知」端点事件数: {n_unknown}）")
    if ("判定规则" in evidence and "实际距离" in evidence) \
            or "不存在「未知」标注" in evidence:
        print("[OK] 证据完整（未知端点已消解或含判定依据）")
    else:
        print("[FAIL] 证据不完整")
        d_ok = False

    # 专项验证 3：距离档位分类（单位换算 / 阈值 / 跨中场语义）
    print("\n[专项] 距离档位分类:")
    from agents.prompts import distance_tier
    cases = [
        ("7号→10号 约28米同侧", distance_tier(313, 681.0, 505.0), "中距离"),
        ("短传 9米", distance_tier(100), "短距离"),
        ("跨中线 41米", distance_tier(450, 600.0, 100.0), "跨越半场的长传"),
        ("同侧转移 32米", distance_tier(355, 600.0, 420.0), "长距离转移"),
    ]
    for name, got, want in cases:
        ok = got == want
        print(f"  {name}: {got} {'[OK]' if ok else f'[FAIL] 期望[{want}]'}")
        d_ok &= ok

    # ── 导出文件验证 ──
    print("\n" + "=" * 66)
    print("导出文件验证")
    print("=" * 66)
    all_ok = True and d_ok
    for name, pp, bp, res, _, cut in results:
        print(f"场景 {name}:")
        all_ok &= verify_prefix_identical(PERSON_CSV, pp, cut, "球员文件")
        all_ok &= verify_prefix_identical(BALL_CSV, bp, cut, "足球文件")
        all_ok &= verify_generated_speeds(pp, cut)
        all_ok &= verify_roundtrip(pp, bp)

    print("\n" + ("[PASS] 全部测试通过" if all_ok else "[FAIL] 存在失败项"))
    return 0 if all_ok else 1
