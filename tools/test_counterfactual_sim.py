"""反事实轨迹生成端到端测试（无 LLM 依赖，直接驱动引擎与导出器）。

测试场景（基于 12s 测试数据，干预帧 102，球静止于 (871, 560)）：
  A. 强制直接射门 → 生成到原始数据结束
  B. 强制短传给队友 → 仅生成后面 5 秒
  C. 无强制动作（纯自然推演）

校验：
  1. 生成段物理校验（运动学限速 / 场地边界 / 球速上限）
  2. 导出 CSV 格式与输入严格一致（列名、列序、小数位）
  3. 干预帧之前的行与原始文件逐字节一致
  4. 生成的 CSV 能被 build_corpus_from_paths 重新加载（格式闭环）
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.player.tracker import build_corpus_from_paths
from agents.professional.mcts.node import Action, ActionType
from agents.professional.simulation import physics
from agents.professional.simulation.engine import TrajectorySimulator
from agents.professional.simulation.exporter import export_simulation

BASE = Path(__file__).resolve().parent.parent
PERSON_CSV = BASE / "TestInput/Files/12s_person2d.csv"
BALL_CSV = BASE / "TestInput/Files/12s_soccer3d.csv"
OUT_DIR = BASE / "Output"
FRAME = 102


def run_scenario(corpus, name, forced, duration=0.0, tag="cf"):
    print("\n" + "=" * 66)
    print(f"场景 {name}")
    print("=" * 66)

    sim = TrajectorySimulator(corpus)
    result = sim.simulate(intervention_frame=FRAME, forced_action=forced,
                          duration_seconds=duration)

    print(f"模拟区间: 帧 {result.start_frame} → {result.end_frame}"
          f"（请求到 {result.requested_end_frame}）  结局: {result.outcome or '-'}")
    print("事件时间线:")
    for e in result.events:
        print(f"  · {e}")

    n_pkf = sum(len(v) for v in result.player_keyframes.values())
    print(f"关键帧统计: 球员 {len(result.player_keyframes)} 人共 {n_pkf} 帧,"
          f" 球 {len(result.ball_keyframes)} 帧")

    person_path, ball_path, warnings = export_simulation(corpus, result, OUT_DIR, tag=tag)
    print(f"导出: {person_path.name}, {ball_path.name}")
    if warnings:
        print(f"!! 物理校验警告 {len(warnings)} 条:")
        for w in warnings[:10]:
            print(f"   - {w}")
    else:
        print("物理校验: 全部生成段通过（无超速/瞬移/越界）")

    return person_path, ball_path, result, warnings


def verify_prefix_identical(src: Path, dst: Path, cut: int, kind: str) -> bool:
    """干预帧之前的行必须与原始文件完全一致。"""
    def rows_before(p: Path):
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r)
            kept = [row for row in r if row and int(float(row[0])) < cut]
        return header, kept

    h1, r1 = rows_before(src)
    h2, r2 = rows_before(dst)
    ok = h1 == h2 and r1 == r2
    print(f"  {kind}: 干预帧前行一致性 {'OK' if ok else 'FAIL'}"
          f"（原始 {len(r1)} 行 / 拼接 {len(r2)} 行）")
    return ok


def verify_generated_speeds(dst: Path, cut: int) -> bool:
    """对生成 CSV 中 frame >= cut 的部分做独立物理复检。"""
    by_tid: dict[str, list[tuple[int, float, float]]] = {}
    with dst.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fr = int(float(row["frame_num"]))
            if fr >= cut:
                by_tid.setdefault(row["track_id"], []).append(
                    (fr, float(row["x"]), float(row["y"])))
    bad = 0
    for tid, pts in by_tid.items():
        pts.sort()
        for i in range(1, len(pts)):
            f1, x1, y1 = pts[i - 1]
            f2, x2, y2 = pts[i]
            if not physics.validate_player_segment(x1, y1, f1, x2, y2, f2):
                bad += 1
    print(f"  生成段独立复检: {'OK' if bad == 0 else f'FAIL ({bad} 段超速)'}")
    return bad == 0


def main() -> int:
    corpus = build_corpus_from_paths(PERSON_CSV, BALL_CSV)
    print(f"数据: {corpus.player_count} 名球员（门将 {len(corpus.goalkeepers())} 人）,"
          f"球帧 {min(corpus.ball_frames)}-{max(corpus.ball_frames)}")
    ball_at = corpus.ball_frames.get(FRAME)
    print(f"干预帧 {FRAME}: 球位于 {ball_at}")

    # 判定干预帧控球者（距球最近球员）
    holder = None
    if ball_at:
        best = (1e9, None)
        for tid, p in corpus.players.items():
            for i, fr in enumerate(p.frames):
                if fr <= FRAME:
                    d = ((p.xs[i] - ball_at[0]) ** 2 + (p.ys[i] - ball_at[1]) ** 2) ** 0.5
                    if d < best[0]:
                        best = (d, tid)
        holder = best[1]
        print(f"干预帧控球者: {holder}号（距球 {best[0]:.0f}px）")

    results = []

    # 场景 A：直接射门，补全到结束
    pa, ba, ra, wa = run_scenario(corpus, "A: 强制直接射门 → 补全到原始结束",
                                  Action(ActionType.SHOOT), tag="shoot")
    results.append(("A", pa, ba, ra, wa, FRAME))

    # 场景 B：短传给队友，仅生成后面 5 秒
    teammate = None
    if holder and holder in corpus.players:
        hc = corpus.players[holder].color
        teammate = next((t for t in corpus.players
                         if t != holder and corpus.players[t].color == hc), None)
    pb, bb, rb, wb = run_scenario(corpus, f"B: 强制短传给 {teammate}号 → 仅生成后 5 秒",
                                  Action(ActionType.PASS, target_player=f"{teammate}号"),
                                  duration=5.0, tag="pass")
    results.append(("B", pb, bb, rb, wb, FRAME))

    # 场景 C：纯自然推演
    pc, bc, rc, wc = run_scenario(corpus, "C: 无强制动作（自然推演）", None, tag="natural")
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
    import re as _re
    pair_count: dict[tuple[str, str], int] = {}
    for e in rd.events:
        m = _re.search(r"(\d+号) (?:长传|短传)给 (\d+号)", e)
        if m:
            pair = tuple(sorted((m.group(1), m.group(2))))
            pair_count[pair] = pair_count.get(pair, 0) + 1
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
        max_dev = max(((x - hx) ** 2 + (y - hy) ** 2) ** 0.5 for _, x, y in gk_kfs)
        if max_dev <= 150:
            print(f"[OK] 门将最大偏离把守位置 {max_dev:.0f}px（≤150）")
        else:
            print(f"[FAIL] 门将偏离把守位置 {max_dev:.0f}px")
            d_ok = False

    # D-4: 攻方球员位置均需物理合法（通用校验）
    _, _, wd = export_simulation(corpus, rd, OUT_DIR, tag="d")
    if wd:
        print(f"[FAIL] 场景D物理警告: {wd[:5]}")
        d_ok = False
    else:
        print("[OK] 场景D物理校验全部通过")
    results.append(("D", *export_simulation(corpus, rd, OUT_DIR, tag="d")[:2], rd, wd, 0))

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
    # 注：数据切片修复（任意帧位置插值）后，12s 数据中的「未知」端点应被
    # 全部消解——此时核验器应明确回答"不存在未知标注"；若仍存在未知端点，
    # 则证据必须包含判定规则与实际距离。两种输出均为合法证据。
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
        # 格式闭环：生成文件可被重新加载
        try:
            re = build_corpus_from_paths(pp, bp)
            print(f"  重新加载闭环: OK（{re.player_count} 名球员）")
        except Exception as exc:
            print(f"  重新加载闭环: FAIL ({exc})")
            all_ok = False

    print("\n" + ("[PASS] 全部测试通过" if all_ok else "[FAIL] 存在失败项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
