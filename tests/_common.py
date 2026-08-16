"""测试 Harness 共享实现。

供 tests/ 下的 pytest 用例与 tools/ 命令行入口共同复用，
保证"pytest 运行"与"独立脚本运行"行为一致、单一实现。
"""

from __future__ import annotations

import csv
import json
import re
from math import hypot
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_INPUT = PROJECT_ROOT / "TestInput" / "Files"
PERSON_CSV = TEST_INPUT / "12s_person2d.csv"
BALL_CSV = TEST_INPUT / "12s_soccer3d.csv"
FRAME = 102  # 反事实干预帧（球静止于 (871, 560)）

# ─────────────────────────────────────────────────────────────────
# 反事实模拟：场景执行 / 导出 / 校验
# ─────────────────────────────────────────────────────────────────

def run_scenario(
    corpus,
    out_dir: Path,
    name: str,
    forced,
    duration: float = 0.0,
    tag: str = "cf",
):
    """执行一次反事实模拟并导出，返回 (person_path, ball_path, result, warnings)。

    与旧 tools/test_counterfactual_sim.py 的 run_scenario 行为完全一致。
    """
    from agents.professional.simulation.engine import TrajectorySimulator
    from agents.professional.simulation.exporter import export_simulation

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

    person_path, ball_path, warnings = export_simulation(corpus, result, out_dir, tag=tag)
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
    from agents.professional.simulation import physics

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


def verify_roundtrip(person_path: Path, ball_path: Path) -> bool:
    """格式闭环：生成的 CSV 能被 build_corpus_from_paths 重新加载。"""
    from agents.player.tracker import build_corpus_from_paths

    try:
        reloaded = build_corpus_from_paths(person_path, ball_path)
        ok = reloaded.player_count > 0
    except Exception:
        ok = False
    print(f"  重新加载闭环: {'OK' if ok else 'FAIL'}")
    return ok


def compute_holder(corpus, frame: int = FRAME):
    """判定干预帧控球者（距球最近球员），返回 (track_id, 距离)。"""
    ball_at = corpus.ball_frames.get(frame)
    if not ball_at:
        return None, float("inf")
    best = (1e9, None)
    for tid, p in corpus.players.items():
        for i, fr in enumerate(p.frames):
            if fr <= frame:
                d = ((p.xs[i] - ball_at[0]) ** 2 + (p.ys[i] - ball_at[1]) ** 2) ** 0.5
                if d < best[0]:
                    best = (d, tid)
    return best[1], best[0]


def count_pass_pairs(events: list[str]) -> dict[tuple[str, str], int]:
    """统计传球对组合次数（用于乒乓互传检测）。"""
    pair_count: dict[tuple[str, str], int] = {}
    for e in events:
        m = re.search(r"(\d+号) (?:长传|短传)给 (\d+号)", e)
        if m:
            pair = tuple(sorted((m.group(1), m.group(2))))
            pair_count[pair] = pair_count.get(pair, 0) + 1
    return pair_count


# ─────────────────────────────────────────────────────────────────
# LLM 决策引擎：mock 工厂
# ─────────────────────────────────────────────────────────────────

FAKE_SCRIPT = json.dumps({
    "narrative": "A队左路渗透，B队中位压迫",
    "attacking_plan": "左路渗透后内切",
    "defending_plan": "中位压迫",
    "key_instructions": {"10号": "多接应"},
    "expected_events": ["6号分边给10号", "10号内切"],
}, ensure_ascii=False)


def make_llm_mock():
    """构造 (calls, fake_call_llm)。

    脚本请求（含 Scriptwriter/战术教练）返回固定剧本；
    决策请求返回"传给第一个队友"的合法 JSON。
    """
    calls: list[str] = []

    def fake_call_llm(system_prompt, user_message, config=None, max_tokens=None, retries=1, **kwargs):
        calls.append("script" if "Scriptwriter" in system_prompt or "战术教练" in system_prompt
                     else "decide")
        if calls[-1] == "script":
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

    return calls, fake_call_llm


def make_bad_llm():
    """始终返回非法内容，验证 LLM 回退启发式。"""
    def bad_call_llm(system_prompt, user_message, config=None, max_tokens=None, retries=1, **kwargs):
        return "这不是JSON"
    return bad_call_llm
