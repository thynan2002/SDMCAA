"""回放回归 — 用 golden 记录驱动 Harness(replay)，逐层比对等价性。

比对维度（全部通过才算 PASS）：
1. LLM 请求序列：replay 实际发出的 (system_prompt, user_message) 序列
   必须与 golden 录制序列逐字节一致
   → 证明 Harness 包装 + 当前代码 发往 LLM 的内容与录制时相同；
2. 逐轮输出：每轮 handle_input/反事实表单/run_once 的响应文本一致；
3. 状态快照：每轮结束后的会话/记忆/语料快照一致（字段级对比）；
4. 产出文件：Output 中确定性命名文件的 sha256 一致；
5. 流量完整性：golden 中的 LLM 记录被全部消费（无遗漏、无多余）。

已知等价性边界（不比对）：耗时、trace 文件、stdout、含时间戳的存档文件名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import HarnessConfig, MODE_REPLAY
from ..core import Harness
from ..mock import GoldenStore, sha256_file
from ..snapshot import compare_snapshots


@dataclass
class TurnDiff:
    seq: int
    input: str
    response_equal: bool
    response_golden: str
    response_replay: str
    snapshot_diffs: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    golden_dir: str
    passed: bool = True
    llm_sequence_equal: bool = True
    llm_golden_count: int = 0
    llm_replay_count: int = 0
    llm_first_mismatch: str = ""
    turn_diffs: list[TurnDiff] = field(default_factory=list)
    file_diffs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False


def execute_scripted_input(harness: Harness, line: str) -> str:
    """执行一条脚本输入（verify 与 CLI 共用同一解释器，保证 record/replay 一致）。

    脚本语法：
        [CF_FORM] subject|time_second|action|target|duration  → 反事实表单
        [ONCE] person|ball|jersey1,jersey2                    → 单次解说
        其他                                                  → handle_input
    """
    if line.startswith("[CF_FORM] "):
        parts = line[len("[CF_FORM] "):].split("|")
        subject, t, action, target, dur = (parts + ["", "0"])[:5]
        return harness.handle_counterfactual_form(
            subject, float(t), action, target, float(dur or 0),
        )
    if line.startswith("[ONCE] "):
        parts = line[len("[ONCE] "):].split("|")
        person, ball = parts[0], parts[1]
        jerseys = [j for j in (parts[2].split(",") if len(parts) > 2 else []) if j]
        harness.run_once(person, ball, jerseys or None)
        return ""
    return harness.handle_input(line)


def verify_golden(
    golden_dir: Path | str,
    trace_dir: Path | str | None = None,
    strict_mock: bool = True,
    trace_enabled: bool = False,
) -> VerificationResult:
    """对 golden_dir 执行回放回归，返回结构化结果。"""
    golden = GoldenStore(golden_dir)
    result = VerificationResult(golden_dir=str(golden_dir))

    meta = golden.load_meta()
    golden_calls = golden.load_llm_calls()
    golden_turns = golden.load_turns()
    golden_files = golden.load_files_manifest()

    dataset = meta.get("dataset") or {}
    person_csv = dataset.get("person")
    ball_csv = dataset.get("ball")
    if not person_csv or not ball_csv:
        result.add_error("golden meta.json 缺少 dataset.person/ball")
        return result

    cfg = HarnessConfig(
        mode=MODE_REPLAY,
        golden_dir=Path(golden_dir),
        seed=meta.get("seed"),
        strict_mock=strict_mock,
        trace_enabled=trace_enabled,
        trace_dir=Path(trace_dir) if trace_dir else Path(golden_dir) / "_traces",
    )

    replay_responses: list[str] = []
    replay_snapshots: list[dict | None] = []
    try:
        with Harness(cfg) as harness:
            harness.load_data(person_csv, ball_csv)
            for turn in golden_turns:
                response = execute_scripted_input(harness, turn["input"])
                replay_responses.append(response)
                replay_snapshots.append(harness.snapshot())
            replay_files = harness.capture_output_files()
            replay_llm = harness.replay_llm
    except Exception as exc:
        result.add_error(f"回放过程异常: {exc!r}")
        return result

    # ── 1. LLM 请求序列 ──
    assert replay_llm is not None
    served = replay_llm.served
    result.llm_golden_count = len(golden_calls)
    result.llm_replay_count = len(served)
    if len(served) != len(golden_calls):
        result.llm_sequence_equal = False
        result.llm_first_mismatch = (
            f"LLM 调用次数不同: golden={len(golden_calls)} replay={len(served)}"
        )
    else:
        for i, (g, s) in enumerate(zip(golden_calls, served)):
            if (g["system_prompt"], g["user_message"]) != (s["system_prompt"], s["user_message"]):
                result.llm_sequence_equal = False
                result.llm_first_mismatch = (
                    f"第 {i} 次 LLM 请求内容不同:\n"
                    f"  golden system[:100]: {g['system_prompt'][:100]!r}\n"
                    f"  replay system[:100]: {s['system_prompt'][:100]!r}"
                )
                break
    if not result.llm_sequence_equal:
        result.passed = False

    # ── 2+3. 逐轮输出与状态快照 ──
    for turn, response, snap in zip(golden_turns, replay_responses, replay_snapshots):
        td = TurnDiff(
            seq=turn["seq"],
            input=turn["input"],
            response_equal=(turn["response"] == response),
            response_golden=turn["response"],
            response_replay=response,
        )
        golden_snap = turn.get("snapshot")
        if golden_snap is not None and snap is not None:
            td.snapshot_diffs = compare_snapshots(golden_snap, snap)
        if not td.response_equal or td.snapshot_diffs:
            result.turn_diffs.append(td)
            result.passed = False

    # ── 4. 产出文件 ──
    for name, g_hash in sorted(golden_files.items()):
        r_hash = replay_files.get(name)
        if r_hash is None:
            result.file_diffs.append(f"{name}: replay 未生成")
        elif r_hash != g_hash:
            result.file_diffs.append(f"{name}: 哈希不同 golden={g_hash[:12]} replay={r_hash[:12]}")
    for name in sorted(set(replay_files) - set(golden_files)):
        result.file_diffs.append(f"{name}: golden 中不存在（replay 多产出）")
    if result.file_diffs:
        result.passed = False

    # ── 5. 流量完整性 ──
    if not replay_llm.exhausted():
        result.add_error("golden 中的 LLM 记录未被全部消费（回放提前结束或路径偏离）")

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m harness.verify.regression <golden_dir> [--report OUT.md]"""
    import argparse

    parser = argparse.ArgumentParser(description="golden 回放回归")
    parser.add_argument("golden_dir", help="golden 数据目录")
    parser.add_argument("--report", default=None, help="markdown 报告输出路径")
    parser.add_argument("--lenient", action="store_true", help="replay 未命中不抛错（返回 None）")
    args = parser.parse_args(argv)

    result = verify_golden(args.golden_dir, strict_mock=not args.lenient)

    from .diff_report import render_markdown
    report = render_markdown(result)
    report_path = args.report or str(Path(args.golden_dir) / "verify_report.md")
    Path(report_path).write_text(report, encoding="utf-8")

    print(f"{'PASS' if result.passed else 'FAIL'} — 报告: {report_path}")
    if result.errors:
        for e in result.errors:
            print(f"  [错误] {e}")
    if not result.llm_sequence_equal:
        print(f"  [LLM] {result.llm_first_mismatch}")
    print(f"  轮次差异: {len(result.turn_diffs)}  文件差异: {len(result.file_diffs)}"
          f"  LLM: {result.llm_replay_count}/{result.llm_golden_count}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
