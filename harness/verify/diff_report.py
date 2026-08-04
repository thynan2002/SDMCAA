"""输出 diff 报告生成器（markdown）。

输入 VerificationResult，输出结构化 markdown：
- 总览（PASS/FAIL + 各维度计数）
- LLM 请求序列比对
- 逐轮输出 diff（unified diff，文本相同则跳过）
- 状态快照字段级差异
- 产出文件哈希比对
- 已知等价性边界声明
"""

from __future__ import annotations

import difflib
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .regression import VerificationResult

MAX_RESPONSE_DIFF_CHARS = 4000  # 报告中单个文本 diff 的最大长度
MAX_DIFF_LINES_PER_TURN = 60


def _unified_diff(golden: str, replay: str, from_name: str, to_name: str) -> str:
    lines = list(difflib.unified_diff(
        golden.splitlines(), replay.splitlines(),
        fromfile=from_name, tofile=to_name, lineterm="",
    ))
    text = "\n".join(lines[:MAX_DIFF_LINES_PER_TURN])
    if len(lines) > MAX_DIFF_LINES_PER_TURN:
        text += f"\n... (省略 {len(lines) - MAX_DIFF_LINES_PER_TURN} 行)"
    if len(text) > MAX_RESPONSE_DIFF_CHARS:
        text = text[:MAX_RESPONSE_DIFF_CHARS] + "\n... (截断)"
    return text


def render_markdown(result: "VerificationResult") -> str:
    """将回归结果渲染为 markdown 报告。"""
    lines: list[str] = []
    add = lines.append

    add("# Harness 等价性验证报告")
    add("")
    add(f"- golden 目录: `{result.golden_dir}`")
    add(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- **结论: {'✅ PASS' if result.passed else '❌ FAIL'}**")
    add("")

    add("## 总览")
    add("")
    add("| 维度 | 结果 | 明细 |")
    add("|------|------|------|")
    add(f"| LLM 请求序列 | {'✅' if result.llm_sequence_equal else '❌'} "
        f"| replay {result.llm_replay_count} 次 / golden {result.llm_golden_count} 次 |")
    add(f"| 逐轮输出 | {'✅' if not any(not t.response_equal for t in result.turn_diffs) else '❌'} "
        f"| {sum(1 for t in result.turn_diffs if not t.response_equal)} 轮不同 |")
    add(f"| 状态快照 | {'✅' if not any(t.snapshot_diffs for t in result.turn_diffs) else '❌'} "
        f"| {sum(len(t.snapshot_diffs) for t in result.turn_diffs)} 处字段差异 |")
    add(f"| 产出文件 | {'✅' if not result.file_diffs else '❌'} "
        f"| {len(result.file_diffs)} 处差异 |")
    add(f"| 异常 | {'✅' if not result.errors else '❌'} | {len(result.errors)} 个 |")
    add("")

    if result.errors:
        add("## 异常")
        add("")
        for e in result.errors:
            add(f"- `{e}`")
        add("")

    if not result.llm_sequence_equal:
        add("## LLM 请求序列差异")
        add("")
        add("```")
        add(result.llm_first_mismatch)
        add("```")
        add("")

    if result.turn_diffs:
        add("## 逐轮差异")
        add("")
        for td in result.turn_diffs:
            add(f"### 轮次 {td.seq}: `{td.input[:60]}`")
            add("")
            if not td.response_equal:
                add("**输出 diff（golden → replay）:**")
                add("")
                add("```diff")
                add(_unified_diff(td.response_golden, td.response_replay, "golden", "replay"))
                add("```")
                add("")
            if td.snapshot_diffs:
                add("**状态快照差异:**")
                add("")
                for d in td.snapshot_diffs[:40]:
                    add(f"- `{d}`")
                if len(td.snapshot_diffs) > 40:
                    add(f"- ... (省略 {len(td.snapshot_diffs) - 40} 处)")
                add("")

    if result.file_diffs:
        add("## 产出文件差异")
        add("")
        for d in result.file_diffs:
            add(f"- `{d}`")
        add("")

    add("## 已知等价性边界（本报告不比对，允许差异）")
    add("")
    add("- 耗时 / 延迟（trace 与快照采集有开销）")
    add("- 内存占用（深拷贝快照、trace 缓冲）")
    add("- 日志与 trace 文件体量（新增 Output/traces/*.jsonl；stdout 内容不在比对范围）")
    add("- 含时间戳的文件名（对话存档、trace 文件名）")
    add("- 真实 LLM 模式的输出文本（temperature 采样不可复现；"
        "逐字节断言仅在 replay 确定性模式下有效）")
    add("")
    return "\n".join(lines)
