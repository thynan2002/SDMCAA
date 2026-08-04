"""等价性验证套件。

- regression:     基于 golden 数据的回放回归（逐轮输出/状态快照/LLM 序列/文件哈希）
- snapshot_diff:  状态快照对比工具（字段级 diff）
- diff_report:    输出 diff 报告生成器（markdown）
"""

from .regression import verify_golden, VerificationResult
from .snapshot_diff import diff_snapshot_files
from .diff_report import render_markdown

__all__ = [
    "verify_golden",
    "VerificationResult",
    "diff_snapshot_files",
    "render_markdown",
]
