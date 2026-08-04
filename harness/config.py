"""Harness 运行配置。

三种运行模式：
- passthrough：生产默认。LLM 分派器仅做只读观测（trace），语义与原系统恒等。
- record：真实调用 LLM，同时把全部 LLM 流量、逐轮输出、状态快照、产出文件
  哈希落盘为 golden 数据集，供 replay 回归使用。
- replay：不触网。LLM 调用由 golden 记录精确回放（确定性），用于回归验证。

mode 仅影响"观测层/注入层"，不影响被包装的智能体代码路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODE_PASSTHROUGH = "passthrough"
MODE_RECORD = "record"
MODE_REPLAY = "replay"
VALID_MODES = (MODE_PASSTHROUGH, MODE_RECORD, MODE_REPLAY)


@dataclass
class HarnessConfig:
    """Harness 配置。

    Attributes:
        mode: passthrough | record | replay
        trace_enabled: 是否写 trace（JSONL）。三种模式均可开启。
        trace_dir: trace 输出目录。
        golden_dir: record 的落盘目录 / replay 的读取目录。
        seed: 确定性注入种子。仅 record/replay 模式生效（MCTS 预留 seed
            通道 + apply_action 无种子回退的确定性包装）；passthrough 下忽略。
        strict_mock: replay 时 prompt 未命中 golden 记录即抛 MockError
            （默认严格：未命中 = 与 golden 发生偏离，正是回归要捕获的）。
        snapshot_enabled: 是否在每个交互轮次边界采集只读状态快照。
    """

    mode: str = MODE_PASSTHROUGH
    trace_enabled: bool = True
    trace_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "Output" / "traces")
    golden_dir: Path | None = None
    seed: int | None = None
    strict_mock: bool = True
    snapshot_enabled: bool = True

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"非法 Harness 模式: {self.mode!r}（可选: {VALID_MODES}）")
        self.trace_dir = Path(self.trace_dir)
        if self.golden_dir is not None:
            self.golden_dir = Path(self.golden_dir)
        if self.mode in (MODE_RECORD, MODE_REPLAY) and self.golden_dir is None:
            raise ValueError(f"mode={self.mode} 需要指定 golden_dir")

    @classmethod
    def from_env(cls) -> "HarnessConfig":
        """从环境变量构建（web 后端等无 CLI 参数的入口使用）。

        HARNESS_MODE       默认 passthrough
        HARNESS_TRACE_DIR  默认 Output/traces
        HARNESS_GOLDEN_DIR record/replay 时必填
        HARNESS_SEED       可选整数
        HARNESS_TRACE      "0"/"false" 关闭 trace
        """
        seed_raw = os.getenv("HARNESS_SEED", "").strip()
        return cls(
            mode=os.getenv("HARNESS_MODE", MODE_PASSTHROUGH).strip() or MODE_PASSTHROUGH,
            trace_enabled=os.getenv("HARNESS_TRACE", "1").strip().lower() not in ("0", "false", "off"),
            trace_dir=Path(os.getenv("HARNESS_TRACE_DIR", str(PROJECT_ROOT / "Output" / "traces"))),
            golden_dir=(Path(os.environ["HARNESS_GOLDEN_DIR"]) if os.getenv("HARNESS_GOLDEN_DIR") else None),
            seed=int(seed_raw) if seed_raw else None,
            strict_mock=os.getenv("HARNESS_STRICT_MOCK", "1").strip().lower() not in ("0", "false", "off"),
        )
