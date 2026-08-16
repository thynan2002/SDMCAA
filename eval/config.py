"""评测配置 — 权重、重复数、Judge、效率基准、路径。

来源优先级：默认值 < eval/config.yaml < CLI 参数。
Judge 模型默认与被测系统相同（用户决策），可经环境变量切换任意
OpenAI 兼容 API：EVAL_JUDGE_MODEL / EVAL_JUDGE_BASE_URL / EVAL_JUDGE_API_KEY。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = Path(__file__).resolve().parent / "cases"
SCORE_ROOT = PROJECT_ROOT / "Score"

# 三系统标识
SYSTEM_BARE = "bare"        # 基线 A：单次 LLM 裸调用（无工具，全量数据入 prompt）
SYSTEM_SINGLE = "single"    # 基线 B：单智能体 + 同套数据工具（function calling 循环）
SYSTEM_MULTI = "multi"      # 被测系统：SessionManager 多智能体流水线
ALL_SYSTEMS = (SYSTEM_BARE, SYSTEM_SINGLE, SYSTEM_MULTI)

SYSTEM_LABELS = {
    SYSTEM_BARE: "单次LLM裸调用",
    SYSTEM_SINGLE: "单智能体+工具",
    SYSTEM_MULTI: "多智能体系统",
}

# 指标家族（1.3 解耦：程序化硬指标 + Judge 软指标）：
# - scored   计入「共同总分」（三系统均可得分，公平可比）
# - extended 仅计入「扩展总分」（工具使用，裸调用结构性无工具）
# - diag     诊断性指标（不计分，单独展示）
# 程序化分量合计 ≈ 0.66、Judge 分量 ≈ 0.34（见 scoring.py 融合注释）
SCORED_METRICS = ("accuracy", "grounding", "completeness", "reasoning",
                  "efficiency", "robustness", "tactical_accuracy")
DIAG_METRICS = ("stability", "tool_use")

DEFAULT_WEIGHTS: dict[str, float] = {
    "accuracy": 0.15,
    "grounding": 0.10,
    "completeness": 0.10,
    "reasoning": 0.20,
    "efficiency": 0.10,
    "robustness": 0.10,
    "tactical_accuracy": 0.25,
}
EXTENDED_TOOL_WEIGHT = 0.10  # 扩展总分中 tool_use 的权重（其余等比缩）


@dataclass
class EfficiencyRefs:
    """效率指标归一化基准（预标定常数，不随本次评测样本变化）。

    评分 = min(1, ref / max(x, ε))，ref 应来自预实验标定
    （运行 `python -m eval calibrate` 或手工填入事件目录 refs.json）。
    """

    latency_ms: float = 120_000.0   # 端到端时延参考（单智能体中位数量级）
    llm_calls: float = 10.0         # LLM 调用次数参考
    total_tokens: float = 12_000.0  # 总 token 参考


# 评测 Judge 默认模型：与被测系统（DEEPSEEK_MODEL，默认 flash）解耦，
# 使用更强的 deepseek-v4-pro 提升评分质量（同 API Key，仅模型 ID 不同）。
DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"


@dataclass
class JudgeConfig:
    model: str | None = DEFAULT_JUDGE_MODEL
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    sample_repeats: int = 2         # 每用例抽评的重复数（从 N 次重复中抽取）
    order_swap: bool = True         # 位置互换重评（消除 Judge 位置偏好）

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            model=os.getenv("EVAL_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
            base_url=os.getenv("EVAL_JUDGE_BASE_URL") or None,
            api_key=os.getenv("EVAL_JUDGE_API_KEY") or None,
        )


@dataclass
class EvalConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    repeats: int = 5
    systems: tuple[str, ...] = ALL_SYSTEMS
    temperature: float = 0.85       # 三系统统一的采样温度（与多智能体默认一致）
    seed: int = 42                  # 交错调度随机种子（记录保证可复现）
    score_root: Path = SCORE_ROOT
    cases_dir: Path = CASES_DIR
    job_timeout_s: int = 900        # 单个子进程作业超时
    max_workers: int = 1            # 并行执行作业数（>1 启用多进程并发，3.1）
    refs: EfficiencyRefs = field(default_factory=EfficiencyRefs)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    judge_enabled: bool = True

    def __post_init__(self) -> None:
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"权重之和必须为 1.0，当前: {sum(self.weights.values())}")
        unknown = set(self.weights) - set(SCORED_METRICS)
        if unknown:
            raise ValueError(f"未知计分指标: {unknown}（可选: {SCORED_METRICS}）")
        missing = set(SCORED_METRICS) - set(self.weights)
        if missing:
            raise ValueError(f"缺少计分指标: {missing}")
        bad = set(self.systems) - set(ALL_SYSTEMS)
        if bad:
            raise ValueError(f"未知系统: {bad}（可选: {ALL_SYSTEMS}）")

    # ── 加载 ──

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalConfig":
        import yaml

        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = cls()
        if "weights" in raw:
            cfg.weights = {**DEFAULT_WEIGHTS, **raw["weights"]}
        for key in ("repeats", "systems", "temperature", "seed", "job_timeout_s",
                    "judge_enabled", "max_workers"):
            if key in raw:
                setattr(cfg, key, raw[key])
        cfg.systems = tuple(cfg.systems)
        if "refs" in raw:
            cfg.refs = EfficiencyRefs(**raw["refs"])
        if "judge" in raw:
            base = JudgeConfig.from_env()
            for key, value in raw["judge"].items():
                setattr(base, key, value)
            cfg.judge = base
        else:
            cfg.judge = JudgeConfig.from_env()
        cfg.__post_init__()
        return cfg

    @classmethod
    def load(cls, override_path: Path | None = None) -> "EvalConfig":
        """默认加载 eval/config.yaml（存在时），可被 override_path 替换。"""
        path = override_path or (Path(__file__).parent / "config.yaml")
        if path.exists():
            return cls.from_yaml(path)
        cfg = cls()
        cfg.judge = JudgeConfig.from_env()
        return cfg
