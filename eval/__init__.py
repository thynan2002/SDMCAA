"""评测框架 — 单 LLM 直接调用 vs 多智能体系统对比评测。

四大能力：
1. 自动化测试（eval.engine）：三系统并行/串行执行同一用例集，子进程隔离、
   交错调度、断点续跑；
2. 多维评价（eval.metrics + eval.judge）：程序化指标 + LLM-as-Judge 双轨；
3. 结构化得分（eval.stats + eval.report）：总分/分项/置信区间/显著性检验，
   HTML/Markdown/Matplotlib 三形态报告输出至 Score/；
4. 中间步骤追溯（eval.tracealign）：统一 IR + 逐步骤对齐时间线。

与业务逻辑解耦：仅依赖 agents/harness 的公共接口；新指标（metrics 注册表）、
新系统（runners 协议）、新用例（eval/cases/*.json）均可扩展。
"""

from .config import EvalConfig

__version__ = "0.1.0"
__all__ = ["EvalConfig"]
