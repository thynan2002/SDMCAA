"""统一 Harness — 足球战术分析智能体的透明包装层。

设计原则（等价性论证概要）：
1. 恒等委托：Harness 的公开方法直接调用 SessionManager 同名方法，
   不做任何预处理/后处理，入参出参原样传递。
2. 纯函数透传包装：LLM 拦截器 g(x) = observe(f(x))，observe 只读记录，
   不修改返回值、不吞异常、不改变重试/流式分支。
3. 只读代理：状态快照经 deepcopy + 规范化序列化在边界取样，无写回路径。
4. 回调复合（tee）：进度/流式回调均与调用方已有回调复合，而非替换。
5. 确定性注入：mock/seed 仅在显式 record/replay 模式启用，passthrough
   模式下注入器为恒等（dispatcher 默认 None 时甚至不经过包装）。

公开接口：
    Harness        — 包装 SessionManager 的统一入口（CLI / API / Web 共用）
    HarnessConfig  — 运行模式与路径配置
    MockError      — replay 未命中等 mock 层异常
"""

from .config import HarnessConfig
from .core import Harness
from .mock import MockError

__all__ = ["Harness", "HarnessConfig", "MockError"]

__version__ = "1.0.0"
