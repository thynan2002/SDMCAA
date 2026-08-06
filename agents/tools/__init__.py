"""agents.tools — 工具注册表与确定性能力工具。

子模块：
- base:     ToolSpec / ToolRegistry / 失败分类（无业务依赖，可安全被 llm_client 引用）
- schemas:  结构化决策工具（submit_* 终止型工具，pydantic 定义参数）
- football: 比赛数据工具（球路分析/战术事实/帧级核验/球员数据/反事实模拟）

football/schemas 会引入较重依赖（tracker/tactical_facts/pydantic），
由使用方按需显式导入；llm_client 仅依赖 base，避免循环导入。
"""

from .base import (
    TOOL_FAILURE_EXEC_ERROR,
    TOOL_FAILURE_INVALID_ARGS,
    TOOL_FAILURE_NO_DATA,
    TOOL_FAILURE_UNKNOWN,
    ToolExecutionError,
    ToolRegistry,
    ToolSpec,
    execute_tool,
    get_default_registry,
    insufficient_data,
    register_tool,
)

__all__ = [
    "TOOL_FAILURE_EXEC_ERROR",
    "TOOL_FAILURE_INVALID_ARGS",
    "TOOL_FAILURE_NO_DATA",
    "TOOL_FAILURE_UNKNOWN",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolSpec",
    "execute_tool",
    "get_default_registry",
    "insufficient_data",
    "register_tool",
]
