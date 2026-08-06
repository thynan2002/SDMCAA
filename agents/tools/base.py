"""工具注册表 — LLM 获取外部信息 / 提交结构化决策的唯一通道。

LLM 需要任何外部信息只能通过工具获取；工具结果不可用时必须如实
返回「数据不足」，不得编造。

工具分两类：
- 数据型工具（terminal=False）：封装确定性分析能力（球路分析、战术
  事实提取、帧级核验等），执行结果以 role=tool 消息回填模型继续生成。
- 终止型工具（terminal=True）：「结构化决策输出」契约。模型通过发起
  该工具调用直接交付最终结构化结果（意图路由、战术决策、档位推断等），
  工具循环在其被调用时即结束，不再产生额外交互轮次。

失败分类（供降级链区分处理）：
- invalid_arguments  工具参数格式非法（JSON 解析/校验失败）
- execution_error    工具执行异常
- insufficient_data  工具所需数据不足（如实声明，不得编造）
- unknown_tool       未注册的工具
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

TOOL_FAILURE_INVALID_ARGS = "invalid_arguments"
TOOL_FAILURE_EXEC_ERROR = "execution_error"
TOOL_FAILURE_NO_DATA = "insufficient_data"
TOOL_FAILURE_UNKNOWN = "unknown_tool"


class ToolExecutionError(RuntimeError):
    """工具执行期可分类失败（kind 取 TOOL_FAILURE_* 常量）。"""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def insufficient_data(message: str) -> dict[str, Any]:
    """构造「数据不足」结果（模型应按提示词纪律如实声明，不得编造）。"""
    return {"status": "数据不足", "message": message}


@dataclass
class ToolSpec:
    """单个工具的规格。

    Attributes:
        name: 工具名（function calling 的 function.name）
        description: 工具用途说明（模型可见）
        parameters: JSON Schema（函数参数）
        handler: args(dict) -> JSON 可序列化结果
        terminal: True = 结构化输出提交工具（其调用即最终答案）
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    terminal: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI Chat Completions tools 条目格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：注册 / 查询 / 执行（执行永不抛异常，失败分类返回）。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.to_openai_schema() for spec in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """执行工具；任何失败都转为带 kind 的错误结果字典。"""
        spec = self._tools.get(name)
        if spec is None:
            return {"error": f"未注册的工具: {name}", "kind": TOOL_FAILURE_UNKNOWN}
        try:
            result = spec.handler(args)
        except ToolExecutionError as exc:
            return {"error": str(exc), "kind": exc.kind}
        except Exception as exc:  # 执行异常不向上传播，交由模型/降级链处理
            return {"error": f"工具执行异常: {exc}", "kind": TOOL_FAILURE_EXEC_ERROR}
        if isinstance(result, dict):
            return result
        return {"result": result}


_DEFAULT_REGISTRY = ToolRegistry()


def get_default_registry() -> ToolRegistry:
    """全局默认注册表（各智能体模块导入时注册各自工具）。"""
    return _DEFAULT_REGISTRY


def register_tool(spec: ToolSpec, registry: ToolRegistry | None = None) -> ToolSpec:
    return (registry or _DEFAULT_REGISTRY).register(spec)


def execute_tool(
    name: str, args: dict[str, Any], registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    return (registry or _DEFAULT_REGISTRY).execute(name, args)
