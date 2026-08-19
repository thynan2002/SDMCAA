# 统一 Harness（透明包装层）🔍

<p align="center">
  <img src="https://img.shields.io/badge/status-implemented-2EA043?style=flat-square" alt="Status: implemented">
  <img src="https://img.shields.io/badge/modes-passthrough%2Frecord%2Freplay-4B8BBE?style=flat-square" alt="Modes: passthrough/record/replay">
  <img src="https://img.shields.io/badge/verify-4--dimensional-8B5CF6?style=flat-square" alt="Verify: 4-dimensional">
</p>

`harness/` 是智能体的**透明包装层**：全链路观测（规划/工具/记忆/LLM/输出）、统一 CLI/API 入口、外部依赖 Mock，且不引入任何语义偏差。Web 后端（`web/backend.py`）已迁移到 Harness 之下，CLI 与 Web 共用同一包装层。系统已升级为工具调用（function calling）驱动架构，等价性论证相应扩展，见[「工具调用（tool_calls）与等价性」](#工具调用tool_calls与等价性)。

## 三种运行模式

| 模式 | 说明 |
|------|------|
| `passthrough` | 生产默认。仅只读观测（trace），语义与原系统恒等 |
| `record` | 真实调用 LLM，同时录制 golden 数据集（LLM 流量/逐轮输出/状态快照/文件哈希） |
| `replay` | 不触网。LLM 由 golden 精确回放（确定性），用于回归验证 |

## 用法

```bash
# 交互 / 单次解说（与 main.py 等价，额外产出 trace）
python -m harness run TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv
python -m harness run TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --once --focus 6

# 录制 golden（真实 API；--seed 启用确定性注入）＝ 工具化重构后的一键重录
python -m harness run TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv \
    --script harness/scripts/standard_12s.txt \
    --mode record --golden-dir harness/golden/standard --seed 42

# 回放回归（不触网；LLM 序列/逐轮输出/状态快照/文件哈希四维比对）
python -m harness verify harness/golden/standard --report report.md

# 状态快照字段级对比
python -m harness snapshot-diff A.json B.json
```

**Python API**：`with Harness(HarnessConfig(mode=...)) as h: h.load_data(...); h.handle_input(...)`。
Web 后端经 `HarnessConfig.from_env()` 读取 `HARNESS_MODE` 等环境变量。

### Trace

交互过程中的完整决策链路以 Span 树（`run → turn → llm/stage/stream/snapshot/output`）写入 `Output/traces/*.jsonl`，含 LLM 请求/响应哈希、耗时、阶段事件与状态快照摘要；工具循环轮次在 LLM span 上追加 `tool_round` 与 `tools` 属性。

## 工具调用（tool_calls）与等价性

工具调用层（`agents/llm_client.py` + `agents/tools/`）接入方式：

- `call_llm(..., tools=[...], tool_choice=..., max_rounds=...)` 显式传入工具，或 `bind_prompt_tools(system_prompt, tools, tool_choice)` 按系统提示词绑定（迁移后的调用点零改动，测试 mock 以固定签名替换 `call_llm` 不受影响）。
- **多轮工具循环**：模型发起 `tool_calls` → 系统经 `agents/tools` 注册表执行 → 结果以 `role=tool` 消息回填 → 继续生成；同一响应内多个独立工具请求（**并行 tool_calls**）一次性全部执行。
- **终止型工具**（结构化决策输出：意图路由/战术决策/剧本/策略表/质疑判断/语义档位）的调用即最终答案，`call_llm` 返回 `json.dumps(工具参数)`，调用方既有 JSON 解析与规则兜底路径原样复用；模型直接返回文本（mock / 旧 golden / 模型行为差异）时走原有文本解析；失败（网络/超时/格式非法/循环超限）返回 `None`，由既有兜底链处理（router 关键词规则、llm_brain 启发式、verifier 关键词）。
- **数据工具**（战术事实/球路分析/帧级核验/球员数据/反事实模拟）把确定性分析能力暴露给 LLM：需要提示词之外的信息时**只能**通过工具获取；工具不可用时如实声明「数据不足」，不得编造。

### 等价性论证（工具层不破坏观测/录制/回放）

| 机制 | 论证 |
|------|------|
| 交换通道 ContextVar | tools 参数与多轮 messages 经 `tool_exchange_var` 传入传输层，解析出的 assistant 消息（含 tool_calls）经同一通道回传——dispatcher / `_call_llm_impl` 的签名与 patch 点完全不变，拦截器与测试 mock 无需感知工具层 |
| 无工具 = 原路径 | `call_llm` 不传 tools 且无绑定时，走与重构前逐字节一致的路径（分派器仅收到原 6 参） |
| 首轮 payload 不变 | 迁移只增加 tools 绑定，`(system_prompt, user_message)` 与重构前完全一致 → 旧 golden 精确匹配 |
| 旧 golden 兼容 | 旧条目无 `assistant_message`/`tools` 字段，回放只返回录制文本 → 文本回退路径，行为不变 |
| 新 golden 兼容 | record 仅在出现工具交互时追加可选字段（`tool_round`/`tools`/`tool_choice`/`assistant_message`）；ReplayLLM 按录制轨迹重现 tool_calls，回放确定性 |
| 文本回退双路径 | 结构化决策工具与文本 JSON 契约字段一致，两条路径产出同一解析结果 |
| 失败分类降级 | 瞬时故障沿用 `retries`；tool_choice 被服务端拒绝（DeepSeek 推理模式 400）时降级为 auto 重试；非法输出/循环超限 → None → 既有规则兜底；数据不足由工具如实返回 |

## 等价性验证

- **A/B 同进程对比**：同一确定性 mock 下，裸 SessionManager 与 Harness 包装的输出/状态快照/LLM 请求序列三者一致（`tests/test_harness_equivalence.py`）
- **golden 回放回归**：`harness/golden/standard`（真实 API 录制，13 次 LLM 调用，覆盖系统命令 / 8 种意图 / 记忆摘要 / 反事实轨迹导出，含工具调用轨迹），回放经四维比对验证当前代码下回放等价，全程不触网
- **工具层单测**：`tests/test_tool_calling.py`（21 例：工具循环/并行 tool_calls/失败分类/文本回退/绑定/回放兼容/流式增量/record→replay 工具闭环）
- 回归报告由 `python -m harness verify` 生成（LLM 请求序列 / 逐轮输出 unified diff / 状态快照字段级差异 / 产出文件哈希 / 已知边界声明）

## 已知等价性边界（允许差异）

- 耗时/延迟、内存占用（trace 写入与快照深拷贝的开销）
- 日志与 trace 体量（新增 `Output/traces/*.jsonl`；stdout 不变）
- 含时间戳的文件名（对话存档、trace 文件名）
- 真实 LLM 模式的输出文本（temperature 采样不可复现；逐字节断言仅在 replay 确定性模式下有效）
- MCTS 默认未 seed：原始版本本身非确定；模拟预算默认 300 次（可配置）；record/replay 注入 seed 属"确定性增强"
- 实网模式下，绑定数据工具的提示词（如综合问答）中模型可能主动发起工具调用获取额外数据，回答文本与纯内嵌事实时不同——语义一致且不编造，属于工具化的预期效果；回放模式下完全确定性、与 golden 逐字节一致

## 模块一览

| 文件 | 职责 |
|------|------|
| `core.py` | `Harness` 类：组合 SessionManager，统一入口 |
| `config.py` | 运行模式配置（passthrough/record/replay） |
| `interceptors.py` | LLM 拦截器（观测/录制/回放三态；录制工具可选字段） |
| `tracing.py` | Span 树 trace（JSONL） |
| `snapshot.py` | 只读状态快照 + 字段级对比 |
| `mock.py` | golden 存储 + ReplayLLM（回放 assistant_message 驱动工具循环） |
| `cli.py` | `python -m harness`（run/verify/snapshot-diff） |
| `verify/` | 回归验证 / 快照对比 / diff 报告 |
| `scripts/` | 录制脚本 |
| `golden/` | golden 回归数据集 |

## 相关文档

- 架构与背景：[project_report.md](project_report.md)
- 工具化重构全程记录：[tool_calling_refactor.md](tool_calling_refactor.md)
- 基于 Harness 的对比评测：[eval.md](eval.md)
