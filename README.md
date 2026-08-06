# Football Agent ⚽ — 足球战术分析多智能体系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/LLM-DeepSeek-4B8BBE?style=flat-square" alt="LLM: DeepSeek">
  <img src="https://img.shields.io/badge/orchestration-LangGraph-2C8C5A?style=flat-square" alt="Orchestration: LangGraph">
  <img src="https://img.shields.io/badge/architecture-tool--calling-8B5CF6?style=flat-square" alt="Architecture: tool-calling">
  <img src="https://img.shields.io/badge/tests-58%20passed-2EA043?style=flat-square" alt="Tests: 58 passed">
</p>

输入球员追踪 CSV 与足球轨迹 CSV，由 LLM 驱动的多智能体流水线自动完成足球比赛数据分析，输出专业级比赛解说、球员表现分析与反事实（What-If）推演。系统以 **function calling（工具调用）** 为架构核心：结构化决策经工具契约交付、外部信息只能经工具获取，全链路经 LangGraph 编排，可通过 CLI 对话、单次解说或 Web 可视化界面使用。

**文档入口**: [统一 Harness 文档](docs/harness.md) · [工具调用重构设计](docs/tool_calling_refactor.md) · [输入数据格式](#输入数据格式) · [测试](#测试)

## 功能特性

- **工具调用（tool_calls）驱动架构** — 意图路由、战术决策、剧本、MCTS 策略表、质疑判断、语义档位等结构化决策输出全部改为**终止型工具**（`submit_*`）契约交付；`call_llm` 支持 tools 参数、多轮工具循环与**并行 tool_calls**（同一响应多个独立请求一次性执行）
- **禁止凭空编造** — 球路分析、战术事实提取、帧级核验、球员数据、反事实模拟等确定性能力封装为**数据工具**（`agents/tools/football.py`），LLM 需要任何外部信息只能通过工具获取；数据不足时如实声明，保留「忠实于数据」提示词纪律
- **失败降级链** — 瞬时故障自动重试 → 模型输出非法回退到既有规则兜底（关键词路由、启发式决策）→ 持续失败如实上报「模型调用失败，并非数据不足」
- **多智能体分析流水线** — 球员运动分析、球互动分析、球路事件识别、对比排名、综合决策与解说稿生成，全链路由 LangGraph 编排
- **反事实推演** — 基于 MCTS 的 What-If 模拟：指定球员在特定时刻改为射门/传球/盘带等不同决策，推演后续轨迹与胜率变化，可导出轨迹 CSV
- **数据核验** — 对系统回答提出质疑时，自动回溯原始数据核验并纠正
- **Web 可视化** — FastAPI + SSE 流式对话，Canvas 逐帧渲染球员/球轨迹
- **统一 Harness** — 透明包装层：全链路 Trace、统一 CLI/API 入口、外部依赖 Mock 与 golden 回放回归（LLM 请求序列/逐轮输出/状态快照/文件哈希四维比对），工具调用不影响等价性

## 安装

```bash
pip install -r requirements.txt   # 或 uv pip install -r requirements.txt
```

复制 `.env`（已含占位模板）并填入你的 API Key：

```bash
# .env
DEEPSEEK_API_KEY=此处更换为你的真实ApiKey
```

## 快速开始

```bash
# 交互模式（推荐）：加载数据后可直接用中文提问
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv

# 单次解说模式（可聚焦特定球员）
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --once
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --focus 6 10

# Web 可视化界面（SSE 流式对话 + 轨迹渲染）
python -m web.backend
# 打开 http://127.0.0.1:8000
```

对话示例：

```
>> 聚焦7号
>> 7号的跑动风格是什么
>> 对比6号和7号谁更积极
>> 如果7号在第3秒改为直接射门会怎样
>> 第90帧发生了什么
```

### 输入数据格式

| 数据 | 列 | 说明 |
|------|-----|------|
| `person.csv` | `frame_num, track_id, x, y, color` | `color`: A=红队, B=蓝队, C=门将 |
| `ball.csv` | `frame_num, x, y, z` | `z`: 球心离地高度（米） |

## 架构组成

- **数据层** — CSV 经帧插值与球员/球数据融合为 `PrefixPlayerCorpus`
- **分析层** — 两条 LangGraph 流水线：球员追踪（运动/互动/球路/对比/决策/解说）与专业分析（采集/风格建模/MCTS 反事实/报告），节点间状态流转由 `StateGraph` + `MemorySaver` 管理
- **LLM 层** — 统一 `call_llm` 客户端（DeepSeek，流式/非流式 + function calling）：`tools` 参数与多轮工具循环经 ContextVar 交换通道透传，不改变传输签名与 harness 钩子；无工具调用时行为与纯文本接口完全一致
- **工具层** — `agents/tools/`：`ToolSpec`/`ToolRegistry` 注册表 + pydantic 决策工具 schema + 6 个数据工具，覆盖失败分类（参数非法/执行异常/数据不足）
- **交互层** — CLI（REPL/单次解说）与 Web（FastAPI + SSE）双入口，均接入统一 Harness（透明包装：trace 记录、golden 录制、mock 回放）

| 智能体 / 模块 | 职责 |
|--------|------|
| PlayerMovementAgent / BallInteractionAgent / BallRelationAgent | 跑动、触球控球、球路事件分析 |
| PlayerComparisonAgent / DecisionAgent / CompositionAgent | 对比排名、态势判断、解说稿生成 |
| QueryRouter | 意图路由（`submit_intent` 工具契约，关键词规则兜底） |
| LLMDecisionEngine / SemanticTierBatcher | MCTS 决策引擎与语义档位推断（工具契约 + 启发式回退） |
| DataCollectorAgent / StyleModelingAgent | 数据采集、球员风格建模 |
| CounterfactualEngineAgent | MCTS 反事实推演与轨迹模拟 |
| ReportGeneratorAgent / GeneralQAAgent / DataVerifierAgent | 报告生成、综合问答（数据工具）、数据核验（质疑判断工具契约） |
| agents/tools | 终止型决策工具（submit_*）+ 数据工具（战术事实/球路/帧核验/球员数据/反事实模拟） |

## 测试

```bash
pytest            # 全部用例（真实 API 冒烟默认跳过，共 58 个：37 原有 + 21 工具调用）
pytest -m smoke   # 需要 DEEPSEEK_API_KEY 的真实 API 冒烟

# Harness golden 回放回归（不触网，验证代码改动不破坏等价性）
python -m harness verify harness/golden/standard
```

## 技术栈

| 领域 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| 智能体编排 | LangGraph（状态图 + 内存检查点） |
| LLM | DeepSeek Chat API（流式/非流式 + function calling） |
| 工具 schema | pydantic v2（JSON Schema + 参数校验） |
| 反事实推演 | MCTS 蒙特卡洛树搜索 |
| 后端 | FastAPI + SSE |
| 前端 | 原生 HTML/CSS/JS，Canvas 逐帧渲染 |
| 数据分析 | NumPy、Pandas |
| 可视化 | Matplotlib |
| 可观测性 | 自研 Harness（Span 树 Trace / golden 回放 / 状态快照） |

## 安全

- API Key 只放在 `.env`，已加入 `.gitignore`；仓库内仅保留占位模板，勿提交真实密钥
- 反事实轨迹导出、解说稿等生成结果输出到 `Output/` 目录，不污染源码

## 文档

| 目标 | 入口 |
|------|------|
| 安装与使用 | 本文件「安装」「快速开始」 |
| 统一 Harness（运行模式/等价性论证/验证套件/已知边界） | [docs/harness.md](docs/harness.md) |
| 工具调用重构（选型理由/迁移方案/执行总结/审查修复记录） | [docs/tool_calling_refactor.md](docs/tool_calling_refactor.md) |
| 输入数据格式 | 本文件「输入数据格式」 |
