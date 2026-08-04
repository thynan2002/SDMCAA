# Football Agent — 足球战术分析智能体系统

基于 **多智能体（Multi-Agent）** 架构的足球比赛数据分析与战术解说系统。
输入球员追踪 CSV 与足球轨迹 CSV，由 LLM 驱动的智能体流水线自动产出专业级比赛解说、球员表现分析与反事实（What-If）推演。

## 功能特性

- **多智能体分析流水线** — 球员运动分析、球互动分析、球路事件识别、对比排名、综合决策与解说稿生成，全链路由 LangGraph 编排
- **自然语言交互** — 加载数据后以对话方式提问（REPL），支持多轮上下文与意图路由（LLM 驱动）
- **专业分析子系统** — 球员风格建模、综合报告生成、数据核验回查
- **反事实推演** — 基于 MCTS 的 What-If 模拟：指定球员在特定时刻做出不同决策（射门/传球/盘带等），推演后续轨迹与胜率变化，可导出轨迹 CSV
- **Web 可视化** — FastAPI + SSE 流式对话，Canvas 逐帧渲染球员/球轨迹
- **统一 Harness** — 透明包装层：全链路 Trace、统一 CLI/API 入口、外部依赖 Mock，且不改变智能体语义

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt   # 或 uv pip install -r requirements.txt

# 配置 API Key（写入 .env）
# DEEPSEEK_API_KEY=your_key_here

# 交互模式（推荐）
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv

# 单次解说模式（可聚焦特定球员）
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --once
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --focus 6 10

# 启动 Web 可视化界面
python -m web.backend
# 访问 http://127.0.0.1:8000
```

### 输入数据格式

| 数据 | 列 | 说明 |
|------|-----|------|
| `person.csv` | `frame_num, track_id, x, y, color` | `color`: A=红队, B=蓝队, C=门将 |
| `ball.csv` | `frame_num, x, y, z` | `z`: 球心离地高度（米） |

## 架构

```
                 ┌──────────────┐
                 │  CSV 输入     │
                 │ (person+ball) │
                 └──────┬───────┘
                        ▼
        ┌───────────────────────────────┐
        │     PrefixPlayerCorpus        │
        │   (帧插值、球员/球数据融合)     │
        └───────┬───────┬───────────────┘
                │       │
        ┌───────▼──┐ ┌──▼────────────┐
        │ Player   │ │ Professional  │
        │ Tracking │ │ Analysis      │
        │ Agent    │ │ Agent         │
        │ Pipeline │ │ Pipeline      │
        │(LangGraph)│ │(LangGraph)    │
        └───────┬──┘ └──┬────────────┘
                │       │
        ┌───────▼───────▼────────────┐
        │    LLM Client (DeepSeek)    │
        │    流式 / 非流式调用        │
        └────────────────────────────┘
                │       │
        ┌───────▼──┐ ┌──▼────────────┐
        │  CLI     │ │  Web (FastAPI)│
        │ 输出/交互 │ │  前端可视化   │
        └──────────┘ └───────────────┘
```

### 智能体组成

| 智能体 | 模块 | 职责 |
|--------|------|------|
| PlayerMovementAgent | `agents/player/` | 跑动距离、速度、冲刺、活动范围 |
| PlayerBallInteractionAgent | `agents/player/` | 触球、控球、争顶等球互动 |
| BallPositionRelationAgent | `agents/player/` | 球路事件识别（传球、射门、盘带、解围） |
| PlayerComparisonAgent | `agents/player/` | 球员跑动积极度对比 |
| PlayerDecisionAgent | `agents/player/` | 比赛态势综合判断（节奏、攻防） |
| PlayerCompositionAgent | `agents/player/` | 最终解说稿生成（时间分段） |
| DataCollectorAgent | `agents/professional/` | 专业分析数据采集 |
| StyleModelingAgent | `agents/professional/` | 球员风格建模 |
| CounterfactualEngineAgent | `agents/professional/` | 反事实推演（MCTS） |
| ReportGeneratorAgent | `agents/professional/` | 专业报告生成 |

## 测试

```bash
pytest            # 全部用例（真实 API 冒烟默认跳过）
pytest -m smoke   # 需要 DEEPSEEK_API_KEY 的真实 API 冒烟
```

旧命令行入口（`tools/test_*.py`）保留为薄包装，与 pytest 共享同一套实现。

## 统一 Harness

`harness/` 是智能体的透明包装层：全链路观测（规划/工具/记忆/LLM/输出）、
统一 CLI/API 入口、外部依赖 Mock 与 golden 回放回归，不引入语义偏差。
CLI 与 Web 均已接入。

```bash
python -m harness run TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv
python -m harness verify harness/golden/standard   # golden 回放回归（不触网）
```

完整文档（运行模式、等价性论证、验证套件、已知边界）见 [docs/harness.md](docs/harness.md)。

## 项目结构

```
football-agent/
├── main.py                    # CLI 入口
├── agents/                    # 智能体核心
│   ├── llm_client.py          #   DeepSeek API 调用（流式+非流式）
│   ├── player/                #   球员追踪智能体（LangGraph）
│   ├── session/               #   交互会话管理（REPL、记忆、路由）
│   └── professional/          #   专业分析子系统（LangGraph + MCTS）
├── harness/                   # 统一 Harness（透明包装层 + golden 数据集）
├── web/                       # FastAPI 后端（SSE 流式）+ 前端
├── tools/                     # 独立分析脚本 / 测试命令行入口
├── tests/                     # pytest 测试
├── TestInput/                 # 测试 CSV 数据
└── Output/                    # 生成结果（解说稿、反事实轨迹、traces）
```

## 技术栈

- **语言**: Python 3.12+
- **编排**: LangGraph（状态图 + 内存检查点）
- **LLM**: DeepSeek Chat API（流式/非流式）
- **后端**: FastAPI + SSE
- **前端**: 原生 HTML/CSS/JS，Canvas 逐帧渲染
- **反事实推演**: MCTS（蒙特卡洛树搜索）
- **数据分析**: NumPy、Pandas
- **可视化**: Matplotlib
