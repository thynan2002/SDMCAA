# Football Agent — 足球战术分析智能体系统

基于 **多智能体（Multi-Agent）** 架构的足球比赛数据分析与战术解说系统。输入球员追踪 CSV 和足球轨迹 CSV，通过 LLM 驱动的智能体流水线，输出专业级比赛解说、球员表现分析和反事实推演。

## 功能

- **多智能体分析流水线** — 球员运动分析、球互动分析、对比排名、综合决策、解说稿生成，全链路由 LangGraph 编排
- **交互式 REPL** — 加载数据后以自然语言对话式提问，支持多轮上下文
- **单次解说模式** — 指定 CSV 文件直接输出完整解说稿，支持聚焦特定球员
- **专业分析子系统** — 球员风格建模、反事实（What-If）推演、综合报告生成
- **反事实推演** — 基于 MCTS 模拟"如果某球员在某时刻做出不同决策"的后续轨迹
- **Web 可视化界面** — FastAPI 后端 + 前端 Canvas 逐帧渲染球员/球轨迹，支持流式对话 SSE
- **轨迹可视化** — 独立 matplotlib 动画脚本 `showAll.py`

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
| PlayerMovementAgent | `agents/player/` | 分析球员跑动距离、速度、冲刺、活动范围 |
| PlayerBallInteractionAgent | `agents/player/` | 分析球员与球的互动（触球、控球、争顶） |
| BallPositionRelationAgent | `agents/player/` | 球路事件识别（传球、射门、盘带、解围） |
| PlayerComparisonAgent | `agents/player/` | 球员跑动积极度对比 |
| PlayerDecisionAgent | `agents/player/` | 比赛态势综合判断（节奏、攻防） |
| PlayerCompositionAgent | `agents/player/` | 最终解说稿生成（时间分段） |
| DataCollectorAgent | `agents/professional/` | 专业分析数据采集 |
| StyleModelingAgent | `agents/professional/` | 球员风格建模 |
| CounterfactualEngineAgent | `agents/professional/` | 反事实推演（MCTS） |
| ReportGeneratorAgent | `agents/professional/` | 专业报告生成 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt  # 或 uv pip install -r requirements.txt

# 设置 API Key（写在 .env 文件中）
# DEEPSEEK_API_KEY=your_key_here

# 交互模式（推荐）
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv

# 单次解说模式
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --once

# 聚焦特定球员解说
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --focus 6 10 1

# 启动 Web 可视化界面
python -m web.backend
# 访问 http://127.0.0.1:8000
```

### 输入 CSV 格式

**球员数据 (`person.csv`)**: `frame_num, track_id, x, y, color`
- `color`: A=红队, B=蓝队, C=门将

**足球数据 (`ball.csv`)**: `frame_num, x, y, z`
- `z`: 球心离地高度（米）

## 项目结构

```
football-agent/
├── main.py                    # CLI 入口
├── showAll.py                 # 轨迹可视化 (matplotlib)
├── .env                       # LLM API Key 配置
├── agents/
│   ├── llm_client.py          # DeepSeek API 调用 (流式+非流式)
│   ├── prompts.py             # 智能体提示词管理
│   ├── types.py               # 共享数据结构 (AgentReport, FinalDecision)
│   ├── constants.py           # 物理常量 (球场尺寸、速度阈值)
│   ├── player/                # 球员追踪智能体 (LangGraph)
│   ├── session/               # 交互会话管理 (REPL, 记忆, 路由)
│   └── professional/          # 专业分析子系统 (LangGraph + MCTS)
├── web/
│   ├── backend.py             # FastAPI 后端 (SSE 流式)
│   └── static/                # 前端资源
├── tools/                     # 独立分析脚本
├── TestInput/                 # 测试 CSV 数据
└── Output/                    # 生成结果
```

## 技术栈

- **语言**: Python 3.12+
- **编排**: LangGraph (状态图 + 内存检查点)
- **LLM**: DeepSeek Chat API (流式/非流式)
- **后端**: FastAPI + SSE (Server-Sent Events)
- **前端**: 原生 HTML/CSS/JS, Canvas 逐帧渲染
- **反事实推演**: MCTS (蒙特卡洛树搜索)
- **数据分析**: NumPy, Pandas
- **可视化**: Matplotlib
