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

### 测试 Harness

标准 pytest 结构（`tests/`），旧命令行入口（`tools/test_*.py`）保留为薄包装，
两者共享 `tests/_common.py` 中的同一套实现，行为一致。

```bash
# 运行全部 pytest 用例（真实 API 冒烟默认跳过）
pytest

# 指定 smoke 标记（需要 DEEPSEEK_API_KEY 的真实 API 冒烟）
pytest -m smoke

# 旧式命令行入口（输出 [OK]/[FAIL]，退出码 0=通过 / 1=失败）
python tools/test_counterfactual_sim.py
python tools/test_llm_decision.py
```

| 文件 | 覆盖内容 |
|------|----------|
| `tests/conftest.py` | 项目路径引导 + 共享夹具（corpus / mock LLM / bad LLM） |
| `tests/_common.py` | pytest 用例与命令行入口共享的模拟/校验/mock 工具 |
| `tests/test_corpus.py` | 数据加载 Harness（语料构建、插值、多数据集、视频预留层） |
| `tests/test_counterfactual_sim.py` | 反事实轨迹端到端（物理校验/前缀一致性/格式闭环/专项验证） |
| `tests/test_llm_decision.py` | LLM 决策引擎接线（mock 全链路/非法回退/纯启发式/冒烟） |
| `tests/test_harness_equivalence.py` | 统一 Harness 等价性套件（A/B 对比/record-replay 闭环/回放契约/工具链） |

## 统一 Harness（透明包装层）

`harness/` 是智能体的**透明包装层**：全链路观测（规划/工具/记忆/LLM/输出）、
统一 CLI/API 入口、外部依赖 Mock，且不引入任何语义偏差。
Web 后端（`web/backend.py`）已迁移到 Harness 之下，CLI 与 Web 共用同一包装层。

### 三种运行模式

| 模式 | 说明 |
|------|------|
| `passthrough` | 生产默认。仅只读观测（trace），语义与原系统恒等 |
| `record` | 真实调用 LLM，同时录制 golden 数据集（LLM 流量/逐轮输出/状态快照/文件哈希） |
| `replay` | 不触网。LLM 由 golden 精确回放（确定性），用于回归验证 |

### 用法

```bash
# 交互 / 单次解说（与 main.py 等价，额外产出 trace）
python -m harness run TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv
python -m harness run TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --once --focus 6

# 录制 golden（真实 API；--seed 启用确定性注入）
python -m harness run TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv \
    --script harness/scripts/standard_12s.txt \
    --mode record --golden-dir harness/golden/standard --seed 42

# 回放回归（不触网；LLM 序列/逐轮输出/状态快照/文件哈希四维比对）
python -m harness verify harness/golden/standard --report report.md

# 状态快照字段级对比
python -m harness snapshot-diff A.json B.json
```

Python API：`with Harness(HarnessConfig(mode=...)) as h: h.load_data(...); h.handle_input(...)`。
Web 后端经 `HarnessConfig.from_env()` 读取 `HARNESS_MODE` 等环境变量。

### 等价性论证（为何不引入语义偏差）

| 机制 | 论证 |
|------|------|
| 恒等委托 | `Harness.handle_input(t) ≡ SessionManager.handle_input(t)`，无预处理/后处理 |
| 纯函数透传包装 | LLM 拦截器 `g(x)=observe(f(x))`；observe 只读记录，不改返回值、不吞异常、不改重试/流式分支 |
| 只读代理 | 状态快照经 deepcopy 边界取样，无写回路径 |
| 回调复合（tee） | 进度/流式回调与调用方回调复合而非替换，Web SSE 行为保留 |
| 确定性注入 | mock/seed 仅 record/replay 启用，且 seed 走作者预留构造参数；passthrough 注入器为恒等 |

对现有代码仅 2 处最小改动：`agents/llm_client.py` 增加调用分派器钩子
（默认 `None` = 与原实现逐字节等价）+ 2 个只读访问器；
`agents/progress.py` 增加 1 个只读访问器。

### 等价性验证

- **A/B 同进程对比**：同一确定性 mock 下，裸 SessionManager 与 Harness 包装的
  输出/状态快照/LLM 请求序列三者一致（`tests/test_harness_equivalence.py`）
- **golden 回放回归**：`harness/golden/standard`（真实 API 录制，39 次 LLM 调用、
  覆盖 8 种意图 + 反事实轨迹导出）回放四维比对全部 PASS

### 已知等价性边界（允许差异）

- 耗时/延迟、内存占用（trace 写入与快照深拷贝的开销）
- 日志与 trace 体量（新增 `Output/traces/*.jsonl`；stdout 不变）
- 含时间戳的文件名（对话存档、trace 文件名）
- 真实 LLM 模式的输出文本（temperature 采样不可复现；逐字节断言仅在 replay 确定性模式下有效）
- MCTS 默认未 seed：原始版本本身非确定；record/replay 注入 seed 属"确定性增强"

## 项目结构

```
football-agent/
├── main.py                    # CLI 入口
├── showAll.py                 # 轨迹可视化 (matplotlib)
├── pyproject.toml             # pytest Harness 配置
├── .env                       # LLM API Key 配置
├── agents/
│   ├── llm_client.py          # DeepSeek API 调用 (流式+非流式, 含 Harness 分派器钩子)
│   ├── prompts.py             # 智能体提示词管理
│   ├── types.py               # 共享数据结构 (AgentReport, FinalDecision)
│   ├── constants.py           # 物理常量 (球场尺寸、速度阈值)
│   ├── player/                # 球员追踪智能体 (LangGraph)
│   ├── session/               # 交互会话管理 (REPL, 记忆, 路由)
│   └── professional/          # 专业分析子系统 (LangGraph + MCTS)
├── harness/                   # 统一 Harness（透明包装层）
│   ├── core.py                #   Harness 类：组合 SessionManager，统一入口
│   ├── config.py              #   运行模式配置 (passthrough/record/replay)
│   ├── interceptors.py        #   LLM 拦截器（观测/录制/回放三态）
│   ├── tracing.py             #   Span 树 trace（JSONL）
│   ├── snapshot.py            #   只读状态快照 + 字段级对比
│   ├── mock.py                #   golden 存储 + ReplayLLM
│   ├── cli.py                 #   python -m harness（run/verify/snapshot-diff）
│   ├── verify/                #   回归验证 / 快照对比 / diff 报告
│   ├── scripts/               #   录制脚本
│   └── golden/                #   golden 回归数据集
├── web/
│   ├── backend.py             # FastAPI 后端 (SSE 流式, 持有 Harness 实例)
│   └── static/                # 前端资源
├── tools/                     # 独立分析脚本 / 测试命令行入口
├── tests/                     # pytest 测试 Harness
├── TestInput/                 # 测试 CSV 数据
└── Output/                    # 生成结果 (含 traces/)
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
