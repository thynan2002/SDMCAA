# 统一 Harness（透明包装层）

`harness/` 是智能体的**透明包装层**：全链路观测（规划/工具/记忆/LLM/输出）、
统一 CLI/API 入口、外部依赖 Mock，且不引入任何语义偏差。
Web 后端（`web/backend.py`）已迁移到 Harness 之下，CLI 与 Web 共用同一包装层。

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

### Trace

交互过程中的完整决策链路以 Span 树（`run → turn → llm/stage/stream/snapshot/output`）
写入 `Output/traces/*.jsonl`，含 LLM 请求/响应哈希、耗时、阶段事件与状态快照摘要。

## 等价性论证（为何不引入语义偏差）

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

## 等价性验证

- **A/B 同进程对比**：同一确定性 mock 下，裸 SessionManager 与 Harness 包装的
  输出/状态快照/LLM 请求序列三者一致（`tests/test_harness_equivalence.py`）
- **golden 回放回归**：`harness/golden/standard`（真实 API 录制，39 次 LLM 调用、
  覆盖 8 种意图 + 反事实轨迹导出）回放四维比对全部 PASS
- 回归报告由 `python -m harness verify` 生成（LLM 请求序列 / 逐轮输出 unified diff /
  状态快照字段级差异 / 产出文件哈希 / 已知边界声明）

## 已知等价性边界（允许差异）

- 耗时/延迟、内存占用（trace 写入与快照深拷贝的开销）
- 日志与 trace 体量（新增 `Output/traces/*.jsonl`；stdout 不变）
- 含时间戳的文件名（对话存档、trace 文件名）
- 真实 LLM 模式的输出文本（temperature 采样不可复现；逐字节断言仅在 replay 确定性模式下有效）
- MCTS 默认未 seed：原始版本本身非确定；record/replay 注入 seed 属"确定性增强"

## 模块一览

| 文件 | 职责 |
|------|------|
| `core.py` | `Harness` 类：组合 SessionManager，统一入口 |
| `config.py` | 运行模式配置（passthrough/record/replay） |
| `interceptors.py` | LLM 拦截器（观测/录制/回放三态） |
| `tracing.py` | Span 树 trace（JSONL） |
| `snapshot.py` | 只读状态快照 + 字段级对比 |
| `mock.py` | golden 存储 + ReplayLLM |
| `cli.py` | `python -m harness`（run/verify/snapshot-diff） |
| `verify/` | 回归验证 / 快照对比 / diff 报告 |
| `scripts/` | 录制脚本 |
| `golden/` | golden 回归数据集 |
