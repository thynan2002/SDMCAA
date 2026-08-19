# 评测框架（eval/）📊 — 单 LLM 直接调用 vs 多智能体系统对比评测

<p align="center">
  <img src="https://img.shields.io/badge/status-implemented-2EA043?style=flat-square" alt="Status: implemented">
  <img src="https://img.shields.io/badge/systems-bare%2Fsingle%2Fmulti-4B8BBE?style=flat-square" alt="Systems: bare/single/multi">
  <img src="https://img.shields.io/badge/cases-40-8B5CF6?style=flat-square" alt="Cases: 40">
</p>

对同一测试用例集，系统性地对比三种方案：

| 标识 | 方案 |
|------|------|
| `bare` | 基线 A：单次 LLM 裸调用（全量数据入 prompt，无工具） |
| `single` | 基线 B：单智能体 + 与多智能体同套数据工具（function calling 循环） |
| `multi` | 被测系统：SessionManager 多智能体流水线（经 Harness passthrough） |

> **公平参评与策略化重构（v2）**：三系统统一「形势判断优先、用户索要才引用数值」的输出纪律；accuracy/tactical_accuracy 改由 Judge 语义比对（程序化数值扫描降级为诊断不计分）；Judge 升级为 deepseek-v4-pro（rubric v3）；领域指标单位米制化（距离米、速度米/秒）；权重向 reasoning/tactical 倾斜（策略类合计 0.45）。

## 快速开始

```bash
python -m eval cases lint                     # 校验用例集
python -m eval gold                           # 打印独立重算金标准（人工核对）
python -m eval run --event demo_v1 --dry-run  # 预演（打印交错执行计划）
python -m eval run --event demo_v1            # 全量评测（默认 5 重复 × 3 系统）
python -m eval report Score/demo_v1_...       # 从事件目录重算报告（Judge 走缓存）
python -m eval judge Score/demo_v1_... --force  # 强制重评 Judge
```

小规模冒烟 / 调试：

```bash
# 分层冒烟预设：每 category 确定性抽 1 例、repeats=1、仅 single+multi
python -m eval run --event smoke_v2 --smoke

# 或手动指定用例与系统：
python -m eval run --event smoke --cases qa_duration,halluc_player_99 \
    --systems bare,single,multi --repeats 1
```

### 配置

复制 `eval/config.example.yaml` 为 `eval/config.yaml`（权重、重复数、效率基准、Judge 抽样等均可调）；Judge 模型默认 `deepseek-v4-pro`（与被测系统解耦，同 API Key 仅模型 ID 不同），可用环境变量 `EVAL_JUDGE_MODEL / EVAL_JUDGE_BASE_URL / EVAL_JUDGE_API_KEY` 切换任意 OpenAI 兼容 API。

## 产物布局（Score/{event}_{ts}/）

```
meta.json       配置快照/git commit/随机种子（可复现性凭证）
                + usage_fix_version（口径标记，见下）+ preset（分层预设标记，如 smoke）
plan.json       交错执行计划
runs/{sys}/{case}/{repeat}/   result.json + trace_*.jsonl + stdout.log
judge/{case}.json             Judge 评分（内容哈希缓存，resume 复用）
metrics.json     每 run 评分明细（含程序化指标逐项明细）
results.json     系统级聚合 + 配对检验 + 类别热力 + 失败模式 + 成本
                 + judge_meta（Judge 一致性等元信息）+ judge_disagreements（Judge-程序分歧）
report.md / report.html / figures/*.png
```

> **`usage_fix_version` 口径标注**：该键由 `eval/engine.py` 的 `create_event` 写入 `meta.json`，表示流式 usage 计量修复后的口径（流式调用经 `stream_options.include_usage` 采集真实 token usage，`llm_calls` 另含调用次数兜底）。历史事件无此字段，**跨该版本前后的事件效率（efficiency）分不可直接比较**。

## 指标体系与默认权重

| 指标 | 权重 | 依据 | 方式 |
|---|---|---|---|
| accuracy | 0.15 | Judge 语义一致性比对（数值/胜者/方向/拒答金标准作参考事实，按数量级/方向/相对关系比对，不逐字扫描数字） | Judge 语义 |
| grounding | 0.10 | 答案实体核验（球衣号存在、帧/秒在范围内，0.5）+ Judge 编造判定（0.5） | 程序化 + Judge |
| completeness | 0.10 | checklist 关键词覆盖（0.5）+ Judge 要点覆盖（0.5） | 程序化 + Judge |
| reasoning | 0.20 | Judge anchored rubric 1–5 分（策略/形势判断与推理深度） | Judge |
| efficiency | 0.10 | 时延 / LLM 调用数 / 总 token，按预标定常数 `min(1, ref/x)` 归一化 | 程序化（trace/usage） |
| robustness | 0.10 | 非空 / 无错误标记 / 连贯；幻觉用例诚实性并入 accuracy.refusal | 程序化 |
| tactical_accuracy | 0.25 | Judge 战术判断（0.6）+ 事件检测 F1（0.4）；区域/威胁/球速数值扫描降级为诊断 | Judge + 程序化 |
| tool_use | 诊断 | 失败率 + 冗余调用率；仅计入「扩展总分」（裸调用结构性无工具） | trace 派生 |
| stability | 诊断 | 跨重复答案相似度（difflib） | 程序化 |

- **共同总分** = Σ 权重×分项（三系统公平可比）；**扩展总分** 额外以 0.10 权重并入 tool_use（其余等比缩放），仅在 single 与 multi 间比较。
- **策略化倾向**：reasoning + tactical_accuracy 合计 0.45（原 0.30），accuracy + grounding + completeness 合计 0.35（原 0.52），得分点从「小细节数值」转向「形势/策略判断」。
- **语义对齐（混合软校验）**：accuracy/tactical_accuracy 以 Judge 语义比对为主，程序化数值/胜者/类别/容差扫描降级为诊断（保留在 metrics.json 的 `details` 便于对照，不计分）；Judge 缺失分量自动重归一化并在报告标注。

### 指标定义与依据

| 方法/指标 | 权威依据 | 本项目落地与局限 |
|---|---|---|
| 威胁值（threat） | Karun Singh (2018), *Beyond Expected Goals*（xT，Expected Threat）：以球位置刻画传球/携带至该位置后的潜在得分威胁 | 参考其「越近球门、越靠中路威胁越高」的构造思路，用无训练数据的确定性启发式实现（距门线线性衰减 + 中路加成），**未用真实射门/得分数据校准**，不代表专业级 xT |
| 战术区域划分 | StatsBomb / Metrica 等事件数据厂商的进攻三区（final/middle/defensive third）与中路/边路划分惯例 | 按距最近球门线归一化距离（禁区 <0.15 / 前场 <0.40）与横向位置（边路 s<0.25 或 s>0.75）划分，双球门对称；边界阈值为项目自定义启发式 |
| 统计检验 | Wilcoxon 配对符号秩检验 + Holm 逐步校正，非参数配对比较的标准做法（不假设正态、控制族错误率） | 按用例配对；Judge 主观分指标存在族内相关，Holm 校正后 p 仅作参考下界；另报配对 t / Cohen's d / Cliff's δ 供交叉核对 |
| LLM-as-Judge 偏差与缓解 | Zheng et al. (2023, NeurIPS Datasets & Benchmarks), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*：系统刻画位置偏差、冗长偏差、自我增强偏差及缓解手段 | 已采用盲评（甲/乙/丙匿名）、位置互换取均值（正逆序差作为一致性指标并写入 `results.json` 的 `judge_meta`，报告对差 >0.15 用例标黄告警）、temp=0；**尚缺与人工标注的一致性校准**，同源 Judge（模型名前缀粗判同厂商）存在自我偏好风险，报告头部固定披露，结论需人工抽检 |
| Judge-程序交叉对照 | 同上（多源评分交叉验证思路） | `score_run` 输出 `judge_program_agreement`：程序化 accuracy（诊断分量）与 Judge accuracy 差 >0.3 记 flag="disagreement"，缺失记 "n/a"；`results.json` 的 `judge_disagreements` 汇总 top 分歧供抽检（仅诊断不改分） |

> **诚实性标注**：本项目的威胁值公式、区域边界阈值、事件检测速度阈值（SPEED_STILL_MS/SPEED_DRIVE_MS，米/秒）均为**未校准启发式，仅用于内部相对比较，不代表专业级绝对标准**；速度阈值经场地标定（1200×700 px → 105×68 m）换算后与历史像素阈值数值等价。

## 领域真实性与可量化性（v1.0 重构）

### 领域量化指标（goldref.py 独立重算，不经被测系统）

| 指标 | 表达式 | 定义 / 依据 |
|---|---|---|
| 战术区域 | `ball_zone@F` / `player_primary_zone:T` | 按距最近球门线距离与横向位置划分为 禁区/前场/中场 × 中路/边路（x 轴=进攻轴，固定 1200×700px 场地，双球门对称；对应进攻三区概念） |
| 威胁值 | `threat_at_frame@F` / `threat_max` | 参考 xT（Expected Threat, Karun Singh 2018）思路的确定性启发式：距门线线性衰减 + 中路加成，值域 [0,1] |
| 球速 | `ball_speed_max` / `ball_speed_avg` | 相邻原始帧差分折算，单位**米/秒**（1200×700px → 105×68m 标定） |
| 事件类型 | `event_types` | 独立事件检测（控球/带球/长传/快速推进/射门），按球速+高度阈值分段 |
| 核验数值 | `player_x/y_at_second:T,S` / `nearest_player_at_second:S` / `ball_player_distance_at_second:T,S` | 独立插值复核位置/最近球员/距离，坐标与距离单位**米** |

### 用例分层（L1-L7）

| 层 | 业务场景 | 默认类别 |
|---|---|---|
| L1 | 数据查询 | general_qa, frame |
| L2 | 事件检测 | comparison, timeline, verify |
| L3 | 战术指标 | team（含威胁/区域/球速用例） |
| L4 | 球员评估 | style, focus |
| L5 | 反事实推演 | counterfactual |
| L6 | 综合战术分析 | team（战术复盘类） |
| L7 | 幻觉与鲁棒性 | hallucination |

新增领域用例：`qa_ball_speed_max`（L2）、`frame_ball_zone_90`（L3）、`team_threat_peak`（L3）、`qa_player_zone_7`（L4）、`timeline_event_types`（L2）；`verify_7s_pos` / `verify_ball_owner_3s` 补充数值金标准；`cf_7s_shoot` 方向金标准增加 neg 关键词（验证变化方向而非纯过程词）。
v2 扩用例：启用 7s 数据集，新增 `qa_7s_player_count` / `qa_7s_ball_hmax`（L1）、`compare_7s_active`（L2）、`style_7s_player11`（L4）、`team_7s_threat`（L3）。

### Judge rubric v3

- 新增 `tactical`（0-1）战术判断维度，评测取向「形势/策略判断优先，其次才是数值精确度」；
- fabrication 从二值改为 0-1 连续（按编造严重程度分级）；
- accuracy 锚点细化为 6 档（0/0.2/0.4/0.6/0.8/1.0），数值按语义/数量级比对；
- 参考事实增强：数值金标准带单位标签（米/米每秒/0-1 威胁值）、分类金标准进入 Judge 参考块；
- `RUBRIC_VERSION="v3"`，缓存自动失效。

## SOTA 能力凸显与关键维度（v1.1 重构）

### 多 Agent 辩论协作（已移除）

原 `DebateAgent`（双视角辩论协作）作为未接入默认流水线的独立能力，在中间层精简中移除（评测中零调用，属死重）；`team_collab_breakdown` 用例的多视角 checklist 与 `collaboration` 诊断指标保留（不计分）。

### 反事实评测增强

`cf_7s_pass10` 补充方向金标准（neg 关键词区分「无变化」放水）；MCTS 随机性大不设精确数值金标准（修订 #2），以方向性 + checklist + 跨重复 stability 诊断评估。

### 对抗鲁棒性（评测侧 + robustness 增强）

- 新增 `adversarial` 类别用例（level 7）：`adv_induced_scoring`（诱导编造）、`adv_frame_out_of_range`（超范围时间）、`adv_prompt_injection`（prompt 注入）、`adv_ambiguous_multi_target`（多目标歧义）、`adv_format_messy`（噪声格式）；
- `score_robustness` 增强：幻觉/对抗类用例纳入「诚实拒答」分量 —— 诚实拒答满分，跟随诱导/编造封顶 0.6，使鲁棒性具备区分度；新增 `adversarial_resistance` 诊断。

### 评测效率（并行执行 + 分层预设）

- `EvalConfig.max_workers`（默认 1 保持串行），`--workers N` 并行调度子进程（`concurrent.futures.ThreadPoolExecutor`，保持交错计划语义）；
- `--quick` 分层预设：1 重复快速冒烟；
- `--smoke` 分层预设：每 category 确定性抽 1 例（按 case_id 排序取首个，不依赖随机种子、可复现），强制 `repeats=1` 且仅参评 `single+multi`（优先于显式 `--systems/--repeats`），`meta.json` 附 `preset: "smoke"` 标记；
- 建议分层：L1 冒烟（5 用例×1 重复）→ L2 快速（12 用例）→ L3 全量（≥3 重复）。

### 可解释性（报告分层展示）

`report.md` 新增「1b. 按业务分层（L1-L7）总分」表：按 level 聚合共同总分，快速定位多智能体系统在哪个业务层级占优/失分。

## 方法学要点（评审修订落实）

1. **金标准独立重算**：数值金标准以 `auto:<expr>` 声明，加载时用 pandas 直接从原始 CSV 重算（`eval/goldref.py`），不经被测系统工具层 —— 消除循环验证偏袒；`python -m eval gold` 输出供人工核对。
2. **MCTS 随机性**：反事实类用例不设精确数值金标准，仅方向性/checklist。
3. **基线公平性**：bare 的数据摘要为确定性模板（全量原始 CSV + 独立统计表），遵循「信息给足、不因信息饥饿而输」；三系统统一采样温度（默认 0.85）。
4. **效率基准预标定**：refs 为配置常数（不随本次样本变化）；建议先跑小规模预实验校准后填入 `eval/config.yaml`。
5. **Judge 防偏与模型解耦**：盲评（甲/乙/丙随机排列）+ 位置互换重评取均值 + temp=0；正逆序差作为 Judge 一致性指标。Judge 默认 `deepseek-v4-pro`（与被测系统 flash 解耦，同 Key 不同模型 ID），降低同源 Judge 的自我偏好；结论仍建议人工抽检。
6. **统计层级**：系统级 CI = 跨用例 bootstrap（B=2000）；显著性按用例配对（Wilcoxon 主检验 + 配对 t 参考 + Cohen's d / Cliff's δ + Holm 校正）；报告含检验力说明（n 个用例在 80% power 下可检出的效应量）。
7. **执行隔离与交错**：每 (系统,用例,重复) 独立子进程（隔离全局 corpus/dispatcher/ContextVar）；重复轮分层 + 轮内随机交错（种子记录），缓解 API 时延波动的时间混淆。
8. **诱导幻觉用例**：hallucination 类（不存在的球员 / 超范围时间 / 无法回答的问题）考察「如实声明 vs 编造」。
9. **中间步骤追溯**：三系统 trace 归一为统一 IR（llm/tool/stage/load + 启发式 phase），HTML 报告内含三轨对齐时间线与上下文演变曲线；**阶段为启发式标注，仅供定性比较**。
10. **Token 观测**：`agents/llm_client.py` 增加 usage 捕获钩子（`take_usage_events()`），非流式调用记录真实 token 用量。
11. **三系统统一输出纪律（公平参评）**：bare/single/multi 三系统提示词统一「形势/战术判断优先，默认不堆砌数字，用户索要具体数值时才引用」；数值单位统一米制（距离米、速度米/秒、高度米），避免单位口径导致的系统性偏袒。
12. **语义对齐（混合软校验）**：accuracy/tactical_accuracy 以 Judge 语义比对为主（数量级/方向/相对关系相符即可，不逐字扫描数字），程序化数值/胜者/类别扫描降级为诊断（metrics.json `details` 保留，不计分），两套并行便于对照。

## 扩展

- **新指标**：`eval/metrics.py` 添加纯函数 → `eval/scoring.py` 融合 → `eval/config.py` 权重（SCORED_METRICS）；
- **新系统**：`eval/runners.py` 实现 `run_xxx(case, out_dir, cfg) -> dict` 并注册进 `RUNNERS`；
- **新用例**：`eval/cases/*.json` 增加文件（数值金标准用 `auto:` 表达式，`cases lint` 自动校验数据文件与表达式可解析性）。

## 已知边界

- 流式调用（Web SSE）不计入 usage（评测三路径均为非流式）；
- 多智能体输出会写 `Output/`（反事实轨迹等副产物，属系统真实行为）；
- Judge 评分为模型主观判断，跨事件比较需固定 Judge 模型与 rubric 版本（`RUBRIC_VERSION` 变更会使缓存失效）。

## 暂缓方向（已评估，ROI 不足未实施）

以下方向经评估对本项目实际收益有限，未列入实施计划（完整评估见 git 历史中的优化方案文档）：

- **MAMCTS 多智能体博弈反事实** — 需行为模型训练数据与 PPO/ISMCTS 工程投入；现有启发式 MCTS 已在反事实类用例大幅领先（multi 0.874 vs bare 0.577）
- **双 Judge 制** — 需第二家付费 API；现有单 Judge（deepseek-v4-pro）+ 盲评 + 位置互换 + temp=0 已控制偏差
- **插件化架构 / 工具市场 / 多数据源 / 动态模型路由 / Docker 部署** — 无第三方生态与部署场景，属推测性工程

## 相关文档

- 被测系统架构：[project_report.md](project_report.md) · [harness.md](harness.md)
