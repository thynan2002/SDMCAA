# 工具调用（tool_calls）驱动架构重构设计 🛠️

<p align="center">
  <img src="https://img.shields.io/badge/status-completed-2EA043?style=flat-square" alt="Status: completed">
  <img src="https://img.shields.io/badge/tool-sites-19-4B8BBE?style=flat-square" alt="Tool sites: 19">
  <img src="https://img.shields.io/badge/golden-39%20calls-8B5CF6?style=flat-square" alt="Golden: 39 calls">
</p>

> **目标**：将足球战术分析多智能体系统从「纯文本 JSON 契约」迁移到「OpenAI 风格 function calling」：tools 参数 + tool_calls 解析 + 多轮工具循环 + 并行 tool_calls，同时保持 harness golden 回放与全部 pytest 用例不回归。

## 一、现状盘点

- 19 处 `call_llm` 调用点（不含定义）：
  - **结构化决策输出（改工具契约）**：`session/router.py`（意图路由 JSON）、`llm_brain.py` 的 `generate_script` / `decide` / `generate_strategy_table` / `SemanticTierBatcher._resolve_batch`（4 处 JSON）、`data_verifier.py` 的 `_judge_challenge`（质疑判断 JSON）。共 7 处。
  - **纯文本生成（保持文本）**：解说稿（player/agents.py）、报告 ×3（report_generator.py）、风格标签/概括（style_modeler.py、data_collector.py）、核验报告 ×3 与质疑回复（data_verifier.py）、反事实分析报告（counterfactual_engine.py）、综合问答正文（general_qa.py）、记忆摘要（memory.py）。
- 关键约束（来自现有测试与 golden）：
  1. `tests/_common.py` 的 mock 以 `(system_prompt, user_message, config, max_tokens, retries)` 5 参签名替换 `llm_brain.call_llm` —— llm_brain 内部调用点的调用签名**不可变更**。
  2. `tests/test_harness_equivalence.py` 以 6 参签名替换 `agents.llm_client._call_llm_impl` 与 `harness.interceptors._call_llm_impl` —— 传输层被 patch 的函数签名**不可变更**。
  3. dispatcher 探针断言收到且仅收到原 6 个参数（不传 tools 时）。
  4. golden 回放以 `(system_prompt, user_message)` 精确匹配、FIFO 消费；现有 `harness/golden/standard` 的 39 条记录全部为 `response: null`（录制时 LLM 不可用），回放验证的是**全兜底链路**。迁移后各调用点的首轮 `(system, user)` 必须保持不变，旧 golden 才能继续回放。

## 二、技术选型与理由

| 候选 | 结论 | 理由 |
|------|------|------|
| langchain / langgraph 工具链（`create_react_agent` / `bind_tools` / `ToolNode`） | **不用于 LLM 调用层** | LangGraph 代理运行时会绕开 `call_llm` 分派器钩子与流式回调 ContextVar，直接破坏 harness 的观测/录制/回放等价性（硬约束 5）。LangGraph 继续承担其现有职责——两个 Orchestrator 的流水线编排（`StateGraph`），不改变。 |
| openai 官方 SDK | **不引入** | 未安装；替换传输层意味着重写 `_call_llm_impl`（测试 patch 点）与流式 SSE 解析语义，引入不可控差异且无 golden 收益。现有 requests 传输层本就是 OpenAI Chat Completions 兼容协议，`tools` 仅是 payload 字段扩展，增量实现成本最低、等价性风险最小。 |
| pydantic（v2，经 langgraph 依赖已存在，2.10.3） | **采用** | 用于定义工具参数模型并导出 JSON Schema（`model_json_schema()`），执行时做参数校验/类型强制。 |
| MCP 框架 | **不引入** | 本项目的工具全部是进程内确定性函数，MCP 的跨进程协议层只会增加集成成本与等价性风险，收益不明确。 |
| 自研最小工具循环 | **采用（受约束的必然选择）** | 任务要求 harness 拦截器钩子继续生效、`call_llm` 签名与流式语义保留、旧 golden 可回放——任何重型框架都无法绕过这些钩子。循环本身约百行，全部行为有单测覆盖。 |

## 三、总体架构

```
调用方（router / llm_brain / verifier / general_qa …）
   │  call_llm(system, user, [tools=...], [stream=...])     ← 签名向后兼容
   ▼
agents/llm_client.py
   ├─ 无 tools 且无绑定 → 原路径（dispatcher/impl，逐字节等价）
   └─ 有 tools（显式参数 或 bind_prompt_tools 绑定）
        ▼  _call_llm_with_tools：多轮工具循环
        每轮：设置 _tool_exchange 上下文变量（tools + 当前 messages）
              → 经 dispatcher（harness 观测/录制/回放）或直调 _call_llm_impl
              → 传输层读取 exchange：payload 加 tools、解析 tool_calls 回填 exchange
        tool_calls → agents/tools 注册表执行（一轮内多个并行调用一次性执行）
              → 终止型工具（结构化输出提交）：返回 json.dumps(结果) 给调用方
              → 数据型工具：结果以 role=tool 消息回填，进入下一轮
        无 tool_calls → 返回文本（含流式契约）
```

### 兼容性机要：`_tool_exchange` 上下文变量

tests 与 harness 均以**固定签名** patch 传输函数与 dispatcher。为不改变任何既有签名，tools 参数与多轮 messages 通过 ContextVar（`_tool_exchange`）传入传输层，解析出的 assistant 消息（含 tool_calls）经同一通道回传。该模式与现有 `_stream_callback` ContextVar 完全同构。由此：

- 不传 tools 时，dispatcher 收到的参数与原 6 参逐一对应（现有断言通过）；
- 被 patch 的 mock 实现忽略 exchange，返回纯文本 → 工具层按「无 tool_calls，文本回退」处理 → 与重构前解析路径一致；
- replay 模式下 ReplayLLM 返回录制文本（旧条目无 tool_calls）→ 同样走文本回退。

### 返回值契约（保持 `str | None`）

| 情形 | 返回 |
|------|------|
| 无 tools（且无绑定） | 与重构前完全一致 |
| 模型经**终止型工具**提交结构化结果 | `json.dumps(工具参数)` —— 调用方现有 JSON 解析器直接复用 |
| 同一响应多个并行终止型工具 | 各结果字典合并后 `json.dumps` |
| 数据型工具循环后的最终文本回答 | 文本（流式调用同时推送增量 chunk） |
| 模型返回纯文本（mock / 旧 golden / 未用工具） | 原文返回 → 调用方文本回退解析 |
| 失败（网络/超时/格式非法/循环超限） | `None` → 调用方规则兜底（router 关键词、llm_brain 启发式、verifier 关键词），持续失败时各调用点已如实上报「模型调用失败，并非数据不足」 |

### 失败降级链

| 失败类型 | 处理 |
|----------|------|
| 网络/超时（瞬时） | 传输层按现有 `retries` 重试（不变） |
| 输出非法（既无 tool_calls 也非可解析文本） | 返回原文/None → 调用方既有规则兜底 |
| 工具参数非法 / 执行异常 | 以 `role=tool` 错误结果回填模型（OpenAI 惯例），受 `max_rounds` 限制；仍失败 → None → 兜底 |
| 数据不足 | 数据工具返回 `{"status": "数据不足", ...}`，由模型按「忠实于数据」提示词纪律如实声明 |

不使用占位内容掩盖失败；不改动任何现有兜底路径。

### 并行 tool_calls

单个 assistant 消息中的多个 tool_calls **全部执行完毕后再回填消息**（一次性）。`SemanticTierBatcher` 的「入队→批量提交」并入该机制：批量推断改用终止型工具 `submit_semantic_tiers`（tool_choice=required）提交，模型也可在同一响应内并行发起多个提交调用（自动合并）；入队/缓存/分批仍是批量请求的组织层，不再是独立的「文本 JSON 推断实现」——重复的文本契约解析路径被工具契约取代（保留为回退）。

### 禁止凭空编造

新增 `agents/tools/`：

- `base.py`：`ToolSpec` / `ToolRegistry` / 失败分类（参数非法、执行异常、数据不足）；
- `schemas.py`：pydantic 决策工具模型（submit_intent / submit_decision / submit_script / submit_strategy / submit_challenge_judgment / submit_semantic_tiers）；
- `football.py`：确定性分析能力封装为**数据工具**，绑定给 general_qa / data_verifier（tool_choice=auto，首轮 payload 不变，模型需要更多信息时**只能**通过工具获取）：`get_tactical_facts`（战术事实提取）、`get_ball_timeline`（球路分析）、`get_frame_snapshot`（帧级核验）、`get_player_raw_data`（球员轨迹核验）、`get_player_profile`（画像/风格模型查询）、`run_counterfactual_simulation`（反事实模拟+轨迹导出，仅在用户明确要求时由模型调用）。
  工具经 `set_active_corpus`（SessionManager.load_data 注入）读取当前语料；语料缺失时返回「数据不足」，不得编造。

## 四、改动文件清单

| 文件 | 改动 | 阶段 |
|------|------|------|
| `agents/llm_client.py` | +tools/tool_choice 参数、`_tool_exchange`、`_call_llm_with_tools` 循环、`bind_prompt_tools`、`_call_llm_impl`/`_call_llm_streaming` 读 exchange（无 exchange 时逐字节等价） | 1 |
| `agents/tools/__init__.py` `base.py` | 注册表骨架（新增） | 1 |
| `harness/interceptors.py` | 录制条目按需追加 `tools`/`assistant_message` 字段（旧格式不受影响）；trace 追加工具属性 | 1 |
| `harness/mock.py` | ReplayLLM：条目含 `assistant_message` 时回填 exchange（旧条目行为不变） | 1 |
| `tests/test_tool_calling.py` | 新增单测（循环/并行/失败分类/回退/回放兼容/流式） | 1 |
| `agents/tools/schemas.py` `football.py` | 决策 schema + 数据工具（新增） | 2 |
| `agents/session/router.py` | 绑定 submit_intent（试点；解析与兜底逻辑不变） | 2 |
| `agents/professional/simulation/llm_brain.py` | 4 个提示词绑定对应提交工具（调用点签名不变，解析/重试/兜底不变） | 3 |
| `agents/professional/agents/data_verifier.py` | 质疑判断绑定 submit_challenge_judgment；核验提示词绑定数据工具（auto） | 3 |
| `agents/professional/agents/general_qa.py` | SYSTEM_PROMPT 绑定数据工具（auto，首轮 payload 不变） | 3 |
| `agents/session/manager.py` `main.py` | load_data / run_once 注入 active corpus（各 1 行） | 3 |
| `docs/harness.md` | 等价性论证更新（工具层 + 一键重录命令） | 收尾 |

## 五、逐智能体迁移方案

| 调用点 | 类型 | 方案 | 回退链 |
|--------|------|------|--------|
| router.parse | 结构化 | 绑定 `submit_intent`（required）；tool 参数→`_build_query_from_json` 复用 | 文本 JSON→同解析；失败→关键词规则→GENERAL_QA |
| llm_brain.generate_script | 结构化 | 绑定 `submit_script`（required） | 文本→`_parse_json`；失败→空剧本 |
| llm_brain.decide | 结构化 | 绑定 `submit_decision`（required） | 文本→`_parse_json`+`_validate_decision`；非法重试1次→None→启发式 |
| llm_brain.generate_strategy_table | 结构化 | 绑定 `submit_strategy`（required） | 文本→`_parse_json`；失败→None→公式权重 |
| SemanticTierBatcher.flush | 结构化 | 绑定 `submit_semantic_tiers`（required） | 文本→`_parse_semantic_tier_result`；失败→阈值档位 |
| data_verifier._judge_challenge | 结构化 | 绑定 `submit_challenge_judgment`（required） | 文本 JSON→同解析；失败→关键词判断 |
| general_qa.answer | 文本+数据工具 | 绑定数据工具（auto），正文仍为文本流式 | 工具不可用→仅用内嵌事实；LLM 失败→「模型调用失败」文案 |
| data_verifier 核验/质疑回复 ×3 | 文本(+数据工具) | 帧级/球员核验绑定数据工具（auto） | 原样保留 |
| 其余 8 处（解说/报告/摘要/风格标签） | 纯文本 | 不迁移 | 不变 |

## 六、golden 兼容策略

1. **旧 golden 直接回放**（主策略）：迁移只增加 tools 绑定，不改任何 `(system_prompt, user_message)` 首轮内容；replay 返回的录制文本/None 走文本回退路径，与重构前行为逐字节一致。现有 `harness/golden/standard`（39 条 response=null 全兜底链路）无需重录即可继续 PASS。
2. **新格式向后兼容**：record 模式仅在出现 tool_calls 时追录 `tools` / `assistant_message` 字段（format_version=1 不变，字段可选）；ReplayLLM 对无该字段的旧条目行为不变。
3. **一键重录**（可选）：`python -m harness run TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --script harness/scripts/standard_12s.txt --mode record --golden-dir harness/golden/standard --seed 42`（真实 API）。

## 七、验收对照

- `call_llm` 支持 tools 参数与 tool_calls 循环；不传 tools 时行为与重构前完全一致（现有 dispatcher 探针用例 + A/B 等价用例直接证明）。
- ≥3 个智能体完成工具化：router、llm_brain.decide、data_verifier（质疑判断）+ general_qa（数据工具）。
- 同一响应并行 tool_calls：循环支持 + 单测证明。
- pytest 全部用例 + golden 四维回放 PASS。

---

# 执行结果（重构总结）

## 1. 选型理由（最终落地）

| 选型 | 结论 |
|------|------|
| 保留手写 requests 传输层，扩展 tools/tool_calls | 测试与 harness 以固定签名 patch `_call_llm_impl`/dispatcher（`tools` 经 `tool_exchange_var` ContextVar 通道传递，不改变任何既有签名）；OpenAI 兼容协议下 tools 仅是 payload 扩展，零新依赖 |
| langgraph 工具链 / openai SDK / MCP | 不引入：会绕开 call_llm 分派器钩子与流式回调通道，破坏 harness 等价性；LangGraph 维持其流水线编排职责 |
| pydantic v2（随 langgraph 依赖已存在） | 定义决策工具参数模型并导出 JSON Schema；校验保持宽容（不设 Literal 枚举），越界值由下游既有容忍解析与规则兜底处理 |
| 自研最小工具循环（受约束的必然选择） | 约 150 行，全部行为有单测覆盖；保证 harness 钩子、流式语义、旧 golden 回放完全兼容 |

## 2. 改动文件（全部提交）

| 文件 | 阶段 | 内容 |
|------|------|------|
| `agents/llm_client.py` | 1/3 | tools/tool_choice/max_rounds 参数、`_call_llm_with_tools` 循环、`bind_prompt_tools`、`tool_exchange_var` 通道、流式 tool_calls 增量累积、tool_choice 400 降级重试 |
| `agents/tools/__init__.py` `base.py` | 1 | ToolSpec/ToolRegistry/失败分类（invalid_arguments/execution_error/insufficient_data/unknown_tool） |
| `harness/interceptors.py` | 1 | 录制条目按需追加 tool_round/tools/tool_choice/assistant_message（旧格式不变）；trace 追加工具属性 |
| `harness/mock.py` | 1 | ReplayLLM 回放 assistant_message 驱动工具循环；format_version=2（读取端不按版本分支） |
| `tests/test_tool_calling.py` | 1/收尾 | 21 个用例（循环/并行/失败分类/绑定/回放/流式/record→replay 工具闭环） |
| `agents/tools/schemas.py` | 2 | 6 个终止型决策工具（submit_intent/decision/script/strategy/challenge_judgment/semantic_tiers） |
| `agents/tools/football.py` | 2 | 6 个数据工具（战术事实/球路时间线/帧快照/球员原始数据/画像/反事实模拟）+ active corpus 注入 |
| `agents/session/router.py` | 2 | 绑定 submit_intent（试点） |
| `agents/professional/simulation/llm_brain.py` | 3 | 绑定 submit_script/decision/strategy/semantic_tiers；修复档位缓存键映射缺陷 |
| `agents/professional/agents/data_verifier.py` | 3 | 绑定 submit_challenge_judgment |
| `agents/professional/agents/general_qa.py` | 3 | 绑定 6 个数据工具（auto），正文保持文本流式 |
| `agents/session/manager.py` `main.py` | 3 | load_data/run_once 注入 active corpus（各 1 行） |
| `docs/harness.md` | 收尾 | 等价性论证扩展（工具层） |

## 3. 迁移的调用点（19 处）

| 调用点 | 迁移方式 |
|--------|----------|
| `session/router.py` 意图路由 | ✅ 绑定 submit_intent（终止型） |
| `llm_brain.generate_script` | ✅ 绑定 submit_script |
| `llm_brain.decide` | ✅ 绑定 submit_decision |
| `llm_brain.generate_strategy_table` | ✅ 绑定 submit_strategy |
| `llm_brain.SemanticTierBatcher._resolve_batch` | ✅ 绑定 submit_semantic_tiers |
| `data_verifier._judge_challenge` | ✅ 绑定 submit_challenge_judgment |
| `general_qa.answer` 正文 | ✅ 数据工具（auto）+ 文本流式（正文仍为文本生成） |
| 其余 12 处（解说/报告/摘要/风格标签/核验回复/反事实分析） | 纯文本生成，保持文本（按任务定义不迁移） |

## 4. golden 兼容策略（落地）

- **旧 golden 直接回放（主策略，零重录）**：迁移仅增加工具绑定，`(system_prompt, user_message)` 全部不变；`harness/golden/standard`（39 条 response=null 的全兜底链路）无需重录，四维比对持续 PASS。
- **新格式向后兼容**：record 仅在出现工具交互时追录可选字段；ReplayLLM 对无该字段的旧条目行为不变；record→replay 工具闭环有专门单测覆盖。
- **一键重录**：`python -m harness run ... --mode record --golden-dir harness/golden/standard --seed 42`（已写入手册 docs/harness.md）。

## 5. 已知边界

- **实网输出文本差异**：绑定数据工具后，模型可能主动调用工具获取额外数据，回答文本与纯内嵌事实时不同（语义一致、不编造，属工具化预期效果）；回放模式完全确定性。
- **tool_choice 兼容**：DeepSeek 推理模式拒绝 `tool_choice=required`（400 Thinking mode），绑定统一使用 auto，传输层对 400 自动降级重试。
- **档位缓存键修复**：迁移中发现并修复既有缺陷（`req_N` 未映射回缓存键导致 LLM 档位描述实网从未生效）；回放路径（全部 None）不受影响。
- **Output/ 残留文件**：golden 文件清单比对以 Output/ 实际文件为准，运行过其他数据集后会留下 `{prefix}_cf_*` 残留（gitignore 产物），影响文件维度比对；测试已做清理，人工运行 verify 前建议清空 Output/。
- **反事实模拟工具耗时**：`run_counterfactual_simulation` 为启发式推演（不触发 LLM 递归），耗时数秒，仅当用户明确描述反事实场景时模型才应调用。

## 6. 审查修复记录（2026-08-06 代码审查后）

| 级别 | 问题 | 修复 |
|------|------|------|
| P1 | 多轮工具循环第 2 轮起丢失 system/user 消息（`messages` 初始为空，工具回填后上下文残缺） | `_call_llm_with_tools` 的 `messages` 全量初始化为 `[system, user]`，每轮完整透传；同步修正 3 个受影响单测的索引断言（改为按 role 定位）并新增消息序列断言 |
| P2 | `get_player_raw_data` 的「所属队伍」输出颜色码（A/B/C）与全局中文队名口径不一致 | 映射为中文队名 |
| P2 | `get_player_profile` 的「战术角色」为占位文案「由数据推断」 | 改用 `classify_position`/`classify_side` 确定性推断位置角色 |
| P3 | 新测试文件 2 个未用导入（pytest、register_tool） | 移除 |

修复后复验：pytest 全部通过、golden 四维回放 PASS、实网冒烟（剧本/决策/档位/质疑判断/general_qa 工具循环）全部正确。

## 相关文档

- 等价性论证与回放回归：[harness.md](harness.md)
- 项目全貌：[project_report.md](project_report.md)
