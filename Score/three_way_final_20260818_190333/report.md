# 评测报告：单 LLM 直接调用 vs 多智能体系统

- 事件目录: `D:\GHY\2.Files\football-agent\Score\three_way_final_20260818_190333`
- 用例数: 11，重复: 1，采样温度: 0.85
- 模型: deepseek-v4-pro；git: `b27434b6`
- 权重: {'accuracy': 0.15, 'grounding': 0.1, 'completeness': 0.1, 'reasoning': 0.2, 'efficiency': 0.1, 'robustness': 0.1, 'tactical_accuracy': 0.25}
- Judge: {'model': 'deepseek-v4-pro', 'sample_repeats': 2, 'order_swap': True, 'temperature': 0.0}
- 【同源风险披露】Judge 模型: deepseek-v4-pro；rubric 版本: v3；被测系统模型: (未配置，默认 deepseek-v4-flash)；Judge 与被测系统非同厂商（模型名前缀粗判）。同源 Judge 存在自我偏好风险，结论需人工抽检。

## 1. 总分

| 系统 | 共同总分 | 95% CI | 扩展总分(+工具) | n |
|---|---|---|---|---|
| 单次LLM裸调用 | 0.739 | [0.644, 0.823] | None | 11 |
| 多智能体系统 | 0.879 | [0.820, 0.921] | 0.816 | 11 |
| 单智能体+工具 | 0.860 | [0.821, 0.898] | 0.874 | 11 |

### 1b. 按业务分层（L1-L7）总分

| 层 | 用例数 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|---|
| L1 | 2 | 0.736 | 0.885 | 0.811 |
| L2 | 3 | 0.81 | 0.911 | 0.821 |
| L3 | 1 | 0.431 | 0.96 | 0.944 |
| L4 | 2 | 0.643 | 0.873 | 0.888 |
| L5 | 1 | 0.696 | 0.635 | 0.855 |
| L7 | 2 | 0.905 | 0.911 | 0.901 |

## 2. 分指标

| 指标 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| accuracy | 0.648 ± [0.468, 0.818] | 0.836 ± [0.696, 0.941] | 0.807 ± [0.691, 0.904] |
| grounding | 0.836 ± [0.707, 0.936] | 0.914 ± [0.860, 0.964] | 0.919 ± [0.877, 0.957] |
| completeness | 0.838 ± [0.720, 0.936] | 0.943 ± [0.893, 0.981] | 0.889 ± [0.793, 0.967] |
| reasoning | 0.600 ± [0.491, 0.718] | 0.818 ± [0.754, 0.873] | 0.754 ± [0.718, 0.791] |
| efficiency | 0.951 ± [0.893, 0.990] | 0.981 ± [0.945, 1.000] | 0.961 ± [0.920, 1.000] |
| robustness | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] |
| tactical_accuracy | 0.636 ± [0.491, 0.764] | 0.823 ± [0.736, 0.895] | 0.846 ± [0.795, 0.900] |

## 3. 配对显著性检验（按用例配对，Wilcoxon 主检验）

n=11（配对用例数），α=0.05，配对 t 检验在 80% power 下约可检出 |d|≥0.84 的效应；更小差异仅作描述性参考。

### 单次LLM裸调用 vs 多智能体系统

- 均值差: -0.1399，paired Cohen's d: -0.765，Cliff's δ: -0.636
- Wilcoxon p=0.01279（Holm 校正后 p=0.26859），配对 t p=0.02956

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.09249 | 1.0 | -0.524 | -0.347 |
| grounding | 0.23672 | 1.0 | -0.317 | -0.157 |
| completeness | 0.01776 | 0.3552 | -0.778 | -0.339 |
| reasoning | 0.01986 | 0.37734 | -1.021 | -0.645 |
| efficiency | 0.24886 | 1.0 | -0.265 | -0.289 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.0498 | 0.7968 | -0.619 | -0.471 |

### 单次LLM裸调用 vs 单智能体+工具

- 均值差: -0.1214，paired Cohen's d: -0.648，Cliff's δ: -0.504
- Wilcoxon p=0.06188（Holm 校正后 p=0.9282），配对 t p=0.05708

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.1763 | 1.0 | -0.507 | -0.273 |
| grounding | 0.4434 | 1.0 | -0.354 | -0.149 |
| completeness | 0.39802 | 1.0 | -0.189 | -0.198 |
| reasoning | 0.03629 | 0.64242 | -0.785 | -0.521 |
| efficiency | 0.46307 | 1.0 | -0.11 | -0.124 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.03569 | 0.64242 | -0.711 | -0.512 |

### 多智能体系统 vs 单智能体+工具

- 均值差: 0.0184，paired Cohen's d: 0.191，Cliff's δ: 0.231
- Wilcoxon p=0.21322（Holm 校正后 p=1.0），配对 t p=0.54099

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.51319 | 1.0 | 0.139 | 0.132 |
| grounding | 0.91628 | 1.0 | -0.056 | 0.017 |
| completeness | 0.34545 | 1.0 | 0.315 | 0.157 |
| reasoning | 0.20139 | 1.0 | 0.467 | 0.397 |
| efficiency | 0.715 | 1.0 | 0.199 | 0.091 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.67414 | 1.0 | -0.198 | -0.025 |

## 4. 类别 × 系统热力

| 类别 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| adversarial | 0.966 | 0.909 | 0.913 |
| comparison | 0.880 | 0.981 | 0.949 |
| counterfactual | 0.696 | 0.635 | 0.855 |
| focus | 0.791 | 0.838 | 0.874 |
| frame | 0.642 | 0.860 | 0.840 |
| general_qa | 0.830 | 0.910 | 0.783 |
| hallucination | 0.845 | 0.913 | 0.889 |
| style | 0.495 | 0.907 | 0.902 |
| team | 0.431 | 0.960 | 0.944 |
| timeline | 0.728 | 0.855 | 0.720 |
| verify | 0.823 | 0.896 | 0.794 |

## 5. 诊断指标

| 系统 | 稳定性(跨重复相似度) | 工具使用得分 | 工具调用数 |
|---|---|---|---|
| 单次LLM裸调用 | None | None | 0 |
| 多智能体系统 | None | 1.0 | 3 |
| 单智能体+工具 | None | 1.0 | 11 |

## 6. 失败模式（Judge 标注 top-3）

- **无依据夸大** ×4（样例: multi/compare_6_7_active#r0, single/focus_player7#r0, single/qa_player_zone_7#r0）
- **逻辑矛盾** ×3（样例: multi/cf_7s_pass10#r0, bare/frame_ball_first#r0, bare/qa_player_zone_7#r0）
- **数据不足未声明** ×2（样例: bare/timeline_ball_height#r0, single/timeline_ball_height#r0）

Judge-程序分歧 top（详见 results.json judge_disagreements）:
- multi/cf_7s_pass10#r0：程序 1.0 vs Judge 0.3（差 0.7）
- bare/timeline_ball_height#r0：程序 0.0 vs Judge 0.4（差 0.4）
- single/timeline_ball_height#r0：程序 0.0 vs Judge 0.4（差 0.4）

## 6b. Judge 一致性（位置互换正逆序差）

阈值：正逆序差 > 0.15 的用例标 ⚠，建议人工抽检（仅告警不改分）。

| 用例 | 系统 | 正逆序差 | 告警 |
|---|---|---|---|
| adv_ambiguous_multi_target | 单次LLM裸调用 | 0.000 |  |
| adv_ambiguous_multi_target | 多智能体系统 | 0.000 |  |
| adv_ambiguous_multi_target | 单智能体+工具 | 0.000 |  |
| cf_7s_pass10 | 单次LLM裸调用 | 0.050 |  |
| cf_7s_pass10 | 多智能体系统 | 0.100 |  |
| cf_7s_pass10 | 单智能体+工具 | 0.050 |  |
| compare_6_7_active | 单次LLM裸调用 | 0.100 |  |
| compare_6_7_active | 多智能体系统 | 0.025 |  |
| compare_6_7_active | 单智能体+工具 | 0.050 |  |
| focus_player7 | 单次LLM裸调用 | 0.075 |  |
| focus_player7 | 多智能体系统 | 0.150 |  |
| focus_player7 | 单智能体+工具 | 0.025 |  |
| frame_ball_first | 单次LLM裸调用 | 0.050 |  |
| frame_ball_first | 多智能体系统 | 0.150 |  |
| frame_ball_first | 单智能体+工具 | 0.200 | ⚠ |
| halluc_frame_600 | 单次LLM裸调用 | 0.000 |  |
| halluc_frame_600 | 多智能体系统 | 0.000 |  |
| halluc_frame_600 | 单智能体+工具 | 0.000 |  |
| qa_7s_ball_hmax | 单次LLM裸调用 | 0.000 |  |
| qa_7s_ball_hmax | 多智能体系统 | 0.000 |  |
| qa_7s_ball_hmax | 单智能体+工具 | 0.000 |  |
| qa_player_zone_7 | 单次LLM裸调用 | 0.000 |  |
| qa_player_zone_7 | 多智能体系统 | 0.000 |  |
| qa_player_zone_7 | 单智能体+工具 | 0.050 |  |
| team_7s_threat | 单次LLM裸调用 | 0.000 |  |
| team_7s_threat | 多智能体系统 | 0.000 |  |
| team_7s_threat | 单智能体+工具 | 0.050 |  |
| timeline_ball_height | 单次LLM裸调用 | 0.000 |  |
| timeline_ball_height | 多智能体系统 | 0.025 |  |
| timeline_ball_height | 单智能体+工具 | 0.100 |  |
| verify_7s_pos | 单次LLM裸调用 | 0.050 |  |
| verify_7s_pos | 多智能体系统 | 0.100 |  |
| verify_7s_pos | 单智能体+工具 | 0.150 |  |

⚠ 一致性告警用例（1）: frame_ball_first

## 7. 成本

| 系统 | 时延 ms | LLM 调用 | 总 token |
|---|---|---|---|
| 单次LLM裸调用 | 59283.7 | 1.0 | 11917.9 |
| 多智能体系统 | 20531.3 | 1.3 | 9500.3 |
| 单智能体+工具 | 16492.9 | 2.7 | 11621.6 |

## 8. 方法与局限说明

- 共同总分 = Σ 权重×分项；效率按预标定常数归一化；缺失分量自动重归一化。
- 系统级 CI 为跨用例 bootstrap（B=2000）；显著性按用例配对（Wilcoxon），Judge 主观分指标有族内相关，Holm 校正仅作参考下界。
- Judge 模型（deepseek-v4-pro）与被测系统解耦，采用盲评+位置互换缓解偏好，结论需人工抽检。
- 数值金标准由 pandas 独立重算（不经被测系统工具层，单位米制）；反事实类因 MCTS 随机性仅用方向性金标准。
- 阶段对齐为启发式（llm 序号/工具属性推断），时间线仅供定性比较。