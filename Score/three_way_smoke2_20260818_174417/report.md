# 评测报告：单 LLM 直接调用 vs 多智能体系统

- 事件目录: `D:\GHY\2.Files\football-agent\Score\three_way_smoke2_20260818_174417`
- 用例数: 11，重复: 1，采样温度: 0.85
- 模型: deepseek-v4-pro；git: `b27434b6`
- 权重: {'accuracy': 0.15, 'grounding': 0.1, 'completeness': 0.1, 'reasoning': 0.2, 'efficiency': 0.1, 'robustness': 0.1, 'tactical_accuracy': 0.25}
- Judge: {'model': 'deepseek-v4-pro', 'sample_repeats': 2, 'order_swap': True, 'temperature': 0.0}
- 【同源风险披露】Judge 模型: deepseek-v4-pro；rubric 版本: v3；被测系统模型: (未配置，默认 deepseek-v4-flash)；Judge 与被测系统非同厂商（模型名前缀粗判）。同源 Judge 存在自我偏好风险，结论需人工抽检。

## 1. 总分

| 系统 | 共同总分 | 95% CI | 扩展总分(+工具) | n |
|---|---|---|---|---|
| 单次LLM裸调用 | 0.739 | [0.639, 0.826] | None | 11 |
| 多智能体系统 | 0.801 | [0.734, 0.858] | 0.816 | 11 |
| 单智能体+工具 | 0.832 | [0.772, 0.891] | 0.849 | 11 |

### 1b. 按业务分层（L1-L7）总分

| 层 | 用例数 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|---|
| L1 | 2 | 0.746 | 0.637 | 0.709 |
| L2 | 3 | 0.874 | 0.84 | 0.785 |
| L3 | 1 | 0.63 | 0.915 | 0.952 |
| L4 | 2 | 0.656 | 0.858 | 0.944 |
| L5 | 1 | 0.627 | 0.81 | 0.823 |
| L7 | 2 | 0.72 | 0.787 | 0.856 |

## 2. 分指标

| 指标 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| accuracy | 0.623 ± [0.446, 0.782] | 0.716 ± [0.559, 0.852] | 0.748 ± [0.625, 0.864] |
| grounding | 0.917 ± [0.822, 0.989] | 0.902 ± [0.809, 0.971] | 0.928 ± [0.892, 0.965] |
| completeness | 0.864 ± [0.779, 0.934] | 0.864 ± [0.773, 0.939] | 0.892 ± [0.822, 0.959] |
| reasoning | 0.618 ± [0.491, 0.746] | 0.645 ± [0.536, 0.746] | 0.718 ± [0.618, 0.818] |
| efficiency | 0.945 ± [0.881, 0.996] | 0.979 ± [0.942, 1.000] | 0.951 ± [0.913, 0.983] |
| robustness | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] |
| tactical_accuracy | 0.596 ± [0.423, 0.736] | 0.759 ± [0.691, 0.823] | 0.795 ± [0.705, 0.873] |

## 3. 配对显著性检验（按用例配对，Wilcoxon 主检验）

n=11（配对用例数），α=0.05，配对 t 检验在 80% power 下约可检出 |d|≥0.84 的效应；更小差异仅作描述性参考。

### 单次LLM裸调用 vs 多智能体系统

- 均值差: -0.0622，paired Cohen's d: -0.299，Cliff's δ: -0.124
- Wilcoxon p=0.37394（Holm 校正后 p=1.0），配对 t p=0.34471

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.47496 | 1.0 | -0.2 | -0.174 |
| grounding | 0.32461 | 1.0 | 0.106 | 0.14 |
| completeness | 0.72184 | 1.0 | 0.0 | -0.066 |
| reasoning | 0.72181 | 1.0 | -0.115 | -0.124 |
| efficiency | 0.13801 | 1.0 | -0.465 | -0.198 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.16825 | 1.0 | -0.494 | -0.314 |

### 单次LLM裸调用 vs 单智能体+工具

- 均值差: -0.0933，paired Cohen's d: -0.418，Cliff's δ: -0.397
- Wilcoxon p=0.286（Holm 校正后 p=1.0），配对 t p=0.19568

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.33225 | 1.0 | -0.299 | -0.289 |
| grounding | 0.86577 | 1.0 | -0.074 | 0.174 |
| completeness | 0.59353 | 1.0 | -0.132 | -0.207 |
| reasoning | 0.21272 | 1.0 | -0.354 | -0.289 |
| efficiency | 0.86577 | 1.0 | -0.057 | 0.132 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.09197 | 1.0 | -0.595 | -0.413 |

### 多智能体系统 vs 单智能体+工具

- 均值差: -0.031，paired Cohen's d: -0.378，Cliff's δ: -0.107
- Wilcoxon p=0.47691（Holm 校正后 p=1.0），配对 t p=0.23864

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.50018 | 1.0 | -0.205 | -0.008 |
| grounding | 0.46521 | 1.0 | -0.214 | 0.033 |
| completeness | 0.44524 | 1.0 | -0.256 | -0.083 |
| reasoning | 0.30458 | 1.0 | -0.318 | -0.256 |
| efficiency | 0.1763 | 1.0 | 0.434 | 0.347 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.39463 | 1.0 | -0.393 | -0.198 |

## 4. 类别 × 系统热力

| 类别 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| adversarial | 0.902 | 0.814 | 0.797 |
| comparison | 0.868 | 0.960 | 0.902 |
| counterfactual | 0.627 | 0.810 | 0.823 |
| focus | 0.858 | 0.826 | 0.966 |
| frame | 0.642 | 0.706 | 0.699 |
| general_qa | 0.850 | 0.568 | 0.719 |
| hallucination | 0.537 | 0.760 | 0.914 |
| style | 0.455 | 0.890 | 0.922 |
| team | 0.630 | 0.915 | 0.952 |
| timeline | 0.853 | 0.724 | 0.677 |
| verify | 0.902 | 0.835 | 0.777 |

## 5. 诊断指标

| 系统 | 稳定性(跨重复相似度) | 工具使用得分 | 工具调用数 |
|---|---|---|---|
| 单次LLM裸调用 | None | None | 0 |
| 多智能体系统 | None | 1.0 | 4 |
| 单智能体+工具 | None | 1.0 | 11 |

## 6. 失败模式（Judge 标注 top-3）

- **答非所问** ×5（样例: bare/cf_7s_pass10#r0, multi/cf_7s_pass10#r0, bare/frame_ball_first#r0）
- **数据不足未声明** ×2（样例: single/adv_ambiguous_multi_target#r0, multi/adv_ambiguous_multi_target#r0）
- **无依据夸大** ×2（样例: multi/frame_ball_first#r0, single/verify_7s_pos#r0）

Judge-程序分歧 top（详见 results.json judge_disagreements）:
- multi/qa_7s_ball_hmax#r0：程序 1.0 vs Judge 0.1（差 0.9）
- bare/cf_7s_pass10#r0：程序 1.0 vs Judge 0.4（差 0.6）
- bare/halluc_frame_600#r0：程序 1.0 vs Judge 0.4（差 0.6）
- multi/timeline_ball_height#r0：程序 0.0 vs Judge 0.6（差 0.6）
- single/timeline_ball_height#r0：程序 0.0 vs Judge 0.6（差 0.6）

## 6b. Judge 一致性（位置互换正逆序差）

阈值：正逆序差 > 0.15 的用例标 ⚠，建议人工抽检（仅告警不改分）。

| 用例 | 系统 | 正逆序差 | 告警 |
|---|---|---|---|
| adv_ambiguous_multi_target | 单次LLM裸调用 | 0.150 |  |
| adv_ambiguous_multi_target | 多智能体系统 | 0.225 | ⚠ |
| adv_ambiguous_multi_target | 单智能体+工具 | 0.225 | ⚠ |
| cf_7s_pass10 | 单次LLM裸调用 | 0.000 |  |
| cf_7s_pass10 | 多智能体系统 | 0.200 | ⚠ |
| cf_7s_pass10 | 单智能体+工具 | 0.050 |  |
| compare_6_7_active | 单次LLM裸调用 | 0.000 |  |
| compare_6_7_active | 多智能体系统 | 0.100 |  |
| compare_6_7_active | 单智能体+工具 | 0.000 |  |
| focus_player7 | 单次LLM裸调用 | 0.050 |  |
| focus_player7 | 多智能体系统 | 0.025 |  |
| focus_player7 | 单智能体+工具 | 0.000 |  |
| frame_ball_first | 单次LLM裸调用 | 0.050 |  |
| frame_ball_first | 多智能体系统 | 0.275 | ⚠ |
| frame_ball_first | 单智能体+工具 | 0.300 | ⚠ |
| halluc_frame_600 | 单次LLM裸调用 | 0.200 | ⚠ |
| halluc_frame_600 | 多智能体系统 | 0.000 |  |
| halluc_frame_600 | 单智能体+工具 | 0.000 |  |
| qa_7s_ball_hmax | 单次LLM裸调用 | 0.000 |  |
| qa_7s_ball_hmax | 多智能体系统 | 0.100 |  |
| qa_7s_ball_hmax | 单智能体+工具 | 0.100 |  |
| qa_player_zone_7 | 单次LLM裸调用 | 0.100 |  |
| qa_player_zone_7 | 多智能体系统 | 0.050 |  |
| qa_player_zone_7 | 单智能体+工具 | 0.050 |  |
| team_7s_threat | 单次LLM裸调用 | 0.000 |  |
| team_7s_threat | 多智能体系统 | 0.000 |  |
| team_7s_threat | 单智能体+工具 | 0.000 |  |
| timeline_ball_height | 单次LLM裸调用 | 0.000 |  |
| timeline_ball_height | 多智能体系统 | 0.200 | ⚠ |
| timeline_ball_height | 单智能体+工具 | 0.200 | ⚠ |
| verify_7s_pos | 单次LLM裸调用 | 0.000 |  |
| verify_7s_pos | 多智能体系统 | 0.100 |  |
| verify_7s_pos | 单智能体+工具 | 0.050 |  |

⚠ 一致性告警用例（5）: adv_ambiguous_multi_target, cf_7s_pass10, frame_ball_first, halluc_frame_600, timeline_ball_height

## 7. 成本

| 系统 | 时延 ms | LLM 调用 | 总 token |
|---|---|---|---|
| 单次LLM裸调用 | 58974.4 | 1.0 | 12279.8 |
| 多智能体系统 | 13265.6 | 1.3 | 8587.0 |
| 单智能体+工具 | 14267.0 | 2.8 | 11744.4 |

## 8. 方法与局限说明

- 共同总分 = Σ 权重×分项；效率按预标定常数归一化；缺失分量自动重归一化。
- 系统级 CI 为跨用例 bootstrap（B=2000）；显著性按用例配对（Wilcoxon），Judge 主观分指标有族内相关，Holm 校正仅作参考下界。
- Judge 模型（deepseek-v4-pro）与被测系统解耦，采用盲评+位置互换缓解偏好，结论需人工抽检。
- 数值金标准由 pandas 独立重算（不经被测系统工具层，单位米制）；反事实类因 MCTS 随机性仅用方向性金标准。
- 阶段对齐为启发式（llm 序号/工具属性推断），时间线仅供定性比较。