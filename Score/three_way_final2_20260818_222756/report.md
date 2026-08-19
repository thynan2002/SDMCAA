# 评测报告：单 LLM 直接调用 vs 多智能体系统

- 事件目录: `D:\GHY\2.Files\football-agent\Score\three_way_final2_20260818_222756`
- 用例数: 11，重复: 1，采样温度: 0.85
- 模型: deepseek-v4-pro；git: `b27434b6`
- 权重: {'accuracy': 0.15, 'grounding': 0.1, 'completeness': 0.1, 'reasoning': 0.2, 'efficiency': 0.1, 'robustness': 0.1, 'tactical_accuracy': 0.25}
- Judge: {'model': 'deepseek-v4-pro', 'sample_repeats': 2, 'order_swap': True, 'temperature': 0.0}
- 【同源风险披露】Judge 模型: deepseek-v4-pro；rubric 版本: v3；被测系统模型: (未配置，默认 deepseek-v4-flash)；Judge 与被测系统同厂商（模型名前缀粗判: deepseek）。同源 Judge 存在自我偏好风险，结论需人工抽检。

## 1. 总分

| 系统 | 共同总分 | 95% CI | 扩展总分(+工具) | n |
|---|---|---|---|---|
| 单次LLM裸调用 | 0.718 | [0.622, 0.805] | None | 11 |
| 多智能体系统 | 0.893 | [0.855, 0.929] | 0.917 | 11 |
| 单智能体+工具 | 0.804 | [0.758, 0.852] | 0.823 | 11 |

### 1b. 按业务分层（L1-L7）总分

| 层 | 用例数 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|---|
| L1 | 2 | 0.704 | 0.835 | 0.734 |
| L2 | 3 | 0.856 | 0.922 | 0.772 |
| L3 | 1 | 0.41 | 0.966 | 0.952 |
| L4 | 2 | 0.615 | 0.853 | 0.83 |
| L5 | 1 | 0.749 | 0.845 | 0.771 |
| L7 | 2 | 0.768 | 0.933 | 0.837 |

## 2. 分指标

| 指标 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| accuracy | 0.616 ± [0.436, 0.786] | 0.850 ± [0.766, 0.929] | 0.732 ± [0.639, 0.830] |
| grounding | 0.902 ± [0.809, 0.966] | 0.920 ± [0.882, 0.959] | 0.900 ± [0.836, 0.955] |
| completeness | 0.827 ± [0.689, 0.931] | 0.948 ± [0.907, 0.984] | 0.833 ± [0.697, 0.941] |
| reasoning | 0.564 ± [0.446, 0.682] | 0.864 ± [0.809, 0.918] | 0.682 ± [0.600, 0.746] |
| efficiency | 0.986 ± [0.962, 1.000] | 0.954 ± [0.904, 1.000] | 0.955 ± [0.912, 0.993] |
| robustness | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] |
| tactical_accuracy | 0.566 ± [0.393, 0.718] | 0.841 ± [0.777, 0.902] | 0.754 ± [0.695, 0.818] |

## 3. 配对显著性检验（按用例配对，Wilcoxon 主检验）

n=11（配对用例数），α=0.05，配对 t 检验在 80% power 下约可检出 |d|≥0.84 的效应；更小差异仅作描述性参考。

### 单次LLM裸调用 vs 多智能体系统

- 均值差: -0.1745，paired Cohen's d: -0.983，Cliff's δ: -0.678
- Wilcoxon p=0.00762（Holm 校正后 p=0.14478），配对 t p=0.00855

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.02484 | 0.3726 | -0.724 | -0.446 |
| grounding | 0.58977 | 1.0 | -0.14 | 0.05 |
| completeness | 0.05041 | 0.65533 | -0.544 | -0.421 |
| reasoning | 0.00489 | 0.10269 | -1.732 | -0.785 |
| efficiency | 0.14413 | 1.0 | 0.376 | 0.14 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.00784 | 0.14478 | -0.882 | -0.595 |

### 单次LLM裸调用 vs 单智能体+工具

- 均值差: -0.0854，paired Cohen's d: -0.377，Cliff's δ: -0.264
- Wilcoxon p=0.286（Holm 校正后 p=1.0），配对 t p=0.2396

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.44936 | 1.0 | -0.263 | -0.207 |
| grounding | 0.91855 | 1.0 | 0.015 | 0.066 |
| completeness | 0.65648 | 1.0 | -0.016 | -0.099 |
| reasoning | 0.20086 | 1.0 | -0.43 | -0.43 |
| efficiency | 0.22492 | 1.0 | 0.373 | 0.215 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.13916 | 1.0 | -0.524 | -0.281 |

### 多智能体系统 vs 单智能体+工具

- 均值差: 0.0891，paired Cohen's d: 1.209，Cliff's δ: 0.587
- Wilcoxon p=0.00585（Holm 校正后 p=0.117），配对 t p=0.00248

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.1221 | 1.0 | 0.642 | 0.405 |
| grounding | 0.36214 | 1.0 | 0.267 | 0.05 |
| completeness | 0.02771 | 0.38794 | 0.513 | 0.372 |
| reasoning | 0.00907 | 0.15419 | 1.297 | 0.702 |
| efficiency | 0.89274 | 1.0 | -0.013 | 0.041 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.02071 | 0.33136 | 0.89 | 0.413 |

## 4. 类别 × 系统热力

| 类别 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| adversarial | 0.953 | 0.937 | 0.788 |
| comparison | 0.833 | 0.976 | 0.904 |
| counterfactual | 0.749 | 0.845 | 0.771 |
| focus | 0.714 | 0.843 | 0.869 |
| frame | 0.618 | 0.795 | 0.745 |
| general_qa | 0.790 | 0.875 | 0.724 |
| hallucination | 0.583 | 0.930 | 0.885 |
| style | 0.515 | 0.863 | 0.790 |
| team | 0.410 | 0.966 | 0.952 |
| timeline | 0.861 | 0.810 | 0.664 |
| verify | 0.875 | 0.980 | 0.748 |

## 5. 诊断指标

| 系统 | 稳定性(跨重复相似度) | 工具使用得分 | 工具调用数 |
|---|---|---|---|
| 单次LLM裸调用 | None | None | 0 |
| 多智能体系统 | None | 1.0 | 5 |
| 单智能体+工具 | None | 1.0 | 11 |

## 6. 失败模式（Judge 标注 top-3）

- **无依据夸大** ×6（样例: multi/frame_ball_first#r0, single/qa_7s_ball_hmax#r0, multi/qa_7s_ball_hmax#r0）
- **答非所问** ×2（样例: bare/frame_ball_first#r0, bare/team_7s_threat#r0）
- **逻辑矛盾** ×1（样例: single/adv_ambiguous_multi_target#r0）

Judge-程序分歧 top（详见 results.json judge_disagreements）:
- bare/halluc_frame_600#r0：程序 1.0 vs Judge 0.4（差 0.6）
- single/timeline_ball_height#r0：程序 0.0 vs Judge 0.45（差 0.45）
- single/qa_7s_ball_hmax#r0：程序 1.0 vs Judge 0.6（差 0.4）
- bare/cf_7s_pass10#r0：程序 1.0 vs Judge 0.625（差 0.375）

## 6b. Judge 一致性（位置互换正逆序差）

阈值：正逆序差 > 0.15 的用例标 ⚠，建议人工抽检（仅告警不改分）。

| 用例 | 系统 | 正逆序差 | 告警 |
|---|---|---|---|
| adv_ambiguous_multi_target | 单次LLM裸调用 | 0.025 |  |
| adv_ambiguous_multi_target | 多智能体系统 | 0.025 |  |
| adv_ambiguous_multi_target | 单智能体+工具 | 0.050 |  |
| cf_7s_pass10 | 单次LLM裸调用 | 0.125 |  |
| cf_7s_pass10 | 多智能体系统 | 0.075 |  |
| cf_7s_pass10 | 单智能体+工具 | 0.050 |  |
| compare_6_7_active | 单次LLM裸调用 | 0.050 |  |
| compare_6_7_active | 多智能体系统 | 0.025 |  |
| compare_6_7_active | 单智能体+工具 | 0.025 |  |
| focus_player7 | 单次LLM裸调用 | 0.025 |  |
| focus_player7 | 多智能体系统 | 0.050 |  |
| focus_player7 | 单智能体+工具 | 0.025 |  |
| frame_ball_first | 单次LLM裸调用 | 0.150 |  |
| frame_ball_first | 多智能体系统 | 0.300 | ⚠ |
| frame_ball_first | 单智能体+工具 | 0.300 | ⚠ |
| halluc_frame_600 | 单次LLM裸调用 | 0.000 |  |
| halluc_frame_600 | 多智能体系统 | 0.000 |  |
| halluc_frame_600 | 单智能体+工具 | 0.000 |  |
| qa_7s_ball_hmax | 单次LLM裸调用 | 0.000 |  |
| qa_7s_ball_hmax | 多智能体系统 | 0.000 |  |
| qa_7s_ball_hmax | 单智能体+工具 | 0.000 |  |
| qa_player_zone_7 | 单次LLM裸调用 | 0.000 |  |
| qa_player_zone_7 | 多智能体系统 | 0.050 |  |
| qa_player_zone_7 | 单智能体+工具 | 0.100 |  |
| team_7s_threat | 单次LLM裸调用 | 0.100 |  |
| team_7s_threat | 多智能体系统 | 0.000 |  |
| team_7s_threat | 单智能体+工具 | 0.000 |  |
| timeline_ball_height | 单次LLM裸调用 | 0.050 |  |
| timeline_ball_height | 多智能体系统 | 0.025 |  |
| timeline_ball_height | 单智能体+工具 | 0.050 |  |
| verify_7s_pos | 单次LLM裸调用 | 0.000 |  |
| verify_7s_pos | 多智能体系统 | 0.000 |  |
| verify_7s_pos | 单智能体+工具 | 0.000 |  |

⚠ 一致性告警用例（1）: frame_ball_first

## 7. 成本

| 系统 | 时延 ms | LLM 调用 | 总 token |
|---|---|---|---|
| 单次LLM裸调用 | 39562.6 | 1.0 | 10092.6 |
| 多智能体系统 | 18923.4 | 1.5 | 11920.9 |
| 单智能体+工具 | 15063.8 | 2.6 | 11466.1 |

## 8. 方法与局限说明

- 共同总分 = Σ 权重×分项；效率按预标定常数归一化；缺失分量自动重归一化。
- 系统级 CI 为跨用例 bootstrap（B=2000）；显著性按用例配对（Wilcoxon），Judge 主观分指标有族内相关，Holm 校正仅作参考下界。
- Judge 模型（deepseek-v4-pro）与被测系统解耦，采用盲评+位置互换缓解偏好，结论需人工抽检。
- 数值金标准由 pandas 独立重算（不经被测系统工具层，单位米制）；反事实类因 MCTS 随机性仅用方向性金标准。
- 阶段对齐为启发式（llm 序号/工具属性推断），时间线仅供定性比较。