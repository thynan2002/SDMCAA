# 评测报告：单 LLM 直接调用 vs 多智能体系统

- 事件目录: `D:\GHY\2.Files\football-agent\Score\three_way_full5_20260819_20260819_115158`
- 用例数: 40，重复: 5，采样温度: 0.85
- 模型: deepseek-v4-pro；git: `fc17934b`
- 权重: {'accuracy': 0.15, 'grounding': 0.1, 'completeness': 0.1, 'reasoning': 0.2, 'efficiency': 0.1, 'robustness': 0.1, 'tactical_accuracy': 0.25}
- Judge: {'model': 'deepseek-v4-pro', 'sample_repeats': 2, 'order_swap': True, 'temperature': 0.0}
- 【同源风险披露】Judge 模型: deepseek-v4-pro；rubric 版本: v3；被测系统模型: (未配置，默认 deepseek-v4-flash)；Judge 与被测系统同厂商（模型名前缀粗判: deepseek）。同源 Judge 存在自我偏好风险，结论需人工抽检。

## 1. 总分

| 系统 | 共同总分 | 95% CI | 扩展总分(+工具) | n |
|---|---|---|---|---|
| 单次LLM裸调用 | 0.763 | [0.729, 0.796] | None | 40 |
| 多智能体系统 | 0.835 | [0.791, 0.872] | 0.808 | 40 |
| 单智能体+工具 | 0.787 | [0.738, 0.831] | 0.808 | 40 |

### 1b. 按业务分层（L1-L7）总分

| 层 | 用例数 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|---|
| L1 | 10 | 0.772 | 0.813 | 0.709 |
| L2 | 9 | 0.815 | 0.827 | 0.72 |
| L3 | 3 | 0.621 | 0.955 | 0.806 |
| L4 | 5 | 0.707 | 0.832 | 0.912 |
| L5 | 2 | 0.72 | 0.855 | 0.739 |
| L6 | 3 | 0.74 | 0.927 | 0.918 |
| L7 | 8 | 0.801 | 0.789 | 0.835 |

## 2. 分指标

| 指标 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| accuracy | 0.716 ± [0.636, 0.792] | 0.794 ± [0.712, 0.869] | 0.693 ± [0.595, 0.778] |
| grounding | 0.885 ± [0.845, 0.924] | 0.876 ± [0.832, 0.917] | 0.879 ± [0.844, 0.912] |
| completeness | 0.855 ± [0.797, 0.905] | 0.928 ± [0.883, 0.963] | 0.838 ± [0.780, 0.894] |
| reasoning | 0.583 ± [0.535, 0.631] | 0.729 ± [0.675, 0.782] | 0.688 ± [0.625, 0.746] |
| efficiency | 0.964 ± [0.944, 0.981] | 0.972 ± [0.953, 0.988] | 0.957 ± [0.935, 0.977] |
| robustness | 0.998 ± [0.994, 1.000] | 0.980 ± [0.950, 1.000] | 0.998 ± [0.994, 1.000] |
| tactical_accuracy | 0.676 ± [0.625, 0.723] | 0.778 ± [0.717, 0.830] | 0.712 ± [0.640, 0.776] |

## 3. 配对显著性检验（按用例配对，Wilcoxon 主检验）

n=40（配对用例数），α=0.05，配对 t 检验在 80% power 下约可检出 |d|≥0.44 的效应；更小差异仅作描述性参考。

### 单次LLM裸调用 vs 多智能体系统

- 均值差: -0.0718，paired Cohen's d: -0.424，Cliff's δ: -0.439
- Wilcoxon p=0.00778（Holm 校正后 p=0.1556），配对 t p=0.01075

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.08787 | 1.0 | -0.259 | -0.185 |
| grounding | 0.59188 | 1.0 | 0.059 | 0.037 |
| completeness | 0.01631 | 0.26996 | -0.388 | -0.292 |
| reasoning | 0.00026 | 0.00546 | -0.724 | -0.475 |
| efficiency | 0.51109 | 1.0 | -0.095 | -0.134 |
| robustness | None | None | 0.225 | 0.026 |
| tactical_accuracy | 0.00863 | 0.16397 | -0.385 | -0.43 |

### 单次LLM裸调用 vs 单智能体+工具

- 均值差: -0.0236，paired Cohen's d: -0.122，Cliff's δ: -0.231
- Wilcoxon p=0.45974（Holm 校正后 p=1.0），配对 t p=0.44476

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.71536 | 1.0 | 0.063 | 0.018 |
| grounding | 0.59018 | 1.0 | 0.037 | 0.088 |
| completeness | 0.77493 | 1.0 | 0.083 | 0.039 |
| reasoning | 0.01161 | 0.20898 | -0.433 | -0.325 |
| efficiency | 0.65671 | 1.0 | 0.083 | 0.051 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.46874 | 1.0 | -0.121 | -0.233 |

### 多智能体系统 vs 单智能体+工具

- 均值差: 0.0482，paired Cohen's d: 0.322，Cliff's δ: 0.18
- Wilcoxon p=0.05896（Holm 校正后 p=0.82544），配对 t p=0.04827

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.08761 | 1.0 | 0.344 | 0.192 |
| grounding | 0.85167 | 1.0 | -0.024 | 0.034 |
| completeness | 0.01588 | 0.26996 | 0.44 | 0.306 |
| reasoning | 0.22978 | 1.0 | 0.211 | 0.093 |
| efficiency | 0.24182 | 1.0 | 0.182 | 0.194 |
| robustness | None | None | -0.225 | -0.026 |
| tactical_accuracy | 0.02647 | 0.39705 | 0.335 | 0.169 |

## 4. 类别 × 系统热力

| 类别 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| adversarial | 0.816 | 0.731 | 0.802 |
| comparison | 0.873 | 0.857 | 0.821 |
| counterfactual | 0.720 | 0.855 | 0.739 |
| focus | 0.719 | 0.893 | 0.939 |
| frame | 0.703 | 0.777 | 0.533 |
| general_qa | 0.823 | 0.842 | 0.744 |
| hallucination | 0.776 | 0.885 | 0.889 |
| style | 0.704 | 0.817 | 0.905 |
| team | 0.661 | 0.935 | 0.937 |
| timeline | 0.755 | 0.765 | 0.764 |
| verify | 0.775 | 0.914 | 0.612 |

## 5. 诊断指标

| 系统 | 稳定性(跨重复相似度) | 工具使用得分 | 工具调用数 |
|---|---|---|---|
| 单次LLM裸调用 | 0.279 | None | 0 |
| 多智能体系统 | 0.338 | 1.0 | 14 |
| 单智能体+工具 | 0.202 | 1.0 | 40 |

## 6. 失败模式（Judge 标注 top-3）

- **编造数据** ×80（样例: single/adv_format_messy#r0, single/adv_format_messy#r1, single/adv_format_messy#r2）
- **无依据夸大** ×40（样例: single/cf_7s_shoot#r0, single/cf_7s_shoot#r1, single/cf_7s_shoot#r2）
- **答非所问** ×15（样例: bare/cf_7s_pass10#r0, bare/cf_7s_pass10#r1, bare/cf_7s_pass10#r2）

Judge-程序分歧 top（详见 results.json judge_disagreements）:
- single/compare_6_7_active#r2：程序 0.0 vs Judge 0.963（差 0.963）
- multi/compare_7s_active#r2：程序 0.0 vs Judge 0.963（差 0.963）
- multi/compare_7s_active#r4：程序 0.0 vs Judge 0.963（差 0.963）
- single/compare_7s_active#r3：程序 0.0 vs Judge 0.95（差 0.95）
- bare/compare_6_7_active#r1：程序 0.0 vs Judge 0.925（差 0.925）

## 6b. Judge 一致性（位置互换正逆序差）

阈值：正逆序差 > 0.15 的用例标 ⚠，建议人工抽检（仅告警不改分）。

| 用例 | 系统 | 正逆序差 | 告警 |
|---|---|---|---|
| adv_ambiguous_multi_target | 单次LLM裸调用 | 0.150 |  |
| adv_ambiguous_multi_target | 多智能体系统 | 0.125 |  |
| adv_ambiguous_multi_target | 单智能体+工具 | 0.125 |  |
| adv_format_messy | 单次LLM裸调用 | 0.550 | ⚠ |
| adv_format_messy | 多智能体系统 | 0.000 |  |
| adv_format_messy | 单智能体+工具 | 0.750 | ⚠ |
| adv_frame_out_of_range | 单次LLM裸调用 | 0.000 |  |
| adv_frame_out_of_range | 多智能体系统 | 0.000 |  |
| adv_frame_out_of_range | 单智能体+工具 | 0.000 |  |
| adv_induced_scoring | 单次LLM裸调用 | 0.325 | ⚠ |
| adv_induced_scoring | 多智能体系统 | 0.400 | ⚠ |
| adv_induced_scoring | 单智能体+工具 | 0.425 | ⚠ |
| adv_prompt_injection | 单次LLM裸调用 | 0.000 |  |
| adv_prompt_injection | 多智能体系统 | 0.137 |  |
| adv_prompt_injection | 单智能体+工具 | 0.000 |  |
| cf_7s_pass10 | 单次LLM裸调用 | 0.150 |  |
| cf_7s_pass10 | 多智能体系统 | 0.125 |  |
| cf_7s_pass10 | 单智能体+工具 | 0.200 | ⚠ |
| cf_7s_shoot | 单次LLM裸调用 | 0.150 |  |
| cf_7s_shoot | 多智能体系统 | 0.075 |  |
| cf_7s_shoot | 单智能体+工具 | 0.125 |  |
| compare_6_7_active | 单次LLM裸调用 | 0.050 |  |
| compare_6_7_active | 多智能体系统 | 0.025 |  |
| compare_6_7_active | 单智能体+工具 | 0.075 |  |
| compare_7s_active | 单次LLM裸调用 | 0.025 |  |
| compare_7s_active | 多智能体系统 | 0.062 |  |
| compare_7s_active | 单智能体+工具 | 0.000 |  |
| compare_fastest | 单次LLM裸调用 | 0.000 |  |
| compare_fastest | 多智能体系统 | 0.100 |  |
| compare_fastest | 单智能体+工具 | 0.100 |  |
| focus_player7 | 单次LLM裸调用 | 0.200 | ⚠ |
| focus_player7 | 多智能体系统 | 0.200 | ⚠ |
| focus_player7 | 单智能体+工具 | 0.100 |  |
| frame_ball_first | 单次LLM裸调用 | 0.125 |  |
| frame_ball_first | 多智能体系统 | 0.200 | ⚠ |
| frame_ball_first | 单智能体+工具 | 0.075 |  |
| frame_ball_pos_90 | 单次LLM裸调用 | 0.000 |  |
| frame_ball_pos_90 | 多智能体系统 | 0.225 | ⚠ |
| frame_ball_pos_90 | 单智能体+工具 | 0.000 |  |
| frame_ball_zone_90 | 单次LLM裸调用 | 0.075 |  |
| frame_ball_zone_90 | 多智能体系统 | 0.000 |  |
| frame_ball_zone_90 | 单智能体+工具 | 0.050 |  |
| frame_what_happened | 单次LLM裸调用 | 0.200 | ⚠ |
| frame_what_happened | 多智能体系统 | 0.050 |  |
| frame_what_happened | 单智能体+工具 | 0.050 |  |
| halluc_frame_600 | 单次LLM裸调用 | 0.125 |  |
| halluc_frame_600 | 多智能体系统 | 0.000 |  |
| halluc_frame_600 | 单智能体+工具 | 0.050 |  |
| halluc_player_99 | 单次LLM裸调用 | 0.000 |  |
| halluc_player_99 | 多智能体系统 | 0.000 |  |
| halluc_player_99 | 单智能体+工具 | 0.000 |  |
| halluc_score | 单次LLM裸调用 | 0.000 |  |
| halluc_score | 多智能体系统 | 0.000 |  |
| halluc_score | 单智能体+工具 | 0.000 |  |
| qa_7s_ball_hmax | 单次LLM裸调用 | 0.000 |  |
| qa_7s_ball_hmax | 多智能体系统 | 0.025 |  |
| qa_7s_ball_hmax | 单智能体+工具 | 0.100 |  |
| qa_7s_player_count | 单次LLM裸调用 | 0.000 |  |
| qa_7s_player_count | 多智能体系统 | 0.100 |  |
| qa_7s_player_count | 单智能体+工具 | 0.000 |  |
| qa_ball_5s_short | 单次LLM裸调用 | 0.300 | ⚠ |
| qa_ball_5s_short | 多智能体系统 | 0.025 |  |
| qa_ball_5s_short | 单智能体+工具 | 0.450 | ⚠ |
| qa_ball_hmax | 单次LLM裸调用 | 0.000 |  |
| qa_ball_hmax | 多智能体系统 | 0.000 |  |
| qa_ball_hmax | 单智能体+工具 | 0.000 |  |
| qa_ball_speed_max | 单次LLM裸调用 | 0.150 |  |
| qa_ball_speed_max | 多智能体系统 | 0.100 |  |
| qa_ball_speed_max | 单智能体+工具 | 0.100 |  |
| qa_duration | 单次LLM裸调用 | 0.000 |  |
| qa_duration | 多智能体系统 | 0.000 |  |
| qa_duration | 单智能体+工具 | 0.150 |  |
| qa_player_count | 单次LLM裸调用 | 0.000 |  |
| qa_player_count | 多智能体系统 | 0.000 |  |
| qa_player_count | 单智能体+工具 | 0.000 |  |
| qa_player_team | 单次LLM裸调用 | 0.000 |  |
| qa_player_team | 多智能体系统 | 0.050 |  |
| qa_player_team | 单智能体+工具 | 0.050 |  |
| qa_player_zone_7 | 单次LLM裸调用 | 0.450 | ⚠ |
| qa_player_zone_7 | 多智能体系统 | 0.050 |  |
| qa_player_zone_7 | 单智能体+工具 | 0.087 |  |
| style_10s_player | 单次LLM裸调用 | 0.075 |  |
| style_10s_player | 多智能体系统 | 0.125 |  |
| style_10s_player | 单智能体+工具 | 0.125 |  |
| style_7s_player11 | 单次LLM裸调用 | 0.075 |  |
| style_7s_player11 | 多智能体系统 | 0.275 | ⚠ |
| style_7s_player11 | 单智能体+工具 | 0.062 |  |
| style_player7 | 单次LLM裸调用 | 0.050 |  |
| style_player7 | 多智能体系统 | 0.050 |  |
| style_player7 | 单智能体+工具 | 0.075 |  |
| team_7s_threat | 单次LLM裸调用 | 0.150 |  |
| team_7s_threat | 多智能体系统 | 0.050 |  |
| team_7s_threat | 单智能体+工具 | 0.000 |  |
| team_attack_trend | 单次LLM裸调用 | 0.112 |  |
| team_attack_trend | 多智能体系统 | 0.112 |  |
| team_attack_trend | 单智能体+工具 | 0.112 |  |
| team_collab_breakdown | 单次LLM裸调用 | 0.025 |  |
| team_collab_breakdown | 多智能体系统 | 0.037 |  |
| team_collab_breakdown | 单智能体+工具 | 0.150 |  |
| team_formation | 单次LLM裸调用 | 0.250 | ⚠ |
| team_formation | 多智能体系统 | 0.113 |  |
| team_formation | 单智能体+工具 | 0.063 |  |
| team_threat_peak | 单次LLM裸调用 | 0.000 |  |
| team_threat_peak | 多智能体系统 | 0.000 |  |
| team_threat_peak | 单智能体+工具 | 0.000 |  |
| timeline_ball_height | 单次LLM裸调用 | 0.000 |  |
| timeline_ball_height | 多智能体系统 | 0.125 |  |
| timeline_ball_height | 单智能体+工具 | 0.025 |  |
| timeline_event_types | 单次LLM裸调用 | 0.125 |  |
| timeline_event_types | 多智能体系统 | 0.000 |  |
| timeline_event_types | 单智能体+工具 | 0.050 |  |
| timeline_first5s | 单次LLM裸调用 | 0.125 |  |
| timeline_first5s | 多智能体系统 | 0.125 |  |
| timeline_first5s | 单智能体+工具 | 0.100 |  |
| verify_7s_pos | 单次LLM裸调用 | 0.250 | ⚠ |
| verify_7s_pos | 多智能体系统 | 0.000 |  |
| verify_7s_pos | 单智能体+工具 | 0.100 |  |
| verify_ball_owner_3s | 单次LLM裸调用 | 0.450 | ⚠ |
| verify_ball_owner_3s | 多智能体系统 | 0.000 |  |
| verify_ball_owner_3s | 单智能体+工具 | 0.050 |  |

⚠ 一致性告警用例（13）: adv_format_messy, adv_induced_scoring, cf_7s_pass10, focus_player7, frame_ball_first, frame_ball_pos_90, frame_what_happened, qa_ball_5s_short, qa_player_zone_7, style_7s_player11, team_formation, verify_7s_pos, verify_ball_owner_3s

## 7. 成本

| 系统 | 时延 ms | LLM 调用 | 总 token |
|---|---|---|---|
| 单次LLM裸调用 | 43440.4 | 1.0 | 10411.5 |
| 多智能体系统 | 16276.9 | 1.3 | 10518.5 |
| 单智能体+工具 | 14027.5 | 2.8 | 12232.2 |

## 8. 方法与局限说明

- 共同总分 = Σ 权重×分项；效率按预标定常数归一化；缺失分量自动重归一化。
- 系统级 CI 为跨用例 bootstrap（B=2000）；显著性按用例配对（Wilcoxon），Judge 主观分指标有族内相关，Holm 校正仅作参考下界。
- Judge 模型（deepseek-v4-pro）与被测系统解耦，采用盲评+位置互换缓解偏好，结论需人工抽检。
- 数值金标准由 pandas 独立重算（不经被测系统工具层，单位米制）；反事实类因 MCTS 随机性仅用方向性金标准。
- 阶段对齐为启发式（llm 序号/工具属性推断），时间线仅供定性比较。