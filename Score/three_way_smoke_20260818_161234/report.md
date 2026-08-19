# 评测报告：单 LLM 直接调用 vs 多智能体系统

- 事件目录: `D:\GHY\2.Files\football-agent\Score\three_way_smoke_20260818_161234`
- 用例数: 11，重复: 1，采样温度: 0.85
- 模型: deepseek-v4-pro；git: `b27434b6`
- 权重: {'accuracy': 0.15, 'grounding': 0.1, 'completeness': 0.1, 'reasoning': 0.2, 'efficiency': 0.1, 'robustness': 0.1, 'tactical_accuracy': 0.25}
- Judge: {'model': 'deepseek-v4-pro', 'sample_repeats': 2, 'order_swap': True, 'temperature': 0.0}
- 【同源风险披露】Judge 模型: deepseek-v4-pro；rubric 版本: v3；被测系统模型: (未配置，默认 deepseek-v4-flash)；Judge 与被测系统非同厂商（模型名前缀粗判）。同源 Judge 存在自我偏好风险，结论需人工抽检。

## 1. 总分

| 系统 | 共同总分 | 95% CI | 扩展总分(+工具) | n |
|---|---|---|---|---|
| 单次LLM裸调用 | 0.732 | [0.605, 0.846] | None | 11 |
| 多智能体系统 | 0.765 | [0.658, 0.863] | 0.795 | 11 |
| 单智能体+工具 | 0.825 | [0.753, 0.895] | 0.843 | 11 |

### 1b. 按业务分层（L1-L7）总分

| 层 | 用例数 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|---|
| L1 | 2 | 0.755 | 0.67 | 0.678 |
| L2 | 3 | 0.906 | 0.818 | 0.751 |
| L3 | 1 | 0.412 | 0.96 | 0.972 |
| L4 | 2 | 0.64 | 0.827 | 0.966 |
| L5 | 1 | 0.786 | 0.382 | 0.883 |
| L7 | 2 | 0.67 | 0.815 | 0.84 |

## 2. 分指标

| 指标 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| accuracy | 0.643 ± [0.420, 0.846] | 0.625 ± [0.427, 0.814] | 0.718 ± [0.550, 0.879] |
| grounding | 0.814 ± [0.643, 0.952] | 0.889 ± [0.795, 0.968] | 0.925 ± [0.875, 0.968] |
| completeness | 0.863 ± [0.760, 0.947] | 0.772 ± [0.614, 0.904] | 0.855 ± [0.755, 0.946] |
| reasoning | 0.609 ± [0.509, 0.700] | 0.627 ± [0.491, 0.754] | 0.718 ± [0.591, 0.846] |
| efficiency | 0.989 ± [0.974, 1.000] | 0.967 ± [0.924, 1.000] | 0.976 ± [0.954, 0.996] |
| robustness | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] |
| tactical_accuracy | 0.586 ± [0.364, 0.777] | 0.734 ± [0.568, 0.873] | 0.793 ± [0.707, 0.879] |

## 3. 配对显著性检验（按用例配对，Wilcoxon 主检验）

n=11（配对用例数），α=0.05，配对 t 检验在 80% power 下约可检出 |d|≥0.84 的效应；更小差异仅作描述性参考。

### 单次LLM裸调用 vs 多智能体系统

- 均值差: -0.0339，paired Cohen's d: -0.105，Cliff's δ: -0.058
- Wilcoxon p=1.0（Holm 校正后 p=1.0），配对 t p=0.73449

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.8938 | 1.0 | 0.029 | -0.008 |
| grounding | 0.8885 | 1.0 | -0.243 | -0.041 |
| completeness | 0.28419 | 1.0 | 0.304 | 0.124 |
| reasoning | 0.72134 | 1.0 | -0.064 | -0.099 |
| efficiency | 0.27332 | 1.0 | 0.401 | 0.107 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.44429 | 1.0 | -0.289 | -0.256 |

### 单次LLM裸调用 vs 单智能体+工具

- 均值差: -0.0938，paired Cohen's d: -0.316，Cliff's δ: -0.223
- Wilcoxon p=0.286（Holm 校正后 p=1.0），配对 t p=0.31894

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.61007 | 1.0 | -0.131 | -0.099 |
| grounding | 0.51467 | 1.0 | -0.367 | -0.033 |
| completeness | 1.0 | 1.0 | 0.029 | -0.041 |
| reasoning | 0.23531 | 1.0 | -0.362 | -0.347 |
| efficiency | 0.34523 | 1.0 | 0.313 | 0.198 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.1389 | 1.0 | -0.475 | -0.331 |

### 多智能体系统 vs 单智能体+工具

- 均值差: -0.0599，paired Cohen's d: -0.321，Cliff's δ: -0.223
- Wilcoxon p=0.37394（Holm 校正后 p=1.0），配对 t p=0.31166

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.63499 | 1.0 | -0.337 | -0.116 |
| grounding | 0.19747 | 1.0 | -0.382 | -0.041 |
| completeness | 0.17295 | 1.0 | -0.417 | -0.132 |
| reasoning | 0.47264 | 1.0 | -0.275 | -0.248 |
| efficiency | 0.60018 | 1.0 | -0.126 | 0.041 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.75315 | 1.0 | -0.21 | -0.083 |

## 4. 类别 × 系统热力

| 类别 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| adversarial | 0.798 | 0.950 | 0.840 |
| comparison | 0.877 | 0.932 | 0.901 |
| counterfactual | 0.786 | 0.382 | 0.883 |
| focus | 0.925 | 0.695 | 0.960 |
| frame | 0.640 | 0.720 | 0.721 |
| general_qa | 0.870 | 0.620 | 0.635 |
| hallucination | 0.542 | 0.680 | 0.840 |
| style | 0.355 | 0.960 | 0.973 |
| team | 0.412 | 0.960 | 0.972 |
| timeline | 0.860 | 0.759 | 0.600 |
| verify | 0.980 | 0.762 | 0.753 |

## 5. 诊断指标

| 系统 | 稳定性(跨重复相似度) | 工具使用得分 | 工具调用数 |
|---|---|---|---|
| 单次LLM裸调用 | None | None | 0 |
| 多智能体系统 | None | 1.0 | 4 |
| 单智能体+工具 | None | 1.0 | 11 |

## 6. 失败模式（Judge 标注 top-3）

- **编造数据** ×5（样例: bare/halluc_frame_600#r0, single/qa_7s_ball_hmax#r0, multi/qa_7s_ball_hmax#r0）
- **无依据夸大** ×2（样例: multi/compare_6_7_active#r0, bare/frame_ball_first#r0）
- **逻辑矛盾** ×2（样例: multi/frame_ball_first#r0, single/timeline_ball_height#r0）

Judge-程序分歧 top（详见 results.json judge_disagreements）:
- multi/qa_7s_ball_hmax#r0：程序 1.0 vs Judge 0.2（差 0.8）
- single/qa_7s_ball_hmax#r0：程序 1.0 vs Judge 0.2（差 0.8）
- bare/halluc_frame_600#r0：程序 1.0 vs Judge 0.4（差 0.6）
- multi/timeline_ball_height#r0：程序 0.0 vs Judge 0.4（差 0.4）
- single/timeline_ball_height#r0：程序 0.0 vs Judge 0.35（差 0.35）

## 6b. Judge 一致性（位置互换正逆序差）

阈值：正逆序差 > 0.15 的用例标 ⚠，建议人工抽检（仅告警不改分）。

| 用例 | 系统 | 正逆序差 | 告警 |
|---|---|---|---|
| adv_ambiguous_multi_target | 单次LLM裸调用 | 0.250 | ⚠ |
| adv_ambiguous_multi_target | 多智能体系统 | 0.000 |  |
| adv_ambiguous_multi_target | 单智能体+工具 | 0.000 |  |
| cf_7s_pass10 | 单次LLM裸调用 | 0.000 |  |
| cf_7s_pass10 | 多智能体系统 | 0.000 |  |
| cf_7s_pass10 | 单智能体+工具 | 0.100 |  |
| compare_6_7_active | 单次LLM裸调用 | 0.050 |  |
| compare_6_7_active | 多智能体系统 | 0.100 |  |
| compare_6_7_active | 单智能体+工具 | 0.075 |  |
| focus_player7 | 单次LLM裸调用 | 0.050 |  |
| focus_player7 | 多智能体系统 | 0.250 | ⚠ |
| focus_player7 | 单智能体+工具 | 0.100 |  |
| frame_ball_first | 单次LLM裸调用 | 0.125 |  |
| frame_ball_first | 多智能体系统 | 0.325 | ⚠ |
| frame_ball_first | 单智能体+工具 | 0.325 | ⚠ |
| halluc_frame_600 | 单次LLM裸调用 | 0.000 |  |
| halluc_frame_600 | 多智能体系统 | 0.000 |  |
| halluc_frame_600 | 单智能体+工具 | 0.000 |  |
| qa_7s_ball_hmax | 单次LLM裸调用 | 0.000 |  |
| qa_7s_ball_hmax | 多智能体系统 | 0.000 |  |
| qa_7s_ball_hmax | 单智能体+工具 | 0.000 |  |
| qa_player_zone_7 | 单次LLM裸调用 | 0.000 |  |
| qa_player_zone_7 | 多智能体系统 | 0.000 |  |
| qa_player_zone_7 | 单智能体+工具 | 0.050 |  |
| team_7s_threat | 单次LLM裸调用 | 0.100 |  |
| team_7s_threat | 多智能体系统 | 0.000 |  |
| team_7s_threat | 单智能体+工具 | 0.000 |  |
| timeline_ball_height | 单次LLM裸调用 | 0.000 |  |
| timeline_ball_height | 多智能体系统 | 0.000 |  |
| timeline_ball_height | 单智能体+工具 | 0.050 |  |
| verify_7s_pos | 单次LLM裸调用 | 0.000 |  |
| verify_7s_pos | 多智能体系统 | 0.100 |  |
| verify_7s_pos | 单智能体+工具 | 0.000 |  |

⚠ 一致性告警用例（3）: adv_ambiguous_multi_target, focus_player7, frame_ball_first

## 7. 成本

| 系统 | 时延 ms | LLM 调用 | 总 token |
|---|---|---|---|
| 单次LLM裸调用 | 35552.5 | 1.0 | 9484.5 |
| 多智能体系统 | 19823.8 | 1.2 | 11333.9 |
| 单智能体+工具 | 14076.6 | 2.5 | 9283.4 |

## 8. 方法与局限说明

- 共同总分 = Σ 权重×分项；效率按预标定常数归一化；缺失分量自动重归一化。
- 系统级 CI 为跨用例 bootstrap（B=2000）；显著性按用例配对（Wilcoxon），Judge 主观分指标有族内相关，Holm 校正仅作参考下界。
- Judge 模型（deepseek-v4-pro）与被测系统解耦，采用盲评+位置互换缓解偏好，结论需人工抽检。
- 数值金标准由 pandas 独立重算（不经被测系统工具层，单位米制）；反事实类因 MCTS 随机性仅用方向性金标准。
- 阶段对齐为启发式（llm 序号/工具属性推断），时间线仅供定性比较。