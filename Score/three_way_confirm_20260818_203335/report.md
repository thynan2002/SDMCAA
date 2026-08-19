# 评测报告：单 LLM 直接调用 vs 多智能体系统

- 事件目录: `D:\GHY\2.Files\football-agent\Score\three_way_confirm_20260818_203335`
- 用例数: 11，重复: 1，采样温度: 0.85
- 模型: deepseek-v4-pro；git: `b27434b6`
- 权重: {'accuracy': 0.15, 'grounding': 0.1, 'completeness': 0.1, 'reasoning': 0.2, 'efficiency': 0.1, 'robustness': 0.1, 'tactical_accuracy': 0.25}
- Judge: {'model': 'deepseek-v4-pro', 'sample_repeats': 2, 'order_swap': True, 'temperature': 0.0}
- 【同源风险披露】Judge 模型: deepseek-v4-pro；rubric 版本: v3；被测系统模型: (未配置，默认 deepseek-v4-flash)；Judge 与被测系统同厂商（模型名前缀粗判: deepseek）。同源 Judge 存在自我偏好风险，结论需人工抽检。

## 1. 总分

| 系统 | 共同总分 | 95% CI | 扩展总分(+工具) | n |
|---|---|---|---|---|
| 单次LLM裸调用 | 0.732 | [0.647, 0.802] | None | 11 |
| 多智能体系统 | 0.843 | [0.781, 0.897] | 0.818 | 11 |
| 单智能体+工具 | 0.792 | [0.722, 0.859] | 0.812 | 11 |

### 1b. 按业务分层（L1-L7）总分

| 层 | 用例数 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|---|
| L1 | 2 | 0.825 | 0.748 | 0.637 |
| L2 | 3 | 0.794 | 0.836 | 0.827 |
| L3 | 1 | 0.44 | 0.96 | 0.925 |
| L4 | 2 | 0.647 | 0.865 | 0.827 |
| L5 | 1 | 0.623 | 0.855 | 0.673 |
| L7 | 2 | 0.831 | 0.865 | 0.854 |

## 2. 分指标

| 指标 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| accuracy | 0.666 ± [0.486, 0.818] | 0.793 ± [0.677, 0.895] | 0.709 ± [0.586, 0.818] |
| grounding | 0.927 ± [0.818, 1.000] | 0.893 ± [0.857, 0.932] | 0.862 ± [0.800, 0.919] |
| completeness | 0.830 ± [0.736, 0.916] | 0.936 ± [0.898, 0.975] | 0.835 ± [0.727, 0.934] |
| reasoning | 0.554 ± [0.455, 0.654] | 0.773 ± [0.673, 0.855] | 0.709 ± [0.609, 0.809] |
| efficiency | 0.936 ± [0.881, 0.983] | 0.966 ± [0.916, 1.000] | 0.942 ± [0.899, 0.981] |
| robustness | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] | 1.000 ± [1.000, 1.000] |
| tactical_accuracy | 0.607 ± [0.464, 0.732] | 0.761 ± [0.659, 0.855] | 0.721 ± [0.607, 0.818] |

## 3. 配对显著性检验（按用例配对，Wilcoxon 主检验）

n=11（配对用例数），α=0.05，配对 t 检验在 80% power 下约可检出 |d|≥0.84 的效应；更小差异仅作描述性参考。

### 单次LLM裸调用 vs 多智能体系统

- 均值差: -0.1115，paired Cohen's d: -0.581，Cliff's δ: -0.504
- Wilcoxon p=0.05046（Holm 校正后 p=0.90378），配对 t p=0.08273

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.32699 | 1.0 | -0.372 | -0.24 |
| grounding | 0.38472 | 1.0 | 0.212 | 0.471 |
| completeness | 0.05021 | 0.90378 | -0.746 | -0.405 |
| reasoning | 0.02553 | 0.53613 | -0.942 | -0.645 |
| efficiency | 0.46307 | 1.0 | -0.234 | -0.314 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.20262 | 1.0 | -0.451 | -0.347 |

### 单次LLM裸调用 vs 单智能体+工具

- 均值差: -0.0604，paired Cohen's d: -0.305，Cliff's δ: -0.24
- Wilcoxon p=0.37394（Holm 校正后 p=1.0），配对 t p=0.33533

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.8938 | 1.0 | -0.121 | -0.017 |
| grounding | 0.31223 | 1.0 | 0.351 | 0.512 |
| completeness | 0.54029 | 1.0 | -0.021 | -0.107 |
| reasoning | 0.12164 | 1.0 | -0.627 | -0.455 |
| efficiency | 0.77943 | 1.0 | -0.051 | -0.091 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.31351 | 1.0 | -0.325 | -0.223 |

### 多智能体系统 vs 单智能体+工具

- 均值差: 0.0512，paired Cohen's d: 0.539，Cliff's δ: 0.24
- Wilcoxon p=0.091（Holm 校正后 p=1.0），配对 t p=0.10436

| 指标 | p (Wilcoxon) | p (Holm) | d | δ |
|---|---|---|---|---|
| accuracy | 0.04614 | 0.87666 | 0.642 | 0.256 |
| grounding | 0.34283 | 1.0 | 0.337 | 0.132 |
| completeness | 0.04252 | 0.8504 | 0.592 | 0.331 |
| reasoning | 0.25909 | 1.0 | 0.377 | 0.24 |
| efficiency | 0.46307 | 1.0 | 0.225 | 0.207 |
| robustness | None | None | None | 0.0 |
| tactical_accuracy | 0.31351 | 1.0 | 0.25 | 0.174 |

## 4. 类别 × 系统热力

| 类别 | 单次LLM裸调用 | 多智能体系统 | 单智能体+工具 |
|---|---|---|---|
| adversarial | 0.843 | 0.819 | 0.820 |
| comparison | 0.886 | 0.990 | 0.905 |
| counterfactual | 0.623 | 0.855 | 0.673 |
| focus | 0.734 | 0.860 | 0.943 |
| frame | 0.859 | 0.647 | 0.562 |
| general_qa | 0.790 | 0.848 | 0.713 |
| hallucination | 0.820 | 0.910 | 0.888 |
| style | 0.560 | 0.870 | 0.710 |
| team | 0.440 | 0.960 | 0.925 |
| timeline | 0.841 | 0.836 | 0.770 |
| verify | 0.655 | 0.682 | 0.805 |

## 5. 诊断指标

| 系统 | 稳定性(跨重复相似度) | 工具使用得分 | 工具调用数 |
|---|---|---|---|
| 单次LLM裸调用 | None | None | 0 |
| 多智能体系统 | None | 1.0 | 4 |
| 单智能体+工具 | None | 0.996 | 11 |

## 6. 失败模式（Judge 标注 top-3）

- **逻辑矛盾** ×3（样例: single/frame_ball_first#r0, bare/team_7s_threat#r0, multi/verify_7s_pos#r0）
- **无依据夸大** ×3（样例: single/qa_player_zone_7#r0, multi/qa_player_zone_7#r0, single/verify_7s_pos#r0）
- **编造数据** ×2（样例: single/qa_7s_ball_hmax#r0, multi/qa_7s_ball_hmax#r0）

Judge-程序分歧 top（详见 results.json judge_disagreements）:
- single/timeline_ball_height#r0：程序 0.0 vs Judge 0.65（差 0.65）
- bare/cf_7s_pass10#r0：程序 1.0 vs Judge 0.45（差 0.55）
- single/cf_7s_pass10#r0：程序 1.0 vs Judge 0.5（差 0.5）

## 6b. Judge 一致性（位置互换正逆序差）

阈值：正逆序差 > 0.15 的用例标 ⚠，建议人工抽检（仅告警不改分）。

| 用例 | 系统 | 正逆序差 | 告警 |
|---|---|---|---|
| adv_ambiguous_multi_target | 单次LLM裸调用 | 0.075 |  |
| adv_ambiguous_multi_target | 多智能体系统 | 0.075 |  |
| adv_ambiguous_multi_target | 单智能体+工具 | 0.150 |  |
| cf_7s_pass10 | 单次LLM裸调用 | 0.250 | ⚠ |
| cf_7s_pass10 | 多智能体系统 | 0.000 |  |
| cf_7s_pass10 | 单智能体+工具 | 0.100 |  |
| compare_6_7_active | 单次LLM裸调用 | 0.025 |  |
| compare_6_7_active | 多智能体系统 | 0.025 |  |
| compare_6_7_active | 单智能体+工具 | 0.000 |  |
| focus_player7 | 单次LLM裸调用 | 0.200 | ⚠ |
| focus_player7 | 多智能体系统 | 0.000 |  |
| focus_player7 | 单智能体+工具 | 0.100 |  |
| frame_ball_first | 单次LLM裸调用 | 0.050 |  |
| frame_ball_first | 多智能体系统 | 0.050 |  |
| frame_ball_first | 单智能体+工具 | 0.050 |  |
| halluc_frame_600 | 单次LLM裸调用 | 0.000 |  |
| halluc_frame_600 | 多智能体系统 | 0.000 |  |
| halluc_frame_600 | 单智能体+工具 | 0.050 |  |
| qa_7s_ball_hmax | 单次LLM裸调用 | 0.000 |  |
| qa_7s_ball_hmax | 多智能体系统 | 0.000 |  |
| qa_7s_ball_hmax | 单智能体+工具 | 0.000 |  |
| qa_player_zone_7 | 单次LLM裸调用 | 0.050 |  |
| qa_player_zone_7 | 多智能体系统 | 0.100 |  |
| qa_player_zone_7 | 单智能体+工具 | 0.150 |  |
| team_7s_threat | 单次LLM裸调用 | 0.100 |  |
| team_7s_threat | 多智能体系统 | 0.000 |  |
| team_7s_threat | 单智能体+工具 | 0.050 |  |
| timeline_ball_height | 单次LLM裸调用 | 0.025 |  |
| timeline_ball_height | 多智能体系统 | 0.025 |  |
| timeline_ball_height | 单智能体+工具 | 0.050 |  |
| verify_7s_pos | 单次LLM裸调用 | 0.100 |  |
| verify_7s_pos | 多智能体系统 | 0.100 |  |
| verify_7s_pos | 单智能体+工具 | 0.100 |  |

⚠ 一致性告警用例（2）: cf_7s_pass10, focus_player7

## 7. 成本

| 系统 | 时延 ms | LLM 调用 | 总 token |
|---|---|---|---|
| 单次LLM裸调用 | 65461.5 | 1.0 | 12631.2 |
| 多智能体系统 | 21384.9 | 1.4 | 11106.8 |
| 单智能体+工具 | 15870.7 | 2.9 | 12625.7 |

## 8. 方法与局限说明

- 共同总分 = Σ 权重×分项；效率按预标定常数归一化；缺失分量自动重归一化。
- 系统级 CI 为跨用例 bootstrap（B=2000）；显著性按用例配对（Wilcoxon），Judge 主观分指标有族内相关，Holm 校正仅作参考下界。
- Judge 模型（deepseek-v4-pro）与被测系统解耦，采用盲评+位置互换缓解偏好，结论需人工抽检。
- 数值金标准由 pandas 独立重算（不经被测系统工具层，单位米制）；反事实类因 MCTS 随机性仅用方向性金标准。
- 阶段对齐为启发式（llm 序号/工具属性推断），时间线仅供定性比较。