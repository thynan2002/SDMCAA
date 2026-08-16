# 文档索引 📚 — Football Agent

<p align="center">
  <img src="https://img.shields.io/badge/status-live-2EA043?style=flat-square" alt="Status: live">
  <img src="https://img.shields.io/badge/docs-5%20pages-4B8BBE?style=flat-square" alt="Docs: 5 pages">
</p>

这里是 Football Agent 的全部技术文档入口。文档按「阅读路径」组织：从安装使用出发，向内逐层深入架构、观测、评测与演进规划。

## 快速导航

| 目标 | 文档 |
|------|------|
| 安装、快速开始、输入数据格式 | [README](../README.md)（[中文](../README.zh-CN.md)） |
| 架构总览、智能体职责、技术栈 | [project_report.md](project_report.md) |
| 统一 Harness：运行模式 / 等价性论证 / 回放回归 | [harness.md](harness.md) |
| 评测框架：指标 / 统计 / 追溯 / 报告 | [eval.md](eval.md) |
| 工具调用（tool_calls）重构：选型 / 迁移 / 总结 | [tool_calling_refactor.md](tool_calling_refactor.md) |

## 推荐阅读顺序

```
README（入门）
  └─ project_report.md（为什么这么做：背景与挑战）
       ├─ harness.md（如何保证可观测、可回归）
       ├─ eval.md（如何证明系统更好）
       └─ tool_calling_refactor.md（架构核心演进的完整记录）
```

## 文档速览

| 文档 | 一句话摘要 | 适合读者 |
|------|-----------|---------|
| [project_report.md](project_report.md) | 项目的完整叙事：背景、五大挑战、五类解法、交付与质量保障 | 新读者、评审、答辩 |
| [harness.md](harness.md) | 透明包装层：三种运行模式（passthrough/record/replay）、四维 golden 回放、工具层等价性论证 | 维护者、重构者 |
| [eval.md](eval.md) | 单 LLM 裸调用 vs 单智能体 vs 多智能体的对比评测体系：指标体系、用例分层、方法学要点 | 评测者、研究者 |
| [tool_calling_refactor.md](tool_calling_refactor.md) | 从纯文本 JSON 契约迁移到 OpenAI 风格 function calling 的全过程：约束盘点、选型、方案、落地、审查修复 | 架构师、贡献者 |

## 目录结构

```
football-agent/
├── main.py                  # CLI 入口（交互 / 单次解说）
├── agents/                  # 智能体核心
│   ├── tools/               #   工具层（submit_* 决策工具 + 数据工具）
│   ├── player/              #   球员追踪流水线
│   ├── professional/        #   专业分析流水线（含 MCTS 反事实）
│   └── session/             #   会话管理与意图路由
├── harness/                 # 统一 Harness（trace / golden / 回放）
├── eval/                    # 对比评测框架
├── web/                     # FastAPI + SSE Web 界面
├── tests/                   # pytest 测试（111 个）
├── TestInput/               # 示例数据
└── docs/                    # 本目录
```

## 快速命令

```bash
# 运行全部离线测试
pytest

# golden 回放回归（不触网）
python -m harness verify harness/golden/standard

# 评测用例校验 / 金标准打印
python -m eval cases lint
python -m eval gold
```
