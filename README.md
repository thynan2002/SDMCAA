# Football Agent ⚽ — LLM-Driven Football Tactical Analysis Multi-Agent System

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue?style=flat-square" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/lang-中文-red?style=flat-square" alt="中文"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/LLM-DeepSeek-4B8BBE?style=flat-square" alt="LLM: DeepSeek">
  <img src="https://img.shields.io/badge/orchestration-LangGraph-2C8C5A?style=flat-square" alt="Orchestration: LangGraph">
  <img src="https://img.shields.io/badge/architecture-tool--calling-8B5CF6?style=flat-square" alt="Architecture: tool-calling">
  <img src="https://img.shields.io/badge/tests-127%20passed-2EA043?style=flat-square" alt="Tests: 127 passed">
</p>

Feed in player-tracking and ball-tracking CSV files, and an LLM-driven multi-agent pipeline turns them into professional match commentary, player performance analysis, and counterfactual (What-If) simulations. The system is built around **function calling (tool calling)**: structured decisions are delivered through tool contracts, and external information can only be obtained through tools — with the full pipeline orchestrated by LangGraph and usable from a CLI chat, one-shot commentary, or a Web visualization interface.

[Documentation](docs/README.md) · [Harness](docs/harness.md) · [Evaluation](docs/eval.md) · [Project Report](docs/project_report.md) · [中文](README.zh-CN.md)

## Features

- **Tool-calling driven architecture** — structured decision outputs (intent routing, tactical decisions, scripts, MCTS strategy tables, challenge judgments, semantic tiers) are delivered as **terminal tools** (`submit_*`) via contract; `call_llm` supports `tools`, multi-round tool loops, and **parallel tool_calls** (multiple independent calls in one response, executed in a single batch)
- **No fabrication by design** — deterministic capabilities (ball-path analysis, tactical fact extraction, frame-level verification, player data, counterfactual simulation) are wrapped as **data tools** (`agents/tools/football.py`); the LLM can only obtain external information through tools and must honestly report when data is insufficient
- **Graceful degradation chain** — transient failures retry automatically → invalid model output falls back to existing rule-based paths (keyword routing, heuristic decisions) → persistent failure is honestly reported as "model call failed, not insufficient data"
- **Multi-agent analysis pipeline** — player movement analysis, ball interaction, ball-path events, comparative ranking, joint decision, and commentary generation, orchestrated end-to-end by LangGraph
- **Counterfactual simulation** — MCTS-based What-If reasoning: change a player's decision at a given moment (shoot / pass / dribble, etc.), replay the trajectory, and inspect win-probability changes; exportable as trajectory CSV
- **Data verification** — challenge the system's answer and it re-verifies against the raw data and corrects itself
- **Web visualization** — FastAPI + SSE streaming chat with Canvas frame-by-frame rendering of player/ball trajectories
- **Unified Harness** — transparent wrapper layer: full-chain tracing, unified CLI/API entry, external dependency mocking, and golden replay regression (4-way comparison: LLM request sequence / per-turn output / state snapshot / file hash), with tool calling preserving equivalence

## Install

```bash
pip install -r requirements.txt   # or: uv pip install -r requirements.txt
```

Copy `.env` (shipped as a placeholder template) and fill in your API key:

```bash
# .env
DEEPSEEK_API_KEY=your_real_api_key_here
```

## Quick start

```bash
# Interactive mode (recommended): load data, then ask questions in natural language
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv

# One-shot commentary (optionally focused on specific players)
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --once
python main.py TestInput/Files/12s_person2d.csv TestInput/Files/12s_soccer3d.csv --focus 6 10

# Web visualization (SSE streaming chat + trajectory rendering)
python -m web.backend
# open http://127.0.0.1:8000
```

Example conversation:

```
>> Focus on #7
>> What is #7's running style?
>> Compare #6 and #7: who is more active?
>> What if #7 had shot directly at second 3?
>> What happened at frame 90?
```

### Input data format

| Data | Columns | Description |
|------|---------|-------------|
| `person.csv` | `frame_num, track_id, x, y, color` | `color`: A = red team, B = blue team, C = goalkeeper |
| `ball.csv` | `frame_num, x, y, z` | `z`: ball height above ground (meters) |

## How it fits together

- **Data layer** — CSV is frame-interpolated and fused into a `PrefixPlayerCorpus` (players + ball)
- **Analysis layer** — two pipelines: player tracking commentary (ball-relation analysis + LLM commentary, slimmed to two steps) and professional analysis (data collection / style modeling / MCTS counterfactual / report, orchestrated with LangGraph)
- **LLM layer** — unified `call_llm` client (DeepSeek, streaming/non-streaming + function calling): `tools` and multi-round tool loops travel through a ContextVar exchange channel, keeping the transport signature and harness hooks untouched; behavior is byte-identical to the plain-text interface when no tools are bound
- **Tool layer** — `agents/tools/`: `ToolSpec` / `ToolRegistry` + pydantic decision tool schemas + 6 data tools, covering a failure taxonomy (invalid arguments / execution error / insufficient data)
- **Interface layer** — CLI (REPL / one-shot) and Web (FastAPI + SSE) entries, both under the unified Harness (transparent wrapping: trace recording, golden capture, mock replay)

| Agent / module | Responsibility |
|----------------|----------------|
| BallPositionRelationAgent / PlayerCompositionAgent | 3D player–ball relation analysis and focused commentary generation |
| QueryRouter | Intent routing (keyword rules first, `submit_intent` tool contract as LLM fallback) |
| LLMDecisionEngine / SemanticTierBatcher | MCTS decision engine and semantic-tier inference (tool contract + heuristic fallback) |
| DataCollectorAgent / StyleModelingAgent | Data collection and player style modeling (feature vectors/style labels are deterministic, zero LLM calls) |
| CounterfactualEngineAgent | MCTS counterfactual reasoning and trajectory simulation |
| ReportGeneratorAgent / GeneralQAAgent / DataVerifierAgent | Report generation, general QA (data tools), data verification (challenge judgment tool contract) |
| `agents/tools` | Terminal decision tools (`submit_*`) + data tools (tactical facts / ball path / frame verification / player data / counterfactual simulation) |

## Testing

```bash
pytest            # full suite, 127 tests (real-API smoke tests skipped by default)
pytest -m smoke   # real-API smoke tests (requires DEEPSEEK_API_KEY)

# Harness golden replay regression (offline; verifies refactors preserve equivalence)
python -m harness verify harness/golden/standard
```

## Security

- The API key lives only in `.env`, which is gitignored; the repository keeps only a placeholder template — never commit real secrets
- Generated outputs (counterfactual trajectory exports, commentary drafts, etc.) go to `Output/` and never pollute the source tree

## Documentation

| Goal | Start here |
|------|-----------|
| Install & usage | This file, «Install» and «Quick start» |
| Docs index & overview | [docs/README.md](docs/README.md) |
| Unified Harness (run modes / equivalence argument / verification suite / known boundaries) | [docs/harness.md](docs/harness.md) |
| Evaluation framework (single-LLM vs multi-agent: metrics / stats / tracing / reports) | [docs/eval.md](docs/eval.md) |
| Tool-calling refactor (rationale / migration plan / execution summary / review fixes) | [docs/tool_calling_refactor.md](docs/tool_calling_refactor.md) |
| Project report (background / challenges / solutions / results) | [docs/project_report.md](docs/project_report.md) |
| Input data format | This file, «Input data format» |

## License

All rights reserved. This project is not open-sourced under a license yet.
