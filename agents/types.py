"""共享数据结构定义。

集中定义单个智能体报告与最终决策的数据模型，
保证各智能体之间以及输出层的数据格式一致。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json


@dataclass
class AgentReport:
    agent_name: str
    summary: str
    score_0_to_100: float
    confidence_0_to_1: float
    key_points: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class FinalDecision:
    recommended_action: str
    overall_score_0_to_100: float
    rationale: str
    next_focus: list[str]
    merged_risks: list[str]
    reports: list[AgentReport]
    reviewer_notes: list[str] = field(default_factory=list)
    input_overview: list[str] = field(default_factory=list)
    composed_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_pretty_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
