"""测试用例 schema + 加载器 + lint。

用例为 JSON（eval/cases/*.json）。数值金标准以 "auto:<expr>" 表达式声明，
加载时经 goldref.resolve 从原始 CSV 独立重算（评审修订 #1：金标准不依赖
被测系统工具层，且永远与当前数据一致）。

schema 示例：
{
  "id": "qa_ball_hmax",
  "category": "general_qa",
  "dataset": {"person": "TestInput/Files/12s_person2d.csv",
              "ball": "TestInput/Files/12s_soccer3d.csv"},
  "input": "比赛中球最高到过多少高度？",
  "numeric": [{"expr": "ball_zmax", "value": "auto:ball_zmax",
               "tolerance": 0.3}],
  "direction": [{"expr": "win_rate_change",
                 "keywords_pos": ["上升", "提高", "增加"],
                 "keywords_neg": ["下降", "降低"]}],
  "refusal": {"keywords": ["不存在", "没有", "未找到", "无法", "数据不足", "范围"]},
  "checklist": [{"text": "提到具体高度数值", "keywords": ["米"]}],
  "notes": "备注"
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import goldref
from .config import PROJECT_ROOT

CATEGORIES = (
    "general_qa", "frame", "focus", "style", "comparison",
    "counterfactual", "timeline", "verify", "hallucination", "team",
    "adversarial",
)

# 业务场景分层（1.2）：L1 数据查询 → L7 幻觉与鲁棒性
LEVEL_NAMES = {
    1: "L1 数据查询", 2: "L2 事件检测", 3: "L3 战术指标",
    4: "L4 球员评估", 5: "L5 反事实推演", 6: "L6 综合战术分析",
    7: "L7 幻觉与鲁棒性",
}
CATEGORY_DEFAULT_LEVEL = {
    "general_qa": 1, "frame": 1, "comparison": 2, "timeline": 2,
    "verify": 2, "style": 4, "focus": 4, "counterfactual": 5,
    "team": 6, "hallucination": 7, "adversarial": 7,
}


@dataclass
class NumericGold:
    expr: str                    # 含义标识（报告展示用）
    value: float | str           # 解析后的金标准（str 用于 winner 类）
    tolerance: float = 0.0       # 数值容差（|ans-gold|<=tol 全分, <=2tol 半分）


@dataclass
class DirectionGold:
    expr: str
    keywords_pos: list[str] = field(default_factory=list)
    keywords_neg: list[str] = field(default_factory=list)


@dataclass
class RefusalGold:
    keywords: list[str] = field(default_factory=list)


@dataclass
class ChecklistItem:
    text: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class CategoryGold:
    """分类金标准（战术区域等类别标签）：答案命中任一别名即得 1，否则 0。"""
    expr: str
    label: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class TestCase:
    id: str
    category: str
    dataset_person: str
    dataset_ball: str
    input: str
    numeric: list[NumericGold] = field(default_factory=list)
    direction: list[DirectionGold] = field(default_factory=list)
    refusal: RefusalGold | None = None
    checklist: list[ChecklistItem] = field(default_factory=list)
    category_gold: list[CategoryGold] = field(default_factory=list)
    level: int = 1                     # 业务场景分层 L1-L7
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.level:
            self.level = CATEGORY_DEFAULT_LEVEL.get(self.category, 1)

    @property
    def dataset_key(self) -> str:
        return f"{self.dataset_person}|{self.dataset_ball}"

    def resolve_dataset_stats(self) -> goldref.DatasetStats:
        return goldref.compute_stats(
            str(PROJECT_ROOT / self.dataset_person), str(PROJECT_ROOT / self.dataset_ball),
        )


# ── 解析 ──

def _parse_case(raw: dict, source: Path) -> TestCase:
    for key in ("id", "category", "input", "dataset"):
        if key not in raw:
            raise ValueError(f"{source.name}: 缺少必填字段 {key!r}")
    if raw["category"] not in CATEGORIES:
        raise ValueError(f"{source.name}: 未知 category {raw['category']!r}（可选: {CATEGORIES}）")
    ds = raw["dataset"]
    case = TestCase(
        id=raw["id"], category=raw["category"],
        dataset_person=ds["person"], dataset_ball=ds["ball"],
        input=raw["input"], notes=raw.get("notes", ""),
        level=int(raw.get("level", 0)),
    )
    for item in raw.get("numeric", []):
        case.numeric.append(
            NumericGold(expr=item["expr"], value=item["value"], tolerance=float(item.get("tolerance", 0.0)))
        )
    for item in raw.get("direction", []):
        case.direction.append(
            DirectionGold(expr=item["expr"], keywords_pos=list(item.get("keywords_pos", [])),
                          keywords_neg=list(item.get("keywords_neg", [])))
        )
    if raw.get("refusal"):
        case.refusal = RefusalGold(keywords=list(raw["refusal"].get("keywords", [])))
    for item in raw.get("checklist", []):
        case.checklist.append(
            ChecklistItem(text=item["text"], keywords=list(item.get("keywords", [])))
        )
    for item in raw.get("category_gold", []):
        case.category_gold.append(
            CategoryGold(expr=item["expr"], label=item.get("label", ""),
                         aliases=list(item.get("aliases", [])))
        )
    return case


def load_cases(cases_dir: Path, select: str | None = None) -> list[TestCase]:
    """加载全部用例；select 支持逗号分隔的 case id 或 category 过滤。"""
    cases: list[TestCase] = []
    for path in sorted(Path(cases_dir).glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        case = _parse_case(raw, path)
        _resolve_auto_gold(case)
        cases.append(case)
    if select:
        wanted = {s.strip() for s in select.split(",") if s.strip()}
        cases = [c for c in cases if c.id in wanted or c.category in wanted]
        known = {c.id for c in cases} | {c.category for c in cases}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"未知的用例/类别: {sorted(unknown)}")
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"用例 id 重复: {c.id}")
        seen.add(c.id)
    return cases


def _resolve_auto_gold(case: TestCase) -> None:
    """把 value 为 "auto:<expr>" 的数值金标准解析为具体值。"""
    stats = case.resolve_dataset_stats()
    for gold in case.numeric:
        if isinstance(gold.value, str) and gold.value.startswith("auto:"):
            gold.value = goldref.resolve_gold(gold.value[len("auto:"):], stats)
    # 数值类 value 统一转 float（winner 类已是 str）
    for gold in case.numeric:
        if not isinstance(gold.value, str):
            gold.value = float(gold.value)


def lint_cases(cases_dir: Path) -> list[str]:
    """校验：schema 合法、数据文件存在、auto 金标准可解析、checklist 非空或可评。"""
    problems: list[str] = []
    cases = load_cases(cases_dir)
    for case in cases:
        for rel in (case.dataset_person, case.dataset_ball):
            if not (PROJECT_ROOT / rel).exists():
                problems.append(f"{case.id}: 数据文件不存在 {rel}")
        for gold in case.numeric:
            if isinstance(gold.value, str):
                continue  # winner 类
            if gold.tolerance < 0:
                problems.append(f"{case.id}: {gold.expr} 容差为负")
        if not (case.numeric or case.direction or case.refusal or case.checklist
                or case.category_gold):
            problems.append(
                f"{case.id}: 无任何可评分金标准（numeric/direction/refusal/"
                f"checklist/category_gold 至少一项）")
        if case.refusal and case.category not in ("hallucination", "adversarial"):
            problems.append(
                f"{case.id}: refusal 金标准仅用于 hallucination/adversarial 类用例")
        if case.level not in LEVEL_NAMES:
            problems.append(f"{case.id}: level={case.level} 非法（应为 1-7）")
        for g in case.category_gold:
            if not g.label and not g.aliases:
                problems.append(f"{case.id}: category_gold 缺少 label/aliases")
    return problems
