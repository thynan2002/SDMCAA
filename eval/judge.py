"""LLM-as-Judge —— 盲评 + 位置互换 + 结构化 rubric（评审修订 #5/#14）。

- 盲评：三系统答案以 甲/乙/丙 匿名呈现，顺序随机；
- 位置互换：同一组答案正序/逆序各评一次取均值（消除 Judge 位置偏好），
  两次评分差作为 Judge 一致性指标；
- Judge 温度 0，配置记录在案；
- 输出结构化 JSON：accuracy / completeness / reasoning(1-5) /
  fabrication(0|1) / failure_mode；
- 评分结果缓存（judge/{case}.json，键=答案内容+rubric 版本哈希），
  resume 时复用，不重复烧 Judge 费用。

Judge 模型默认与被测系统相同（用户决策），可经 EVAL_JUDGE_* 环境变量
切换任意 OpenAI 兼容 API；报告中标注同源 Judge 的自我偏好风险。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

RUBRIC_VERSION = "v3"

JUDGE_SYSTEM_PROMPT = (
    "你是一个严格、中立的足球数据分析评测裁判。你将看到同一个问题、参考事实与多个匿名答案"
    "（甲/乙/丙）。请对每个答案独立评分，互不影响。只依据参考事实与答案内容评分，"
    "不要因答案长度或风格偏好打分。"
    "评测取向：优先考察答案的**形势判断与策略分析**是否正确（进攻态势、攻防转换、威胁程度、"
    "球员战术角色、区域利用），其次才是具体数值的精确度；不要因为答案没有逐字复述参考数值就"
    "判低分，只要语义一致（数量级、方向、相对关系正确）即可。"
    "输出严格的 JSON（不要包裹代码块，不要输出其他文字），格式：\n"
    '{\n'
    '  "甲": {"accuracy": 0到1小数, "completeness": 0到1小数, "reasoning": 1到5整数,\n'
    '         "fabrication": 0到1小数, "tactical": 0到1小数,\n'
    '         "failure_mode": "无|编造数据|答非所问|数据不足未声明|逻辑矛盾|格式混乱|无依据夸大"},\n'
    '  "乙": {...}, "丙": {...}\n'
    "}\n"
    "评分锚点（rubric v3）：\n"
    "- accuracy（与参考事实的语义一致性）: 完全一致=1.0；主体正确仅细节偏差=0.8；"
    "主体正确有少量偏差=0.6；部分正确/遗漏关键=0.4；大方向错但局部对=0.2；错误或编造=0；"
    "数值类参考事实按语义/数量级比对，方向类按相对关系比对；\n"
    "- completeness（要点覆盖）: 全部子问题与参考要点=1.0；绝大部分=0.8；大部分=0.6；"
    "约一半=0.4；小部分=0.2；遗漏主体=0；\n"
    "- reasoning（策略/形势判断与推理深度 1-5）: 1=只有结论无依据；2=结论+单一依据；"
    "3=有因果链且引用数据支撑形势判断；4=多步推理、区分事实与推断、能权衡攻防/威胁/转换；"
    "5=多步推理+权衡不确定性/反例/明确数据局限，并给出战术层面的洞察；\n"
    "- tactical（战术判断质量 0-1）: 攻防态势、威胁程度、区域利用、阵型站位、事件类型等"
    "战术结论是否正确且有依据=1.0；主体正确=0.8；基本正确有偏差=0.6；含糊或部分错=0.4；"
    "错误或无战术判断=0.2；与事实严重不符=0；该问题无战术维度时给 0.8（中性基准）；\n"
    "- fabrication（编造程度 0-1）: 未编造=0；轻微夸大/添油加醋=0.3；"
    "编造不存在的球员/事件/数值=0.7；大段编造与事实严重不符=1.0；\n"
    "- failure_mode: 标记最主要的失败模式，无则为「无」。"
)


# 数值金标准 expr → 人读标签（含单位，帮助 Judge 理解参考事实）
_EXPR_LABELS = {
    "player_count": "球员人数",
    "duration_s": "片段时长(秒)",
    "ball_zmax": "球最大高度(米)",
    "ball_speed_max": "球峰值速度(米/秒)",
    "ball_speed_avg": "球平均速度(米/秒)",
    "threat_max": "球路最大威胁值(0-1，越接近1越危险)",
    "event_types": "球路事件类型",
    "speed_winner_top": "峰值速度最快球员",
    "distance_winner_top": "累计位移最多球员",
}


def _fmt_numeric(g) -> str:
    """把数值金标准格式化为 Judge 友好的参考事实（带单位）。"""
    expr = g.expr
    value = g.value
    if isinstance(value, str) and expr.startswith(("distance_winner", "speed_winner")):
        dim = "累计位移更多者" if expr.startswith("distance") else "峰值速度更快者"
        return f"{dim}：{value}"
    if expr in _EXPR_LABELS:
        return f"{_EXPR_LABELS[expr]}：{value}"
    m = re.fullmatch(r"ball_([xy])@(\d+)", expr)
    if m:
        axis = "x坐标(米)" if m.group(1) == "x" else "y坐标(米)"
        return f"第{m.group(2)}帧球{axis}：{value}"
    m = re.fullmatch(r"player_([xy])_at_second:(\d+),([\d.]+)", expr)
    if m:
        axis = "x坐标(米)" if m.group(1) == "x" else "y坐标(米)"
        return f"{m.group(2)}号第{m.group(3)}秒{axis}：{value}"
    m = re.fullmatch(r"ball_player_distance_at_second:(\d+),([\d.]+)", expr)
    if m:
        return f"第{m.group(2)}秒球与{m.group(1)}号距离(米)：{value}"
    return f"{expr}={value}"


def _reference_block(case) -> str:
    lines: list[str] = []
    if case.numeric:
        facts = ", ".join(_fmt_numeric(g) for g in case.numeric)
        lines.append(f"参考事实（程序独立核算，答案的语义/数值应与之相符）: {facts}")
    if case.category_gold:
        cats = ", ".join(f"{g.label}(同义词: {'/'.join(g.aliases) or g.label})"
                         for g in case.category_gold)
        lines.append(f"参考分类标签（答案应使用正确类别）: {cats}")
    if case.refusal:
        lines.append(
            "注意：该问题数据不足以回答，正确行为是如实声明数据不足/不存在，"
            "编造具体内容应判 fabrication 高、accuracy 低分。"
        )
    if case.checklist:
        lines.append("参考要点（completeness 评分依据）:")
        for item in case.checklist:
            lines.append(f"  - {item.text}")
    return "\n".join(lines) if lines else "（无特定参考，按通用质量评）"


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


_LABELS = ("甲", "乙", "丙")


class JudgeScores(dict):
    """dict[system, dict]：accuracy / completeness / reasoning / fabrication /
    failure_mode / consistency(|正序-逆序| 均值)。"""


def score_case_with_judge(case, answers_by_system: dict[str, list[str]],
                          judge_cfg, rng: random.Random) -> tuple[JudgeScores, dict]:
    """对单用例做盲评。返回 (每系统评分, 元信息)。"""
    from agents.llm_client import LlmConfig, call_llm, load_llm_config

    base = load_llm_config()
    if base is None:
        raise RuntimeError("Judge 需要 API Key（DEEPSEK_API_KEY 或 EVAL_JUDGE_API_KEY）")
    jcfg = LlmConfig(
        api_key=judge_cfg.api_key or base.api_key,
        model=judge_cfg.model or base.model,
        base_url=judge_cfg.base_url or base.base_url,
        temperature=judge_cfg.temperature,
    )

    systems = sorted(answers_by_system)
    sample_n = max(1, judge_cfg.sample_repeats)
    # 抽取重复样本（全系统取相同下标，公平；以最少的可用重复数为限）
    sampled: dict[str, list[str]] = {}
    for sys_name in systems:
        pool = list(answers_by_system[sys_name])
        idx = sorted(rng.sample(range(len(pool)), min(sample_n, len(pool)))) if pool else []
        sampled[sys_name] = [pool[i] for i in idx] or [None]
    n_reps = min((len(sampled[s]) for s in systems), default=0)

    n_sys = len(systems)
    rounds: list[dict[str, dict]] = []   # 每轮: system -> scores
    orders: list[list[str]] = []

    # 正序（随机排列）与逆序互换
    order1 = systems[:]
    rng.shuffle(order1)
    order2 = list(reversed(order1)) if judge_cfg.order_swap else order1
    for order in ([order1, order2] if judge_cfg.order_swap else [order1]):
        for rep_idx in range(n_reps):
            block_lines = [
                f"【问题】{case.input}", "",
                "【参考事实与要点】", _reference_block(case), "",
            ]
            for li, sys_name in enumerate(order):
                ans = sampled[sys_name][rep_idx] or "（该系统未产生答案）"
                block_lines += [f"——— 答案{_LABELS[li]} ———", ans, ""]
            block_lines.append("请输出 JSON 评分。")
            raw = call_llm(
                JUDGE_SYSTEM_PROMPT, "\n".join(block_lines), config=jcfg, retries=2,
            )
            parsed = _extract_json(raw or "")
            if not parsed:
                continue
            round_scores: dict[str, dict] = {}
            for li, sys_name in enumerate(order):
                item = parsed.get(_LABELS[li]) or {}
                if not isinstance(item, dict):
                    continue
                try:
                    fab = item.get("fabrication", 0)
                    fab_v = float(fab) if not isinstance(fab, bool) else (1.0 if fab else 0.0)
                    round_scores[sys_name] = {
                        "accuracy": max(0.0, min(1.0, float(item.get("accuracy", 0.3)))),
                        "completeness": max(0.0, min(1.0, float(item.get("completeness", 0.3)))),
                        "reasoning": max(1.0, min(5.0, float(item.get("reasoning", 3)))),
                        "fabrication": max(0.0, min(1.0, fab_v)),
                        "tactical": max(0.0, min(1.0, float(item.get("tactical", 0.8)))),
                        "failure_mode": str(item.get("failure_mode", "无")),
                    }
                except (TypeError, ValueError):
                    continue
            if round_scores:
                rounds.append(round_scores)
                orders.append(order)

    # 聚合：跨轮均值；一致性 = 正逆序同名指标差的均值
    result: JudgeScores = JudgeScores()
    for sys_name in systems:
        per = [r[sys_name] for r in rounds if sys_name in r]
        if not per:
            result[sys_name] = {"judge_failed": True}
            continue
        agg: dict[str, float | str] = {}
        for key in ("accuracy", "completeness", "reasoning", "fabrication", "tactical"):
            vals = [p[key] for p in per]
            agg[key] = round(sum(vals) / len(vals), 3)
        modes: dict[str, int] = {}
        for p in per:
            modes[p["failure_mode"]] = modes.get(p["failure_mode"], 0) + 1
        agg["failure_mode"] = max(modes, key=modes.get)
        agg["judge_rounds"] = len(per)
        result[sys_name] = agg
    # 一致性（正逆序差）：按出现顺序配对相邻两轮（同 rep 不同序）
    consistency: dict[str, float] = {}
    if judge_cfg.order_swap and len(rounds) >= 2:
        for i in range(0, len(rounds) - 1, 2):
            r1, r2 = rounds[i], rounds[i + 1]
            for sys_name in systems:
                if sys_name in r1 and sys_name in r2:
                    d = abs(r1[sys_name]["accuracy"] - r2[sys_name]["accuracy"])
                    consistency[sys_name] = round((consistency.get(sys_name, 0) + d) / 2, 3)
    meta = {
        "orders": [list(o) for o in orders], "sample_repeats": sample_n,
        "judge_model": jcfg.model, "judge_temperature": jcfg.temperature,
        "rubric_version": RUBRIC_VERSION, "consistency": consistency,
    }
    return result, meta


# ── 缓存 ────────────────────────────────────────────────────

def cache_key(case, answers_by_system: dict[str, list[str]]) -> str:
    h = hashlib.sha256()
    h.update(case.id.encode("utf-8"))
    h.update(RUBRIC_VERSION.encode())
    for sys_name in sorted(answers_by_system):
        h.update(sys_name.encode())
        for ans in answers_by_system[sys_name]:
            h.update(hashlib.sha256((ans or "").encode("utf-8")).hexdigest().encode())
    return h.hexdigest()


def load_cached(judge_dir: Path, case_id: str, key: str) -> JudgeScores | None:
    path = judge_dir / f"{case_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("key") == key:
            scores = JudgeScores(data.get("scores", {}))
            return scores
    except (json.JSONDecodeError, OSError):
        pass
    return None


def save_cached(judge_dir: Path, case_id: str, key: str,
                scores: JudgeScores, meta: dict) -> None:
    judge_dir.mkdir(parents=True, exist_ok=True)
    (judge_dir / f"{case_id}.json").write_text(
        json.dumps({"key": key, "scores": dict(scores), "meta": meta},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
