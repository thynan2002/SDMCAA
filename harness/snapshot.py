"""状态快照 — 只读代理 + 规范化序列化 + 字段级对比。

等价性论证（只读代理）：
- snapshot_session 通过 copy.deepcopy 在交互边界取样，绝不持有/返回
  业务对象的可变引用，无任何写回路径；
- 快照内容覆盖 Harness 需要验证的全部状态面：对话历史、聚焦球员、
  记忆（摘要/轮次/缓存键/原始历史）、语料统计摘要。
- 语料（corpus）由输入 CSV 经确定性加载器构建，同输入必然同语料，
  因此只取统计摘要（球员 id/帧数/距离等），不做全量坐标拷贝。

规范化：浮点统一 round(4)，字典按键排序，保证跨进程可文本对比。
"""

from __future__ import annotations

import copy
import json
from typing import Any

FLOAT_PRECISION = 4


# ─────────────────────────────────────────────────────────────────
# 采集
# ─────────────────────────────────────────────────────────────────

def snapshot_session(session: Any) -> dict[str, Any]:
    """对 SessionManager 做只读状态快照（deepcopy 取样，无写回）。"""
    snap: dict[str, Any] = {}

    # ── 会话状态 ──
    snap["focus_jerseys"] = copy.deepcopy(list(session.focus_jerseys))
    snap["conversation_history"] = copy.deepcopy(session.conversation_history)

    # ── 记忆 ──
    memory = session.memory
    snap["memory"] = {
        "context_summary": copy.deepcopy(memory.context_summary),
        "turn_count": memory.turn_count,
        "cached_profiles": sorted(memory.analyzed_profiles.keys()),
        "cached_models": sorted(memory.analyzed_models.keys()),
        "full_history": copy.deepcopy(memory._full_history),
    }

    # ── 语料统计摘要 ──
    corpus = session.corpus
    if corpus is None:
        snap["corpus"] = None
    else:
        players = []
        for p in corpus.sorted_players():
            players.append({
                "track_id": p.track_id,
                "jersey": p.jersey_label,
                "color": p.color,
                "is_goalkeeper": p.is_goalkeeper,
                "frame_count": p.frame_count,
                "total_distance": _r(p.total_distance),
                "avg_speed": _r(p.avg_speed),
                "max_speed": _r(p.max_speed),
            })
        snap["corpus"] = {
            "prefix": corpus.prefix,
            "player_count": corpus.player_count,
            "ball_frame_count": len(corpus.ball_frames),
            "players": players,
        }
    return snap


def _r(value: Any) -> Any:
    return round(value, FLOAT_PRECISION) if isinstance(value, float) else value


# ─────────────────────────────────────────────────────────────────
# 规范化序列化
# ─────────────────────────────────────────────────────────────────

def canonical_json(snap: dict[str, Any]) -> str:
    """规范化 JSON（键排序、统一缩进），保证跨进程文本一致。"""
    return json.dumps(snap, ensure_ascii=False, indent=2, sort_keys=True, default=str)


# ─────────────────────────────────────────────────────────────────
# 字段级对比
# ─────────────────────────────────────────────────────────────────

def compare_snapshots(a: Any, b: Any, path: str = "$") -> list[str]:
    """递归对比两个快照（或任意 JSON 兼容结构），返回字段级差异描述列表。"""
    diffs: list[str] = []

    if type(a) is not type(b):
        # int/float 跨类型容忍（JSON 往返可能出现 1 vs 1.0）
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if float(a) != float(b):
                diffs.append(f"{path}: 数值不同 {a!r} != {b!r}")
            return diffs
        diffs.append(f"{path}: 类型不同 {type(a).__name__} != {type(b).__name__}")
        return diffs

    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                diffs.append(f"{path}.{key}: 仅存在于右侧")
            elif key not in b:
                diffs.append(f"{path}.{key}: 仅存在于左侧")
            else:
                diffs.extend(compare_snapshots(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: 长度不同 {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(compare_snapshots(x, y, f"{path}[{i}]"))
    elif isinstance(a, float):
        if abs(a - b) > 10 ** (-FLOAT_PRECISION):
            diffs.append(f"{path}: 浮点不同 {a!r} != {b!r}")
    else:
        if a != b:
            diffs.append(f"{path}: 值不同 {a!r} != {b!r}")
    return diffs
