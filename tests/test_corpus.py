"""数据加载 Harness 测试。

覆盖：
1. PrefixPlayerCorpus 构建（球员/球融合、source_files 记录）
2. 球路事件分析产物
3. 任意帧位置插值
"""

from __future__ import annotations

from agents.player.tracker import build_corpus_from_paths, position_at_frame

from tests._common import BALL_CSV, PERSON_CSV, PROJECT_ROOT


# ── PrefixPlayerCorpus 构建 ─────────────────────────────────────

def test_corpus_builds(corpus) -> None:
    assert corpus.player_count >= 2
    assert corpus.ball_frames
    assert min(corpus.ball_frames) >= 0
    assert corpus.source_files == (str(PERSON_CSV), str(BALL_CSV))


def test_corpus_has_ball_events(corpus) -> None:
    events = corpus.ball_analysis.events
    assert events, "球路事件分析为空"


def test_position_interpolation(corpus) -> None:
    """任意帧线性插值：首帧与末帧之间必可取得坐标。"""
    p = next(iter(corpus.players.values()))
    assert len(p.frames) >= 2
    mid = (p.frames[0] + p.frames[-1]) // 2
    pos = position_at_frame(p, mid)
    assert pos is not None
    assert len(pos) == 2


def test_corpus_build_other_datasets() -> None:
    """同一 Harness 应能加载其他测试数据集（格式兼容）。"""
    for name in ("5s", "7s", "10s"):
        person = PROJECT_ROOT / "TestInput" / "Files" / f"{name}-person.csv"
        ball = PROJECT_ROOT / "TestInput" / "Files" / f"{name}-ball.csv"
        if not person.exists() or not ball.exists():
            continue
        c = build_corpus_from_paths(person, ball)
        assert c.player_count >= 2
        assert c.ball_frames
