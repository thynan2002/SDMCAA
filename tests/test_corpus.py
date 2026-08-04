"""数据加载 Harness 测试。

覆盖：
1. PrefixPlayerCorpus 构建（球员/球融合、source_files 记录）
2. 球路事件分析产物
3. 任意帧位置插值
4. 视频接口预留层（发现 / 匹配 / 上下文）
"""

from __future__ import annotations

import pytest

from agents.player.tracker import build_corpus_from_paths, position_at_frame
from agents.video_loader import build_video_context, discover_video_files, match_video

from tests._common import BALL_CSV, PERSON_CSV, PROJECT_ROOT

VIDEO_DIR = PROJECT_ROOT / "TestInput" / "Video"


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


# ── 视频接口预留层 ─────────────────────────────────────────────

def test_video_discovery() -> None:
    videos = discover_video_files(VIDEO_DIR)
    assert {"10s", "12s", "5s"} <= set(videos)
    assert all(v.suffix == ".mp4" for v in videos.values())


def test_video_match() -> None:
    ok = match_video("12s", VIDEO_DIR)
    assert ok.available
    assert ok.video_path is not None and ok.video_path.name == "12s.mp4"

    missing = match_video("no-such-prefix", VIDEO_DIR)
    assert not missing.available
    assert "未找到" in missing.note


def test_video_context() -> None:
    ctx = build_video_context(VIDEO_DIR, ["12s", "no-such-prefix"])
    assert ctx["video_interface_ready"] is True
    assert ctx["video_decode_enabled"] is False
    assert "12s" in ctx["available_videos"]
    assert "no-such-prefix" in ctx["missing_video_prefixes"]


def test_video_context_no_dir() -> None:
    ctx = build_video_context(None, ["12s"])
    assert ctx["available_videos"] == {}
    assert ctx["missing_video_prefixes"] == ["12s"]
