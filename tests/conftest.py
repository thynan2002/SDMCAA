"""pytest 共享夹具与路径引导。

- 将项目根目录插入 sys.path（兼容未安装包环境）
- 提供 12s 测试数据 corpus、mock LLM 等会话级/函数级夹具
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests._common import (  # noqa: E402
    BALL_CSV,
    FRAME,
    PERSON_CSV,
    make_bad_llm,
    make_llm_mock,
)


@pytest.fixture(scope="session")
def corpus():
    """12s 测试数据构建的 PrefixPlayerCorpus（全会话复用一次）。"""
    from agents.player.tracker import build_corpus_from_paths

    return build_corpus_from_paths(PERSON_CSV, BALL_CSV)


@pytest.fixture
def llm_mock(monkeypatch):
    """将 llm_brain.call_llm 替换为可控 mock。

    Returns:
        (调用记录列表, mock 函数)；脚本请求返回固定剧本，
        决策请求返回"传给第一个队友"。
    """
    from agents.professional.simulation import llm_brain

    calls, fake = make_llm_mock()
    monkeypatch.setattr(llm_brain, "call_llm", fake)
    return calls, fake


@pytest.fixture
def bad_llm(monkeypatch):
    """将 llm_brain.call_llm 替换为始终返回非法内容的函数。"""
    from agents.professional.simulation import llm_brain

    fake = make_bad_llm()
    monkeypatch.setattr(llm_brain, "call_llm", fake)
    return fake


@pytest.fixture
def frame() -> int:
    return FRAME
