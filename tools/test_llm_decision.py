"""LLM 决策引擎接线测试 — 命令行入口（Harness 规范化后）。

完整用例已迁移至 tests/test_llm_decision.py（可用 `pytest` 运行）；
本脚本保留旧契约：`python tools/test_llm_decision.py`

- 用例 1: mock LLM 全链路接线
- 用例 2: LLM 非法输出 → 回退启发式
- 用例 3: 无 LLM 引擎 → 纯启发式
- 用例 4: 环境有 DEEPSEEK_API_KEY 时执行真实 API 冒烟
- 输出 [OK]/[FAIL] 明细，退出码 0=全部通过 / 1=存在失败项
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_llm_decision import run_standalone  # noqa: E402


def main() -> int:
    return run_standalone()


if __name__ == "__main__":
    sys.exit(main())
