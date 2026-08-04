"""反事实轨迹生成端到端测试 — 命令行入口（Harness 规范化后）。

完整用例已迁移至 tests/test_counterfactual_sim.py（可用 `pytest` 运行）；
本脚本保留旧契约：`python tools/test_counterfactual_sim.py`

- 场景 A/B/C/D + 专项验证 + 导出文件验证
- 输出 [OK]/[FAIL] 明细，退出码 0=全部通过 / 1=存在失败项
- 导出文件写入项目 Output/ 目录
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_counterfactual_sim import run_standalone  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "Output"


def main() -> int:
    return run_standalone(out_dir=OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
