"""子进程作业入口 —— 由 engine 经 `python -m eval.exec_job` 拉起。

每次作业 = (system, case_id, repeat) 在全新进程内执行：
- 彻底隔离全局状态（active corpus / llm dispatcher / ContextVar）；
- stdout/stderr 重定向到作业目录 stdout.log，不打断父进程；
- 产出 result.json（无论成败）；进程退出码 0 表示作业框架正常运转
  （被测系统失败以 result.error 记录，属于有效评测样本）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="eval.exec_job")
    parser.add_argument("--system", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cases-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"

    # 被测系统的控制台输出落盘（多智能体会打印进度）
    log = open(log_path, "w", encoding="utf-8", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = log

    try:
        from dotenv import load_dotenv

        load_dotenv()

        from eval.cases import load_cases
        from eval.config import CASES_DIR, EvalConfig
        from eval.runners import run_job

        cases = load_cases(Path(args.cases_dir) if args.cases_dir else CASES_DIR)
        case = next(c for c in cases if c.id == args.case_id)
        cfg = EvalConfig.load()
        result = run_job(args.system, {
            "id": case.id, "category": case.category, "input": case.input,
            "dataset": {"person": case.dataset_person, "ball": case.dataset_ball},
        }, out_dir, cfg)
        result["repeat"] = args.repeat
        (out_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
        )
        return 0
    except Exception as exc:  # 作业框架级失败
        import traceback

        err = {
            "system": args.system, "case_id": args.case_id, "repeat": args.repeat,
            "answer": None, "error": f"job_crash: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "out_dir": str(out_dir),
        }
        try:
            (out_dir / "result.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
            )
        except Exception:
            pass
        return 1
    finally:
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
