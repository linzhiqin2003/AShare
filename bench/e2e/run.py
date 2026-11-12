"""端到端性能剖析：一条命令重跑全部或单个阶段，结果写成一份带时间戳的 JSON。

每个阶段在全新的子进程里运行（多线程库初始化后不 fork；峰值内存互不干扰）。
缓存和中间文件放在 --work（默认仓库内被忽略的 .cache/e2e/），结束时删除大文件。

    python bench/e2e/run.py                      # 全部阶段
    python bench/e2e/run.py --stage factors      # 单个阶段
    python bench/e2e/run.py --stage scaling      # 线程扩展与能效核（不在默认集合里）
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT, default_data, environment  # noqa: E402

STAGES = ("data", "factors", "evaluation", "risk", "portfolio", "backtest", "e2e")
EXTRA = ("scaling",)
DEFAULT_WORK = ROOT / ".cache/e2e"


def child(args):
    os.environ.setdefault("PYTHONHASHSEED", "0")
    module = importlib.import_module("stage_" + args.child)
    ctx = {
        "data": args.data,
        "work": args.work,
        "threads": args.threads,
        "repeat": lambda n: max(1, round(n * args.repeat_scale)),
        "quick": args.quick,
    }
    args.work.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cases = module.run(ctx)
    record = {
        "stage": args.child,
        "seconds": time.perf_counter() - start,
        "cases": cases,
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, default=float))


def cleanup(work):
    """删除缓存目录里的大文件（因子缓存、风险模型缓存、临时面板），保留文本。"""
    for path in sorted(work.rglob("*"), reverse=True):
        if {"handoff", "results"} & set(path.relative_to(work).parts):
            continue
        if path.is_file() and path.suffix not in (".txt", ".json", ".log"):
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", choices=STAGES + EXTRA)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--work",
        type=Path,
        default=Path(os.environ.get("ASHARE_E2E_WORK", DEFAULT_WORK)),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=0, help="0 表示各库的默认线程数")
    parser.add_argument("--repeat-scale", type=float, default=1.0)
    parser.add_argument("--quick", action="store_true", help="缩小规模的冒烟运行")
    parser.add_argument("--timeout", type=int, default=1800, help="单个阶段的秒数上限")
    parser.add_argument("--keep", action="store_true", help="不删除缓存大文件")
    parser.add_argument("--child", choices=STAGES + EXTRA, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data = args.data or default_data()
    args.work = args.work.resolve()
    if args.child:
        return child(args)

    stages = args.stage or list(STAGES)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output or args.work / "results" / f"{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 1,
        "environment": environment(args.threads),
        "options": {
            "stages": stages,
            "threads": args.threads,
            "repeat_scale": args.repeat_scale,
            "quick": args.quick,
        },
        "stages": {},
    }
    total = time.perf_counter()
    for stage in stages:
        part = args.work / f"stage-{stage}.json"
        part.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            stage,
            "--data",
            str(args.data),
            "--work",
            str(args.work),
            "--output",
            str(part),
            "--threads",
            str(args.threads),
            "--repeat-scale",
            str(args.repeat_scale),
            *(["--quick"] if args.quick else []),
        ]
        print(f"[{time.strftime('%H:%M:%S')}] {stage} ...", flush=True)
        start = time.perf_counter()
        try:
            subprocess.run(command, check=True, timeout=args.timeout)
            report["stages"][stage] = json.loads(part.read_text())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            report["stages"][stage] = {"stage": stage, "error": str(exc), "cases": []}
        print(f"    {time.perf_counter() - start:.1f} s", flush=True)
        part.unlink(missing_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=float)
        )
        if not args.keep:
            cleanup(args.work)
    report["total_seconds"] = time.perf_counter() - total
    report["environment"]["load_end"] = list(os.getloadavg())
    output.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    print(f"写入 {output}（{report['total_seconds']:.0f} s）")
    if not args.keep:
        shutil.rmtree(args.work / "handoff", ignore_errors=True)


if __name__ == "__main__":
    main()
