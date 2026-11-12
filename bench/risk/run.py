"""完整规模重跑到被忽略目录，与冻结 JSON 逐项比较耗时。"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from compare_kernels import run as risk_run
from consolidate import assemble
from factor_kernels import run as factor_run
from gram_benchmark import run as gram_run
from pipeline_benchmark import run as pipeline_run
from rebuild_benchmark import run as rebuild_run
from report import render
from review_comparisons import run as review_run
from structure_benchmark import run as structure_run
from varying_regression import run as varying_run
from window_benchmark import run as window_run

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "bench/risk/results/baseline.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / ".cache/risk-bench" / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, fn in (
        ("kernels", risk_run),
        ("factor-kernels", factor_run),
        ("gram", gram_run),
        ("windows", window_run),
        ("pipeline", pipeline_run),
        ("varying-regression", varying_run),
        ("rebuild", rebuild_run),
        ("structure", structure_run),
        ("review-comparisons", review_run),
    ):
        fn(output / (name + ".json"))
    for name, extra in (
        ("optimizers", []),
        ("optimizers-auto", ["--methods", "auto", "--no-parallel"]),
    ):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench/risk/optimizer_benchmark.py"),
                "--output",
                str(output / (name + ".json")),
                *extra,
            ],
            check=True,
            cwd=ROOT,
        )
    subprocess.run(
        [sys.executable, "-m", "ashare.risk.validate", "--output", str(output)],
        check=True,
        cwd=ROOT,
    )
    before = json.loads(args.baseline.read_text())
    after = assemble(output)
    comparisons = []

    def compare(old, new, path=""):
        if isinstance(old, dict) and isinstance(new, dict):
            for key in old.keys() & new.keys():
                compare(old[key], new[key], path + "/" + key)
        elif (
            isinstance(old, (int, float)) and isinstance(new, (int, float)) and old > 0
        ):
            if path.endswith(("/median", "_seconds", "/seconds")):
                comparisons.append(
                    dict(
                        case=path,
                        baseline_seconds=old,
                        new_seconds=new,
                        ratio=new / old,
                    )
                )

    compare(before, after)
    (output / "comparison.json").write_text(json.dumps(comparisons, indent=2))
    (output / "baseline.json").write_text(json.dumps(after, indent=2))
    render(output / "baseline.json", output / "benchmark-report.md")
    print(output / "comparison.json")


if __name__ == "__main__":
    main()
