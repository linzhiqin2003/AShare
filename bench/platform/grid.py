"""串行复测同时变体数 × 算子线程数；支持 .cache 中的旧版包作为配对基准。"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 3, 4, 0])
    p.add_argument("--ops-threads", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument(
        "--scenarios", nargs="+", choices=("four", "eight"), default=["four", "eight"]
    )
    args = p.parse_args()
    if not args.output.resolve().is_relative_to(Path(".cache").resolve()):
        p.error("结果只能写入本 worktree 的 .cache")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    versions = ["current"] if args.reference is None else ["reference", "current"]
    for concurrency in args.concurrency:
        for threads in args.ops_threads:
            if concurrency == 0 and threads != 8:
                continue
            for scenario in args.scenarios:
                for version in versions:
                    folder = (
                        args.output / f"{version}-{scenario}-c{concurrency}-t{threads}"
                    )
                    metrics = folder / "metrics.json"
                    env = dict(os.environ, ASHARE_PLATFORM_OPS_THREADS=str(threads))
                    if version == "reference":
                        prefix = [
                            str((args.reference / "src").resolve()),
                            str(args.reference.resolve()),
                        ]
                        env["PYTHONPATH"] = os.pathsep.join(
                            prefix + [env.get("PYTHONPATH", "")]
                        )
                    windows = "[20, 60]" if scenario == "four" else "[5, 20, 60, 120]"
                    if not metrics.exists():
                        with (args.output / f"{folder.name}.log").open("w") as log:
                            subprocess.run(
                                [
                                    sys.executable,
                                    "-m",
                                    "bench.platform.speed",
                                    "--output",
                                    str(folder),
                                    "--modes",
                                    "threads",
                                    "--attempts",
                                    str(args.attempts),
                                    "--concurrency",
                                    str(concurrency),
                                    "--source",
                                    f"vol = ts_std(returns, {windows})\nalpha = -vol",
                                ],
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=180,
                            )
                    measurements = json.loads(metrics.read_text())
                    assert len(measurements) == args.attempts, "运行中断，请换目录复测"
                    warm = [r for r in measurements if r["attempt"] == "warm"]
                    row = dict(
                        version=version,
                        scenario=scenario,
                        concurrency=concurrency,
                        ops_threads=threads,
                        wall_seconds=statistics.median(r["wall_seconds"] for r in warm),
                        worker_seconds=statistics.median(
                            r["worker_seconds"] for r in warm
                        ),
                        evaluation_seconds=statistics.median(
                            r["evaluation_seconds"] for r in warm
                        ),
                        coordinator_peak_bytes=max(
                            r["coordinator_peak_bytes"] for r in warm
                        ),
                        peak_tree_bytes=max(
                            (r["peak_tree_bytes"] or 0 for r in warm), default=0
                        )
                        or None,
                    )
                    rows.append(row)
                    (args.output / "grid.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
