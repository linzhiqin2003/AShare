"""按独立进程顺序测量，避免各阶段互相抢 CPU；原始结果只放 .cache。"""

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=("real", "synthetic"), default="real")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    folder = root / ".cache/groupbt/final"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    # Paired, alternating fresh processes. Every process includes a cold call
    # and two subsequent calls; report both, without silently discarding either.
    for round_ in range(3):
        for backend in ("main", "simulation", "current"):
            jobs.append(
                (f"{backend}-{round_}", ["--backend", backend, "--repeat", "3"])
            )
    for backend in ("main", "simulation", "current"):
        jobs.append((f"{backend}-profile", ["--backend", backend, "--profile"]))
    for backend in ("main", "current"):
        jobs.append((f"{backend}-trace", ["--backend", backend, "--trace"]))
    for threads in (1, 2, 4, 8):
        jobs.append(
            (
                f"threads-{threads}",
                ["--threads", str(threads), "--jobs", "8", "--repeat", "3"],
            )
        )
    for threads in (1, 2, 4):
        jobs.append(
            (
                f"platform-{threads}",
                ["--threads", str(threads), "--platform", "--repeat", "3"],
            )
        )
    for name, options in jobs:
        print(f"测量 {name}", flush=True)
        output = folder / f"{name}.json"
        command = [
            sys.executable,
            "-m",
            "bench.groupbt.measure",
            "--data",
            args.data,
            *options,
            "-o",
            str(output),
        ]
        with output.with_suffix(".log").open("w") as log:
            child = subprocess.Popen(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            if code:
                raise RuntimeError(f"{name} 失败，见 .cache/groupbt/final/{name}.log")
    print("测量完成", flush=True)


if __name__ == "__main__":
    main()
