"""在本 worktree 内变异 Rust 换手，确认冻结 main 的全输出对照能抓到。"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def command(args, seconds, log):
    with log.open("w") as output:
        process = subprocess.Popen(
            args, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", type=Path, default=Path(".cache/groupbt/mutation.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if Path.cwd() != root or not args.o.resolve().is_relative_to(root / ".cache"):
        parser.error("必须在本 worktree 运行，原始记录只能写入 .cache")
    cache = root / ".cache/groupbt/mutation"
    cache.mkdir(parents=True, exist_ok=True)
    source = root / "crates/panelbt-core/src/weights.rs"
    original = source.read_text()
    old = "(pnl, sum(&scratch.select))"
    assert original.count(old) == 1
    build = [
        "uv",
        "run",
        "--no-sync",
        "maturin",
        "develop",
        "--release",
        "--manifest-path",
        "packages/panelbt/Cargo.toml",
    ]
    tests = [
        sys.executable,
        "-m",
        "pytest",
        "test/factor/test_groupbt_bitwise.py",
        "-k",
        "all_outputs_bitwise",
        "-xq",
    ]
    started = time.perf_counter()
    assert command(build, 600, cache / "initial-build.log") == 0
    assert command(tests, 300, cache / "baseline.log") == 0
    caught = False
    try:
        source.write_text(original.replace(old, "(pnl, 0.5 * sum(&scratch.select))"))
        assert command(build, 600, cache / "mutant-build.log") == 0
        code = command(tests, 300, cache / "mutant.log")
        caught = (
            code == 1
            and "逐位不一致：turnover/long_short" in (cache / "mutant.log").read_text()
        )
        assert caught, "变异未被目标断言抓到，见 .cache/groupbt/mutation"
    finally:
        source.write_text(original)
        assert command(build, 600, cache / "restore-build.log") == 0
    assert command(tests, 300, cache / "restored.log") == 0
    report = dict(caught=caught, restored=True, elapsed=time.perf_counter() - started)
    args.o.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
