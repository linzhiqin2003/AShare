"""复核返工：工作块漏行和 NumPy 求和次序变异；始终恢复源码与构建。"""

import argparse
import json
import sys
import time
from pathlib import Path

from .mutate import command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", type=Path, default=Path(".cache/groupbt-review/mutations.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if Path.cwd() != root or not args.o.resolve().is_relative_to(root / ".cache"):
        parser.error("必须在本 worktree 运行，原始记录只能写入 .cache")
    folder = args.o.parent / "mutation-logs"
    folder.mkdir(parents=True, exist_ok=True)
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
    assert command(build, 600, folder / "initial-build.log") == 0
    assert command(tests, 300, folder / "baseline.log") == 0
    results = {}
    mutations = [
        (
            "block",
            "crates/panelbt-core/src/group_weights.rs",
            "let end = (begin + block).min(t);",
            "let end = (begin + block - 1).min(t);",
            tests,
            "逐位不一致：group_pnl",
            "panelbt-core",
            "group_weights::tests::every_row_survives_work_block_boundaries",
        ),
        (
            "sum",
            "crates/panelops-core/src/numpy_exact.rs",
            "((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))",
            "r.iter().fold(0.0, |acc, v| acc + v)",
            [sys.executable, "-c", "import panelbt"],
            "ImportError: panelbt: NumPy float64 summation order has changed",
            "panelops-core",
            "numpy_exact::tests::sum_preserves_numpy_tree_and_identity",
        ),
    ]
    for name, relative, old, new, test, message, crate, rust_test in mutations:
        source = root / relative
        original = source.read_text()
        assert original.count(old) == 1
        try:
            source.write_text(original.replace(old, new))
            assert command(build, 600, folder / f"{name}-build.log") == 0
            code = command(test, 300, folder / f"{name}-mutant.log")
            caught = (
                code == 1 and message in (folder / f"{name}-mutant.log").read_text()
            )
            assert caught, f"{name} 变异未被目标断言抓到"
            rust_code = command(
                ["cargo", "test", "-p", crate, rust_test],
                300,
                folder / f"{name}-rust.log",
            )
            assert (
                rust_code == 101
                and f"{rust_test} ... FAILED"
                in (folder / f"{name}-rust.log").read_text()
            )
            results[name] = dict(python_caught=caught, rust_caught=True)
            print(f"{name}: Python 与 Rust 均捕获变异", flush=True)
        finally:
            source.write_text(original)
            assert command(build, 600, folder / f"{name}-restore-build.log") == 0
    # Restore both extensions' source fingerprints after the shared-core mutation.
    panelops_build = [*build[:-1], "packages/panelops/Cargo.toml"]
    assert command(panelops_build, 600, folder / "panelops-build.log") == 0
    assert command(tests, 300, folder / "restored.log") == 0
    results.update(restored=True, elapsed=time.perf_counter() - started)
    args.o.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
