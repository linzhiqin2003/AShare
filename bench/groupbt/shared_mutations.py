"""共享键、输入所有权、归约和批量调仓变异；结束后恢复源码与扩展。"""

import argparse
import json
import sys
from pathlib import Path

from .mutate import command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", type=Path, default=Path(".cache/groupbt-fusion/mutations.json")
    )
    args = parser.parse_args()
    if not args.o.resolve().is_relative_to(Path(".cache").resolve()):
        parser.error("记录只能写入本 worktree 的 .cache")
    folder = args.o.parent / "mutation-logs"
    folder.mkdir(parents=True, exist_ok=True)
    shared = "test/factor/test_groupbt_shared.py"
    analysis = "test/platform/test_analysis.py"
    api = "src/ashare/factor/groupbt/api.py"
    native = "crates/panelbt-core/src/group_weights.rs"
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
    cases = [
        (
            "readonly-memoryview",
            api,
            "base = base.obj",
            "return x",
            shared,
            "readonly_memoryview",
            False,
        ),
        (
            "readonly-alias",
            api,
            "if x.flags.c_contiguous and not x.flags.writeable:",
            "if x.flags.c_contiguous and not x.flags.writeable:\n        return x",
            shared,
            "readonly_flag_does_not",
            False,
        ),
        (
            "pool-key",
            api,
            '("pool", window_key, name, pool_start)',
            '("pool", window_key, pool_start)',
            shared,
            "shared_pools_windows",
            False,
        ),
        (
            "window-key",
            api,
            '("logcap", window_key)',
            '("logcap",)',
            shared,
            "shared_pools_windows",
            False,
        ),
        (
            "expression-window",
            "src/ashare/platform/runner.py",
            "digest, alpha = next(evaluated)",
            "digest, alpha = next(evaluated)\n            if not prepared:\n                first_alpha = alpha\n            alpha = first_alpha",
            analysis,
            "config_variants_share",
            False,
        ),
        (
            "exposure-pool",
            native,
            "cache.means[s * pools + pool]",
            "cache.means[s * pools]",
            shared,
            "shared_pools_windows",
            True,
        ),
        (
            "ledger-schedule",
            native,
            "let scheduled = s.is_multiple_of(config.rebalance);",
            "let scheduled = s.is_multiple_of(variants[0].rebalance);",
            shared,
            "one_native_batch",
            True,
        ),
    ]
    result = {}
    tests = [sys.executable, "-m", "pytest", shared, analysis, "-q"]
    assert command(tests, 180, folder / "baseline.log") == 0
    for name, file, old, new, test, selector, rust in cases:
        path = Path(file)
        original = path.read_text()
        assert original.count(old) == 1, name
        try:
            path.write_text(original.replace(old, new))
            if rust:
                assert command(build, 600, folder / f"{name}-build.log") == 0
            code = command(
                [sys.executable, "-m", "pytest", test, "-k", selector, "-xq"],
                180,
                folder / f"{name}.log",
            )
            text = (folder / f"{name}.log").read_text()
            assert code == 1 and "FAILED" in text and "ERROR collecting" not in text, (
                name
            )
            result[name] = {"caught": True}
            print(f"{name}: 变异被测试捕获", flush=True)
        finally:
            path.write_text(original)
            if rust:
                assert command(build, 600, folder / f"{name}-restore.log") == 0
    path = Path("crates/panelops-core/src/numpy_exact.rs")
    original = path.read_text()
    old = "sum(&self.temp)\n    }\n}"
    assert original.count(old) == 1
    try:
        path.write_text(
            original.replace(
                old,
                "sum(&self.temp.iter().copied().filter(|v| *v != 0.0).collect::<Vec<_>>())\n    }\n}",
            )
        )
        code = command(
            [
                "cargo",
                "test",
                "-p",
                "panelops-core",
                "shared_means_retain_full_width_reduction",
            ],
            180,
            folder / "zero-compaction.log",
        )
        assert (
            code == 101
            and "shared_means_retain_full_width_reduction ... FAILED"
            in (folder / "zero-compaction.log").read_text()
        )
        result["zero-compaction"] = {"caught": True}
    finally:
        path.write_text(original)
    assert command(tests, 180, folder / "restored.log") == 0
    assert (
        command(
            [
                "cargo",
                "test",
                "-p",
                "panelops-core",
                "shared_means_retain_full_width_reduction",
            ],
            180,
            folder / "rust-restored.log",
        )
        == 0
    )
    result["restored"] = True
    args.o.write_text(json.dumps(result, indent=2) + "\n")
    print("全部变异被捕获；源码与扩展已恢复", flush=True)


if __name__ == "__main__":
    main()
