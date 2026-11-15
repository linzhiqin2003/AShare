"""C7 定点源码变异；导入路径预检、基线/恢复测试、拒绝把收集错误算作杀死。"""

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "test/crypto/test_minute.py"
BASE = "src/ashare/data/crypto/"
MUTATIONS = [
    (
        "timestamp_scale",
        "minute.py",
        "times = stamp // 1000",
        "times = stamp // 1",
        "test_units_and_headers",
    ),
    (
        "unrecorded_gaps",
        "minute.py",
        "rows=frame.height, gaps=gaps",
        "rows=frame.height, gaps=[]",
        "test_gaps_are_full_month_segments",
    ),
    (
        "duplicate_accepted",
        "minute.py",
        "if times.n_unique() != len(times):",
        "if False:",
        "test_duplicates_and_unsorted_raise",
    ),
    (
        "daily_overwrites_monthly",
        "minute.py",
        "if day // DAY in present_days:",
        "if False:",
        "test_daily_only_missing_days_and_resume",
    ),
    (
        "hour_open_last",
        "minute.py",
        'pl.col("open").first()',
        'pl.col("open").last()',
        "test_aggregate_ohlcv_and_incomplete",
    ),
    (
        "hour_close_first",
        "minute.py",
        'pl.col("close").last()',
        'pl.col("close").first()',
        "test_aggregate_ohlcv_and_incomplete",
    ),
    (
        "hour_volume_mean",
        "minute.py",
        "pl.col(f).sum() for f in fields",
        "pl.col(f).mean() for f in fields",
        "test_aggregate_ohlcv_and_incomplete",
    ),
    (
        "verifier_ignores_gaps",
        "verify_minute.py",
        'if part["quality"] != dict(',
        'if False and part["quality"] != dict(',
        "test_independent_verifier_rejects_resigned_damage[gap]",
    ),
    (
        "forget_old_failure",
        "minute.py",
        'failures = [f for f in old.get("failures", []) if not retried(f["key"])]',
        "failures = []",
        "test_failure_is_not_cleared_by_unrelated_sync",
    ),
]


def run(paths):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths, "--tb=short"],
        cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )


def clear_cache(path):
    for cached in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        cached.unlink()


def main():
    imports = []
    for name in (
        "ashare.data.crypto.minute",
        "ashare.data.crypto.verify_minute",
        "ashare.data.crypto.sources",
        "ashare.data.crypto.storage",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError("变异前发现导入了 worktree 外的代码")
        imports.append(str(path.relative_to(ROOT)))
    baseline = run([TEST])
    if baseline.returncode:
        raise RuntimeError("baseline failed: " + baseline.stdout[-2000:])
    results = []
    for name, filename, before, after, test in MUTATIONS:
        path = ROOT / BASE / filename
        original = path.read_bytes()
        content = original.decode()
        if content.count(before) != 1:
            raise RuntimeError("变异锚点不唯一: " + name)
        started = time.perf_counter()
        try:
            mutant = content.replace(before, after).encode()
            compile(mutant, filename, "exec")
            path.write_bytes(mutant)
            clear_cache(path)
            result = run([TEST + "::" + test])
            output = result.stdout + result.stderr
            failed = re.findall(r"^FAILED ([^\s]+)", output, re.M)
            invalid = any(
                s in output
                for s in (
                    "SyntaxError",
                    "ImportError",
                    "ModuleNotFoundError",
                    "ERROR collecting",
                )
            )
            killed = result.returncode == 1 and bool(failed) and not invalid
            results.append(
                dict(
                    name=name,
                    file=BASE + filename,
                    killed=killed,
                    invalid=invalid,
                    failed_tests=failed,
                    exit_code=result.returncode,
                    before_sha256=hashlib.sha256(original).hexdigest(),
                    mutant_sha256=hashlib.sha256(mutant).hexdigest(),
                    seconds=time.perf_counter() - started,
                )
            )
            print(name, "KILLED" if killed else "NOT KILLED", flush=True)
        finally:
            path.write_bytes(original)
            clear_cache(path)
            assert path.read_bytes() == original
    restored = run([TEST])
    report = dict(
        imports=imports,
        baseline_passed=True,
        mutations=results,
        killed=sum(r["killed"] for r in results),
        total=len(results),
        restored_tests_passed=restored.returncode == 0,
    )
    (ROOT / "bench/results/mutations_crypto_minute.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if report["killed"] != len(results) or restored.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
