"""C3 Python 真实源码变异；先验证本 worktree 导入，语法/导入错误不算杀死。"""

from __future__ import annotations

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
SCHEDULE = "src/ashare/backtest/crypto_schedule.py"
RUN = "src/ashare/backtest/crypto_run.py"
PROFILE = "packages/panelbt/python/panelbt/markets/binance.py"
TESTS = [
    "test/crypto/test_backtest.py",
    "test/crypto/test_binance_profile.py",
    "test/crypto/test_c3_data_fixes.py",
    "test/crypto/test_c3_review.py",
    "test/crypto/test_c3_review_r2.py",
]
MUTATIONS = [
    (
        "daily_double_shift",
        SCHEDULE,
        "times = days // DAY * DAY + offset + DAY",
        "times = days // DAY * DAY + offset + 2 * DAY",
        TESTS[0],
    ),
    (
        "daily_carry",
        RUN,
        "replace(m.config, carry_blocked=False)",
        "replace(m.config, carry_blocked=True)",
        TESTS[0],
    ),
    (
        "known_at_early",
        RUN,
        "asof(m.events, int(t))",
        "asof(m.events, int(t + HOUR))",
        TESTS[0],
    ),
    (
        "delist_buy_override",
        RUN,
        "target[exits] = 0.0",
        "target[exits] = target[exits]",
        TESTS[0],
    ),
    (
        "ranking_window",
        SCHEDULE,
        '(pl.col("time") >= at - window)',
        '(pl.col("time") >= at - 2 * window)',
        TESTS[0],
    ),
    (
        "fee_transition",
        PROFILE,
        "taker = np.where(later, 0.0005, 0.0004)",
        "taker = np.where(later, 0.0004, 0.0004)",
        TESTS[1],
    ),
    (
        "assumption_gate",
        PROFILE,
        'elif missing.any() and policy == "previous" and records:',
        "elif missing.any() and records:",
        TESTS[1],
    ),
    (
        "roll_rounding",
        SCHEDULE,
        "return (cutoff - offset) // DAY * DAY + offset",
        "return (cutoff - offset + DAY - 1) // DAY * DAY + offset",
        TESTS[0],
    ),
    (
        "tier_validity",
        RUN,
        'if diag["tier_cap_exceeded"] and not allow_tier_exceeded:',
        "if False and not allow_tier_exceeded:",
        TESTS[0],
    ),
    (
        "migration_ratio",
        "src/ashare/backtest/crypto_migration.py",
        'ratio *= e["ratio"]',
        "ratio *= 1.",
        TESTS[0],
    ),
    (
        "partial_halt_evidence",
        "src/ashare/data/crypto/panel.py",
        "inferred_halt = (\n            bar is None",
        "inferred_halt = (\n            False and bar is None",
        TESTS[2],
    ),
    (
        "cms_resume_page",
        "src/ashare/data/crypto/announcements.py",
        'progress["next_pages"].get(str(catalog), 1)',
        "1",
        TESTS[2],
    ),
]

MUTATIONS += [
    (
        "r1_p0_1_migration_event_scope",
        "src/ashare/backtest/crypto_run.py",
        'execution_names = set(m.arrays["execution_symbol"].ravel())',
        "execution_names = set(m.symbols)",
        "test/crypto/test_c3_review.py",
    ),
    (
        "r1_p0_2_auxiliary_halt",
        "src/ashare/data/crypto/panel.py",
        '        inferred_halt = (\n            bar is None\n            and trade_partial_before\n            and not (\n                futures\n                and mark is not None\n                and mark.get("time_quality", "normal") == "normal"\n            )\n        )\n',
        "        inferred_halt = missing\n",
        "test/crypto/test_c3_review.py",
    ),
    (
        "r1_p0_3_expiry_price_overwrite",
        "src/ashare/backtest/crypto.py",
        'if not a["expires"][i, j]:',
        "if True:",
        "test/crypto/test_c3_review.py",
    ),
    (
        "r1_p0_4_fetch_before_pending",
        "src/ashare/data/crypto/announcements.py",
        "                count, reviews = _drain_article(store, ledger, a)\n                imported += count\n                pending += reviews\n                response = client.json(",
        "                count, reviews = (0, 0)\n                imported += count\n                pending += reviews\n                response = client.json(",
        "test/crypto/test_c3_review.py",
    ),
    (
        "r1_p1_5_left_boundary_delivery_fee",
        "src/ashare/backtest/crypto.py",
        '(times + HOUR).astype("datetime64[ms]")',
        'times.astype("datetime64[ms]")',
        "test/crypto/test_c3_review.py",
    ),
    (
        "r1_p1_6_raw_daily_signal",
        "src/ashare/data/crypto/dataset.py",
        "frame = signal_frame(self.read(**kwargs), invalid_prices=invalid_prices)",
        "frame = self.read(**kwargs)",
        "test/crypto/test_c3_review.py",
    ),
    (
        "r1_p1_7_double_clearance",
        "bench/crypto_examples.py",
        '            "other_fee",',
        '            "other_fee", "clearance_fee",',
        "test/crypto/test_c3_review.py",
    ),
    (
        "r1_p1_8_circular_funding_oracle",
        "bench/crypto_examples.py",
        'expected_by_hour[row] = contracts * unit * float(price["markPrice"]) * x["rate"]',
        'expected_by_hour[row] = contracts * m.arrays["funding"][row, 0]',
        "test/crypto/test_c3_review.py",
    ),
    (
        "r1_p2_9_missing_rounding_diagnostic",
        "packages/panelbt/python/panelbt/markets/binance.py",
        "approximation=bool(effective != applied),",
        "approximation=False,",
        "test/crypto/test_c3_review.py",
    ),
]


MUTATIONS += [
    (
        "r2_p1_keep_delisted_pending_buy",
        RUN,
        "or np.any(exits[state.pending_cols])",
        "or False",
        TESTS[4],
    ),
    (
        "r2_p2_mark_before_gap_refutes_build",
        "src/ashare/data/crypto/panel.py",
        "            bar is None\n            and futures\n            and mark is not None\n",
        "            futures\n            and mark is not None\n",
        TESTS[4],
    ),
    (
        "r2_p2_mark_before_gap_refutes_verify",
        "src/ashare/data/crypto/verify.py",
        'normal_mark_times[mark_cursor - 1] > preceding["time"]',
        'normal_mark_times[mark_cursor - 1] >= preceding["time"]',
        TESTS[4],
    ),
]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clear_cache(path):
    for pyc in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        pyc.unlink()


def run_tests(paths):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths, "--tb=short"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )


def check_imports():
    sys.path.insert(0, str(ROOT))
    paths = []
    for name in (
        "panelbt",
        "panelbt._core",
        "panelbt.markets.binance",
        "ashare.backtest.crypto",
        "ashare.backtest.crypto_run",
        "ashare.backtest.crypto_schedule",
        "ashare.backtest.crypto_migration",
        "ashare.data.crypto.announcements",
        "ashare.data.crypto.dataset",
        "ashare.data.crypto.panel",
        "ashare.data.crypto.verify",
        "bench.crypto_examples",
    ):
        module = importlib.import_module(name)
        p = Path(module.__file__).resolve()
        if not p.is_relative_to(ROOT):
            raise RuntimeError("测试导入了 worktree 外的代码")
        paths.append(str(p.relative_to(ROOT)))
    return paths


def main():
    imports = check_imports()
    baseline = run_tests(TESTS)
    if baseline.returncode:
        raise RuntimeError("baseline failed: " + baseline.stdout[-3000:])
    results = []
    for name, relative, before, after, test in MUTATIONS:
        path = ROOT / relative
        original = path.read_text()
        sha = fingerprint(path)
        if original.count(before) != 1:
            raise RuntimeError("变异锚点不唯一: " + name)
        started = time.monotonic()
        try:
            path.write_text(original.replace(before, after))
            clear_cache(path)
            changed = fingerprint(path)
            assert sha != changed
            proc = run_tests([test])
            output = proc.stdout + proc.stderr
            failures = re.findall(r"^FAILED ([^\s]+)", output, re.M)
            invalid = any(
                x in output
                for x in (
                    "SyntaxError",
                    "ImportError",
                    "ModuleNotFoundError",
                    "ERROR collecting",
                )
            )
            killed = proc.returncode == 1 and bool(failures) and not invalid
            results.append(
                dict(
                    name=name,
                    file=relative,
                    before_sha256=sha,
                    mutant_sha256=changed,
                    killed=killed,
                    failed_tests=failures,
                    exit_code=proc.returncode,
                    invalid=invalid,
                    seconds=round(time.monotonic() - started, 3),
                )
            )
            print(name, "KILLED" if killed else "NOT KILLED", flush=True)
        finally:
            path.write_text(original)
            clear_cache(path)
            assert fingerprint(path) == sha
    restored = run_tests(TESTS)
    report = dict(
        imports=imports,
        baseline_passed=True,
        mutations=results,
        killed=sum(r["killed"] for r in results),
        total=len(results),
        restored_tests_passed=restored.returncode == 0,
    )
    output = ROOT / "bench/results/mutations_crypto_backtest.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if report["killed"] != len(results) or restored.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
