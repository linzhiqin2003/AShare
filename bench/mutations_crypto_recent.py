"""C5 真实源码变异；检查本 worktree 导入，语法/收集错误不算杀死。"""

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
BASE = "src/ashare/data/crypto/"
TEST = "test/crypto/test_recent.py"
SYNC_TEST = "test/crypto/test_recent_sync.py"
REVIEW_TEST = "test/crypto/test_c5_review_r1.py"
REVIEW2_TEST = "test/crypto/test_c5_review_r2.py"
MUTATIONS = [
    (
        "spacing",
        "recent.py",
        "if t - previous != step or t % step:",
        "if t % step:",
        TEST + "::test_stop_at_first_untrusted_settlement[missing_middle-24]",
    ),
    (
        "default_only_8h",
        "recent.py",
        '(not entries and last["interval"] == 8)',
        "(not entries)",
        TEST + "::test_stop_at_first_untrusted_settlement[info_missing_4h-19]",
    ),
    (
        "missing_mark",
        "recent.py",
        "if not math.isfinite(price) or price <= 0 or not math.isfinite(rate):",
        "if not math.isfinite(rate):",
        TEST + "::test_stop_at_first_untrusted_settlement[missing_mark-24]",
    ),
    (
        "tail_shrink",
        "recent.py",
        "if failure is None and previous + step <= active_cap:",
        "if False:",
        TEST + "::test_stop_at_first_untrusted_settlement[missing_last-40]",
    ),
    (
        "schedule_tail_guard",
        "funding.py",
        "if expected_last <= active_end:",
        "if False:",
        "test/crypto/test_panel.py::test_funding_missing_first_middle_last_is_error[2]",
    ),
    (
        "economic_difference",
        "reconcile.py",
        "if old_values.get(t) != new_values.get(t):",
        "if False:",
        TEST + "::test_archive_reconciliation_keeps_old_and_does_not_block[rate]",
    ),
    (
        "default_closed",
        "build.py",
        "allow_inferred_funding=False",
        "allow_inferred_funding=True",
        TEST + "::test_strict_unchanged_default_and_separate_latest",
    ),
    (
        "reader_default_closed",
        "dataset.py",
        "def open(root, version=None, *, allow_inferred_funding=False):",
        "def open(root, version=None, *, allow_inferred_funding=True):",
        TEST + "::test_strict_unchanged_default_and_separate_latest",
    ),
    (
        "independent_terminal",
        "verify_recent.py",
        "if end != min(caps):",
        "if False:",
        TEST + "::test_verify_terminal_formula_even_without_new_settlement",
    ),
    (
        "archive_replaces_cached_rate",
        "funding.py",
        'and not record.get("archive_rate_override")',
        "and True",
        TEST + "::test_archive_rate_revision_overrides_cached_rest_rate",
    ),
    (
        "daily_non_overlap",
        "archive.py",
        "if day // DAY in present_days:",
        "if False:",
        SYNC_TEST + "::test_daily_only_missing_days_and_no_rest",
    ),
    (
        "daily_checksum",
        "sources.py",
        "if hashlib.sha256(payload).hexdigest() != parts[0]:",
        "if False:",
        SYNC_TEST + "::test_daily_bad_checksum_raises",
    ),
    (
        "ended_funding_caps_panel",
        "recent.py",
        "if strict_end < stops[name]:",
        "if True:",
        REVIEW_TEST + "::test_ended_column_does_not_cap_panel[False-settlement]",
    ),
    (
        "ended_prices_cap_panel",
        "recent.py",
        "if max(archived) < stops[name]:",
        "if True:",
        REVIEW_TEST + "::test_ended_column_does_not_cap_panel[True-delist]",
    ),
    (
        "archive_publication_uses_stale_rest",
        "sync.py",
        "refetch=new_archives if previous and not refresh else (),",
        "refetch=(),",
        REVIEW_TEST + "::test_sync_refetches_prices_when_archive_published",
    ),
    (
        "archive_tail_shrinks_audit",
        "reconcile.py",
        "if coverage is not None",
        "if False",
        REVIEW_TEST + "::test_reconciliation_audits_declared_month_tail",
    ),
    (
        "archive_refetch_clipped_to_new_start",
        "sync.py",
        'lo = window["start"]',
        'lo = max(start, window["start"])',
        REVIEW_TEST + "::test_sync_archive_refetch_uses_full_declared_month",
    ),
    (
        "rest_source_not_checked",
        "verify_raw.py",
        "if economic(original) != economic(",
        "if False and economic(original) != economic(",
        REVIEW_TEST + "::test_verify_rest_input_against_frozen_response[markPrice-200]",
    ),
    (
        "verifier_accepts_shortened_strict",
        "verify_recent.py",
        "if end != min(caps):",
        "if end > min(caps):",
        REVIEW_TEST + "::test_verifier_rejects_ended_column_truncation[False]",
    ),
    (
        "short_horizon_skips_rest",
        "recent.py",
        "        candidates = sorted(",
        "        if strict_end >= stops[name]:\n            continue\n        candidates = sorted(",
        REVIEW2_TEST
        + "::test_rest_inside_short_horizon_stops_extension[confirmed-request_end]",
    ),
    (
        "verifier_short_horizon_skips_rest",
        "verify_recent.py",
        "        if opted_in:",
        "        if opted_in and strict_bound < stops[name]:",
        REVIEW2_TEST
        + "::test_verifier_rejects_unchecked_rest_inside_short_horizon[confirmed-known_exit]",
    ),
    (
        "verifier_unconfirmed_info_skips_rest",
        "verify_recent.py",
        "        if opted_in:",
        "        if authorized:",
        REVIEW2_TEST
        + "::test_verifier_rejects_unchecked_rest_inside_short_horizon[missing-request_end]",
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
        "ashare.data.crypto.build",
        "ashare.data.crypto.recent",
        "ashare.data.crypto.verify_recent",
        "ashare.data.crypto.reconcile",
        "ashare.data.crypto.sync",
        "panelbt",
        "panelbt._core",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError("变异前发现导入了 worktree 外的代码")
        imports.append(str(path.relative_to(ROOT)))
    baseline = run([TEST, SYNC_TEST, REVIEW_TEST, REVIEW2_TEST])
    if baseline.returncode:
        raise RuntimeError("baseline failed: " + baseline.stdout[-3000:])
    results = []
    for name, relative, before, after, test in MUTATIONS:
        path = ROOT / BASE / relative
        original = path.read_bytes()
        content = original.decode()
        if content.count(before) != 1:
            raise RuntimeError("变异锚点不唯一: " + name)
        started = time.monotonic()
        try:
            mutant = content.replace(before, after).encode()
            path.write_bytes(mutant)
            clear_cache(path)
            result = run([test])
            output = result.stdout + result.stderr
            failed = re.findall(r"^FAILED ([^\s]+)", output, re.M)
            invalid = any(
                x in output
                for x in (
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
                    file=BASE + relative,
                    killed=killed,
                    invalid=invalid,
                    failed_tests=failed,
                    exit_code=result.returncode,
                    before_sha256=hashlib.sha256(original).hexdigest(),
                    mutant_sha256=hashlib.sha256(mutant).hexdigest(),
                    seconds=round(time.monotonic() - started, 3),
                )
            )
            print(name, "KILLED" if killed else "NOT KILLED", flush=True)
        finally:
            path.write_bytes(original)
            clear_cache(path)
            assert path.read_bytes() == original
    restored = run([TEST, SYNC_TEST, REVIEW_TEST, REVIEW2_TEST])
    report = dict(
        imports=imports,
        baseline_passed=True,
        mutations=results,
        killed=sum(r["killed"] for r in results),
        total=len(results),
        restored_tests_passed=restored.returncode == 0,
    )
    (ROOT / "bench/results/mutations_crypto_recent.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if report["killed"] != len(results) or restored.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
