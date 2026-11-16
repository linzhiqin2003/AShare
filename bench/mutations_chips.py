"""D3 定向变异：先跑未修改对照并核对导入位置，只改 worktree/tmp 的源码副本。

筹码基准未证实，成本/收盘比值字段尚未实现；对应分母、快照不变性变异明确记为 deferred。
已落地的极值诊断、判定隔离、百分数、覆盖、单调、ETF 单位/时点/版本逐个变异。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (
        "chip_basis_must_remain_unverified",
        "data/chip_verify.py",
        'status="unverified",',
        'status="pass" if _audit_chip_extrema(store, sample_size=sample_size)["outside_tolerance"] == 0 else "fail",',
        "test_chips.py::test_chip_boundaries_are_diagnostic_only",
    ),
    (
        "chip_extrema_must_remain_diagnostic",
        "data/chip_verify.py",
        'status="diagnostic",',
        'status="fail" if details.get("outside_tolerance", 0) else "pass",',
        "test_chips.py::test_chip_boundaries_are_diagnostic_only",
    ),
    (
        "chip_extrema_excluded_from_failure_count",
        "data/verify.py",
        'chip_extrema = chip_basis.pop("extrema_diagnostic")',
        'chip_extrema = chip_basis.pop("extrema_diagnostic")\n    audit.add("chip_adjustment_basis", chip_extrema["checked"], [], failed=chip_extrema["outside_tolerance"])',
        "test_chips.py::test_verify_boundary_changes_do_not_change_failures",
    ),
    (
        "chip_diagnostic_nonfinite_json",
        "data/chip_verify.py",
        "pl.when(floats.is_finite()).then(floats).otherwise(None)",
        "floats.fill_nan(None)",
        "test_chips.py::test_chip_diagnostic_nonfinite_boundaries_are_json_safe",
    ),
    (
        "basis_latest_ignores_fetched_at",
        "data/chip_verify.py",
        'adjustment.filter(pl.col("trade_date") <= fetched)',
        "adjustment",
        "test_chips.py::test_basis_uses_partition_timestamp_and_only_past_extremes",
    ),
    (
        "basis_extrema_denominator",
        "data/chip_verify.py",
        "lows[valid].min() / basis",
        "lows[valid].min()",
        "test_chips.py::test_basis_uses_partition_timestamp_and_only_past_extremes",
    ),
    (
        "winner_percent",
        "data/chip_verify.py",
        'pl.col("winner_rate").cast(pl.Float64) / 100',
        'pl.col("winner_rate").cast(pl.Float64)',
        "test_chips.py::test_chip_values_positive_and_null",
    ),
    (
        "coverage_start",
        "data/chip_verify.py",
        'pl.col("trade_date") < "20180102"',
        'pl.col("trade_date") < "20170102"',
        "test_chips.py::test_chip_values_negative",
    ),
    (
        "quantile_order",
        "data/chip_verify.py",
        "pl.col(a) > pl.col(b)",
        "pl.lit(False)",
        "test_chips.py::test_chip_values_negative",
    ),
    (
        "etf_share_units",
        "data/fund_panel.py",
        'pl.col("total_share").cast(pl.Float64) * 10000',
        'pl.col("total_share").cast(pl.Float64)',
        "test_etf_data.py::test_etf_fields_units_timing_and_index_lookup",
    ),
    (
        "etf_size_units",
        "data/fund_panel.py",
        'pl.col("total_size").cast(pl.Float64) * 10000',
        'pl.col("total_size").cast(pl.Float64)',
        "test_etf_data.py::test_etf_fields_units_timing_and_index_lookup",
    ),
    (
        "etf_known_at",
        "data/fund_schema.py",
        "ETF 专题份额，按所属交易日记录；次日可知",
        "ETF 专题份额，按所属交易日记录；当日盘前可知",
        "test_etf_data.py::test_etf_fields_units_timing_and_index_lookup",
    ),
    (
        "etf_content_dependency",
        "data/fund_panel.py",
        'v.get("dataset") in FUND_DATASETS',
        'v.get("dataset", "").startswith(("fund_", "cb_"))',
        "test_etf_data.py::test_etf_changes_version_but_not_old_fields",
    ),
    (
        "etf_empty_refresh_guard",
        "data/store.py",
        "if rows == 0 or (",
        "if rows != 0 and (",
        "test_etf_data.py::test_successful_empty_share_sync_warns_and_keeps_revision",
    ),
    (
        "etf_same_partition_guard",
        "data/store.py",
        'if spec.partition == "fut_day" or name in ("etf_basic", "etf_share_size"):',
        'if spec.partition == "fut_day":',
        "test_etf_data.py::test_successful_empty_or_collapsed_refresh_keeps_revision",
    ),
    (
        "etf_shrink_guard",
        "data/store.py",
        "rows < before * spec.drop_check_ratio",
        "False",
        "test_etf_data.py::test_successful_empty_or_collapsed_refresh_keeps_revision",
    ),
    (
        "etf_status_shrink_guard",
        "data/fund_sync.py",
        "if len(previous_basic):",
        "if False:",
        "test_etf_data.py::test_successful_status_shrink_keeps_complete_table",
    ),
    (
        "etf_empty_source_guard",
        "data/fund_verify.py",
        'if required and not len(raw):\n        bad.append(dict(table="etf_share_size", problem="empty_source"))',
        'if False:\n        bad.append(dict(table="etf_share_size", problem="empty_source"))',
        "test_etf_data.py::test_verify_rejects_empty_source_and_empty_panel",
    ),
    (
        "etf_all_nan_panel_guard",
        "data/fund_verify.py",
        "if required and not np.isfinite(got).any():",
        "if False:",
        "test_etf_data.py::test_verify_rejects_all_nan_fields",
    ),
    (
        "etf_first_empty_unverified",
        "data/store.py",
        "suspicious = etf_coverage(self).required(partition)",
        "suspicious = False",
        "test_etf_coverage.py::test_empty_etf_partition_remains_unverified",
    ),
    (
        "etf_legacy_zero_retry",
        "data/store.py",
        'entry.get("dataset") == "etf_share_size"',
        "False",
        "test_etf_coverage.py::test_empty_etf_partition_is_retried_without_refresh",
    ),
    (
        "etf_small_snapshot_shrink",
        "data/catalog.py",
        "# 同一日的早期小样本也不应被截断覆盖；不与相邻交易日比较。\n            drop_check_min_rows=1,",
        "drop_check_min_rows=100,",
        "test_etf_coverage.py::test_small_etf_snapshot_shrink_is_rejected",
    ),
    (
        "etf_daily_partition_guard",
        "data/fund_verify.py",
        'if name == "etf_share_size":',
        "if False:",
        "test_etf_coverage.py::test_verify_rejects_each_empty_etf_day",
    ),
    (
        "etf_daily_raw_finite_guard",
        "data/fund_verify.py",
        "or str(day) not in finite",
        "or False",
        "test_etf_coverage.py::test_verify_rejects_each_nan_etf_day",
    ),
    (
        "etf_daily_panel_finite_guard",
        "data/fund_verify.py",
        "if not np.isfinite(got[dates[day], columns]).any():",
        "if False:",
        "test_etf_coverage.py::test_verify_rejects_each_nan_etf_day",
    ),
    (
        "etf_unknown_listing_is_not_absence",
        "data/etf_coverage.py",
        "return self.eligible(day) != set()",
        "return bool(self.eligible(day))",
        "test_etf_coverage.py::test_empty_etf_partition_remains_unverified",
    ),
    (
        "history_missing_day_guard",
        "data/chip_verify.py",
        "if missing:",
        "if False:",
        "test_chips.py::test_basis_rejects_internal_gap_and_truncated_tail",
    ),
    (
        "history_calendar_continuity",
        "data/chip_verify.py",
        "if len(days) != (last - first).days + 1:",
        "if False:",
        "test_chips.py::test_basis_requires_complete_calendar",
    ),
    (
        "history_listing_interval",
        "data/chip_verify.py",
        'past = past.filter(pl.col("trade_date") >= start)',
        "past = past",
        "test_chips.py::test_basis_excludes_prices_before_listing_interval",
    ),
]


def run(arguments, env):
    return subprocess.run(
        [sys.executable, *arguments],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def check_import(env, expected):
    code = "import ashare.data.chip_verify as c; print(c.__file__)"
    result = run(["-c", code], env)
    if result.returncode or Path(result.stdout.strip()).resolve() != expected.resolve():
        raise RuntimeError("导入位置不符，停止变异")


def main():
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "TUSHARE_TOKEN" and not k.startswith("ASHARE_")
    }
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        POLARS_MAX_THREADS="2",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
    )
    check_import(env, ROOT / "src/ashare/data/chip_verify.py")
    baseline = run(
        [
            "-m",
            "pytest",
            "test/data/test_chips.py",
            "test/data/test_etf_data.py",
            "test/data/test_etf_coverage.py",
            "--basetemp=tmp/d3/pytest-mutation-zero",
            "-q",
            "-n",
            "0",
        ],
        env,
    )
    logs = ROOT / "tmp/d3/mutations"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "0.log").write_text(baseline.stdout + baseline.stderr)
    if baseline.returncode:
        raise RuntimeError("变异 0 失败；见 tmp/d3/mutations/0.log")
    results = [
        dict(
            mutation="0",
            status="passed",
            imported="src/ashare/data/chip_verify.py",
            summary=baseline.stdout.strip().splitlines()[-1],
        )
    ]
    with tempfile.TemporaryDirectory(prefix="d3-mutants-", dir=ROOT / "tmp") as temp:
        overlay = Path(temp) / "src"
        shutil.copytree(
            ROOT / "src/ashare",
            overlay / "ashare",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        env["PYTHONPATH"] = str(overlay)
        for name, relative, old, new, test in CASES:
            path = overlay / "ashare" / relative
            saved = path.read_text()
            if saved.count(old) != 1:
                raise ValueError(f"{name} 的锚点必须唯一")
            path.write_text(saved.replace(old, new, 1))
            try:
                check_import(env, overlay / "ashare/data/chip_verify.py")
                result = run(
                    [
                        "-m",
                        "pytest",
                        "test/data/" + test,
                        "-q",
                        "-n",
                        "0",
                        "--basetemp=tmp/d3/pytest-mutant",
                    ],
                    env,
                )
                (logs / f"{name}.log").write_text(result.stdout + result.stderr)
                failures = re.findall(r"^FAILED (\S+)", result.stdout, re.MULTILINE)
                killed = (
                    result.returncode == 1
                    and bool(failures)
                    and "ERROR collecting" not in result.stdout
                )
                results.append(
                    dict(
                        mutation=name,
                        file="src/ashare/" + relative,
                        status="killed" if killed else "survived_or_invalid",
                        failures=failures,
                    )
                )
                print(name, results[-1]["status"], flush=True)
            finally:
                path.write_text(saved)
    report = dict(
        results=results,
        deferred=["筹码成本/前复权收盘比值的分母", "两种抓取基准快照的筹码比值不变性"],
        reason="his_low/high 历史极值判据已作废；仍缺分布、实际除权跨快照及异常尺度证据，不发布 chip_* 字段",
    )
    (ROOT / "bench/results/mutations_chips.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if any(r["status"] == "survived_or_invalid" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
