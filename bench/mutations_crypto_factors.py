"""C6 定向变异：先证明导入本 worktree，再隔离副本，全部合成离线测试。"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = "src/ashare/factor/crypto.py"
E = "src/ashare/factor/crypto_eval.py"
S = "src/ashare/data/crypto/signals.py"
MUTATIONS = [
    (
        "annual_interval",
        C,
        "rates * 8760, intervals",
        "rates * 8760, np.full_like(intervals, 8)",
    ),
    (
        "unsettled_event",
        C,
        'rates = a["funding_rate"][:, 1::2]',
        'rates = np.roll(a["funding_rate"][:, 1::2], -1, axis=0)',
    ),
    ("daily_no_shift", C, "return at[use] - DAY, x[use]", "return at[use], x[use]"),
    (
        "daily_double_shift",
        C,
        "return at[use] - DAY, x[use]",
        "return at[use] - 2*DAY, x[use]",
    ),
    ("z_today", C, "lag = np.vstack([np.full((1, k), np.nan), v[:-1]])", "lag = v"),
    (
        "z_future",
        C,
        "lag = np.vstack([np.full((1, k), np.nan), v[:-1]])",
        "lag = np.vstack([v[1:], np.full((1, k), np.nan)])",
    ),
    ("roll_early", C, "remaining > 336", "remaining > 337"),
    ("roll_late", C, "remaining > 336", "remaining >= 336"),
    (
        "remaining_left_edge",
        C,
        '(meta["delivery_at"] - at) / HOUR',
        '(meta["delivery_at"] - times) / HOUR',
    ),
    ("cross_split", E, "entries + horizon * DAY < BOUNDARY", "entries < BOUNDARY"),
    ("gap_mask", C, "& BAD_PRICE", "& (BAD_PRICE & ~1)"),
    ("1b_missing_past", E, "(count == horizon)", "(count >= horizon-1)"),
    (
        "1b_regression_residual",
        E,
        "e = g - np.mean(g[valid_g])",
        "e = g - coef[0] - coef[1] * standardized",
    ),
    (
        "1b_divide_T",
        E,
        "standard_error = np.linalg.norm(influence[:, 1])",
        "standard_error = np.linalg.norm(influence[:, 1]) / np.sqrt(n)",
    ),
    (
        "calibrate_oos",
        E,
        'ins = sample_mask(entries, 1, "in")',
        "ins = np.ones(len(entries), bool)",
    ),
    (
        "holm_smaller_family",
        E,
        "p = np.array([pvalues.get(key, np.nan) for key in CONFIRM_FAMILY])",
        "p = np.array([pvalues.get(key, np.nan) for key in CONFIRM_FAMILY]); p = np.nan_to_num(p, nan=1.0); p = p / 8",
    ),
    (
        "near_gap_fallback",
        C,
        'selected[choose, j] = meta["symbol"]',
        'selected[choose, j] = meta["symbol"]\n                missing_times = frame.filter((pl.col("symbol") == meta["symbol"]) & (pl.col("quality") != 0))["time"].to_numpy()\n                selected[np.isin(times, missing_times), j] = ""',
    ),
    (
        "listing_query_start",
        S,
        "normalized.append(item)",
        'item["listed_at"] = max(item["listed_at"], start) if item["listed_at"] is not None else None\n        normalized.append(item)',
    ),
    (
        "accept_post_delivery_trade",
        S,
        "last_trade > delivery",
        "last_trade > delivery + 24 * HOUR",
    ),
    (
        "labels_use_signal_rows_instead_of_calendar",
        E,
        "daily_returns = returns_at(calendar_entries, 1)",
        "calendar_entries = entries.copy()\n    daily_returns = returns_at(calendar_entries, 1)",
    ),
    (
        "labels_accept_uncovered_entry",
        E,
        "exact[inside] = t[indices[inside]] == moments[inside]",
        "exact[inside] = True",
    ),
    (
        "1b_uncentered_normal_equation",
        E,
        "coef = np.linalg.solve(upper, orthogonal.T @ y[use])",
        "raw_z = np.column_stack([np.ones(n), x[use]])\n    raw_coef = np.linalg.inv(raw_z.T @ raw_z) @ raw_z.T @ y[use]\n    coef = np.array([raw_coef[0] + raw_coef[1] * (anchor + shift), raw_coef[1] * scale])",
    ),
    ("z_59_instead_of_60", C, "count >= 60", "count >= 59"),
    (
        "quarter_quality_uses_future",
        S,
        "by_time.get(t - HOUR)",
        "by_time.get(t + HOUR)",
    ),
]


def main():
    from ashare.data.crypto import signals
    from ashare.factor import crypto, crypto_eval

    for module in (crypto, crypto_eval, signals):
        if not Path(module.__file__).resolve().is_relative_to(ROOT / "src"):
            raise RuntimeError("测试未导入当前 worktree")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "test/crypto/test_factors_c6.py",
        "test/crypto/test_eval_c6.py",
        "test/crypto/test_signals_c6.py",
        "test/crypto/test_review_c6.py",
    ]
    baseline = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    if baseline.returncode:
        raise RuntimeError("基线测试失败：" + baseline.stdout[-4000:])
    rows = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="c6-mutations-", dir=ROOT / "tmp"
    ) as folder:
        folder = Path(folder)
        shutil.copytree(
            ROOT / "src/ashare",
            folder / "ashare",
            ignore=shutil.ignore_patterns("__pycache__", "*.so"),
        )
        env = {
            **os.environ,
            "PYTHONPATH": str(folder),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        for name, path, old, new in MUTATIONS:
            destination = folder / Path(path).relative_to("src")
            original = destination.read_text()
            pattern = re.compile(r"\s+".join(re.escape(part) for part in old.split()))
            if pattern.search(original) is None:
                raise RuntimeError("变异锚点失效：" + name)
            destination.write_text(
                pattern.sub(lambda _, replacement=new: replacement, original, count=1)
            )
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                killed = (
                    result.returncode == 1
                    and "failed" in result.stdout
                    and "ERROR collecting" not in result.stdout
                )
                rows.append(
                    dict(
                        name=name,
                        killed=killed,
                        returncode=result.returncode,
                        summary=result.stdout.splitlines()[-1]
                        if result.stdout
                        else result.stderr[-200:],
                    )
                )
                print(name, "killed" if killed else "SURVIVED/ERROR", flush=True)
            finally:
                destination.write_text(original)
    output = dict(
        worktree_import_verified=True,
        baseline="pass",
        mutations=rows,
        killed=sum(r["killed"] for r in rows),
        total=len(rows),
        seconds=time.monotonic() - started,
    )
    (ROOT / "bench/results/mutations_crypto_factors.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    if not all(r["killed"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
