"""C8 全历史 metrics 误差审计与汇总；真实行只保留在显式数据根。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ashare.data.crypto import metrics as m
from ashare.data.crypto import minute
from ashare.data.crypto.timeutil import DAY, millis, months
from ashare.data.crypto.verify_metrics import verify
from ashare.data.manifest import Manifest
from ashare.data.store import atomic_json


def audit(root, minute_root):
    manifest = Manifest(root, readonly=True)
    try:
        refs = manifest.read()
        source = {
            key: {k: v[k] for k in ("sha256", "bytes", "checksum", "checksum_text")}
            for key, v in refs.items()
            if v.get("status") == "complete"
        }
    finally:
        if manifest.db is not None:
            manifest.db.close()
    mv = minute.metadata(root=minute_root)["version"]
    report = {}
    for symbol in m.SYMBOLS:
        keys = sorted(k for k in source if f"/{symbol}/" in k)
        if not keys:
            continue
        start, end = millis(keys[0][-14:-4]), millis(keys[-1][-14:-4]) + DAY
        rows = []
        for label in months(start, end):
            frames = [
                m.parse_archive(m._raw(root, key, source[key]), symbol, key[-14:-4])[0]
                for key in keys
                if key[-14:-7] == label
            ]
            if not frames:
                continue
            frame = pl.concat(frames).sort("create_time")
            bars = m._load_minutes(
                symbol,
                millis(label + "-01") - m.STEP,
                minute.month_end(label) + m.STEP,
                root=minute_root,
                version=mv,
            )
            error = m.errors(frame, m.minute_windows(bars))
            rows.append(np.column_stack([frame["create_time"].to_numpy(), error]))
        data = np.concatenate(rows)
        # 完整逐行证据留在外部数据集，仓库只输出分布。
        np.save(root / f"audit-{symbol}.npy", data)
        minimum = np.fmin(data[:, 1], data[:, 2])
        maximum = np.fmax(data[:, 1], data[:, 2])
        report[symbol] = dict(
            rows=len(data),
            finite_minimum=int(np.isfinite(minimum).sum()),
            minimum_quantiles=np.quantile(
                minimum[np.isfinite(minimum)], [0, 0.5, 0.9, 0.99, 0.999, 1]
            ).tolist(),
            maximum_quantiles=np.quantile(
                maximum[np.isfinite(maximum)], [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
            ).tolist(),
            min_above_match=int(np.count_nonzero(minimum >= m.MATCH_ERROR)),
            decisive=int(
                np.count_nonzero((minimum < m.MATCH_ERROR) & (maximum > m.REJECT_ERROR))
            ),
        )
    return report


def run(root, minute_root, *, prior_download_seconds=0):
    began = time.perf_counter()
    try:
        version = m.build(root=root, minute_root=minute_root)
    except ValueError as exc:
        atomic_json(
            root / "build_report.json",
            dict(ok=False, error=str(exc), seconds=time.perf_counter() - began),
        )
        raise
    build_seconds = time.perf_counter() - began
    began = time.perf_counter()
    verification = verify(root=root, minute_root=minute_root, version=version)
    verify_seconds = time.perf_counter() - began
    return summarize(
        root,
        version,
        verification,
        build_seconds,
        verify_seconds,
        prior_download_seconds,
    )


def summarize(
    root, version, verification, build_seconds, verify_seconds, prior_download_seconds
):
    meta = m.metadata(root=root, version=version)
    symbols = {}
    for symbol, q in meta["quality"].items():
        symbols[symbol] = {
            k: q[k]
            for k in (
                "rows",
                "duplicates",
                "null_rows",
                "null_cells",
                "inherited_rows",
                "switch_at",
                "compared",
                "mismatches",
                "mismatch_ratio",
                "comparable_mismatch_ratio",
                "days",
                "years",
                "undetermined_days",
                "inherited_days",
                "determined_days",
                "segments",
            )
        }
        symbols[symbol].update(
            registered_switch_at=meta["registered_switches"][symbol],
            gap_segments=len(q["gaps"]),
            gap_rows=sum(g["rows"] for g in q["gaps"]),
            normalized_gap_rows=sum(g["rows"] for g in q["normalized_gaps"]),
            uncomparable=q["rows"] - q["compared"],
            archives=sum(f"/{symbol}/" in k for k in meta["archives"]),
            opposite_rows=sum(d["opposite_rows"] for d in q["days"]),
            opposite_max_run=max(d["opposite_max_run"] for d in q["days"]),
            paired_zero_days=sum(d["both_compared"] == 0 for d in q["days"]),
            paired_nonzero_quantiles=np.quantile(
                [d["both_compared"] for d in q["days"] if d["both_compared"]],
                [0, 0.01, 0.05, 0.5, 0.95, 1],
            ).tolist(),
        )
    differences = verification["differences"]
    parquet_bytes = sum(r["bytes"] for r in meta["partitions"].values())
    sync = json.loads((root / "sync_report.json").read_text())
    return dict(
        ok=int(verification["ok"]),
        built=int(version is not None),
        differences=differences,
        symbols=symbols,
        match_error=m.MATCH_ERROR,
        reject_error=m.REJECT_ERROR,
        mismatch_error=m.MISMATCH_ERROR,
        min_paired_rows=m.MIN_PAIRED_ROWS,
        opposite_run_limit=m.OPPOSITE_RUN_LIMIT,
        switch_window_ms=m.SWITCH_WINDOW,
        archives=len(meta["archives"]),
        download_seconds=prior_download_seconds + sync["seconds"],
        prior_download_seconds=prior_download_seconds,
        download_bytes=sum(r["bytes"] for r in meta["archives"].values()),
        build_seconds=build_seconds,
        verify_seconds=verify_seconds,
        disk_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        parquet_bytes=parquet_bytes,
    )


def collect(root, version, *, prior_download_seconds=0):
    """分币报告只能在版本、分钟版本和全集覆盖一致时合并为完整验证。"""
    meta = m.metadata(root=root, version=version)
    available = {k.rsplit("/", 2)[1] for k in meta["archives"]}
    if available != set(meta["quality"]):
        raise ValueError("verification coverage differs from metadata")
    reports = []
    for symbol in sorted(available):
        report = json.loads(
            (root / "verification" / f"{version}-{symbol}.json").read_text()
        )
        if (
            report["version"] != version
            or report["minute_version"] != meta["minute_version"]
            or set(report["symbols"]) != {symbol}
        ):
            raise ValueError("verification scope/version mismatch")
        reports.append(report)
    combined = dict(
        version=version,
        minute_version=meta["minute_version"],
        complete=True,
        ok=all(r["ok"] for r in reports),
        differences=sum(r["differences"] for r in reports),
        symbols={s: q for r in reports for s, q in r["symbols"].items()},
        details=[d for r in reports for d in r["details"]],
        taker_mismatches=[d for r in reports for d in r["taker_mismatches"]],
    )
    if combined["differences"] != len(combined["details"]):
        raise ValueError("verification detail count mismatch")
    atomic_json(root / "verification" / f"{version}.json", combined)
    build = json.loads((root / "build_report.json").read_text())
    if build["version"] != version:
        raise ValueError("build timing belongs to another version")
    return summarize(
        root,
        version,
        combined,
        build["seconds"],
        sum(r["seconds"] for r in reports),
        prior_download_seconds,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--minute-root", type=Path, required=True)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--version")
    p.add_argument("--verify-symbol", choices=m.SYMBOLS)
    p.add_argument("--collect", action="store_true")
    p.add_argument("--prior-download-seconds", type=float, default=0)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results/crypto_metrics_validation.json"),
    )
    a = p.parse_args()
    if a.verify_symbol:
        if not a.version:
            p.error("--verify-symbol requires --version")
        began = time.perf_counter()
        result = verify(
            root=a.root,
            minute_root=a.minute_root,
            version=a.version,
            symbols=[a.verify_symbol],
        )
        path = a.root / "verification" / f"{a.version}-{a.verify_symbol}.json"
        detailed = json.loads(path.read_text())
        detailed["seconds"] = time.perf_counter() - began
        atomic_json(path, detailed)
        print(
            json.dumps(
                dict(
                    ok=result["ok"],
                    differences=result["differences"],
                    seconds=detailed["seconds"],
                )
            )
        )
        raise SystemExit(0 if result["ok"] else 1)
    if a.collect:
        if not a.version:
            p.error("--collect requires --version")
        result = collect(
            a.root, a.version, prior_download_seconds=a.prior_download_seconds
        )
    else:
        result = (
            audit(a.root, a.minute_root)
            if a.audit
            else run(
                a.root, a.minute_root, prior_download_seconds=a.prior_download_seconds
            )
        )
    atomic_json(a.output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "symbols"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if "ok" in result and not result["ok"]:
        raise SystemExit(1)
