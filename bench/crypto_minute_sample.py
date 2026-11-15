"""C7 小样本验收；真实逐小时差异只写被忽略的 tmp，公开报告仅含汇总。"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from ashare.data.crypto import minute
from ashare.data.crypto.timeutil import DAY, millis, month_end
from ashare.data.crypto.verify_minute import verify


def estimate_download_seconds(report, raw_bytes, full_raw_bytes):
    """按首次实际下载吞吐外推；缓存命中不能当作网络性能。"""
    if raw_bytes <= 0 or report["downloaded_bytes"] < raw_bytes:
        return None
    return report["seconds"] * full_raw_bytes / max(raw_bytes, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="tmp/crypto_minute")
    parser.add_argument("--hourly-root", required=True)
    parser.add_argument("--recent-start", default=None)
    parser.add_argument("--recent-days", type=int, default=3)
    parser.add_argument("--skip-sync", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    # 防止误把真实行情产物写进可跟踪的目录。
    workspace = Path(__file__).resolve().parents[1]
    if not root.resolve().is_relative_to(workspace / "tmp"):
        raise ValueError("验收原始数据与完整报告必须位于本 worktree 的 tmp 下")
    now = datetime.now(UTC)
    recent = args.recent_start or now.strftime("%Y-%m-01")
    windows = [
        ("2021-05-01", "2021-06-01"),
        ("2025-03-01", "2025-04-01"),
        (millis(recent), millis(recent) + args.recent_days * DAY),
    ]
    hourly_root = Path(args.hourly_root)
    # 先固定严格指针；后续从不再次解析 latest，不写小时库。
    hourly_version = json.loads((hourly_root / "latest.json").read_text())["version"]
    timings = {}
    if not args.skip_sync:
        report = minute.sync(
            root=root,
            windows=windows,
            max_bytes=150_000_000,
            progress=lambda part: print("sync", part, flush=True),
        )
        timings["sync_seconds"] = report["seconds"]
    else:
        report = json.loads((root / "sync_report.json").read_text())
        timings["sync_seconds"] = report["seconds"]
    started = time.perf_counter()
    version = minute.build(root=root)
    timings["build_seconds"] = time.perf_counter() - started
    print("build done", timings["build_seconds"], flush=True)
    started = time.perf_counter()
    validation = verify(root=root, version=version)
    timings["verify_seconds"] = time.perf_counter() - started
    if not validation["ok"]:
        raise ValueError(validation)
    print("verify done", timings["verify_seconds"], flush=True)
    started = time.perf_counter()
    reconciliation = verify(
        root=root,
        version=version,
        hourly_root=hourly_root,
        hourly_version=hourly_version,
    )
    timings["verify_and_reconcile_seconds"] = time.perf_counter() - started
    (root / "reconciliation.json").write_text(
        json.dumps(reconciliation, ensure_ascii=False, indent=2) + "\n"
    )
    summaries = []
    for r in reconciliation["reconciliation"]:
        reasons = {}
        for d in r["differences"]:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        summaries.append(
            {
                k: r[k]
                for k in (
                    "partition",
                    "fields",
                    "compared_hours",
                    "max_absolute",
                    "material_differences",
                    "roundoff_differences",
                    "matches",
                )
            }
            | dict(
                incomplete_hours=len(r["incomplete_hours"]),
                hourly_unavailable=len(r["hourly_unavailable"]),
                reasons=reasons,
            )
        )
    meta = minute.metadata(root=root, version=version)
    refs = {
        ref["sha256"]: ref
        for p in meta["partitions"].values()
        for ref in p["archives"].values()
    }
    raw_bytes = sum(r["bytes"] for r in refs.values())
    parquet_bytes = sum(p["bytes"] for p in meta["partitions"].values())
    # 历史两整月的压缩率外推；包含每日分区尾部的全样本行数外推 CPU。
    monthly_parts = [
        p for p in meta["partitions"].values() if p["month"] in ("2021-05", "2025-03")
    ]
    month_raw = sum(r["bytes"] for p in monthly_parts for r in p["archives"].values())
    month_parquet = sum(p["bytes"] for p in monthly_parts)
    month_minutes = sum(
        (month_end(p["month"]) - millis(p["month"] + "-01")) // 60000
        for p in monthly_parts
    )
    total_minutes = (millis(now) // DAY * DAY - millis("2020-01-01")) // 60000 * 8
    scale = total_minutes / month_minutes
    full_raw, full_parquet = round(month_raw * scale), round(month_parquet * scale)
    summary = dict(
        version=version,
        hourly_version=hourly_version,
        local_validation=validation,
        timings=timings,
        raw_bytes=raw_bytes,
        parquet_bytes=parquet_bytes,
        archive_count=len(refs),
        downloaded_bytes=report["downloaded_bytes"],
        missing_archives=len(report["missing"]),
        reconciliation=summaries,
        extrapolation=dict(
            start="2020-01-01",
            end=now.strftime("%Y-%m-%d"),
            series=8,
            calendar_minutes=total_minutes,
            raw_bytes=full_raw,
            parquet_bytes=full_parquet,
            disk_bytes=full_raw + full_parquet,
            sync_seconds=estimate_download_seconds(report, raw_bytes, full_raw),
            build_seconds=timings["build_seconds"]
            * total_minutes
            / max(validation["rows"], 1),
            verify_seconds=timings["verify_seconds"]
            * total_minutes
            / max(validation["rows"], 1),
            model="串行同带宽；整月样本压缩率；CPU 按行数线性外推；不含重试和上游修订",
        ),
    )
    path = workspace / "bench/results/crypto_minute_sample.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "timings",
                    "raw_bytes",
                    "parquet_bytes",
                    "archive_count",
                    "missing_archives",
                    "extrapolation",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not reconciliation["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
