"""在显式指定的原始库副本上补日归档、限定可验证终点并 build/verify；不写逐行行情到报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ashare.data.crypto.archive import archive_key
from ashare.data.crypto.build import build
from ashare.data.crypto.sources import PublicClient
from ashare.data.crypto.storage import RawStore
from ashare.data.crypto.sync import load_plan
from ashare.data.crypto.timeutil import DAY, HOUR, iso
from ashare.data.crypto.verify import verify
from ashare.data.store import atomic_json


def gaps(rows):
    return [
        dict(
            start=a["time"] + HOUR,
            end=b["time"],
            previous_quality=a.get("time_quality", "normal"),
        )
        for a, b in zip(rows, rows[1:], strict=False)
        if b["time"] > a["time"] + HOUR
    ]


def prepare(
    root,
    *,
    repair_daily=False,
    allow_inferred_funding=False,
    accept_spot_gaps=False,
    output="bench/results/crypto_data_validation.json",
):
    root = Path(root)
    plan, data = load_plan(root)
    original = root / "sync_plan.before_c3.json"
    if not original.exists():
        atomic_json(original, plan)
    observations = []
    repairs = []
    exceptions = list(plan.get("exceptions", []))
    for p in plan["profiles"]:
        name = p["instrument"]
        datasets = data[name]
        for field in ("klines", "markPriceKlines", "indexPriceKlines"):
            if field not in datasets:
                continue
            rows = datasets[field]
            holes = gaps(rows)
            observations.append(
                dict(
                    instrument=name,
                    dataset=field,
                    rows=len(rows),
                    partial_halt=sum(
                        r.get("time_quality") == "partial_halt" for r in rows
                    ),
                    invalid_close_time=sum(
                        r.get("time_quality") == "invalid_close_time" for r in rows
                    ),
                    subsecond_end_offset=sum(
                        r.get("time_quality") == "normal"
                        and r.get("close_time", r["time"] + HOUR - 1)
                        != r["time"] + HOUR - 1
                        for r in rows
                    ),
                    gaps=holes,
                )
            )
            if p["market"] == "spot" and accept_spot_gaps:
                for hole in holes:
                    if hole["previous_quality"] == "partial_halt":
                        continue
                    record = dict(
                        instrument=name,
                        start=hole["start"],
                        end=hole["end"],
                        reason="Observed archive gap; explicit flat/untradable research assumption, not confirmed exchange halt",
                        source_url="https://data.binance.vision/?prefix=data/spot/monthly/klines/"
                        + p["symbol"]
                        + "/1h/",
                    )
                    if not any(
                        e["instrument"] == name
                        and e["start"] == record["start"]
                        and e["end"] == record["end"]
                        for e in exceptions
                    ):
                        exceptions.append(record)
            if p["market"] == "usdm" and repair_daily:
                with PublicClient(max_bytes=30_000_000) as client:
                    store = RawStore(root)
                    try:
                        for hole in holes:
                            if hole["start"] % DAY or hole["end"] % DAY:
                                raise ValueError(
                                    "非完整 UTC 日缺口：不能把重叠原始分区静默覆盖"
                                )
                            for day in range(hole["start"], hole["end"], DAY):
                                symbol = (
                                    p.get("index_symbol", p["symbol"].split("_")[0])
                                    if field == "indexPriceKlines"
                                    else p["symbol"]
                                )
                                key = archive_key(
                                    "usdm",
                                    field,
                                    symbol,
                                    iso(day)[:10],
                                    frequency="daily",
                                )
                                found = client.archive(store, key) is not None
                                repairs.append(
                                    dict(
                                        instrument=name,
                                        dataset=field,
                                        day=iso(day)[:10],
                                        found=found,
                                    )
                                )
                                if found and key not in plan["series"][name][field]:
                                    plan["series"][name][field].append(key)
                        plan["raw_version"] = store.freeze()
                    finally:
                        store.close()
    plan["exceptions"] = exceptions
    atomic_json(root / "sync_plan.json", plan)
    ds = build(
        root,
        allow_missing=accept_spot_gaps,
        allow_inferred_funding=allow_inferred_funding,
    )
    report = verify(root, ds.version)
    result = dict(
        version=ds.version,
        start=iso(ds.meta["start"]),
        end=iso(ds.meta["end"]),
        rows=ds.meta["rows"],
        verdict=report["verdict"],
        error_count=len(report["errors"]),
        diagnostic_count=len(report["diagnostics"]),
        archives_checked=report["archives_checked"],
        observations=observations,
        repairs=repairs,
        explicit_gap_intervals=exceptions,
        terminal_policy=(
            "explicit inferred continuation; min archived complete hour and last accepted settlement on first failure/missing tail"
            if allow_inferred_funding
            else "stop before first funding settlement without historical interval evidence; no REST gap inference"
        ),
    )
    atomic_json(Path(output), result)
    if report["errors"]:
        raise ValueError(
            "verify failed; inspect the returned error count and run crypto verify"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="tmp/crypto",
        help="原始库副本；命令会在此目录写计划、面板和校验结果",
    )
    parser.add_argument(
        "--repair-daily", action="store_true", help="仅补 U 本位完整日缺口的官方日归档"
    )
    parser.add_argument(
        "--accept-spot-gaps",
        action="store_true",
        help="显式允许非停机自证的现货归档缺口，逐段记录为假设",
    )
    parser.add_argument("--allow-inferred-funding", action="store_true")
    parser.add_argument("--output", default="bench/results/crypto_data_validation.json")
    args = parser.parse_args()
    report = prepare(
        args.data,
        repair_daily=args.repair_daily,
        allow_inferred_funding=args.allow_inferred_funding,
        accept_spot_gaps=args.accept_spot_gaps,
        output=args.output,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "verdict",
                    "rows",
                    "error_count",
                    "archives_checked",
                    "start",
                    "end",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
