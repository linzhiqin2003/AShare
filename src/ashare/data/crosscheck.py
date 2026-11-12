"""小规模价格抽样与两个交易日的全市场状态交叉核对。"""

import hashlib
import json
import random

import polars as pl

from .crosscheck_compare import compare_rows
from .sources.baostock import BaoStock
from .store import atomic_json


def iso(day):
    return f"{day[:4]}-{day[4:6]}-{day[6:]}"


def fingerprint(store, sample_size=300, status_days=2):
    inputs = {
        key: value["content_hash"]
        for key, value in store.manifest().items()
        if value["dataset"] in ("daily", "stock_basic", "stock_st", "suspend_d")
        and value["status"] == "complete"
    }
    return hashlib.sha256(
        json.dumps(
            dict(
                inputs=inputs,
                sample_size=sample_size,
                status_days=status_days,
                sampling_version=2,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def sample_quotes(store, entries, selected_days, sample_size):
    # 在所有原始行的序号中均匀抽样，只解码涉及的分区，不构造全历史 Python 字典。
    rng = random.Random(20261112)
    ordinals = rng.sample(
        range(sum(e["rows"] for e in entries)),
        min(sample_size * 2, sum(e["rows"] for e in entries)),
    )
    priority = {value: rank for rank, value in enumerate(ordinals)}
    ordinals.sort()
    rows, sampled, offset, candidates = {}, set(), 0, []
    for entry in entries:
        indices = [i - offset for i in ordinals if offset <= i < offset + entry["rows"]]
        if indices or entry["partition"] in selected_days:
            day_rows = store.read("daily", entry["partition"], entry["partition"])
            for i in indices:
                row = day_rows.row(i, named=True)
                key = (row["ts_code"], row["trade_date"])
                if not key[0].endswith(".BJ"):
                    candidates.append((priority[offset + i], key, row))
            if entry["partition"] in selected_days:
                rows.update(
                    {
                        (r["ts_code"], r["trade_date"]): r
                        for r in day_rows.iter_rows(named=True)
                    }
                )
        offset += entry["rows"]
    for _, key, row in sorted(candidates, key=lambda item: item[0])[:sample_size]:
        sampled.add(key)
        rows[key] = row
    return rows, sampled


async def crosscheck(store, sample_size=300, status_days=2):
    store = store.snapshot()
    input_fingerprint = fingerprint(store, sample_size, status_days)
    entries = sorted(
        (
            e
            for e in store.entries("daily").values()
            if e["status"] == "complete" and e["rows"]
        ),
        key=lambda e: e["partition"],
    )
    dates = [e["partition"] for e in entries]
    selected_days = dates[-status_days:]
    securities = store.read("stock_basic").filter(
        ~pl.col("ts_code").str.ends_with(".BJ")
    )
    securities = securities.filter(
        (pl.col("list_date") <= dates[-1])
        & (pl.col("delist_date").is_null() | (pl.col("delist_date") > selected_days[0]))
    )
    rows, sampled = sample_quotes(store, entries, selected_days, sample_size)
    st = {
        (r["ts_code"], r["trade_date"])
        for r in store.read("stock_st", selected_days[0], selected_days[-1]).iter_rows(
            named=True
        )
    }
    suspended = {
        (r["ts_code"], r["trade_date"])
        for r in store.read("suspend_d", selected_days[0], selected_days[-1]).iter_rows(
            named=True
        )
        if r["suspend_type"] == "S" and r["suspend_timing"] in (None, "", "全天")
    }
    expected_status = {
        (r["ts_code"], d)
        for r in securities.iter_rows(named=True)
        for d in selected_days
        if r["list_date"] <= d and (not r.get("delist_date") or d < r["delist_date"])
    }
    target = store.root / "reports/baostock-progress.json"
    progress = (
        __import__("json").loads(target.read_text())
        if target.exists()
        else {"completed": [], "rows": [], "calls": 0}
    )
    if progress.get("input_fingerprint") != input_fingerprint:
        progress = dict(
            input_fingerprint=input_fingerprint, completed=[], rows=[], calls=0
        )
    completed = set(progress["completed"])
    source = BaoStock()
    try:
        # 每只股票一次串行查询；范围覆盖抽样日与状态日，返回后只保留目标行。
        for symbol in sorted(set(securities["ts_code"]) | {s for s, d in sampled}):
            if symbol in completed:
                continue
            code, exchange = symbol.split(".")
            symbol_dates = [d for s, d in sampled if s == symbol]
            start = min(symbol_dates + [selected_days[0]])
            result = await source.fetch(
                "daily",
                dict(
                    code=f"{exchange.lower()}.{code}",
                    fields="date,code,open,high,low,close,volume,tradestatus,isST",
                    start_date=iso(start),
                    end_date=iso(dates[-1]),
                    frequency="d",
                    adjustflag="3",
                ),
            )
            for row in result.table.iter_rows(named=True):
                day = row["date"].replace("-", "")
                if day in selected_days or (symbol, day) in sampled:
                    progress["rows"].append({**row, "symbol": symbol, "date": day})
            completed.add(symbol)
            progress["completed"] = sorted(completed)
            progress["calls"] += 1
            if len(completed) % 100 == 0:
                atomic_json(target, progress)
                print(
                    f"BaoStock 串行核对 {len(completed)}/{len(securities)}", flush=True
                )
        atomic_json(target, progress)
    finally:
        await source.close()
    comparison = compare_rows(
        rows, sampled, st, suspended, selected_days, progress["rows"], expected_status
    )
    report = dict(
        status="complete",
        input_fingerprint=input_fingerprint,
        sample_size=sample_size,
        requested_status_days=status_days,
        sample_seed=20261112,
        sampled_pairs=len(sampled),
        status_days=selected_days,
        calls=progress["calls"],
        **comparison,
        excluded="北交所：BaoStock 不覆盖；退市和未上市证券不在当前状态截面内",
    )
    atomic_json(store.root / "reports/baostock.json", report)
    return report
