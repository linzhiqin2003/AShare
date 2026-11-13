"""由目录声明生成采集任务；存储不负责重试和翻页。"""

import asyncio
import calendar
from collections import defaultdict, deque
from datetime import date, timedelta

import polars as pl

from .catalog import (
    DAILY_DATASETS,
    DATASETS,
    FUTURES_DATASETS,
    INDEX_DAILY_CODES,
    INDEXES,
)
from .store import Store, lock
from .sync_policy import month_final, should_refresh


def compact(value):
    return date.fromisoformat(value).strftime("%Y%m%d") if "-" in value else value


def months(start, end):
    year, month = int(start[:4]), int(start[4:6])
    while f"{year:04}{month:02}01" <= end:
        yield (
            f"{year:04}{month:02}01",
            f"{year:04}{month:02}{calendar.monthrange(year, month)[1]:02}",
        )
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def plan(spec, start, end, days, store):
    kind = spec.partition
    if kind == "year":
        # 上市交易日数必须用完整日历，不能从样本起点重新计数。
        for year in range(1990, int(end[:4]) + 1):
            yield str(year), dict(start_date=f"{year}0101", end_date=f"{year}1231")
    elif kind == "status":
        for status in ("L", "D", "P"):
            yield status, {"list_status": status}
    elif kind == "static":
        yield "all", {}
    elif kind == "classification":
        for src in ("SW2014", "SW2021"):
            yield src, {"src": src, "level": "L1"}
    elif kind == "industry":
        for code in sorted(set(store.read("index_classify")["index_code"])):
            for flag in ("Y", "N"):
                yield f"{code}_{flag}", {"l1_code": code, "is_new": flag}
    elif kind == "legacy_industry":
        for code in sorted(
            set(
                store.read("index_classify").filter(pl.col("src") == "SW2014")[
                    "index_code"
                ]
            )
        ):
            for flag in ("Y", "N"):
                yield f"{code}_{flag}", {"index_code": code, "is_new": flag}
    elif kind == "day":
        for day in days:
            if not spec.coverage_start or day >= spec.coverage_start:
                yield day, {spec.date: day}
    elif kind == "index_month_daily":
        # 日线：股票池指数与基准名单的并集；成分权重（index_month）仍只取股票池指数。
        for begin, finish in months(start, end):
            for code in INDEX_DAILY_CODES:
                yield (
                    f"{code}_{begin[:6]}",
                    dict(ts_code=code, start_date=begin, end_date=min(finish, end)),
                )
    elif kind == "index_month":
        # 额外取上月快照，样本首日只继承已经公开的权重。
        first = date(int(start[:4]), int(start[4:6]), 1) - timedelta(days=1)
        for begin, finish in months(first.strftime("%Y%m01"), end):
            for code in INDEXES.values():
                yield (
                    f"{code}_{begin[:6]}",
                    dict(index_code=code, start_date=begin, end_date=min(finish, end)),
                )
    elif kind == "report":
        for year in range(int(start[:4]), int(end[:4]) + 1):
            for quarter in ("0331", "0630", "0930", "1231"):
                period = f"{year}{quarter}"
                if start <= period <= end:
                    yield period, {"period": period}
    else:
        raise ValueError(f"未知分区方式 {kind}")


async def sync(
    source,
    store: Store,
    start,
    end,
    names=None,
    *,
    force=False,
    refresh=True,
    workers=24,
    now=None,
):
    """跨接口并发采集，写 Parquet 在线程内完成；断点判断只做一次清单读取。"""
    start, end = compact(start), compact(end)
    if start > end:
        raise ValueError("start 必须不晚于 end")
    selected = list(names or DAILY_DATASETS)
    futures = set(selected) & FUTURES_DATASETS.keys()
    if futures:
        raise ValueError(
            f"期货数据集 {sorted(futures)} 由 ashare.data.futures_sync 采集；"
            "命令行用 python -m ashare.data sync-futures"
        )
    unknown = set(selected) - DATASETS.keys()
    if unknown:
        raise ValueError(f"未知数据集 {sorted(unknown)}；可用：{sorted(DATASETS)}")
    selected = list(
        dict.fromkeys(
            ["trade_cal"]
            + (
                ["index_classify"]
                if any(n in selected for n in ("index_member_all", "index_member"))
                else []
            )
            + selected
        )
    )
    results = []
    failures = []
    manifest = dict(store.manifest())

    async def download(name, partition, params):
        spec = DATASETS[name]
        key = f"{name}/{partition}"
        previous = manifest.get(key, {})
        if previous.get("status") != "complete":
            await asyncio.to_thread(
                store.mark,
                key,
                dict(
                    dataset=name,
                    partition=partition,
                    params=params,
                    status="pending",
                ),
            )
        try:
            result = await source.fetch(
                spec.api, params, fields=spec.columns, page_size=spec.limit
            )
            entry = await asyncio.to_thread(
                store.write, name, partition, params, result
            )
            if spec.partition in ("index_month", "index_month_daily"):
                entry = store.mark(key, {"finalized": month_final(entry)})
            results.append(entry)
        except Exception as exc:
            change = (
                dict(refresh_error=type(exc).__name__)
                if previous.get("status") == "complete"
                else dict(status="failed", error=type(exc).__name__)
            )
            await asyncio.to_thread(store.mark, key, change)
            failures.append(exc)

    async def run_group(names):
        tasks = defaultdict(deque)
        days = (
            []
            if names == ["trade_cal"]
            else sorted(
                set(
                    store.read("trade_cal").filter(
                        (pl.col("is_open") == 1)
                        & pl.col("cal_date").is_between(pl.lit(start), pl.lit(end))
                    )["cal_date"]
                )
            )
        )
        for name in names:
            spec = DATASETS[name]
            for partition, params in plan(spec, start, end, days, store):
                key = f"{name}/{partition}"
                entry = manifest.get(key, {})
                if (
                    not force
                    and store.complete(key, verify_hash=False)
                    and not (
                        refresh
                        and should_refresh(spec, partition, entry, days, end, now=now)
                    )
                ):
                    continue
                tasks[name].append((name, partition, {**spec.params, **params}))

        # 每个接口独立取任务，限频等待不能耗尽其他接口的工作者。
        async def worker(group):
            while group and not failures:
                await download(*group.popleft())

        per_api = max(2, workers // max(1, len(tasks)))
        await asyncio.gather(
            *(
                worker(group)
                for group in tasks.values()
                for _ in range(min(per_api, len(group)))
            )
        )
        await asyncio.to_thread(store.validate_partitions, names)
        if failures:
            raise failures[0]

    with lock(store.root / ".sync.lock"):
        await run_group(["trade_cal"])
        if "index_classify" in selected:
            await run_group(["index_classify"])
        await run_group(
            [n for n in selected if n not in ("trade_cal", "index_classify")]
        )
    return results
