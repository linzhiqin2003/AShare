"""场内基金（ETF、LOF）与可转债原始库的采集。

沿用股票、期货采集的做法：每个分区一个不可变 Parquet 修订，清单记状态，完成的分区跳过
（断点续跑），每个接口独立限速并发。分区方式：

- ``fund_calendar``：上交所日历一个分区（一次取 1990 年至明年年底，翻页）。
- ``fund_basic``：整表一个分区（只取场内基金，``market=E``，含全部上市状态）。
- 日分区（``fund_daily``、``fund_adj``、``fund_share``、``cb_daily``；``fund_div`` 按除息日）：
  交易日取上交所日历，每天一次请求（单日行数都在单页上限以内，满页时自动翻页）。
- ``cb_basic``、``cb_issue``、``cb_call``、``cb_rate``：整表一个分区，翻页取完。
- ``cb_price_chg``：源要求给代码、支持逗号分隔多个代码，按「代码前四位 + 交易所」成组请求。
- ``cb_share``：整表超过源的 offset 上限（十万行），按单个债券分区。

刷新：日分区重抓最近 ``refresh`` 个交易日；整表、状态分区每次重抓；按债券的分区只重抓
仍在存续（或退市不足 ``live_days`` 天）的债券。

最短例子::

    import asyncio
    from ashare.data.sources import Tushare
    from ashare.data.store import Store
    from ashare.data.fund_sync import sync_funds

    store = Store("/path/to/dev-root")          # 原始库写在 <root>/cn
    async def main():
        async with Tushare(rate=200) as source:
            await sync_funds(source, store, "2026-11-01", "2026-11-12")
    asyncio.run(main())
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import date, timedelta

import polars as pl

from .catalog import FUND_DATASETS
from .store import Store, lock
from .sync import compact

LOG = logging.getLogger(__name__)
FIRST = ("fund_calendar",)
SECOND = ("fund_basic", "cb_basic")  # 按代码分区的数据集依赖它们


def fund_days(store, start, end):
    """上交所开市日；日历分区缺失时报错。"""
    frame = store.read("fund_calendar")
    return sorted(
        set(
            frame.filter(
                (pl.col("exchange") == "SSE")
                & (pl.col("is_open").cast(pl.Int64) == 1)
                & pl.col("cal_date").is_between(pl.lit(start), pl.lit(end))
            )["cal_date"].to_list()
        )
    )


def cb_groups(store):
    """可转债代码按「前四位 + 交易所」成组，返回 {分区名: [代码...]}（代码升序）。"""
    codes = sorted(set(store.read("cb_basic")["ts_code"].to_list()))
    groups = defaultdict(list)
    for code in codes:
        groups[f"{code[:4]}_{code.split('.')[-1]}"].append(code)
    return dict(groups)


def live_codes(store, end, live_days=60):
    """未退市或退市不足 live_days 天的可转债；它们的转股、转股价分区需要刷新。"""
    cutoff = (
        date.fromisoformat(f"{end[:4]}-{end[4:6]}-{end[6:]}")
        - timedelta(days=live_days)
    ).strftime("%Y%m%d")
    frame = store.read("cb_basic")
    return set(
        frame.filter(
            pl.col("delist_date").is_null() | (pl.col("delist_date") >= cutoff)
        )["ts_code"].to_list()
    )


def plan(spec, start, end, days, store):
    """生成 (分区名, 请求参数)。"""
    kind = spec.partition
    if kind == "fund_exchange":
        yield (
            "SSE",
            dict(
                exchange="SSE",
                start_date="19900101",
                end_date=f"{int(end[:4]) + 1}1231",
            ),
        )
    elif kind == "fund_status":
        # 不带 status 的请求返回全部状态（实测 3020 行），比分 L/D/I 三次请求多出约 50 个
        # 其他状态的基金，所以只取一个分区。
        yield "all", {}
    elif kind == "static":
        yield "all", {}
    elif kind == "day":
        for day in days:
            if not spec.coverage_start or day >= spec.coverage_start:
                yield day, {spec.date: day}
    elif kind == "cb_batch":
        for name, codes in cb_groups(store).items():
            yield name, {"ts_code": ",".join(codes)}
    elif kind == "cb_code":
        for code in sorted(set(store.read("cb_basic")["ts_code"].to_list())):
            yield code, {"ts_code": code}
    else:
        raise ValueError(f"未知基金/可转债分区方式 {kind}")


def needs_refresh(spec, partition, days, live, refresh=True):
    """已完成分区是否重抓。"""
    if not refresh:
        return False
    kind = spec.partition
    if kind in ("fund_exchange", "fund_status", "static"):
        return True
    if kind == "day":
        recent = set(days[-spec.refresh :]) if spec.refresh > 0 else set()
        return partition in recent
    if kind == "cb_code":
        return partition in live
    if kind == "cb_batch":
        prefix, exchange = partition.split("_")
        return any(c[:4] == prefix and c.endswith(exchange) for c in live)
    return False


async def sync_funds(
    source,
    store: Store,
    start,
    end,
    names=None,
    *,
    force=False,
    refresh=True,
    workers=8,
    progress=None,
):
    """采集 [start, end] 内的基金与可转债原始库；返回写入的清单条目及按接口的统计。

    已完成的分区跳过；某个分区失败时其余接口继续跑完，最后抛出第一个错误，清单里记为
    failed（已完成分区的刷新失败只记 refresh_error，旧修订保留），下次重跑补齐。
    `progress` 是可选回调，参数为 (数据集, 已完成数, 总数)。
    """
    start, end = compact(start), compact(end)
    if start > end:
        raise ValueError("start 必须不晚于 end")
    selected = list(names or FUND_DATASETS)
    unknown = set(selected) - FUND_DATASETS.keys()
    if unknown:
        raise ValueError(
            f"未知基金/可转债数据集 {sorted(unknown)}；可用：{sorted(FUND_DATASETS)}"
        )
    needs_cb = any(
        FUND_DATASETS[n].partition in ("cb_batch", "cb_code") for n in selected
    )
    selected = list(
        dict.fromkeys(["fund_calendar", *(["cb_basic"] if needs_cb else []), *selected])
    )
    manifest = dict(store.manifest())
    results, failures = [], []
    stats = defaultdict(lambda: dict(partitions=0, rows=0, calls=0, seconds=0.0))

    async def download(name, partition, params):
        spec = FUND_DATASETS[name]
        key = f"{name}/{partition}"
        previous = manifest.get(key, {})
        if previous.get("status") != "complete":
            await asyncio.to_thread(
                store.mark,
                key,
                dict(
                    dataset=name, partition=partition, params=params, status="pending"
                ),
            )
        try:
            result = await source.fetch(
                spec.api, params, fields=spec.columns, page_size=spec.limit
            )
            entry = await asyncio.to_thread(
                store.write, name, partition, params, result
            )
            results.append(entry)
            stats[name]["partitions"] += 1
            stats[name]["rows"] += entry["rows"]
            stats[name]["calls"] += result.calls
        except Exception as exc:  # noqa: BLE001 - 记录后统一抛出
            LOG.warning("partition=%s error=%s", key, type(exc).__name__)
            change = (
                dict(refresh_error=type(exc).__name__)
                if previous.get("status") == "complete"
                else dict(status="failed", error=type(exc).__name__)
            )
            await asyncio.to_thread(store.mark, key, change)
            failures.append(exc)

    async def run_group(group, days):
        tasks = defaultdict(deque)
        live = (
            live_codes(store, end)
            if any(FUND_DATASETS[n].partition in ("cb_batch", "cb_code") for n in group)
            else set()
        )
        for name in group:
            spec = FUND_DATASETS[name]
            for partition, params in plan(spec, start, end, days, store):
                key = f"{name}/{partition}"
                if (
                    not force
                    and store.complete(key, verify_hash=False)
                    and not needs_refresh(spec, partition, days, live, refresh)
                ):
                    continue
                tasks[name].append((name, partition, {**spec.params, **params}))
        totals = {name: len(queue) for name, queue in tasks.items()}
        clocks = {name: time.monotonic() for name in tasks}

        async def worker(name, queue):
            while queue:
                await download(*queue.popleft())
                if progress is not None:
                    progress(name, totals[name] - len(queue), totals[name])
            stats[name]["seconds"] = time.monotonic() - clocks[name]

        per_api = max(2, workers // max(1, len(tasks)))
        await asyncio.gather(
            *(
                worker(name, queue)
                for name, queue in tasks.items()
                for _ in range(min(per_api, len(queue)))
            )
        )
        await asyncio.to_thread(
            store.validate_partitions,
            [n for n in group if FUND_DATASETS[n].partition == "day"],
        )

    with lock(store.root / ".sync.lock"):
        await run_group(["fund_calendar"], [])
        if failures:
            raise failures[0]
        days = fund_days(store, start, end)
        second = [n for n in selected if n in SECOND]
        if second:
            await run_group(second, days)
            if failures:
                raise failures[0]
        rest = [n for n in selected if n not in FIRST and n not in SECOND]
        await run_group(rest, days)
    if failures:
        raise failures[0]
    return dict(entries=results, stats=dict(stats))


__all__ = [
    "cb_groups",
    "fund_days",
    "live_codes",
    "needs_refresh",
    "plan",
    "sync_funds",
]
