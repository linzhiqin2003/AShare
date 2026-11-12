"""境内期货原始库采集：交易所日历、合约表、日线、主力映射、结算参数。

沿用股票采集的做法：每个分区一个不可变 Parquet 修订，清单记状态，完成的分区跳过
（断点续跑），每个接口独立限速并发。与股票同步的区别只在分区计划：

- 日历和合约表按交易所分区（每个交易所一次请求，合约表单次上限 10000 行，最大的
  交易所约 3600 行）。
- 日线、主力映射、结算参数按交易日分区。交易日取六个期货交易所日历的并集，
  不用股票日历。2026-11 实测单日最多约 1100 / 200 / 900 行，都在单页上限
  （2000 / 5000 / 至少 1000）以内，按日一次请求最省；按合约切分要多出约一倍请求。

最短例子::

    import asyncio
    from ashare.data.sources import Tushare
    from ashare.data.store import Store
    from ashare.data.futures_sync import sync_futures

    store = Store("/path/to/dev-root")          # 原始库写在 <root>/cn
    async def main():
        async with Tushare(rate=400) as source:
            await sync_futures(source, store, "2026-11-01", "2026-11-12")
    asyncio.run(main())
"""

import asyncio
import logging
import time
from collections import defaultdict, deque

import polars as pl

from .catalog import FUTURES_DATASETS, FUTURES_EXCHANGES
from .store import Store, lock
from .sync import compact
from .sync_policy import should_refresh

LOG = logging.getLogger(__name__)
ORDER = ("futures_calendar", "futures_contracts")


def futures_days(store, start, end):
    """六个期货交易所开市日的并集；日历分区缺失时报错，不退回股票日历。"""
    frame = store.read("futures_calendar")
    missing = set(FUTURES_EXCHANGES) - set(frame["exchange"].unique().to_list())
    if missing:
        raise FileNotFoundError(
            f"缺少交易所日历 {sorted(missing)}；先同步 futures_calendar"
        )
    return sorted(
        set(
            frame.filter(
                (pl.col("is_open").cast(pl.Int64) == 1)
                & pl.col("cal_date").is_between(pl.lit(start), pl.lit(end))
            )["cal_date"].to_list()
        )
    )


def plan(spec, start, end, days):
    """生成 (分区名, 请求参数)；日分区只排到数据集已知的覆盖起点之后。"""
    if spec.partition == "fut_exchange":
        for exchange in FUTURES_EXCHANGES:
            params = {"exchange": exchange}
            if spec.api == "trade_cal":
                params.update(start_date="19900101", end_date=f"{int(end[:4]) + 1}1231")
            yield exchange, params
    elif spec.partition == "fut_day":
        for day in days:
            if not spec.coverage_start or day >= spec.coverage_start:
                yield day, {"trade_date": day}
    else:
        raise ValueError(f"未知期货分区方式 {spec.partition}")


async def sync_futures(
    source,
    store: Store,
    start,
    end,
    names=None,
    *,
    force=False,
    refresh=True,
    workers=12,
    now=None,
    progress=None,
):
    """采集 [start, end] 内的期货原始库；返回本次写入的清单条目及按接口的统计。

    已完成的分区跳过；最近 `refresh` 个交易日（数据集声明）与按交易所分区的表每次重抓。
    某个分区失败时其余接口继续跑完，最后抛出第一个错误，清单里记为 failed，下次重跑。
    `progress` 是可选的回调，参数为 (数据集, 已完成数, 总数)。
    """
    start, end = compact(start), compact(end)
    if start > end:
        raise ValueError("start 必须不晚于 end")
    selected = list(names or FUTURES_DATASETS)
    unknown = set(selected) - FUTURES_DATASETS.keys()
    if unknown:
        raise ValueError(
            f"未知期货数据集 {sorted(unknown)}；可用：{sorted(FUTURES_DATASETS)}"
        )
    # 日历永远先跑：日分区由它决定。
    selected = list(dict.fromkeys(["futures_calendar", *selected]))
    manifest = dict(store.manifest())
    results, failures = [], []
    stats = defaultdict(lambda: dict(partitions=0, rows=0, calls=0, seconds=0.0))

    async def download(name, partition, params):
        spec = FUTURES_DATASETS[name]
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
        for name in group:
            spec = FUTURES_DATASETS[name]
            for partition, params in plan(spec, start, end, days):
                key = f"{name}/{partition}"
                if (
                    not force
                    and store.complete(key, verify_hash=False)
                    and not (
                        refresh
                        and should_refresh(
                            spec, partition, manifest.get(key, {}), days, end, now=now
                        )
                    )
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

    with lock(store.root / ".sync.lock"):
        await run_group(["futures_calendar"], [])
        if failures:
            raise failures[0]
        days = futures_days(store, start, end)
        rest = [n for n in selected if n != "futures_calendar"]
        await run_group(rest, days)
    if failures:
        raise failures[0]
    return dict(entries=results, stats=dict(stats))


__all__ = ["futures_days", "plan", "sync_futures"]
