"""港股、美股原始库采集（第 2 批）。

数据根是 ``Store(root, "hk")`` / ``Store(root, "us")``，与 A 股的 ``<root>/cn`` 各有一份清单。
沿用 A 股的不可变 Parquet 修订、清单状态、断点续跑和按接口限速；与 A 股不同的只有三处：

- **样本历史按证券取**：开发库只取「全部证券近 3 年」（按交易日分区）加「2010 年起的 200 只
  样本」（每只一次请求，``*_hist`` 数据集）。样本名单由 ``select_sample`` 按当前市值分层抽出，
  写进 ``<root>/<市场>/sample.json``，同步时读它。
- **复权因子是前复权快照**：Tushare 港美股的 ``cum_adjfactor`` 以最新交易日为 1，每次除权都把
  整段历史改写一遍，不同时刻取的两天不能相除。所以因子数据集每次同步都在请求范围内整段重抓，
  每个分区记同一个 ``snapshot``（本次运行的编号），面板构建和校验要求区间内快照唯一。
- **交易日用本市场的日历**（``hk_tradecal`` / ``us_tradecal``），不用 A 股日历。

最短例子::

    import asyncio
    from ashare.data.sources import Tushare
    from ashare.data.store import Store
    from ashare.data.overseas_sync import select_sample, sync_overseas

    store = Store("/path/to/dev-root", "hk")          # 原始库写在 <root>/hk
    async def main():
        async with Tushare(rate=200) as source:
            await sync_overseas(source, store, "2023-11-13", "2026-11-12",
                                ["hk_calendar", "hk_securities"])
            await select_sample(source, store, "2026-11-12")
            await sync_overseas(source, store, "2023-11-13", "2026-11-12")
    asyncio.run(main())
"""

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime

import polars as pl

from .catalog import OVERSEAS_DATASETS
from .store import Store, atomic_json, lock
from .sync import compact

LOG = logging.getLogger(__name__)

HIST_START = "20100101"
FX_START = "20100101"
MARKET_SPECS = {
    "hk": dict(
        calendar="hk_calendar",
        securities="hk_securities",
        daily="hk_daily",
        factor="hk_adjfactor",
        daily_hist="hk_daily_hist",
        factor_hist="hk_adjfactor_hist",
        extra=("hk_connect_sh", "hk_connect_sz", "hk_connect_hold"),
        fx_daily=("USDHKD.FXCM", "USDCNH.FXCM"),
        fx_midrate=("CNYHKD",),
        currency="HKD",
    ),
    "us": dict(
        calendar="us_calendar",
        securities="us_securities",
        daily="us_daily",
        factor="us_adjfactor",
        daily_hist="us_daily_hist",
        factor_hist="us_adjfactor_hist",
        extra=(),
        fx_daily=("USDCNH.FXCM",),
        fx_midrate=("CNYUSD",),
        currency="USD",
    ),
}
# 校验用的「两地都熟悉的大票」，同时强制进样本（拆股历史要从 2010 年起才看得到）。
ANCHORS = {
    "hk": (
        "00700.HK",
        "00005.HK",
        "00941.HK",
        "00388.HK",
        "01299.HK",
        "02318.HK",
        "00939.HK",
        "00883.HK",
        "00001.HK",
        "00016.HK",
    ),
    "us": ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "JPM", "XOM", "KO", "WMT"),
}
US_MAJOR_EXCHANGES = ("NYS", "NAS", "ASE")
SAMPLE_SIZE = 200
SAMPLE_DELISTED = 20
SAMPLE_STRATA = 5
SAMPLE_SEED = "ashare-overseas-2026-11-13"


def spec_for(market):
    if market not in MARKET_SPECS:
        raise ValueError(f"港美股采集只支持 hk、us，收到 {market!r}")
    return MARKET_SPECS[market]


def market_datasets(market):
    """该市场的全部数据集名（日历、证券表在前）。"""
    m = spec_for(market)
    return [
        m["calendar"],
        m["securities"],
        m["daily"],
        m["factor"],
        m["daily_hist"],
        m["factor_hist"],
        *m["extra"],
        "overseas_fx_daily",
        "overseas_fx_midrate",
    ]


def snapshot_datasets(market):
    """前复权因子：每次同步整段重抓并记同一快照编号。"""
    m = spec_for(market)
    return {m["factor"], m["factor_hist"]}


def safe_partition(code):
    return code.replace("/", "_").replace(" ", "_")


def trading_days(store, market, start, end):
    """本市场日历里 [start, end] 的开市日；日历缺失时报错，不退回 A 股日历。"""
    name = spec_for(market)["calendar"]
    frame = store.read(name)
    return sorted(
        set(
            frame.filter(
                (pl.col("is_open").cast(pl.Int64) == 1)
                & pl.col("cal_date").is_between(pl.lit(start), pl.lit(end))
            )["cal_date"].to_list()
        )
    )


def read_sample(store):
    path = store.root / "sample.json"
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text())


def plan(spec, market, start, end, days, sample):
    """生成 (分区名, 请求参数)。"""
    m = spec_for(market)
    kind = spec.partition
    if kind == "static":
        if spec.api in ("hk_tradecal", "us_tradecal"):
            yield "all", dict(start_date="19800101", end_date=f"{int(end[:4]) + 1}1231")
        else:
            yield "all", {}
    elif kind == "status":
        for status in ("L", "D", "P"):
            yield status, {"list_status": status}
    elif kind == "day":
        for day in days:
            if not spec.coverage_start or day >= spec.coverage_start:
                yield day, {"trade_date": day}
    elif kind == "symbol":
        if sample is None:
            raise FileNotFoundError(
                "缺少 sample.json；先运行 select_sample（命令行 select-sample-overseas）"
            )
        for code in sample["symbols"]:
            yield (
                safe_partition(code),
                dict(
                    ts_code=code,
                    start_date=sample.get("start", HIST_START),
                    end_date=end,
                ),
            )
    elif kind == "fx_pair":
        pairs = m["fx_daily"] if spec.api == "fx_daily" else m["fx_midrate"]
        for pair in pairs:
            yield (
                safe_partition(pair),
                dict(ts_code=pair, start_date=FX_START, end_date=end),
            )
    else:
        raise ValueError(f"未知分区方式 {kind}")


async def sync_overseas(
    source,
    store: Store,
    start,
    end,
    names=None,
    *,
    force=False,
    refresh=True,
    workers=12,
    progress=None,
):
    """采集 [start, end] 内港股或美股（按 ``store.market``）的原始库。

    已完成的日分区跳过，最近 ``refresh`` 个交易日（数据集声明）重抓；日历、证券表、样本历史、
    汇率每次重抓；复权因子（前复权快照）每次整段重抓并记同一 ``snapshot``。某个分区失败时
    其余继续跑完，最后抛出第一个错误，清单里记 failed，下次重跑。返回写入的条目、按数据集的
    统计和本次的快照编号。
    """
    market = getattr(store, "market", None)
    m = spec_for(market)
    start, end = compact(start), compact(end)
    if start > end:
        raise ValueError("start 必须不晚于 end")
    available = market_datasets(market)
    selected = list(names or available)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"{market} 没有数据集 {sorted(unknown)}；可用：{available}")
    selected = list(dict.fromkeys([m["calendar"], *selected]))
    manifest = dict(store.manifest())
    snapshot = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    snapshots = snapshot_datasets(market)
    results, failures = [], []
    stats = defaultdict(lambda: dict(partitions=0, rows=0, calls=0, seconds=0.0))

    async def download(name, partition, params):
        spec = OVERSEAS_DATASETS[name]
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
            if name in snapshots:
                entry = await asyncio.to_thread(store.mark, key, {"snapshot": snapshot})
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

    def needs(name, partition, days):
        spec = OVERSEAS_DATASETS[name]
        key = f"{name}/{partition}"
        if force or not store.complete(key, verify_hash=False):
            return True
        if not refresh:
            return False
        if name in snapshots or spec.partition != "day":
            return True
        return partition in set(days[-spec.refresh :])

    async def run_group(group, days):
        sample = read_sample(store)
        tasks = defaultdict(deque)
        for name in group:
            spec = OVERSEAS_DATASETS[name]
            for partition, params in plan(spec, market, start, end, days, sample):
                if needs(name, partition, days):
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
        await asyncio.to_thread(store.validate_partitions, group)

    with lock(store.root / ".sync.lock"):
        await run_group([m["calendar"]], [])
        if failures:
            raise failures[0]
        days = trading_days(store, market, start, end)
        await run_group([n for n in selected if n != m["calendar"]], days)
    if failures:
        raise failures[0]
    return dict(entries=results, stats=dict(stats), snapshot=snapshot)


def _rank_key(code):
    return hashlib.sha256(f"{SAMPLE_SEED}:{code}".encode()).hexdigest()


def stratified_pick(universe, *, size, strata, exclude=()):
    """按市值分层的确定性抽样：市值降序切成 strata 层，每层按种子哈希取前若干只。

    ``universe`` 有 ts_code、mv 两列。返回 [(code, stratum)]，层号 0 是市值最大的一层。
    """
    frame = (
        universe.filter(pl.col("mv").is_finite() & (pl.col("mv") > 0))
        .filter(~pl.col("ts_code").is_in(list(exclude)))
        .unique("ts_code")
        .sort(["mv", "ts_code"], descending=[True, False])
    )
    n = len(frame)
    if n == 0 or size <= 0:
        return []
    frame = frame.with_columns(
        stratum=(pl.int_range(0, n) * strata // n).cast(pl.Int64)
    )
    per, extra = divmod(size, strata)
    picked = []
    for stratum in range(strata):
        members = sorted(
            frame.filter(pl.col("stratum") == stratum)["ts_code"].to_list(),
            key=_rank_key,
        )
        take = per + (1 if stratum < extra else 0)
        picked.extend((code, stratum) for code in members[:take])
    return picked


async def select_sample(source, store, asof, *, size=SAMPLE_SIZE):
    """抽 2010 年起的历史样本并写 ``sample.json``（需要先同步证券表）。

    固定包含 ``ANCHORS`` 的 10 只大票，再抽 ``SAMPLE_DELISTED`` 只 2011 年后退市的证券，
    其余按 ``asof`` 当天的总市值分 5 层等量抽取（种子固定，结果可复现）。
    """
    market = getattr(store, "market", None)
    m = spec_for(market)
    asof = compact(asof)
    anchors = list(ANCHORS[market])
    if market == "hk":
        securities = store.read(m["securities"])
        mv = (
            await source.fetch(
                "hk_daily_adj",
                {"trade_date": asof},
                fields="ts_code,total_mv",
                page_size=6000,
            )
        ).table
        listed = securities.filter(pl.col("list_status") == "L")
        universe = mv.join(listed.select("ts_code"), on="ts_code").select(
            "ts_code", pl.col("total_mv").cast(pl.Float64).alias("mv")
        )
        delisted_pool = sorted(
            securities.filter(
                (pl.col("list_status") == "D")
                & (pl.col("delist_date") >= "20110101")
                & (pl.col("delist_date") <= asof)
                & (pl.col("list_date") < pl.col("delist_date"))
            )["ts_code"].to_list(),
            key=_rank_key,
        )
        currency_note = "总市值来自 hk_daily_adj.total_mv（港元）"
    else:
        daily = (
            await source.fetch(
                "us_daily",
                {"trade_date": asof},
                fields="ts_code,total_mv",
                page_size=6000,
            )
        ).table
        factor = (
            await source.fetch(
                "us_adjfactor",
                {"trade_date": asof},
                fields="ts_code,exchange",
                page_size=15000,
            )
        ).table
        securities = store.read(m["securities"])
        kinds = securities.group_by("ts_code").agg(
            pl.col("classify").first(), pl.len().alias("n")
        )
        # 当天重复出现的代码（代码复用、源重复行）不进样本。
        clean = daily.group_by("ts_code").agg(
            pl.col("total_mv").first(), pl.len().alias("rows")
        )
        clean = clean.filter(pl.col("rows") == 1)
        majors = factor.filter(pl.col("exchange").is_in(US_MAJOR_EXCHANGES)).unique(
            "ts_code"
        )
        universe = (
            clean.join(majors.select("ts_code"), on="ts_code")
            .join(
                kinds.filter(
                    (pl.col("n") == 1) & pl.col("classify").is_in(["EQ", "ADR"])
                ),
                on="ts_code",
            )
            .select("ts_code", pl.col("total_mv").cast(pl.Float64).alias("mv"))
        )
        old = (
            await source.fetch(
                "us_adjfactor",
                {"trade_date": "20150102"},
                fields="ts_code,exchange",
                page_size=15000,
            )
        ).table
        present = set(factor["ts_code"].to_list()) | set(daily["ts_code"].to_list())
        # 优先股（代码带 ^）不算普通股退市样本；源的退市代码没有证券表记录。
        delisted_pool = sorted(
            {
                c
                for c in old.filter(pl.col("exchange").is_in(US_MAJOR_EXCHANGES))[
                    "ts_code"
                ].to_list()
                if "^" not in c
            }
            - present,
            key=_rank_key,
        )
        currency_note = "总市值来自 us_daily.total_mv（美元）；只抽纽交所、纳斯达克、美交所的 EQ/ADR"
    delisted = [c for c in delisted_pool if c not in anchors][:SAMPLE_DELISTED]
    rest = size - len(anchors) - len(delisted)
    picked = stratified_pick(
        universe, size=rest, strata=SAMPLE_STRATA, exclude=anchors + delisted
    )
    symbols = sorted(set(anchors) | set(delisted) | {c for c, _ in picked})
    value = dict(
        market=market,
        asof=asof,
        start=HIST_START,
        seed=SAMPLE_SEED,
        symbols=symbols,
        anchors=anchors,
        delisted=delisted,
        strata={code: s for code, s in picked},
        method=f"固定 {len(anchors)} 只大票 + {len(delisted)} 只 2011 年后退市（种子哈希排序）+ 按 {asof} 总市值降序分 {SAMPLE_STRATA} 层等量抽取；{currency_note}",
    )
    atomic_json(store.root / "sample.json", value)
    return value


__all__ = [
    "ANCHORS",
    "MARKET_SPECS",
    "market_datasets",
    "plan",
    "select_sample",
    "stratified_pick",
    "sync_overseas",
    "trading_days",
]
