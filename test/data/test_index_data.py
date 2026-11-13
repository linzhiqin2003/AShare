"""基准指数名单（catalog.BENCHMARKS）的下载与读取；全部用合成数据。"""

import asyncio

import polars as pl
import pytest

import ashare.data as ad
from ashare.data import sync as sync_module
from ashare.data.catalog import (
    BENCHMARKS,
    DATASETS,
    DEFAULT_BENCHMARK,
    INDEX_DAILY_CODES,
    INDEXES,
)
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store
from ashare.data.sync import plan, sync
from ashare.data.sync_policy import month_final

JANUARY = ["20260105", "20260106", "20260107", "20260108", "20260109"]
FEBRUARY = ["20260202", "20260203"]
DAYS = JANUARY + FEBRUARY
FETCHED = "2026-02-04T00:00:00+00:00"  # 一月已过定稿宽限期，二月是当月
POOL = list(INDEXES.values())
EXTRA = [c for c in BENCHMARKS if c not in POOL]


def bar(code, day):
    close = 1000.0 + INDEX_DAILY_CODES.index(code)
    return dict(
        ts_code=code,
        trade_date=day,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        pre_close=close,
        vol=3.0,
        amount=4.0,
    )


class FakeSource:
    """合成接口：交易日历按年，指数日线按代码和月份。"""

    def __init__(self, short=()):
        self.calls = []
        self.short = set(short)  # 这些代码的当月只返回一天，模拟数据没发全

    async def fetch(self, api, params, *, fields="", page_size=None):
        self.calls.append((api, dict(params)))
        if api == "trade_cal":
            days = [d for d in DAYS if d.startswith(params["start_date"][:4])]
            return FetchResult(
                pl.DataFrame(
                    dict(
                        exchange=["SSE"] * len(days),
                        cal_date=days,
                        is_open=[1] * len(days),
                        pretrade_date=days,
                    )
                ),
                FETCHED,
                1,
            )
        if api == "index_daily":
            assert set(params) == {"ts_code", "start_date", "end_date"}
            days = [d for d in DAYS if params["start_date"] <= d <= params["end_date"]]
            if params["ts_code"] in self.short and days and days[0] in FEBRUARY:
                days = days[:1]
            return FetchResult(
                pl.DataFrame([bar(params["ts_code"], d) for d in days]), FETCHED, 1
            )
        raise AssertionError(f"测试未覆盖的接口 {api}")


def run(store, source, **kwargs):
    return asyncio.run(
        sync(source, store, DAYS[0], DAYS[-1], ["index_daily"], **kwargs)
    )


def index_calls(source):
    return [p for api, p in source.calls if api == "index_daily"]


def test_catalog_lists():
    # 股票池指数不变；基准名单另列，下载的是两者的并集，股票池指数排在前面。
    assert INDEXES == {
        "csi300": "000300.SH",
        "csi500": "000905.SH",
        "csi1000": "000852.SH",
        "csi2000": "932000.CSI",
        "csi_all": "000985.CSI",
    }
    assert DEFAULT_BENCHMARK == "000001.SH" and DEFAULT_BENCHMARK in BENCHMARKS
    assert list(INDEX_DAILY_CODES[: len(POOL)]) == POOL
    assert set(INDEX_DAILY_CODES) == set(POOL) | set(BENCHMARKS)
    assert len(INDEX_DAILY_CODES) == len(set(INDEX_DAILY_CODES))
    assert all(name for name in BENCHMARKS.values())


def test_plan_includes_benchmarks_weights_stay_pool(tmp_path):
    store = Store(tmp_path)
    daily = list(plan(DATASETS["index_daily"], DAYS[0], DAYS[-1], DAYS, store))
    assert [p for p, _ in daily] == [
        f"{code}_{month}"
        for month in ("202601", "202602")
        for code in INDEX_DAILY_CODES
    ]
    assert daily[0][1] == dict(
        ts_code=POOL[0], start_date="20260101", end_date="20260131"
    )
    assert daily[-1][1] == dict(
        ts_code=INDEX_DAILY_CODES[-1], start_date="20260201", end_date=DAYS[-1]
    )
    weights = list(plan(DATASETS["index_weight"], DAYS[0], DAYS[-1], DAYS, store))
    assert {p.rsplit("_", 1)[0] for p, _ in weights} == set(POOL)


def test_sync_downloads_every_benchmark(tmp_path):
    store = Store(tmp_path)
    source = FakeSource()
    run(store, source)
    assert sorted({p["ts_code"] for p in index_calls(source)}) == sorted(
        INDEX_DAILY_CODES
    )
    table = ad.open(store.root.parent).table("index_daily").collect()
    assert set(table["symbol"]) == set(INDEX_DAILY_CODES)
    assert table.select("symbol", "date").is_duplicated().sum() == 0


def test_pool_rows_identical_to_pool_only_download(tmp_path, monkeypatch):
    # 原 5 个指数：只下股票池（main 的做法）与加上基准名单后，读出来逐位相同。
    with monkeypatch.context() as ctx:
        ctx.setattr(sync_module, "INDEX_DAILY_CODES", tuple(POOL))
        before = Store(tmp_path / "a")
        run(before, FakeSource())
    after = Store(tmp_path / "b")
    run(after, FakeSource())
    assert {
        e["partition"].rsplit("_", 1)[0] for e in before.entries("index_daily").values()
    } == set(POOL)

    def pool(store, **kwargs):
        return (
            ad.open(store.root.parent)
            .table("index_daily", **kwargs)
            .filter(pl.col("symbol").is_in(POOL))
            .collect()
            .sort("symbol", "date")
        )

    assert pool(before).equals(pool(after))
    assert pool(before).equals(pool(after, symbols=POOL))
    assert pool(before, start="20260106", end="20260202").equals(
        pool(after, start="20260106", end="20260202", symbols=POOL)
    )
    # 原始分区本身（内容哈希）也相同。
    hashes = {
        k: v["content_hash"]
        for k, v in before.manifest().items()
        if v["dataset"] == "index_daily"
    }
    assert all(after.manifest()[k]["content_hash"] == h for k, h in hashes.items())


def test_symbols_open_only_their_partitions(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run(store, FakeSource())
    opened = []
    original = pl.scan_parquet

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", spy)
    one = (
        ad.open(store.root.parent)
        .table("index_daily", symbols=["399006.SZ"], start="20260201")
        .collect()
    )
    assert one["date"].to_list() == FEBRUARY
    assert len(opened) == 1 and "399006.SZ_202602" in opened[0]
    opened.clear()
    missing = (
        ad.open(store.root.parent).table("index_daily", symbols=["X.SH"]).collect()
    )
    assert missing.height == 0 and "close" in missing.columns


def test_incremental_refreshes_only_current_month(tmp_path):
    store = Store(tmp_path)
    run(store, FakeSource())
    source = FakeSource()
    run(store, source, refresh=False)
    assert index_calls(source) == []
    source = FakeSource()
    run(store, source)
    # 一月已定稿不重下；二月是当月，每个指数重抓一次。
    again = index_calls(source)
    assert sorted(p["ts_code"] for p in again) == sorted(INDEX_DAILY_CODES)
    assert {p["start_date"] for p in again} == {"20260201"}


def test_short_current_month_is_not_finalized(tmp_path):
    # 新增指数的当月分区行数偏少：照逐月分区规则不定稿，下次同步重抓补齐。
    store = Store(tmp_path)
    code = EXTRA[0]
    run(store, FakeSource(short=[code]))
    entry = store.manifest()[f"index_daily/{code}_202602"]
    assert entry["rows"] == 1 and entry["finalized"] is False
    assert not month_final(entry)
    assert store.manifest()[f"index_daily/{code}_202601"]["finalized"] is True
    source = FakeSource()
    run(store, source)
    assert f"{code}" in {p["ts_code"] for p in index_calls(source)}
    assert store.manifest()[f"index_daily/{code}_202602"]["rows"] == len(FEBRUARY)


def test_indexes_lists_only_benchmarks_and_searches(tmp_path):
    store = Store(tmp_path)
    run(store, FakeSource())
    # 名单外的指数即使本地有日线也不列。
    store.write(
        "index_daily",
        "999999.SH_202601",
        {},
        FetchResult(
            pl.DataFrame([bar(POOL[0], JANUARY[0]) | {"ts_code": "999999.SH"}]),
            FETCHED,
            1,
        ),
    )
    data = ad.open(store.root.parent)
    listing = data.indexes()
    assert listing.columns == ["symbol", "name", "first_date", "last_date"]
    assert listing["symbol"].to_list() == sorted(BENCHMARKS)
    row = listing.filter(pl.col("symbol") == DEFAULT_BENCHMARK).row(0, named=True)
    assert (row["name"], row["first_date"], row["last_date"]) == (
        BENCHMARKS[DEFAULT_BENCHMARK],
        DAYS[0],
        DAYS[-1],
    )
    assert data.indexes("创业板指")["symbol"].to_list() == ["399006.SZ"]
    assert data.indexes("399006")["symbol"].to_list() == ["399006.SZ"]
    assert data.indexes("bj")["symbol"].to_list() == ["899050.BJ"]
    assert set(data.indexes("红利")["symbol"]) == {
        "000922.CSI",
        "000015.SH",
        "H30269.CSI",
    }
    assert data.indexes("999999").height == 0
    assert data.indexes("不存在").height == 0


def test_indexes_skips_benchmarks_without_local_data(tmp_path, monkeypatch):
    with monkeypatch.context() as ctx:
        ctx.setattr(sync_module, "INDEX_DAILY_CODES", tuple(POOL))
        store = Store(tmp_path)
        run(store, FakeSource())
    assert sorted(ad.open(store.root.parent).indexes()["symbol"]) == sorted(POOL)
    empty = ad.open(Store(tmp_path / "none").root.parent).indexes()
    assert empty.height == 0 and empty.columns == [
        "symbol",
        "name",
        "first_date",
        "last_date",
    ]


@pytest.mark.parametrize("code", ["000001.SH", "399001.SZ", "000688.SH", "000922.CSI"])
def test_required_benchmarks_present(code):
    assert code in BENCHMARKS


def test_window_end_month_is_read(tmp_path):
    # 窗口终点所在的月份也要打开：终点落在二月第一天，二月分区必须在内。
    store = Store(tmp_path)
    run(store, FakeSource())
    rows = (
        ad.open(store.root.parent)
        .table("index_daily", symbols=["399006.SZ"], start="20260106", end="20260202")
        .sort("date")
        .collect()
    )
    assert rows["date"].to_list() == JANUARY[1:] + ["20260202"]


class TushareEmpty(FakeSource):
    """像真实接口那样返回「有列名、没有行」的空表（列类型为 Null）。"""

    def __init__(self, empty_code):
        super().__init__()
        self.empty_code = empty_code

    async def fetch(self, api, params, *, fields="", page_size=None):
        if api == "index_daily" and params["ts_code"] == self.empty_code:
            if params["start_date"] < "20260201":
                self.calls.append((api, dict(params)))
                return FetchResult(
                    pl.DataFrame([], schema=fields.split(","), orient="row"),
                    FETCHED,
                    1,
                )
        return await super().fetch(api, params, fields=fields, page_size=page_size)


def test_empty_window_has_normal_types(tmp_path):
    store = Store(tmp_path)
    code = "899050.BJ"
    run(store, TushareEmpty(code))
    entry = store.manifest()[f"index_daily/{code}_202601"]
    assert entry["rows"] == 0
    data = ad.open(store.root.parent)
    empty = data.table(
        "index_daily", symbols=[code], start="20260105", end="20260109"
    ).collect()
    assert empty.height == 0
    expected = {
        "symbol": pl.String,
        "date": pl.String,
        **{
            c: pl.Float64
            for c in (
                "open",
                "high",
                "low",
                "close",
                "previous_close",
                "volume",
                "amount",
            )
        },
    }
    assert dict(empty.schema) == expected
    # 有数据的窗口不受影响：类型和数值照原始列换算。
    full = data.table("index_daily", symbols=[code], start="20260201").collect()
    assert dict(full.schema) == expected
    assert full["volume"].to_list() == [300.0] * len(FEBRUARY)
    assert full["amount"].to_list() == [4000.0] * len(FEBRUARY)
