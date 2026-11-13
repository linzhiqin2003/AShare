"""指数名单（index_basic）与全市场指数日线（index_daily_all）；全部用合成数据。"""

import asyncio
from datetime import date

import polars as pl
import pytest

import ashare.data as ad
from ashare.data.catalog import (
    DAILY_DATASETS,
    DATASETS,
    DEFAULT_BENCHMARK,
    INDEXES,
    PUBLIC_TABLES,
)
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store
from ashare.data.sync import plan, sync
from ashare.data.sync_policy import should_refresh

DAYS = [
    "20260105",
    "20260106",
    "20260107",
    "20260108",
    "20260109",
    "20260112",
    "20260113",
    "20260114",
]
POOL = "000300.SH"  # 股票池指数之一，两份来源都有
WIDE = ["000001.SH", "399006.SZ", "H00001.CSI"]
BASIC_FIELDS = DATASETS["index_basic"].columns.split(",")


def basic_rows():
    rows = []
    for code, name, full, market in (
        ("000001.SH", "上证指数", "上海证券综合指数", "SSE"),
        ("399006.SZ", "创业板指", "创业板指数", "SZSE"),
        (POOL, "沪深300", "沪深300指数", "SSE"),
        ("999999.CSI", "合成终止指数", "只在名单里的指数", "CSI"),
    ):
        row = dict.fromkeys(BASIC_FIELDS)
        row.update(
            ts_code=code,
            name=name,
            fullname=full,
            market=market,
            category="规模指数",
            base_date="20000101",
            base_point=1000.0,
            list_date="20000101",
        )
        rows.append(row)
    return pl.DataFrame(rows)


def bar(code, day, close):
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
    """合成接口：全市场按日返回，股票池指数按代码和月份返回，数值能分辨来源。"""

    def __init__(self):
        self.calls = []

    async def fetch(self, api, params, *, fields="", page_size=None):
        self.calls.append((api, dict(params), page_size))
        if api == "trade_cal":
            year = params["start_date"][:4]
            days = [d for d in DAYS if d.startswith(year)]
            return FetchResult(
                pl.DataFrame(
                    dict(
                        exchange=["SSE"] * len(days),
                        cal_date=days,
                        is_open=[1] * len(days),
                        pretrade_date=days,
                    )
                ),
                "2026-01-20T00:00:00+00:00",
                1,
            )
        if api == "index_basic":
            return FetchResult(basic_rows(), "2026-01-20T00:00:00+00:00", 1)
        if api == "index_daily" and "trade_date" in params:
            assert "ts_code" not in params
            day = params["trade_date"]
            rows = [bar(c, day, 100.0 + i) for i, c in enumerate(WIDE)]
            rows[-1].update(open=None, high=None, low=None)  # 只给收盘价的指数
            rows.append(bar(POOL, day, 9999.0))  # 全市场里的股票池指数，读取时应被略去
            return FetchResult(pl.DataFrame(rows), "2026-01-20T00:00:00+00:00", 1)
        if api == "index_daily":
            days = [d for d in DAYS if params["start_date"] <= d <= params["end_date"]]
            return FetchResult(
                pl.DataFrame([bar(params["ts_code"], d, 3000.0) for d in days]),
                "2026-01-20T00:00:00+00:00",
                1,
            )
        raise AssertionError(f"测试未覆盖的接口 {api}")


def run(store, source, names, **kwargs):
    return asyncio.run(sync(source, store, DAYS[0], DAYS[-1], names, **kwargs))


@pytest.fixture
def synced(tmp_path):
    store = Store(tmp_path)
    source = FakeSource()
    run(store, source, ["index_basic", "index_daily", "index_daily_all"])
    return store, source


def test_catalog_keeps_pool_indexes_separate():
    # 股票池用的指数不变；全市场日线、名单是另外的数据集，默认每日同步都会带上。
    assert INDEXES == {
        "csi300": "000300.SH",
        "csi500": "000905.SH",
        "csi1000": "000852.SH",
        "csi2000": "932000.CSI",
        "csi_all": "000985.CSI",
    }
    assert DATASETS["index_daily"].partition == "index_month_daily"
    assert DATASETS["index_daily_all"].partition == "day"
    assert DATASETS["index_daily_all"].api == "index_daily"
    assert {"index_basic", "index_daily_all"} <= set(DAILY_DATASETS)
    assert DEFAULT_BENCHMARK == "000001.SH"


def test_plan_wide_by_day_pool_by_month(tmp_path):
    store = Store(tmp_path)
    wide = list(plan(DATASETS["index_daily_all"], DAYS[0], DAYS[-1], DAYS, store))
    assert wide == [(d, {"trade_date": d}) for d in DAYS]
    pool = list(plan(DATASETS["index_daily"], DAYS[0], DAYS[-1], DAYS, store))
    assert pool == [
        (f"{code}_202601", dict(ts_code=code, start_date="20260101", end_date=DAYS[-1]))
        for code in INDEXES.values()
    ]
    assert list(plan(DATASETS["index_basic"], DAYS[0], DAYS[-1], DAYS, store)) == [
        ("all", {})
    ]


def test_sync_requests_and_raw_layout(synced):
    store, source = synced
    wide_calls = [
        c for c in source.calls if c[0] == "index_daily" and "trade_date" in c[1]
    ]
    assert sorted(c[1]["trade_date"] for c in wide_calls) == DAYS
    assert {c[2] for c in wide_calls} == {8000}
    pool_calls = [
        c for c in source.calls if c[0] == "index_daily" and "ts_code" in c[1]
    ]
    assert sorted(c[1]["ts_code"] for c in pool_calls) == sorted(INDEXES.values())
    entries = store.entries("index_daily_all")
    assert sorted(e["partition"] for e in entries.values()) == DAYS
    assert all(e["file"].startswith("raw/index_daily_all/") for e in entries.values())
    assert all(
        e["file"].startswith("raw/index_daily/")
        for e in store.entries("index_daily").values()
    )


def test_incremental_does_not_refetch_old_days(synced):
    store, source = synced
    source.calls.clear()
    run(store, source, ["index_daily_all"], refresh=False)
    assert [c for c in source.calls if c[0] == "index_daily"] == []
    # 默认刷新只重抓最近 refresh 个交易日；更早的已完成分区不动。
    source.calls.clear()
    run(store, source, ["index_daily_all"])
    again = sorted(c[1]["trade_date"] for c in source.calls if c[0] == "index_daily")
    assert again == DAYS[-DATASETS["index_daily_all"].refresh :]
    assert len(again) < len(DAYS)
    spec = DATASETS["index_daily_all"]
    entry = store.manifest()[f"index_daily_all/{DAYS[0]}"]
    assert not should_refresh(spec, DAYS[0], entry, DAYS, DAYS[-1])


def test_basic_public_table(synced):
    store, _ = synced
    table = ad.open(store.root.parent).table("index_basic").collect()
    assert set(table.columns) == {
        public for public, _ in PUBLIC_TABLES["index_basic"].values()
    }
    row = table.filter(pl.col("symbol") == "000001.SH").row(0, named=True)
    assert row["name"] == "上证指数" and row["full_name"] == "上海证券综合指数"
    assert row["market"] == "SSE" and row["listing_date"] == "20000101"
    assert "ts_code" not in table.columns
    with pytest.raises(ValueError):
        ad.open(store.root.parent).table("index_basic", start="20260101")


def test_union_keeps_pool_from_original_partitions(synced):
    store, _ = synced
    data = ad.open(store.root.parent)
    table = data.table("index_daily").collect()
    assert table.select("symbol", "date").is_duplicated().sum() == 0
    pool = table.filter(pl.col("symbol") == POOL)
    assert pool["close"].to_list() == [3000.0] * len(
        DAYS
    )  # 来自逐月分区，不是全市场的 9999
    assert set(table["symbol"]) == set(INDEXES.values()) | set(WIDE)
    one = data.table("index_daily", symbols=["399006.SZ"]).sort("date").collect()
    assert one["date"].to_list() == DAYS and one["close"].to_list() == [101.0] * len(
        DAYS
    )
    assert one["volume"].to_list() == [300.0] * len(DAYS)  # 单位换算与原表相同（手→股）
    window = data.table("index_daily", start="2026-01-06", end="20260107").collect()
    assert sorted(set(window["date"])) == DAYS[1:3]


def test_pool_rows_identical_with_or_without_wide(tmp_path):
    # 原来的 5 个指数：有没有全市场日线，读出来逐位相同。
    before_store = Store(tmp_path / "a")
    run(before_store, FakeSource(), ["index_daily"])
    after_store = Store(tmp_path / "b")
    run(after_store, FakeSource(), ["index_daily", "index_daily_all", "index_basic"])
    codes = list(INDEXES.values())

    def pool(store):
        return (
            ad.open(store.root.parent)
            .table("index_daily", symbols=codes)
            .collect()
            .sort("symbol", "date")
        )

    assert pool(before_store).equals(pool(after_store))


def test_only_wide_or_nothing(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(FileNotFoundError):
        ad.open(store.root.parent).table("index_daily").collect()
    run(store, FakeSource(), ["index_daily_all"])
    table = ad.open(store.root.parent).table("index_daily").collect()
    assert set(table["symbol"]) == set(WIDE)  # 股票池指数只认逐月分区，这里没有就不出现
    listing = ad.open(store.root.parent).indexes()
    assert listing["symbol"].to_list() == sorted(WIDE)
    assert listing["name"].null_count() == len(WIDE)  # 没同步名单时简称为空


def test_local_indexes_listing_and_search(synced):
    store, _ = synced
    data = ad.open(store.root.parent)
    listing = data.indexes()
    assert listing.columns == [
        "symbol",
        "name",
        "full_name",
        "market",
        "category",
        "first_date",
        "last_date",
        "has_open",
    ]
    assert listing.filter(~pl.col("has_open"))["symbol"].to_list() == ["H00001.CSI"]
    # 只列有日线的：名单里有、本地没日线的 999999.CSI 不出现。
    assert listing["symbol"].to_list() == sorted(set(INDEXES.values()) | set(WIDE))
    first = listing.filter(pl.col("symbol") == "000001.SH").row(0, named=True)
    assert (first["name"], first["first_date"], first["last_date"]) == (
        "上证指数",
        DAYS[0],
        DAYS[-1],
    )
    assert data.indexes("创业")["symbol"].to_list() == ["399006.SZ"]
    assert data.indexes("h00001")["symbol"].to_list() == ["H00001.CSI"]
    assert data.indexes("上海证券综合")["symbol"].to_list() == ["000001.SH"]
    assert data.indexes("不存在").height == 0
    assert DEFAULT_BENCHMARK in listing["symbol"].to_list()


def test_empty_wide_partition_is_suspect(tmp_path):
    store = Store(tmp_path)
    empty = FetchResult(
        pl.DataFrame(schema={c: pl.String for c in ("ts_code", "trade_date")}),
        date(2026, 1, 20).isoformat(),
        1,
    )
    entry = store.write("index_daily_all", DAYS[0], {"trade_date": DAYS[0]}, empty)
    assert entry["status"] == "suspect"


def test_audit_reports_overlap_values_and_names(synced):
    from ashare.data.indexes import audit_index_daily
    from ashare.data.verify_tables import Audit

    store, _ = synced
    audit = Audit()
    audit_index_daily(store, audit)
    checks = audit.checks
    # 合成的全市场分区里股票池指数故意写成另一个数，逐日都应报出差异。
    assert checks["index_daily_pool_overlap"]["checked"] == len(DAYS)
    assert checks["index_daily_pool_overlap"]["failed"] == len(DAYS)
    assert checks["index_daily_all_values"]["checked"] == len(DAYS) * (len(WIDE) + 1)
    assert checks["index_daily_all_values"]["failed"] == 0
    assert checks["index_basic_covers_daily"]["failed"] == 1
    assert checks["index_basic_covers_daily"]["examples"] == [{"ts_code": "H00001.CSI"}]


def test_audit_overlap_passes_when_values_match(tmp_path):
    from ashare.data.indexes import audit_index_daily
    from ashare.data.verify_tables import Audit

    class Same(FakeSource):
        async def fetch(self, api, params, **kwargs):
            out = await super().fetch(api, params, **kwargs)
            if api == "index_daily" and "trade_date" in params:
                table = out.table.with_columns(
                    pl.when(pl.col("ts_code") == POOL)
                    .then(3000.0)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in ("open", "close", "pre_close")
                ).with_columns(
                    pl.when(pl.col("ts_code") == POOL)
                    .then(pl.col("close") + d)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c, d in (("high", 1), ("low", -1))
                )
                out = FetchResult(table, out.fetched_at, out.calls)
            return out

    store = Store(tmp_path)
    run(store, Same(), ["index_daily", "index_daily_all"])
    audit = Audit()
    audit_index_daily(store, audit)
    assert audit.checks["index_daily_pool_overlap"]["checked"] == len(DAYS)
    assert audit.checks["index_daily_pool_overlap"]["failed"] == 0
    assert "index_basic_covers_daily" not in audit.checks


def test_date_window_prunes_wide_partitions(synced, monkeypatch):
    # 全市场日线有上万个按日分区，带日期读取时只打开窗口内的分区。
    store, _ = synced
    seen = []
    original = Store.scan

    def spy(self, name, start=None, end=None):
        seen.append((name, start, end))
        return original(self, name, start, end)

    monkeypatch.setattr(Store, "scan", spy)
    ad.open(store.root.parent).table(
        "index_daily", start="2026-01-06", end="2026-01-07"
    ).collect()
    assert ("index_daily_all", "20260106", "20260107") in seen
