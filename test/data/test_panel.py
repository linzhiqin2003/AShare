from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ashare
import ashare.data as ad
from ashare.data.panel import build
from ashare.data.sources.base import FetchResult

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def test_cells_roundtrip_and_metadata(tables, make_store):
    store = make_store(tables)
    build(store)
    data = ad.open(store.root.parent)
    p = data.panel(data.fields())
    assert p.shape == (4, 3)
    assert p["volume"][0, 0] == 200
    assert p["amount"][0, 0] == 20
    assert p["vwap"][0, 0] == 0.1
    assert p["total_market_cap"][0, 0] == 200000
    assert p["float_market_cap"][0, 0] == 200000
    assert p["total_shares"][0, 0] == 20000
    assert p["turnover"][0, 0] == np.float32(0.02)
    assert np.isnan(p["close"][2, 0])
    assert p["is_suspended"][2, 0]
    assert not p["is_suspended"][1, 0]
    assert p["is_st"][:, 0].tolist() == [False, True, False, False]
    assert p["is_listed"][:, 1].tolist() == [False, True, True, False]
    assert p["delist"][:, 1].tolist() == [False, False, False, True]
    assert np.isnan(p["volume"][0, 1]) and np.isnan(p["close"][3, 1])
    np.testing.assert_array_equal(p["listing_age"][:, 1], [0, 1, 2, 0])
    np.testing.assert_array_equal(p["industry"][:, 0], [801010, 801010, 801020, 801020])
    assert p["weight_csi300"][0, 0] == 0.6
    np.testing.assert_array_equal(p["weight_csi300"][1:, 0], [0.6, 0.6, 0])
    assert p["weight_csi300"][3, 2] == 1
    assert p["member_csi300"][:, 0].tolist() == [True, True, True, False]
    assert p["split"][3, 2] == 1.1 and p["cash_dividend"][3, 2] == 0.1
    for name in data.fields():
        assert p[name].flags.c_contiguous
        assert not p[name].flags.writeable
        assert isinstance(p[name], np.memmap)
    subset = data.panel(["close"], start="2026-01-05", end="2026-01-06")
    assert isinstance(subset["close"], np.memmap)
    assert subset["close"].shape == (2, 3)
    for row in tables["daily"].iter_rows(named=True):
        i = list(p.dates).index(row["trade_date"])
        j = list(p.symbols).index(row["ts_code"])
        for field in ("open", "high", "low", "close"):
            assert p[field][i, j] == row[field]
    assert data.field("volume")["unit"] == "股"
    assert data.table("dividend").collect()["cash_dividend"][0] == 0.1
    with pytest.raises(KeyError, match="字段"):
        data.panel(["does_not_exist"])
    with pytest.raises(ValueError, match="超出"):
        data.panel(["close"], start="19900101")
    with pytest.raises(ValueError):
        ad.open(store.root.parent, market="jp")


def test_incremental_bitwise(tables, make_store):
    short = {
        k: (
            v.filter(pl.col("trade_date") < "20260107")
            if "trade_date" in v.columns
            else v
        )
        for k, v in tables.items()
    }
    store = make_store(short, "incremental")
    build(store, end="20260106")
    for name in (
        "daily",
        "adj_factor",
        "daily_basic",
        "stk_limit",
        "stock_st",
        "suspend_d",
    ):
        store.write(
            name,
            "20260107",
            {"trade_date": "20260107"},
            FetchResult(
                tables[name].filter(pl.col("trade_date") == "20260107"), "now", 1
            ),
        )
    meta = build(store, incremental=True)
    assert meta["reused_rows"] == 3
    full = make_store(tables, "full")
    build(full)
    a, b = ad.open(store.root.parent), ad.open(full.root.parent)
    pa, pb = a.panel(a.fields()), b.panel(b.fields())
    for name in a.fields():
        assert pa[name].tobytes() == pb[name].tobytes(), name
    # 旧历史被修订后必须重新计算受影响的日期。
    changed = (
        tables["daily"]
        .filter(pl.col("trade_date") == "20260105")
        .with_columns(pl.lit(20.0).alias("close"))
    )
    store.write("daily", "20260105", {}, FetchResult(changed, "later", 1))
    assert build(store, incremental=True)["reused_rows"] == 1
    assert ad.open(store.root.parent).panel(["close"])["close"][1, 0] == 20


def test_new_column_incremental(tables, make_store):
    tables["stock_basic"] = tables["stock_basic"].with_columns(
        pl.when(pl.col("ts_code") == "000003.SZ")
        .then(pl.lit("20260107"))
        .otherwise(pl.col("list_date"))
        .alias("list_date")
    )
    for name, table in list(tables.items()):
        if "ts_code" in table.columns and "trade_date" in table.columns:
            tables[name] = table.filter(
                (pl.col("ts_code") != "000003.SZ")
                | (pl.col("trade_date") == "20260107")
            )
    # 测试市场每天保留至少一条 ST 记录，新增证券以前也满足全市场完整性要求。
    tables["stock_st"] = pl.DataFrame(
        dict(
            ts_code=["000001.SZ"] * 4,
            trade_date=["20260102", "20260105", "20260106", "20260107"],
            type=["ST"] * 4,
        )
    )
    old = {
        k: (v.filter(pl.col("ts_code") != "000003.SZ") if "ts_code" in v.columns else v)
        for k, v in tables.items()
    }
    store = make_store(old, "growing")
    build(store, end="20260106")
    store.write(
        "stock_basic", "all", {}, FetchResult(tables["stock_basic"], "later", 1)
    )
    for name in ("daily", "adj_factor", "daily_basic", "stk_limit"):
        store.write(
            name,
            "20260107",
            {},
            FetchResult(
                tables[name].filter(pl.col("trade_date") == "20260107"), "later", 1
            ),
        )
    store.write("dividend", "20260107", {}, FetchResult(tables["dividend"], "later", 1))
    meta = build(store, incremental=True)
    assert meta["reused_rows"] == 3
    full = make_store(tables, "full")
    build(full)
    a, b = ad.open(store.root.parent), ad.open(full.root.parent)
    for field in a.fields():
        assert a.panel([field])[field].tobytes() == b.panel([field])[field].tobytes(), (
            field
        )


def test_missing_partition_and_late_dividend(tables, make_store):
    tables["dividend"] = tables["dividend"].with_columns(
        pl.lit("20260108").alias("imp_ann_date")
    )
    store = make_store(tables)
    build(store)
    assert (
        ad.open(store.root.parent).panel(["cash_dividend"])["cash_dividend"][3, 2] == 0
    )
    store.mark("stock_st/20260106", dict(status="failed"))
    with pytest.raises(FileNotFoundError, match="stock_st"):
        build(store)


def test_handwritten_industry_boundary_and_projected_mmap(tables, make_store):
    from ashare.data.panel_events import industry

    rows = [
        dict(
            ts_code="EXAMPLE",
            l1_code="801150.SI",
            in_date="20210101",
            out_date="20210610",
        ),
        dict(ts_code="EXAMPLE", l1_code="801750.SI", in_date="20210611", out_date=None),
    ]
    array = industry(
        pl.DataFrame(rows), np.array(["20210610", "20210611"]), ["EXAMPLE"]
    )
    np.testing.assert_array_equal(array[:, 0], [801150, 801750])
    store = make_store(tables)
    build(store)
    p = ad.open(store.root.parent).panel(["close"], symbols=["000003.SZ", "000001.SZ"])
    assert isinstance(p["close"], np.memmap)
    assert p["close"].flags.c_contiguous
    assert not p["close"].flags.writeable
    assert np.isnan(p["close"][2, 1])


def test_clone_does_not_change_old_reader(tmp_path):
    from ashare.data.panel_io import clone_append

    original = tmp_path / "old.npy"
    target = tmp_path / "new.npy"
    np.save(original, np.arange(12, dtype=np.float64).reshape(4, 3))
    old = np.load(original, mmap_mode="r")
    cloned = clone_append(original, target, (5, 3))
    if cloned is None:
        cloned = np.lib.format.open_memmap(
            target, mode="w+", dtype=old.dtype, shape=(5, 3)
        )
        cloned[:4] = old
    cloned[0, 0] = 99
    cloned[4] = 22
    cloned.flush()
    assert old[0, 0] == 0
    assert np.load(target)[4, 0] == 22
