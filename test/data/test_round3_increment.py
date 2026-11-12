"""覆盖被审查变异遗漏的增量分支，逐字段按位对照独立全量。"""

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


def put(store, name, table, partition="all"):
    store.write(name, partition, {}, FetchResult(table, "2026-02-10T00:00:00Z", 1))


def equal_to_full(store, expected_reuse):
    incremental = build(store, incremental=True)
    assert incremental["reused_rows"] == expected_reuse
    data = ad.open(store.root.parent)
    old = data.panel(data.fields())
    # 保持 mmap 引用；只移开 current 代目录，独立重建会重新从原始表读取。
    import shutil

    version = store.root / "panel" / incremental["version"]
    backup = store.root / "comparison"
    version.rename(backup)
    try:
        full = build(store)
        fresh = ad.open(store.root.parent).panel(data.fields())
        for field in data.fields():
            assert old[field].tobytes() == fresh[field].tobytes(), field
        assert incremental["diagnostics"]["actions"] == full["diagnostics"]["actions"]
    finally:
        shutil.rmtree(backup)


@pytest.mark.parametrize(
    "kind", ["delist", "backdated_security", "middle_column", "weights", "industry"]
)
def test_increment_revision_branches(tables, make_store, kind):
    if kind == "middle_column":
        # 原列 000001、000003；补入 000002，必须把原第三只向右移动。
        tables["stock_basic"] = tables["stock_basic"].filter(
            pl.col("ts_code") != "000002.SZ"
        )
    store = make_store(tables)
    build(store)
    if kind == "delist":
        changed = tables["stock_basic"].with_columns(
            pl.when(pl.col("ts_code") == "000001.SZ")
            .then(pl.lit("20260106"))
            .otherwise(pl.col("delist_date"))
            .alias("delist_date")
        )
        put(store, "stock_basic", changed)
        reuse = 2
    elif kind in ("backdated_security", "middle_column"):
        symbol = "000002.SZ" if kind == "middle_column" else "000000.SZ"
        extra = (
            tables["stock_basic"]
            .head(1)
            .with_columns(
                pl.lit(symbol).alias("ts_code"), pl.lit("20260105").alias("list_date")
            )
        )
        put(store, "stock_basic", pl.concat([tables["stock_basic"], extra]))
        reuse = 1
    elif kind == "weights":
        # 自然月布局，测试该分支而非泛用 all 分区。
        for key in list(store.entries("index_weight")):
            store.mark(key, {"status": "failed"})
        # 删除旧通用条目，不让完整性检查掩盖增量分支。
        store._manifest.db.execute(
            "DELETE FROM partitions WHERE key='index_weight/all'"
        )
        store._manifest.db.commit()
        store._manifest.cache = None
        put(
            store,
            "index_weight",
            tables["index_weight"].with_columns(pl.lit(20.0).alias("weight")),
            "000300.SH_202601",
        )
        reuse = 0
    else:
        change = tables["index_member_all"].with_columns(
            pl.when(pl.col("in_date") == "20260106")
            .then(pl.lit("801030.SI"))
            .otherwise(pl.col("l1_code"))
            .alias("l1_code")
        )
        put(store, "index_member_all", change)
        reuse = 2
    equal_to_full(store, reuse)


def test_cross_month_missing_weight_then_increment(tables, make_store):
    from ashare.data.catalog import INDEXES

    # 1月初尚无12月快照时沿用11月；补抓后必须回退到全体受影响日期。
    tables["index_weight"] = (
        tables["index_weight"]
        .filter(pl.col("trade_date") == "20251231")
        .with_columns(pl.lit("20251128").alias("trade_date"))
    )
    store = make_store(tables)
    meta = build(store)
    assert meta["diagnostics"]["index_snapshots"]["csi300"]["snapshot"] == "20251128"
    assert meta["diagnostics"]["index_snapshots"]["csi300"]["age_days"] == 40
    assert (
        ad.open(store.root.parent).panel(["weight_csi300"])["weight_csi300"][0, 0]
        == 0.6
    )
    with pytest.raises(ValueError, match="快照"):
        build(store, strict=True)
    replacement = tables["index_weight"].with_columns(
        pl.lit("20251231").alias("trade_date"), pl.lit(40.0).alias("weight")
    )
    put(store, "index_weight", replacement, "000300.SH_202512")
    equal_to_full(store, 0)
    assert (
        ad.open(store.root.parent).panel(["weight_csi300"])["weight_csi300"][0, 0]
        == 0.4
    )
    assert len(INDEXES) == 5


def test_quote_tradable_and_integer_storage(tables, make_store):
    tables["daily"] = tables["daily"].with_columns(
        pl.when(
            (pl.col("ts_code") == "000001.SZ") & (pl.col("trade_date") == "20260105")
        )
        .then(0.0)
        .otherwise(pl.col("vol"))
        .alias("vol")
    )
    store = make_store(tables)
    build(store)
    data = ad.open(store.root.parent)
    assert "st_known" not in data.fields()
    p = data.panel(data.fields())
    assert p["has_quote"][:, 0].tolist() == [True, True, False, True]
    assert p["tradable"][:, 0].tolist() == [True, False, False, True]
    assert "st_known" not in data.fields()
    for field in data.fields():
        if p[field].dtype == bool:
            assert p[field].dtype.itemsize == 1
    assert p["action_source"].dtype == np.uint8
    assert p["industry_source"].dtype == np.uint8
    assert p["industry"].dtype == np.uint32
    assert p["listing_age"].dtype == np.uint16


@pytest.mark.parametrize("dataset", ["dividend", "index_weight"])
def test_event_cache_hash_invalidation(tables, make_store, dataset):
    store = make_store(tables)
    if dataset == "index_weight":
        store._manifest.db.execute(
            "DELETE FROM partitions WHERE key='index_weight/all'"
        )
        store._manifest.db.commit()
        store._manifest.cache = None
        put(store, dataset, tables[dataset], "000300.SH_202601")
    build(store)
    if dataset == "dividend":
        changed = tables[dataset].with_columns(
            pl.lit(0.0).alias("stk_div"), pl.lit(1.0).alias("cash_div_tax")
        )
        put(store, dataset, changed, "20260107")
        equal_to_full(store, 3)
        assert (
            ad.open(store.root.parent).panel(["cash_dividend"])["cash_dividend"][3, 2]
            == 1.0
        )
    else:
        put(
            store,
            dataset,
            tables[dataset].with_columns(pl.lit(25.0).alias("weight")),
            "000300.SH_202601",
        )
        equal_to_full(store, 0)
        assert (
            ad.open(store.root.parent).panel(["weight_csi300"])["weight_csi300"][3, 2]
            == 0.25
        )
