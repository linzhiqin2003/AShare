"""第四轮：换码优先级与拦截、行业来源边界、增量的两个布局分支。全部为合成夹具。"""

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ashare
import ashare.data as ad
import ashare.data.aliases as aliases
from ashare.data.aliases import alias_context, canonical
from ashare.data.panel import build
from ashare.data.panel_events import industry_mixing, stitched_industry
from ashare.data.sources.base import FetchResult

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)

ALIAS = dict(old="OLD.SZ", new="000003.SZ", old_last="20260105", switch="20260106")
CONTEXT = [dict(**ALIAS, factor_scale=1.0)]


def put(store, name, table, partition="all"):
    store.write(name, partition, {}, FetchResult(table, "2026-02-10T00:00:00Z", 1))


def equal_to_full(store, expected_reuse):
    incremental = build(store, incremental=True)
    assert incremental["reused_rows"] == expected_reuse
    data = ad.open(store.root.parent)
    old = data.panel(data.fields())
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


def alias_tables(tables):
    """000003.SZ 在 20260106 由 OLD.SZ 换码；旧码因子是新码的 1/10。"""
    for name, table in list(tables.items()):
        if "ts_code" in table.columns and "trade_date" in table.columns:
            if name == "adj_factor":
                table = table.with_columns(
                    pl.when(pl.col("ts_code") == "000003.SZ")
                    .then(pl.col("adj_factor") * 10)
                    .otherwise(pl.col("adj_factor"))
                    .alias("adj_factor")
                )
            old = table.filter(
                (pl.col("ts_code") == "000003.SZ") & (pl.col("trade_date") < "20260106")
            ).with_columns(pl.lit("OLD.SZ").alias("ts_code"))
            if name == "adj_factor":
                old = old.with_columns((pl.col("adj_factor") / 10).alias("adj_factor"))
            # 新码只保留换码后的行；源回写的旧历史由各测试自己加。
            new = table.filter(
                (pl.col("ts_code") != "000003.SZ")
                | (pl.col("trade_date") >= "20260106")
            )
            tables[name] = pl.concat([new, old])
    return tables


def test_alias_priority_before_and_after_switch(monkeypatch):
    monkeypatch.setattr(aliases, "SYMBOL_ALIASES", (ALIAS,))
    # X14：新码回写了旧历史时，接缝前取旧码，接缝后取新码。
    daily = pl.DataFrame(
        dict(
            ts_code=["OLD.SZ", "000003.SZ", "OLD.SZ", "000003.SZ"],
            trade_date=["20260105", "20260105", "20260106", "20260106"],
            close=[10.0, 99.0, 77.0, 9.0],
        )
    )
    out = canonical(daily, "daily", CONTEXT).sort("trade_date")
    assert out["ts_code"].to_list() == ["000003.SZ"] * 2
    assert out["close"].to_list() == [10.0, 9.0]


def test_alias_dividend_full_column_dedup(monkeypatch):
    monkeypatch.setattr(aliases, "SYMBOL_ALIASES", (ALIAS,))
    # X18：新旧码各一条相同记录只记一次；同主键不同条款的两条都保留。
    base = dict(
        end_date="20251231",
        ex_date="20260107",
        div_proc="实施",
        ann_date="20260101",
        imp_ann_date="20260105",
        stk_div=0.0,
    )
    same = pl.DataFrame(
        [dict(base, ts_code=c, cash_div_tax=0.5) for c in ("OLD.SZ", "000003.SZ")]
    )
    assert len(canonical(same, "dividend", CONTEXT)) == 1
    split = pl.DataFrame(
        [
            dict(base, ts_code="OLD.SZ", cash_div_tax=0.5),
            dict(base, ts_code="000003.SZ", cash_div_tax=0.2),
        ]
    )
    out = canonical(split, "dividend", CONTEXT)
    assert sorted(out["cash_div_tax"].to_list()) == [0.2, 0.5]


def test_alias_seam_action_and_missing_quote_rejected(tables, make_store, monkeypatch):
    monkeypatch.setattr(aliases, "SYMBOL_ALIASES", (ALIAS,))
    tables = alias_tables(tables)
    store = make_store(tables)
    assert alias_context(store)[0]["factor_scale"] == 10
    # X15：接缝日有公司行为。
    seam = tables["dividend"].with_columns(pl.lit("20260106").alias("ex_date"))
    put(store, "dividend", seam, "20260106")
    with pytest.raises(ValueError, match="接缝有公司行为"):
        alias_context(store)
    put(store, "dividend", seam.head(0), "20260106")
    # 接缝在库内却缺一边的行情：不能静默按因子 1 拼接。
    day = tables["daily"].filter(
        (pl.col("trade_date") == "20260106") & (pl.col("ts_code") != "000003.SZ")
    )
    put(store, "daily", day, "20260106")
    with pytest.raises(ValueError, match="缺行情"):
        alias_context(store)


def test_alias_context_change_forces_full_rebuild(tables, make_store, monkeypatch):
    # X19：换码因子基准变化（接缝日新码因子被源修订）必须全量重建。
    monkeypatch.setattr(aliases, "SYMBOL_ALIASES", (ALIAS,))
    tables = alias_tables(tables)
    store = make_store(tables)
    build(store)
    revised = tables["adj_factor"].with_columns(
        pl.when(pl.col("ts_code") == "000003.SZ")
        .then(pl.col("adj_factor") * 2)
        .otherwise(pl.col("adj_factor"))
        .alias("adj_factor")
    )
    for day in ("20260106", "20260107"):
        put(store, "adj_factor", revised.filter(pl.col("trade_date") == day), day)
    equal_to_full(store, 0)
    factor = ad.open(store.root.parent).panel(["adj_factor"])["adj_factor"][:, 2]
    np.testing.assert_allclose(factor, [20, 20, 20, 200 / 9], rtol=1e-14)


def test_industry_source_forced_before_sw2014(tables, make_store):
    # X13：SW2014 发布前的旧版历史也只能标来源 3。
    tables["index_member_all"] = (
        tables["index_member_all"]
        .head(1)
        .with_columns(
            pl.lit("19900101").alias("in_date"),
            pl.lit(None, dtype=pl.String).alias("out_date"),
        )
    )
    store = make_store(tables)
    old = pl.DataFrame(
        dict(
            con_code=["000001.SZ"],
            index_code=["801020.SI"],
            in_date=["20100101"],
            out_date=["20200101"],
            is_new=["N"],
        )
    )
    store.write("index_member", "old", {}, FetchResult(old, "now", 1))
    dates = np.array(["20130104", "20140220", "20140221", "20210104"])
    values, sources, _ = stitched_industry(store, dates, ["000001.SZ"])
    assert values[:, 0].tolist() == [801020, 801020, 801020, 801010]
    assert sources[:, 0].tolist() == [3, 3, 1, 3]


def test_industry_mixing_counts_foreign_codes():
    dates = np.array(["20130104", "20140221", "20200601", "20210802"])
    values = np.array([[801010, 801970, 0], [801010, 801970, 801980]] * 2)
    sources = np.array([[3, 3, 0], [1, 3, 3], [1, 3, 1], [2, 2, 2]])
    listed = np.ones((4, 3), bool)
    out = industry_mixing(values, sources, listed, dates)
    # 第二天来源1只有 801010：801970、801980 两格；第三天 801970 一格；第四天已在切换日后。
    assert out["foreign_code_cells"] == 3
    assert out["restated_before_20140221"] == 2


def test_weight_revision_in_generic_partition(tables, make_store):
    # 指数权重分区名不是「代码_年月」时，修订必须整体重算。
    store = make_store(tables)
    build(store)
    put(
        store,
        "index_weight",
        tables["index_weight"].with_columns(pl.lit(30.0).alias("weight")),
    )
    equal_to_full(store, 0)
    weight = ad.open(store.root.parent).panel(["weight_csi300"])["weight_csi300"]
    assert weight[3, 2] == pytest.approx(0.3)


def test_industry_null_typed_column_increment(tables, make_store):
    # 旧分区 out_date 整列为空（Null 类型）时增量不得报 SchemaError。
    first = (
        tables["index_member_all"]
        .head(1)
        .with_columns(pl.Series("out_date", [None], dtype=pl.Null))
    )
    store = make_store({**tables, "index_member_all": first})
    build(store)
    put(store, "index_member_all", tables["index_member_all"])
    equal_to_full(store, 0)
