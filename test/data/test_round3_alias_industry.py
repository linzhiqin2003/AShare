"""换码的全部证券数据和行业来源必须有一致的规范化语义。"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ashare
import ashare.data as ad
from ashare.data.aliases import alias_context, canonical
from ashare.data.panel import build
from ashare.data.panel_events import stitched_industry
from ashare.data.sources.base import FetchResult

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def test_alias_prices_factors_all_datasets(tables, make_store, monkeypatch):
    import ashare.data.aliases as module

    alias = dict(old="OLD.SZ", new="000003.SZ", old_last="20260105", switch="20260106")
    monkeypatch.setattr(module, "SYMBOL_ALIASES", (alias,))
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
            tables[name] = pl.concat([table, old])
    store = make_store(tables)
    hashes = {k: e["content_hash"] for k, e in store.manifest().items()}
    context = alias_context(store)
    assert context[0]["factor_scale"] == 10
    meta = build(store)
    p = ad.open(store.root.parent).panel(
        ["close", "adj_factor", "upper_limit", "is_st"]
    )
    assert p["close"][:, 2].tolist() == [10, 10, 10, 9]
    np.testing.assert_allclose(
        p["adj_factor"][:, 2], [10, 10, 10, 100 / 9], rtol=0, atol=1e-14
    )
    assert p["adj_factor"][2, 2] / p["adj_factor"][1, 2] == 1
    assert np.isfinite(p["upper_limit"][:, 2]).all()
    assert meta["aliases"][0]["old_close"] == meta["aliases"][0]["new_previous_close"]
    assert hashes == {k: e["content_hash"] for k, e in store.manifest().items()}
    for dataset in (
        "daily",
        "daily_basic",
        "adj_factor",
        "stk_limit",
        "stock_st",
        "suspend_d",
        "dividend",
        "namechange",
        "index_member_all",
        "index_member",
        "index_weight",
        "margin_secs",
    ):
        column = (
            "con_code" if dataset in ("index_member", "index_weight") else "ts_code"
        )
        from ashare.data.catalog import DATASETS

        values = {c: ["x"] for c in DATASETS[dataset].keys}
        values[column] = ["OLD.SZ"]
        if dataset == "adj_factor":
            values["adj_factor"] = [2.0]
        frame = canonical(pl.DataFrame(values), dataset, context)
        assert frame[column].to_list() == ["000003.SZ"], dataset
    changed = tables["daily"].with_columns(
        pl.when(
            (pl.col("ts_code") == "000003.SZ") & (pl.col("trade_date") == "20260106")
        )
        .then(1.0)
        .otherwise(pl.col("pre_close"))
        .alias("pre_close")
    )
    store.write(
        "daily",
        "20260106",
        {},
        FetchResult(changed.filter(pl.col("trade_date") == "20260106"), "later", 1),
    )
    with pytest.raises(ValueError, match="接缝参考价"):
        alias_context(store)


def test_industry_backfill_source(tables, make_store):
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
            in_date=["20140221"],
            out_date=["20200101"],
            is_new=["N"],
        )
    )
    store.write("index_member", "old", {}, FetchResult(old, "now", 1))
    dates = np.array(["20130104", "20140221", "20210104", "20210802"])
    values, sources, _ = stitched_industry(store, dates, ["000001.SZ", "UNKNOWN"])
    assert values[:, 0].tolist() == [801010, 801020, 801010, 801010]
    assert sources[:, 0].tolist() == [3, 1, 3, 2]
    assert sources[:, 1].tolist() == [0, 0, 0, 0]
    assert np.isnan(values[:, 1]).all()
