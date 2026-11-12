import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ashare
import ashare.data as ad
from ashare.data.catalog import FIELDS
from ashare.data.maintenance import collect_garbage, disk_usage
from ashare.data.panel import build, scatter
from ashare.data.panel_events import daily_mask, dividends, listing, stitched_industry
from ashare.data.panel_quality import securities_table
from ashare.data.sources.base import FetchResult
from ashare.data.verify import audit_tables

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def test_real_duplicate_and_distinct_distributions():
    table = pl.DataFrame(
        json.loads(
            (Path(__file__).parent / "fixtures/dividend-301393.json").read_text()
        )
    )
    diagnostics = {}
    split, cash = dividends(
        table, np.array(["20261110"]), ["301393.SZ"], diagnostics=diagnostics
    )
    assert cash[0, 0] == 0.15 and split[0, 0] == 1
    assert diagnostics["duplicate_rows_removed"] == 1
    distinct = table.head(1).with_columns(pl.lit(0.2).alias("cash_div_tax"))
    _, cash = dividends(
        pl.concat([table, distinct]), np.array(["20261110"]), ["301393.SZ"]
    )
    assert cash[0, 0] == 0.35
    stock = pl.concat(
        [
            table.head(1).with_columns(pl.lit(0.1).alias("stk_div")),
            distinct.with_columns(pl.lit(0.2).alias("stk_div")),
        ]
    )
    split, _ = dividends(stock, np.array(["20261110"]), ["301393.SZ"])
    assert split[0, 0] == 1.3


def test_weekend_delist_full_calendar():
    calendar = np.array(["20260102", "20260105", "20260106"])
    table = pl.DataFrame(
        dict(ts_code=["A"], list_date=["20200101"], delist_date=["20260103"])
    )
    _, _, full = listing(calendar, ["A"], table, calendar)
    _, _, tail = listing(calendar[1:], ["A"], table, calendar)
    assert full[:, 0].tolist() == [False, True, False]
    assert tail.tobytes() == full[1:].tobytes()


def test_non_listed_mask_even_when_raw_contains_values(tables, make_store):
    extra = tables["daily"].head(1).with_columns(pl.lit("000002.SZ").alias("ts_code"))
    tables["daily"] = pl.concat([tables["daily"], extra])
    tables["stock_st"] = pl.concat(
        [
            tables["stock_st"],
            pl.DataFrame(
                dict(ts_code=["000002.SZ"], trade_date=["20260102"], type=["ST"])
            ),
        ]
    )
    store = make_store(tables)
    build(store)
    data = ad.open(store.root.parent)
    panel = data.panel(data.fields())
    for field in data.fields():
        if field in ("delist", "is_listed"):
            continue
        value = panel[field][0, 1]
        assert (
            np.isnan(value)
            if np.issubdtype(panel[field].dtype, np.floating)
            else value == 0
        ), field


def test_resumption_and_intraday_are_not_all_day_suspension():
    table = pl.DataFrame(
        dict(
            ts_code=["A", "B", "C"],
            trade_date=["20260102"] * 3,
            suspend_type=["S", "R", "S"],
            suspend_timing=[None, None, "09:30-10:00"],
        )
    )
    assert daily_mask(
        table, np.array(["20260102"]), ["A", "B", "C"], suspension=True
    ).tolist() == [[True, False, False]]


def test_stale_index_and_empty_partition_refused(tables, make_store):
    missing = tables.copy()
    missing["index_weight"] = tables["index_weight"].filter(
        pl.col("index_code") != "000300.SH"
    )
    with pytest.raises(ValueError, match="快照"):
        build(make_store(missing, "missing"), strict=True)
    stale = tables.copy()
    stale["index_weight"] = tables["index_weight"].filter(
        pl.col("trade_date") != "20251231"
    )
    with pytest.raises(ValueError, match="202512"):
        build(make_store(stale, "stale"), strict=True)
    store = make_store(tables, "empty")
    with pytest.raises(ValueError, match="可疑空表"):
        store.write(
            "stock_st",
            "20260102",
            {},
            FetchResult(tables["stock_st"].head(0), "now", 1),
        )
    store.mark("stock_st/20260102", {"rows": 0})
    with pytest.raises(ValueError, match="空分区"):
        build(store)


def test_zero_weight_membership_and_join_diagnostics(tables, make_store):
    tables["index_weight"] = tables["index_weight"].with_columns(
        pl.lit(0.0).alias("weight")
    )
    store = make_store(tables)
    meta = build(store)
    panel = ad.open(store.root.parent).panel(["weight_csi300", "member_csi300"])
    assert panel["weight_csi300"][0, 0] == 0 and panel["member_csi300"][0, 0]
    diag = {}
    list(
        scatter(
            tables["daily"],
            np.array(["20260102"]),
            ["000001.SZ"],
            [FIELDS["close"]],
            diagnostics=diag,
        )
    )
    assert diag["unknown_symbol_rows"] == 1
    assert meta["code_version"]


def test_rename_preserves_incremental_and_gc(tables, make_store):
    store = make_store(tables)
    old = build(store, end="20260106")
    changed = tables["stock_basic"].with_columns(pl.lit("更名").alias("name"))
    store.write("stock_basic", "all", {}, FetchResult(changed, "later", 1))
    meta = build(store, incremental=True)
    assert meta["reused_rows"] == 3 and meta["version"] != old["version"]
    assert len([p for p in (store.root / "panel").iterdir() if p.is_dir()]) == 1
    assert ad.open(store.root.parent).securities().collect()["name"][0] == "更名"
    (store.root / "panel/.building-crashed").mkdir()
    (store.root / "projections/orphan").mkdir(parents=True)
    collect_garbage(store)
    assert not (store.root / "panel/.building-crashed").exists()
    assert not (store.root / "projections").exists()
    assert disk_usage(store.root)["panel"]["bytes"] > 0


def test_missing_announcement_and_factor_event_detection(tables):
    changed = tables["dividend"].with_columns(
        pl.lit(None, dtype=pl.String).alias("imp_ann_date")
    )
    stats = {}
    _, cash = dividends(
        changed, np.array(["20260107"]), ["000003.SZ"], diagnostics=stats
    )
    assert cash[0, 0] == 0.1 and stats["missing_announcement"] == 1
    baseline = audit_tables(tables, ["20260102", "20260105", "20260106", "20260107"])
    assert baseline.checks["factor_change_has_event"]["failed"] == 0
    tables["dividend"] = tables["dividend"].head(0)
    broken = audit_tables(tables, ["20260102", "20260105", "20260106", "20260107"])
    assert broken.checks["factor_change_has_event"]["failed"] == 1


def test_security_determinism_calendar_and_long_tables(tables, make_store):
    store = make_store(tables)
    other = (
        tables["stock_basic"]
        .head(1)
        .with_columns(
            pl.lit("后来").alias("name"), pl.lit("20260106").alias("delist_date")
        )
    )
    store.write("stock_basic", "D", {}, FetchResult(other, "2026-01-09T00:00:00Z", 1))
    assert securities_table(store)["name"][0] == "后来"
    build(store)
    data = ad.open(store.root.parent)
    assert data.calendar().collect()["date"].to_list() == [
        "20260102",
        "20260105",
        "20260106",
        "20260107",
    ]
    store.write("index_daily", "sample", {}, FetchResult(tables["daily"], "now", 1))
    data = ad.open(store.root.parent)
    assert data.table("index_daily").collect()["volume"][0] == 200
    assert "ts_code" not in data.table("index_daily").collect_schema()


def test_legacy_industry_stitch(tables, make_store):
    store = make_store(tables)
    legacy = pl.DataFrame(
        dict(
            con_code=["000001.SZ"],
            index_code=["801020.SI"],
            in_date=["19900101"],
            out_date=[None],
            is_new=["N"],
        )
    )
    store.write("index_member", "old", {}, FetchResult(legacy, "now", 1))
    current = (
        tables["index_member_all"]
        .head(1)
        .with_columns(
            pl.lit("801950.SI").alias("l1_code"),
            pl.lit("19900101").alias("in_date"),
            pl.lit(None, dtype=pl.String).alias("out_date"),
        )
    )
    store.write("index_member_all", "all", {}, FetchResult(current, "now", 1))
    array, sources, caveat = stitched_industry(
        store, np.array(["20210730", "20210802"]), ["000001.SZ"]
    )
    assert array[:, 0].tolist() == [801020, 801950]
    assert "2014-02-21前" in caveat
    assert sources[:, 0].tolist() == [1, 2]


def test_code_version_invalidates_cached_build(tables, make_store, monkeypatch):
    import ashare.data.panel as module

    store = make_store(tables)
    first = build(store)
    monkeypatch.setattr(module, "code_version", lambda: "changed-build-code")
    second = build(store, incremental=True)
    assert first["version"] != second["version"] and second["reused_rows"] == 0


def test_month_partition_layout_missing_index(tables, make_store):
    from ashare.data.catalog import INDEXES

    subset = {k: v for k, v in tables.items() if k != "index_weight"}
    store = make_store(subset)
    for name, code in INDEXES.items():
        if name == "csi300":
            continue
        table = tables["index_weight"].filter(
            (pl.col("index_code") == code) & (pl.col("trade_date") == "20251231")
        )
        store.write(
            "index_weight",
            f"{code}_202512",
            {"start_date": "20251201", "end_date": "20251231"},
            FetchResult(table, "2026-01-08T00:00:00Z", 1),
        )
    with pytest.raises(ValueError, match="csi300.*202512"):
        build(store, strict=True)
