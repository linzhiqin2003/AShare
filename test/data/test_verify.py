from pathlib import Path

import polars as pl
import pytest

import ashare
from ashare.data.rules import limit_price, limit_rates
from ashare.data.verify import Audit, audit_limits, audit_tables

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


@pytest.mark.parametrize(
    "rule",
    [
        "daily_coverage",
        "unique_keys",
        "ohlc_high",
        "ohlc_low",
        "adjustment_continuity",
        "ex_dividend_reference",
        "suspension_no_trade",
        "missing_quote_has_suspension",
        "no_post_delist_rows",
        "market_cap",
    ],
)
def test_detect_each_violation(tables, rule):
    baseline = audit_tables(tables, ["20260102", "20260105", "20260106", "20260107"])
    assert baseline.checks[rule]["failed"] == 0
    daily = tables["daily"]
    if rule == "daily_coverage":
        tables["daily"] = daily.filter(pl.col("trade_date") != "20260105")
    elif rule == "unique_keys":
        tables["daily"] = pl.concat([daily, daily.head(1)])
    elif rule == "ohlc_high":
        tables["daily"] = daily.with_columns(pl.lit(0.0).alias("high"))
    elif rule == "ohlc_low":
        tables["daily"] = daily.with_columns(pl.lit(999.0).alias("low"))
    elif rule == "adjustment_continuity":
        tables["adj_factor"] = tables["adj_factor"].with_columns(
            pl.when(pl.col("trade_date") == "20260105")
            .then(2.0)
            .otherwise(1.0)
            .alias("adj_factor")
        )
    elif rule == "ex_dividend_reference":
        tables["dividend"] = tables["dividend"].with_columns(
            pl.lit(5.0).alias("cash_div_tax")
        )
    elif rule == "suspension_no_trade":
        tables["suspend_d"] = tables["suspend_d"].with_columns(
            pl.lit("20260105").alias("trade_date")
        )
    elif rule == "missing_quote_has_suspension":
        tables["suspend_d"] = tables["suspend_d"].head(0)
    elif rule == "no_post_delist_rows":
        tables["stock_basic"] = tables["stock_basic"].with_columns(
            pl.lit("20260106").alias("delist_date")
        )
    elif rule == "market_cap":
        tables["daily_basic"] = tables["daily_basic"].with_columns(
            pl.lit(1.0).alias("total_mv")
        )
    report = audit_tables(tables, ["20260102", "20260105", "20260106", "20260107"])
    assert report.checks[rule]["failed"] > 0
    assert report.checks[rule]["examples"]


def test_price_rules_and_violation(tables):
    assert limit_rates("000001.SZ", "20260703", "19900101", 100, True) == (0.05, 0.05)
    assert limit_rates("000001.SZ", "20260706", "19900101", 100, True) == (0.1, 0.1)
    assert limit_rates("300001.SZ", "20200824", "20090101", 100, True) == (0.2, 0.2)
    assert limit_rates("688001.SH", "20260102", "20260102", 1, False) is None
    assert limit_rates("000001.SZ", "20230501", "20230501", 5, False) is None
    assert limit_rates("920001.BJ", "20260102", "20260101", 2, True) == (0.3, 0.3)
    assert limit_price(10.05, 0.1) == 11.06
    tables["stock_basic"] = tables["stock_basic"].with_columns(
        pl.lit("19900101").alias("list_date")
    )
    baseline = Audit()
    audit_limits(tables, ["20260102", "20260105", "20260106", "20260107"], baseline)
    assert baseline.checks["price_limits"]["failed"] == 0
    tables["stk_limit"] = tables["stk_limit"].with_columns(
        pl.lit(999.0).alias("up_limit")
    )
    tables["stock_basic"] = tables["stock_basic"].with_columns(
        pl.lit("19900101").alias("list_date")
    )
    audit = Audit()
    audit_limits(tables, ["20260102", "20260105", "20260106", "20260107"], audit)
    assert audit.checks["price_limits"]["failed"] > 0


def test_non_reformed_and_minimum_tick():
    assert limit_rates("600182.SH", "20261111", "19990507", 100, False, True) == (
        0.05,
        0.05,
    )
    assert limit_price(0.04, 0.1) == 0.05
    assert limit_price(0.04, 0.1, upper=False) == 0.03


def test_bse_inward_rounding_real_sample():
    assert limit_price(10.82, 0.3, inward=True) == 14.06
    assert limit_price(10.82, 0.3, upper=False, inward=True) == 7.58


def test_report_missing_partition_and_bad_file(tables, make_store):
    from ashare.data.verify import verify

    store = make_store(tables)
    store.mark("stock_st/20260106", {"status": "failed"})
    entry = store.manifest()["adj_factor/20260102"]
    # 可解码但内容被改写：哈希检查必须识别。
    path = store.root / entry["file"]
    table = pl.read_parquet(path)
    table.with_columns(pl.lit(2.0).alias("adj_factor")).write_parquet(path)
    report = verify(store)
    assert "20260106" in report["coverage"]["stock_st"]["missing_partitions"]
    assert "20260102" in report["coverage"]["margin_secs"]["missing_partitions"]
    assert report["checks"]["manifest_files"]["failed"] == 2
    assert (store.root / "reports/verify.md").exists()
    assert (store.root / "reports/verify.json").exists()


def test_baostock_cache_invalidated_on_revision(tables, make_store):
    from ashare.data.crosscheck import fingerprint
    from ashare.data.store import atomic_json
    from ashare.data.verify import verify

    store = make_store(tables)
    atomic_json(
        store.root / "reports/baostock.json",
        {"status": "complete", "input_fingerprint": fingerprint(store)},
    )
    assert verify(store)["baostock"]["status"] == "complete"
    store.mark("daily/20260102", {"content_hash": "changed"})
    assert verify(store)["baostock"]["status"] == "stale"
