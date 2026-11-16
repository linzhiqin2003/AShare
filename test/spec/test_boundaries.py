"""D2 补充边界：类型词典、非行情可知时点、引用、公共质量位与只报告检查。"""

from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from ashare.data.spec import check, open
from ashare.data.spec.constants import FIELDS
from test.spec.conftest import (
    assert_bad,
    assert_good,
    item,
    read_manifest,
    resign,
    write_manifest,
)
from test.spec.test_contract import add_field, change, hourly


def test_s5_1_dictionary_complete():
    # 来源：D2 开发首次发现词典漏 previous_close/trade_count/open_interest/settle。
    assert set(FIELDS) == set(
        "open high low close previous_close vwap volume amount trade_count open_interest settle adj_factor limit_up limit_down suspended taker_buy_volume".split()
    )


@pytest.mark.parametrize(
    "field,dtype,unit,value",
    [
        ("trade_count", pl.Int64, "count", 2),
        ("open_interest", pl.Float64, "share", 100.0),
        ("settle", pl.Float64, "CNY/share", 12.0),
        ("previous_close", pl.Float64, "CNY/share", 10.0),
    ],
)
def test_s5_1_all_dictionary_fields(package, field, dtype, unit, value):
    add_field(
        package,
        dict(name=field, type="integer" if dtype == pl.Int64 else "number", unit=unit),
        pl.lit(value, dtype=dtype),
    )
    assert_good(package)
    m = read_manifest(package)
    m["resources"][0]["schema"]["fields"][-1]["type"] = "string"
    write_manifest(package, m)
    assert_bad(package, "dictionary")


@pytest.mark.parametrize("field", ["asset_type", "currency", "symbol"])
def test_s4_instrument_required_columns(package, field):
    m = read_manifest(package)
    r = m["resources"][1]
    r["schema"]["fields"] = [f for f in r["schema"]["fields"] if f["name"] != field]
    p = package / "instruments/all.parquet"
    pl.read_parquet(p).drop(field).write_parquet(p)
    resign(package, m)
    assert_bad(package, "instruments_schema")


@pytest.mark.parametrize(
    "field,value", [("asset_type", "not-an-asset"), ("currency", "YUAN")]
)
def test_s4_instrument_vocabulary(package, field, value):
    change(
        package,
        lambda f: f.with_columns(pl.lit(value).alias(field)),
        "instruments/all.parquet",
    )
    assert_bad(package, "instruments_schema")


def test_s2_column_name_matches_manifest(package):
    change(package, lambda f: f.rename({"close": "wrong"}))
    assert_bad(package, "columns")


def test_s6_primary_key_null(package):
    change(package, lambda f: f.with_columns(pl.lit(None, dtype=pl.Date).alias("date")))
    assert_bad(package, "primary_key")


def test_s3_1_nanosecond_not_supported(package):
    hourly(package)
    change(
        package, lambda f: f.with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
    )
    assert_bad(package, "time_type")


def test_s3_3_known_at_reference_must_exist(package):
    m = read_manifest(package)
    m["resources"][0]["schema"]["fields"][2]["knownAt"] = "column:missing_time"
    write_manifest(package, m)
    assert_bad(package, "known_at")


def events(root):
    m = read_manifest(root)
    fields = [
        dict(
            name=k,
            type=t,
            knownAt="end" if k == "available_at" else "column:available_at",
        )
        for k, t in [
            ("symbol", "string"),
            ("ex_date", "date"),
            ("kind", "string"),
            ("split", "number"),
            ("cash_dividend", "number"),
            ("available_at", "datetime"),
            ("source", "string"),
        ]
    ]
    m["resources"].append(
        dict(
            name="corporate_actions",
            kind="corporate_actions",
            schema=dict(fields=fields, primaryKey=["symbol", "ex_date"]),
        )
    )
    p = root / "corporate_actions/all.parquet"
    p.parent.mkdir()
    pl.DataFrame(
        dict(
            symbol=["000001.SZ"],
            ex_date=[date(2024, 1, 3)],
            kind=["split"],
            split=[2.0],
            cash_dividend=[0.0],
            available_at=[datetime(2024, 1, 3, 7, tzinfo=UTC)],
            source=["合成事件"],
        )
    ).write_parquet(p)
    resign(root, m)
    assert_good(root)


@pytest.mark.parametrize(
    "damage", ["no_available", "wrong_known", "null_time", "wrong_kind"]
)
def test_s3_3_s5_5_nonbar_events(package, damage):
    events(package)
    if damage == "no_available":
        m = read_manifest(package)
        m["resources"][-1]["schema"]["fields"] = [
            f
            for f in m["resources"][-1]["schema"]["fields"]
            if f["name"] != "available_at"
        ]
        p = package / "corporate_actions/all.parquet"
        pl.read_parquet(p).drop("available_at").write_parquet(p)
        resign(package, m)
        assert_bad(package, "available_at")
    elif damage == "wrong_known":
        m = read_manifest(package)
        m["resources"][-1]["schema"]["fields"][0]["knownAt"] = "start"
        write_manifest(package, m)
        assert_bad(package, "known_at")
    elif damage == "null_time":
        change(
            package,
            lambda f: f.with_columns(
                pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("available_at")
            ),
            "corporate_actions/all.parquet",
        )
        assert_bad(package, "available_at")
    else:
        change(
            package,
            lambda f: f.with_columns(pl.lit("numeric-enum-2").alias("kind")),
            "corporate_actions/all.parquet",
        )
        assert_bad(package, "actions_schema")


def test_s8_5_corporate_comparison_warning_only(package):
    events(package)
    add_field(
        package,
        dict(name="adj_factor", type="number"),
        pl.when(pl.col("date") >= date(2024, 1, 3)).then(2.0).otherwise(1.0),
    )
    assert_good(package)
    report = check(package)
    assert item(report, "actions_consistency")["status"] == "pass"
    change(package, lambda f: f.with_columns(pl.lit(1.0).alias("adj_factor")))
    report = check(package)
    assert report["exit_code"] == 0
    assert item(report, "actions_consistency")["status"] == "warning"


def test_s8_5_jumps_warning_only(package):
    change(
        package,
        lambda f: f.with_columns(
            pl.when(pl.col("date") >= date(2024, 1, 3))
            .then(pl.col(k) * 4)
            .otherwise(pl.col(k))
            .alias(k)
            for k in ("open", "high", "low", "close")
        ),
    )
    report = check(package)
    assert report["ok"] and item(report, "price_jumps")["status"] == "warning"


@pytest.mark.parametrize("damage", ["timezone", "reversed", "no_reason", "no_table"])
def test_s6_gap_declaration_validity(package, damage):
    m = read_manifest(package)
    gap = dict(
        table="bars",
        start="2023-01-01T00:00:00Z",
        end="2023-01-02T00:00:00Z",
        reason="已知来源缺口",
    )
    m["gaps"] = [gap]
    write_manifest(package, m)
    assert_good(package)
    if damage == "timezone":
        gap["start"] = "2023-01-01T00:00:00"
    elif damage == "reversed":
        gap["start"], gap["end"] = gap["end"], gap["start"]
    elif damage == "no_reason":
        del gap["reason"]
    else:
        gap["table"] = "absent"
    write_manifest(package, m)
    assert_bad(package, "gaps")


def test_s6_quality_uint32_not_int64(package):
    add_field(package, dict(name="quality", type="integer"), pl.lit(0, dtype=pl.UInt32))
    assert_good(package)
    change(package, lambda f: f.with_columns(pl.col("quality").cast(pl.Int64)))
    assert_bad(package, "column_types")


def test_s9_trading_masks(package):
    add_field(package, dict(name="suspended", type="boolean"), pl.lit(False))
    add_field(package, dict(name="limit_up", type="number"), pl.col("open"))
    add_field(package, dict(name="limit_down", type="number"), pl.col("open") / 2)
    p = open(package).panel(["can_buy", "can_sell"])
    assert not p["can_buy"].any()
    assert p["can_sell"].all()
    change(package, lambda f: f.with_columns(pl.lit(True).alias("suspended")))
    p = open(package).panel(["can_buy", "can_sell"])
    assert not p["can_sell"].any()


def test_s5_1_fixed_dictionary_unit_is_not_price(package):
    # 来源：D2 A 股导出对拍发现，固定单位 1 不应再套用价格单位推导。
    add_field(package, dict(name="adj_factor", type="number", unit="1"), pl.lit(1.0))
    assert_good(package)


@pytest.mark.parametrize(
    "key,value,code",
    [
        ("unit", None, "units"),
        ("knownAt", None, "known_at"),
        ("unit", {}, "units"),
        ("knownAt", [], "known_at"),
        ("type", {}, "column_types"),
    ],
)
def test_s2_invalid_declaration_values_report_not_crash(package, key, value, code):
    # 来源：D2 自审；合法 JSON 中字段属性可以是错误类型，仍需结构化报告。
    m = read_manifest(package)
    m["resources"][0]["schema"]["fields"][2][key] = value
    write_manifest(package, m)
    assert_bad(package, code)


def test_s9_masks_without_restrictions_use_presence_only(package):
    # 来源：9.3 没有停牌/涨跌停信息时，只按这一格是否有行得到掩码。
    change(
        package, lambda f: f.with_columns(pl.lit(None, dtype=pl.Float64).alias("open"))
    )
    panel = open(package).panel(["can_buy", "can_sell"])
    assert np.array_equal(panel["can_buy"], panel["has_quote"])
    assert np.array_equal(panel["can_sell"], panel["has_quote"])


def test_s4_cn_prefixed_exchange_code(package):
    # 来源：真实面板中 T600018.SH；规范的 cn 是“代码.交易所”，没有限定代码必须六位纯数字。
    for table, partition in [("bars", "2024"), ("instruments", "all")]:
        change(
            package,
            lambda f, table=table: f.with_columns(
                pl.when(pl.col("symbol") == "600001.SH")
                .then(pl.lit("T600018.SH"))
                .otherwise(pl.col("symbol"))
                .alias("symbol")
            ).sort(["symbol"] + (["date"] if table == "bars" else [])),
            f"{table}/{partition}.parquet",
        )
    assert_good(package)


def test_s2_resource_names_follow_data_package_v2(package):
    # 来源：D2 自审；snake_case 要求针对扩展字段，不应拒绝 v2 的连字符资源名。
    m = read_manifest(package)
    m["resources"][0]["name"] = "daily-bars"
    (package / "bars").rename(package / "daily-bars")
    resign(package, m)
    assert_good(package)
    assert open(package).panel("close", table="daily-bars").shape == (40, 3)


def test_s8_4_gaps_cover_event_tables_too(package):
    # 来源：8.4 缺口约束没有限定仅行情表。
    events(package)
    m = read_manifest(package)
    m["gaps"] = [
        dict(
            table="corporate_actions",
            start="2024-01-03T01:30:00Z",
            end="2024-01-03T07:00:00Z",
            reason="合成缺口",
        )
    ]
    write_manifest(package, m)
    assert_bad(package, "gaps")
