"""DATA_SPEC 每项可观测 MUST/不得的正反例；反例仅改变合规基线的一处。"""

import importlib
import json
import shutil
from datetime import date

import polars as pl
import pyarrow.parquet as pq
import pytest

from ashare.data.spec import check
from test.spec.conftest import (
    assert_bad,
    assert_good,
    item,
    read_manifest,
    resign,
    write_manifest,
)


def change(root, transform, name="bars/2024.parquet"):
    path = root / name
    transform(pl.read_parquet(path)).write_parquet(path, compression="zstd")
    resign(root)


def add_field(root, desc, expression):
    m = read_manifest(root)
    m["resources"][0]["schema"]["fields"].append(desc)
    path = root / "bars/2024.parquet"
    pl.read_parquet(path).with_columns(expression.alias(desc["name"])).write_parquet(
        path
    )
    resign(root, m)


@pytest.mark.parametrize(
    "key,value",
    [
        ("specVersion", "2"),
        ("symbology", "unregistered"),
        ("licenses", []),
        ("source", ""),
        ("name", "../bad"),
        ("created", "2024-01-01"),
    ],
)
def test_s2_s11_manifest_required(package, key, value):
    m = read_manifest(package)
    m[key] = value
    write_manifest(package, m)
    assert_bad(package, "manifest")


def test_s2_unreadable_structure_exit2(package):
    (package / "datapackage.json").write_text("{")
    assert_bad(package, "manifest", 2)


def test_s2_duplicate_json_keys(package):
    path = package / "datapackage.json"
    path.write_text(
        path.read_text().replace(
            '"specVersion": "1"', '"specVersion": "1", "specVersion": "1"'
        )
    )
    assert_bad(package, "manifest", 2)


@pytest.mark.parametrize("filename", ["2024-13", "today", "202401", "2024-01-02"])
def test_s2_partition_names(package, filename):
    path = package / "bars/2024.parquet"
    path.rename(path.with_name(filename + ".parquet"))
    resign(package)
    assert_bad(package, "paths", 2)


def test_s2_unregistered_file(package):
    (package / "extra.txt").write_text("unregistered")
    assert_bad(package, "registered_files")


def test_s2_symlink_and_traversal(package):
    path = package / "bars/2024.parquet"
    target = package.parent / "outside.parquet"
    path.rename(target)
    path.symlink_to(target)
    assert_bad(package, "paths", 2)


@pytest.mark.parametrize("compression", ["gzip", "uncompressed"])
def test_s2_parquet_compression(package, compression):
    path = package / "bars/2024.parquet"
    pl.read_parquet(path).write_parquet(path, compression=compression)
    resign(package)
    assert_bad(package, "compression")


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Decimal(precision=18, scale=2)])
def test_s5_3_reject_narrow_and_decimal(package, dtype):
    change(package, lambda f: f.with_columns(pl.col("close").cast(dtype)))
    assert_bad(package, "column_types")


def test_s5_3_reject_dictionary_category(package):
    path = package / "bars/2024.parquet"
    table = pq.read_table(path)
    table = table.set_column(0, "symbol", table["symbol"].dictionary_encode())
    pq.write_table(table, path, compression="zstd")
    resign(package)
    assert_bad(package, "column_types")


@pytest.mark.parametrize("dtype", [pl.Int64, pl.String, pl.Datetime("us")])
def test_s3_1_reject_wrong_daily_time(package, dtype):
    change(package, lambda f: f.with_columns(pl.col("date").cast(dtype)))
    assert_bad(package, "time_type")


def hourly(root, unit="us", tz="UTC"):
    m = read_manifest(root)
    m["calendar"] = dict(name="24x7", tz="UTC")
    r = m["resources"][0]
    r["frequency"] = "1h"
    r["schema"]["fields"][1] = dict(name="ts", type="datetime")
    r["schema"]["primaryKey"] = ["symbol", "ts"]
    p = root / "bars/2024.parquet"
    pl.read_parquet(p).with_columns(
        pl.col("date").cast(pl.Datetime(unit)).dt.replace_time_zone(tz).alias("ts")
    ).drop("date").select(
        "symbol", "ts", "open", "high", "low", "close", "volume"
    ).write_parquet(p)
    resign(root, m)


@pytest.mark.parametrize("unit", ["ms", "us"])
def test_s3_1_utc_timestamp_positive_and_timezone_negative(package, unit):
    hourly(package, unit)
    assert_good(package)
    change(
        package,
        lambda f: f.with_columns(pl.col("ts").dt.replace_time_zone("Asia/Shanghai")),
    )
    assert_bad(package, "time_type")


def test_s3_1_timestamp_integer_epoch_rejected(package):
    hourly(package)
    assert_good(package)
    change(package, lambda f: f.with_columns(pl.col("ts").cast(pl.Int64)))
    assert_bad(package, "time_type")


def test_s3_1_exactly_one_primary_time(package):
    add_field(
        package,
        dict(name="ts", type="datetime"),
        pl.col("date").cast(pl.Datetime("us", "UTC")),
    )
    assert_bad(package, "time_column")


@pytest.mark.parametrize("frequency", ["tick", "2d", "1H", None])
def test_s3_2_frequency_vocabulary(package, frequency):
    m = read_manifest(package)
    m["resources"][0]["frequency"] = frequency
    write_manifest(package, m)
    assert_bad(package, "frequency")


@pytest.mark.parametrize(
    "known", ["start", "end", "end+1d", "end+15m", "column:available_at"]
)
def test_s3_3_known_at_grammar(package, known):
    if known.startswith("column:"):
        add_field(
            package,
            dict(name="available_at", type="datetime", knownAt="end"),
            pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(hours=7),
        )
    m = read_manifest(package)
    m["resources"][0]["schema"]["fields"][2]["knownAt"] = known
    write_manifest(package, m)
    assert_good(package)
    m["resources"][0]["schema"]["fields"][2]["knownAt"] = "unkown"
    write_manifest(package, m)
    assert_bad(package, "known_at")


@pytest.mark.parametrize("unit", ["CNY/share", "USD/share"])
def test_s5_1_dictionary_cannot_change_units(package, unit):
    m = read_manifest(package)
    m["resources"][0]["schema"]["fields"][2]["unit"] = unit
    write_manifest(package, m)
    if unit == "CNY/share":
        assert_good(package)
    else:
        assert_bad(package, "dictionary")


@pytest.mark.parametrize(
    "unit", ["ratio", "1", "count", "USDT", "contract", "USD/contract"]
)
def test_s5_2_extension_units_positive_negative(package, unit):
    # 审查 P2-4：count 的合规正例必须声明 integer，并实际存成 int64。
    add_field(
        package,
        dict(
            name="custom_value",
            type="integer" if unit == "count" else "number",
            unit=unit,
            knownAt="end",
            description="合成扩展",
        ),
        pl.lit(1, dtype=pl.Int64) if unit == "count" else pl.lit(1.0),
    )
    assert_good(package)
    m = read_manifest(package)
    m["resources"][0]["schema"]["fields"][-1]["unit"] = "percent"
    write_manifest(package, m)
    assert_bad(package, "units")


@pytest.mark.parametrize("remove", ["unit", "knownAt", "description"])
def test_s5_4_extension_declarations_required(package, remove):
    add_field(
        package,
        dict(
            name="custom_value",
            type="number",
            unit="1",
            knownAt="end",
            description="合成值",
        ),
        pl.lit(1.0),
    )
    assert_good(package)
    m = read_manifest(package)
    del m["resources"][0]["schema"]["fields"][-1][remove]
    write_manifest(package, m)
    assert_bad(package, "extension")


def test_s5_4_snake_case(package):
    add_field(
        package,
        dict(
            name="BadName", type="number", unit="1", knownAt="end", description="坏名字"
        ),
        pl.lit(1.0),
    )
    assert_bad(package, "extension")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_s6_null_positive_nan_infinity_negative(package, value):
    change(
        package,
        lambda f: f.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        ),
    )
    assert_good(package)
    change(package, lambda f: f.with_columns(pl.col("close").fill_null(value)))
    assert_bad(package, "nan" if value != value else "finite")


def test_s6_pk_unique(package):
    change(package, lambda f: pl.concat([f.head(1), f]).sort(["symbol", "date"]))
    assert_bad(package, "primary_key")


def test_s6_pk_sorted(package):
    change(package, lambda f: f.reverse())
    report = assert_bad(package, "sorted")
    assert len(item(report, "sorted")["examples"]) == 20


def test_s6_cross_partition_unique(package):
    shutil.copyfile(package / "bars/2024.parquet", package / "bars/all.parquet")
    resign(package)
    assert_bad(package, "cross_partition")


def test_s2_partition_membership(package):
    (package / "bars/2024.parquet").rename(package / "bars/2025.parquet")
    resign(package)
    assert_bad(package, "partition_range")


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("high", 1.0, "prices"),
        ("low", 1000.0, "prices"),
        ("open", 0.0, "prices"),
        ("volume", -1.0, "nonnegative"),
    ],
)
def test_s6_price_and_quantity_constraints(package, field, value, code):
    change(package, lambda f: f.with_columns(pl.lit(value).alias(field)))
    assert_bad(package, code)


def test_s6_negative_price_opt_in(package):
    change(
        package,
        lambda f: f.with_columns(
            -pl.col("open"),
            (-pl.col("low")).alias("high"),
            (-pl.col("high")).alias("low"),
            -pl.col("close"),
        ),
    )
    m = read_manifest(package)
    for f in m["resources"][0]["schema"]["fields"]:
        if f["name"] in {"open", "high", "low", "close"}:
            f["allowNegative"] = True
    write_manifest(package, m)
    assert_good(package)
    m["resources"][0]["schema"]["fields"][2].pop("allowNegative")
    write_manifest(package, m)
    assert_bad(package, "prices")


def test_s3_4_daily_calendar_required(package):
    m = read_manifest(package)
    m["resources"] = [r for r in m["resources"] if r["kind"] != "calendar"]
    shutil.rmtree(package / "calendar")
    resign(package, m)
    assert_bad(package, "calendar_schema")


def test_s8_4_calendar_membership(package):
    change(package, lambda f: f.slice(1), "calendar/all.parquet")
    assert_bad(package, "calendar_membership")


def test_s4_lifecycle_inclusive(package):
    m = read_manifest(package)
    r = m["resources"][1]
    r["schema"]["fields"].extend(
        [dict(name="list_date", type="date"), dict(name="delist_date", type="date")]
    )
    p = package / "instruments/all.parquet"
    pl.read_parquet(p).with_columns(
        pl.lit(date(2024, 1, 2)).alias("list_date"),
        pl.lit(date(2024, 2, 26)).alias("delist_date"),
    ).write_parquet(p)
    resign(package, m)
    assert_good(package)
    change(
        package,
        lambda f: f.with_columns(pl.lit(date(2024, 1, 3)).alias("list_date")),
        "instruments/all.parquet",
    )
    assert_bad(package, "lifecycle")


def test_s4_custom_requires_instruments(package):
    m = read_manifest(package)
    m["symbology"] = "custom"
    write_manifest(package, m)
    assert_good(package)
    m["resources"] = [r for r in m["resources"] if r["kind"] != "instruments"]
    shutil.rmtree(package / "instruments")
    resign(package, m)
    assert_bad(package, "instruments_schema")


@pytest.mark.parametrize(
    "symbology,symbol",
    [
        ("cn", "H30269.CSI"),
        ("hk", "00700.HK"),
        ("us", "BRK.B"),
        ("futures", "SC2612.INE"),
        ("crypto", "spot:BTCUSDT"),
        ("custom", "own-code"),
    ],
)
def test_s4_symbology_patterns(package, symbology, symbol):
    m = read_manifest(package)
    m["symbology"] = symbology
    for table, filename in [("bars", "2024"), ("instruments", "all")]:
        p = package / table / f"{filename}.parquet"
        pl.read_parquet(p).filter(pl.col("symbol") == "000001.SZ").with_columns(
            pl.lit(symbol).alias("symbol")
        ).write_parquet(p)
    resign(package, m)
    assert_good(package)
    change(package, lambda f: f.with_columns(pl.lit("bad code").alias("symbol")))
    assert_bad(package, "symbols")


@pytest.mark.parametrize("intraday", [False, True])
def test_s3_3_bars_available_at_lower_bound(package, intraday):
    # 来源：D2 审查决定，只有行情表检查区间开始下界。
    if intraday:
        hourly(package)
        start = pl.col("ts")
    else:
        start = pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(
            hours=1, minutes=30
        )
    add_field(package, dict(name="available_at", type="datetime", knownAt="end"), start)
    assert_good(package)
    change(
        package,
        lambda f: f.with_columns(pl.col("available_at") - pl.duration(microseconds=1)),
    )
    assert_bad(package, "available_at")


def test_s6_gaps_absence(package):
    m = read_manifest(package)
    gap = dict(
        table="bars",
        symbol="000001.SZ",
        start="2024-01-02T01:30:00Z",
        end="2024-01-02T07:00:00Z",
        reason="合成已知缺口",
    )
    m["gaps"] = [gap]
    original = pl.read_parquet(package / "bars/2024.parquet")
    original.slice(1).write_parquet(package / "bars/2024.parquet")
    resign(package, m)
    assert_good(package)
    original.write_parquet(package / "bars/2024.parquet")
    resign(package)
    assert_bad(package, "gaps")


@pytest.mark.parametrize("quality", [1, 2, 8, 65536])
def test_s6_quality_bits(package, quality):
    add_field(
        package, dict(name="quality", type="integer"), pl.lit(quality, dtype=pl.UInt32)
    )
    m = read_manifest(package)
    m["qualityFlags"] = {"65536": "自定义来源标记"}
    write_manifest(package, m)
    assert_good(package)
    change(
        package, lambda f: f.with_columns(pl.lit(16, dtype=pl.UInt32).alias("quality"))
    )
    assert_bad(package, "quality")


def test_s6_synthetic_null_prices(package):
    add_field(package, dict(name="quality", type="integer"), pl.lit(4, dtype=pl.UInt32))
    change(
        package,
        lambda f: f.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(k)
            for k in ("open", "high", "low", "close")
        ),
    )
    assert_good(package)
    change(package, lambda f: f.with_columns(pl.lit(10.0).alias("close")))
    assert_bad(package, "quality")


@pytest.mark.parametrize(
    "key,code,value",
    [
        ("bytes", "file_bytes", 0),
        ("rows", "file_rows", 0),
        ("hash", "file_hash", "sha256:" + "0" * 64),
    ],
)
def test_s7_file_integrity(package, key, code, value):
    m = read_manifest(package)
    m["resources"][0]["files"][0][key] = value
    write_manifest(package, m)
    assert_bad(package, code)


def test_s7_version_metadata_change(package):
    m = read_manifest(package)
    m["source"] += " changed"
    write_manifest(package, m, version=False)
    assert_bad(package, "version")


def test_s8_independent_checker(package, monkeypatch):
    for module, name in [
        ("seal", "seal"),
        ("manifest", "version"),
        ("manifest", "load"),
        ("build", "build"),
        ("reader", "_verify"),
    ]:
        monkeypatch.setattr(
            importlib.import_module("ashare.data.spec." + module),
            name,
            lambda *a, **k: pytest.fail("校验器使用了写入/读取实现"),
        )
    assert_good(package)


def test_s8_report_json_never_nan(package):
    change(package, lambda f: f.with_columns(pl.lit(float("nan")).alias("close")))
    json.dumps(check(package), allow_nan=False)
