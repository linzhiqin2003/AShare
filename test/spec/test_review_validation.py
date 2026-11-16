"""§5、§6、§8：review_d2.md 四项 P2 的完整合规基线正反例。"""

import numpy as np
import polars as pl
import pytest

from ashare.data.spec import open
from test.spec.conftest import (
    assert_bad,
    assert_good,
    read_manifest,
    resign,
    write_manifest,
)
from test.spec.test_boundaries import events
from test.spec.test_contract import add_field, change


def test_s9_negative_price_trading_mask(package):
    add_field(package, dict(name="suspended", type="boolean"), pl.lit(False))
    manifest = read_manifest(package)
    for desc in manifest["resources"][0]["schema"]["fields"]:
        if desc["name"] in {"open", "high", "low", "close"}:
            desc["allowNegative"] = True
    write_manifest(package, manifest)
    change(
        package,
        lambda f: f.with_columns(
            -pl.col("open"),
            (-pl.col("low")).alias("high"),
            (-pl.col("high")).alias("low"),
            -pl.col("close"),
        ),
    )
    assert_good(package)
    p = open(package).panel(["can_buy", "can_sell"])
    np.testing.assert_array_equal(p["can_buy"], p["has_quote"])
    np.testing.assert_array_equal(p["can_sell"], p["has_quote"])


@pytest.mark.parametrize(
    "unit,good_type,bad_type,bad_value",
    [
        ("CNY/share", "number", "integer", 5),
        ("share", "number", "string", "1"),
        ("CNY", "number", "integer", 5),
        ("count", "integer", "number", 1.25),
    ],
)
def test_s5_extension_unit_enforces_type(package, unit, good_type, bad_type, bad_value):
    add_field(
        package,
        dict(
            name="vendor_value",
            type=good_type,
            unit=unit,
            knownAt="end",
            description="合成扩展",
        ),
        pl.lit(
            5.0 if good_type == "number" else 5,
            dtype=pl.Float64 if good_type == "number" else pl.Int64,
        ),
    )
    assert_good(package)
    m = read_manifest(package)
    m["resources"][0]["schema"]["fields"][-1]["type"] = bad_type
    write_manifest(package, m)
    change(
        package,
        lambda f: f.with_columns(
            pl.lit(
                bad_value,
                dtype={"integer": pl.Int64, "number": pl.Float64, "string": pl.String}[
                    bad_type
                ],
            ).alias("vendor_value")
        ),
    )
    assert_bad(package, "column_types")


@pytest.mark.parametrize("allow_negative,bad", [(False, -5.0), (True, 0.0)])
def test_s6_extension_price_domain(package, allow_negative, bad):
    add_field(
        package,
        dict(
            name="vendor_price",
            type="number",
            unit="CNY/share",
            knownAt="end",
            description="合成价格",
            **({"allowNegative": True} if allow_negative else {}),
        ),
        pl.lit(-5.0 if allow_negative else 5.0),
    )
    assert_good(package)
    change(package, lambda f: f.with_columns(pl.lit(bad).alias("vendor_price")))
    assert_bad(package, "prices")


def test_s5_resource_units_ignore_unreferenced_instruments(package):
    change(package, lambda f: f.filter(pl.col("symbol") != "600002.SH"))
    m = read_manifest(package)
    next(f for f in m["resources"][0]["schema"]["fields"] if f["name"] == "open")[
        "unit"
    ] = "CNY/share"
    write_manifest(package, m)
    change(
        package,
        lambda f: f.with_columns(
            pl.when(pl.col("symbol") == "600002.SH")
            .then(pl.lit("USD"))
            .otherwise(pl.col("currency"))
            .alias("currency")
        ),
        "instruments/all.parquet",
    )
    assert_good(package)
    info = open(package).info()
    close = next(f for f in info["resources"][0]["fields"] if f["name"] == "close")
    assert close["unit"] == "CNY/share"
    change(
        package,
        lambda f: f.with_columns(
            pl.when(pl.col("symbol") == "000001.SZ")
            .then(pl.lit("USD"))
            .otherwise(pl.col("currency"))
            .alias("currency")
        ),
        "instruments/all.parquet",
    )
    assert_bad(package, "dictionary")


@pytest.mark.parametrize("kind", ["events", "corporate_actions"])
@pytest.mark.parametrize("bad", ["public_bit", "undeclared_custom", "malformed_flags"])
def test_s6_event_quality_flags(package, kind, bad):
    events(package)
    m = read_manifest(package)
    resource = m["resources"][-1]
    resource["kind"] = kind
    if kind == "events":
        # events 也使用普通 date 主键；其他公司行为列保留为声明完整的扩展。
        for desc in resource["schema"]["fields"]:
            if desc["name"] == "ex_date":
                desc["name"] = "date"
            elif desc["name"] in {"kind", "split", "cash_dividend", "source"}:
                desc.update(unit="1", description="合成事件属性")
        resource["schema"]["primaryKey"] = ["symbol", "date"]
    resource["schema"]["fields"].append(
        dict(name="quality", type="integer", knownAt="column:available_at")
    )
    path = package / "corporate_actions/all.parquet"
    frame = pl.read_parquet(path)
    if kind == "events":
        frame = frame.rename({"ex_date": "date"})
    frame.with_columns(
        pl.lit(65536 | 1, dtype=pl.UInt32).alias("quality")
    ).write_parquet(path)
    m["qualityFlags"] = {"65536": "合成来源标记"}
    resign(package, m)
    assert_good(package)
    if bad == "malformed_flags":
        m = read_manifest(package)
        m["qualityFlags"] = {"65536": ""}
        write_manifest(package, m)
    else:
        change(
            package,
            lambda f: f.with_columns(
                pl.lit(16 if bad == "public_bit" else 131072, dtype=pl.UInt32).alias(
                    "quality"
                )
            ),
            "corporate_actions/all.parquet",
        )
    assert_bad(package, "quality")


@pytest.mark.parametrize(
    "bit", ["²", "9" * 5000], ids=["non_ascii_digits", "oversized"]
)
def test_s6_invalid_quality_flag_key_reports_without_crashing(package, bit):
    # 来源：将 qualityFlags 提到包级后自审，isdigit 不保证 int() 可解析。
    m = read_manifest(package)
    m["qualityFlags"] = {"65536": "合成标记"}
    write_manifest(package, m)
    assert_good(package)
    m["qualityFlags"] = {bit: "非法位值"}
    write_manifest(package, m)
    assert_bad(package, "quality")
