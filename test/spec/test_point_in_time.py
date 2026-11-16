"""§3.3、§9.3：独立审查 P1 的复现与时点读取边界。"""

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from ashare.data.spec import open
from ashare.factor import Evaluator, field
from test.spec.conftest import assert_good, read_manifest, write_manifest
from test.spec.test_contract import add_field, hourly


def known(root, name, value):
    manifest = read_manifest(root)
    next(f for f in manifest["resources"][0]["schema"]["fields"] if f["name"] == name)[
        "knownAt"
    ] = value
    write_manifest(root, manifest)


def availability(root, instant):
    add_field(
        root, dict(name="available_at", type="datetime", knownAt="end"), pl.lit(instant)
    )
    known(root, "close", "column:available_at")
    assert_good(root)


def test_s9_review_delayed_close_never_enters_same_day_factor(package):
    # 来源：review_d2.md P1-1 原样复现，内存声明不改 parquet。
    ds = open(package)
    next(f for f in ds.resources["bars"]["schema"]["fields"] if f["name"] == "close")[
        "knownAt"
    ] = "end+1d"
    panel = ds.panel("close", end="20240102")
    assert np.isnan(Evaluator().evaluate(field("close"), panel)).all()


def test_s9_review_future_announcement_is_masked(package):
    # 来源：review_d2.md P1-1 的完整合规包复现。
    availability(package, datetime(2030, 1, 1, tzinfo=UTC))
    ds = open(package)
    assert ds.read(end="20240102")["close"].null_count() == 0  # 原始行不是时点视图
    assert np.isnan(ds.panel("close", end="20240102")["close"]).all()
    assert ds.read(end="20240102", as_of="2024-01-02T07:00:00Z").is_empty()


def test_s9_delay_calendar_shift_and_left_boundary(package):
    original = open(package).panel("close")["close"]
    known(package, "close", "end+1d")
    ds = open(package)
    shifted = ds.panel("close")
    assert np.isnan(shifted["close"][0]).all()
    np.testing.assert_array_equal(shifted["close"][1:], original[:-1])
    # 周五的值在周一收盘可知；请求范围不应丢掉前一行的来源值。
    monday = ds.panel("close", start="20240108", end="20240108")
    np.testing.assert_array_equal(monday["close"][0], original[3])
    np.testing.assert_array_equal(shifted["has_quote"], np.ones((40, 3), dtype=bool))


def test_s9_delay_intraday_right_edge_and_rounding(package):
    hourly(package)
    known(package, "close", "end+90m")
    p = open(package).panel("close", "2024-01-02", "2024-01-02T04:00:00Z")
    assert np.isnan(p["close"][[0, 1, 3]]).all()
    np.testing.assert_array_equal(p["close"][2], [10.5, 15.5, 20.5])
    sliced = open(package).panel(
        "close", "2024-01-02T02:00:00Z", "2024-01-02T03:00:00Z"
    )
    np.testing.assert_array_equal(sliced["close"], p["close"][2:3])


def test_s9_available_at_close_boundary(package):
    availability(package, datetime(2024, 2, 26, 7, tzinfo=UTC))
    ds = open(package)
    p = ds.panel("close")
    assert np.isnan(p["close"][:-1]).all()
    np.testing.assert_array_equal(p["close"][-1], [15.375, 20.375, 25.375])
    assert ds.read(as_of="2024-02-26T06:59:59.999999Z").is_empty()
    assert ds.read(as_of="2024-02-26T07:00:00Z").height == 120


def test_s9_as_of_masks_fields_by_known_at(package):
    known(package, "volume", "end+1d")
    ds = open(package)
    early = ds.read(
        end="20240102", fields=["open", "close", "volume"], as_of="2024-01-02T01:30:00Z"
    )
    assert early.height == 3 and early["open"].null_count() == 0
    assert early["close"].null_count() == early["volume"].null_count() == 3
    end = ds.read(
        end="20240102", fields=["close", "volume"], as_of="2024-01-02T07:00:00Z"
    )
    assert end["close"].null_count() == 0 and end["volume"].null_count() == 3
    later = ds.read(end="20240102", fields="volume", as_of="2024-01-03T07:00:00Z")
    assert later["volume"].to_list() == [1000.0] * 3


def test_s9_unknown_rejected_unless_explicitly_allowed(package):
    known(package, "close", "unknown")
    ds = open(package)
    with pytest.raises(ValueError, match="unknown.*close|close.*unknown"):
        ds.panel("close")
    p = ds.panel("close", allow_unknown=True)
    assert p.field_metadata["close"]["knownAt"] == "unknown"
    assert np.isfinite(p["close"]).all()
    # 未请求的 unknown 列不应阻止安全字段使用。
    assert np.isfinite(ds.panel("open")["open"]).all()


def test_s9_unknown_mask_dependency_rejected(package):
    add_field(
        package,
        dict(name="limit_up", type="number", knownAt="unknown"),
        pl.col("high") + 1,
    )
    with pytest.raises(ValueError, match="unknown"):
        open(package).panel("can_buy")
    assert (
        open(package)
        .panel("can_buy", allow_unknown=True)
        .field_metadata["can_buy"]["knownAt"]
        == "unknown"
    )


def test_s9_as_of_event_announcement_precedes_event_date(package):
    from test.spec.test_boundaries import events
    from test.spec.test_contract import change

    events(package)
    change(
        package,
        lambda f: f.with_columns(pl.col("available_at") - pl.duration(days=1)),
        "corporate_actions/all.parquet",
    )
    assert_good(package)
    ds = open(package)
    assert ds.read("corporate_actions", as_of="2024-01-02T06:59:59Z").is_empty()
    result = ds.read("corporate_actions", as_of="2024-01-02T07:00:00Z")
    assert result.height == 1 and result["ex_date"][0].isoformat() == "2024-01-03"


def test_s9_as_of_unknown_and_primary_key_projection(package):
    known(package, "close", "unknown")
    ds = open(package)
    with pytest.raises(ValueError, match="unknown"):
        ds.read(fields="close", as_of="2024-01-02T07:00:00Z")
    assert ds.read(fields=[], as_of="2024-01-01T00:00:00Z").is_empty()
    assert ds.read(fields="open", as_of="2024-01-02T07:00:00Z").height == 3


def test_s9_delayed_collision_uses_latest_source_and_slice_matches(package):
    known(package, "close", "end+2d")
    ds = open(package)
    full = ds.panel("close")
    monday = ds.panel("close", start="20240108", end="20240108")
    # 周四和周五的值都在周一到期，采用周五的 10.875。
    np.testing.assert_array_equal(monday["close"][0], [10.875, 15.875, 20.875])
    np.testing.assert_array_equal(monday["close"], full["close"][4:5])


def test_s9_delay_without_instruments_keeps_historical_symbol(package):
    # 来源：实现回读时发现，仅由请求范围推导代码轴会漏掉历史到期值的 symbol。
    import shutil

    from test.spec.test_contract import change

    known(package, "close", "end+1d")
    m = read_manifest(package)
    m["resources"] = [r for r in m["resources"] if r["kind"] != "instruments"]
    shutil.rmtree(package / "instruments")
    write_manifest(package, m)
    change(package, lambda f: f.filter(pl.col("date") == pl.date(2024, 1, 2)))
    assert_good(package)
    p = open(package).panel("close", "20240103", "20240103")
    assert p.symbols.tolist() == ["000001.SZ", "600001.SH", "600002.SH"]
    np.testing.assert_array_equal(p["close"][0], [10.5, 15.5, 20.5])
    assert not p["has_quote"].any()


def test_s9_session_seconds_determine_decision_time(package):
    # 来源：实现自审，合法 ISO sessionClose 可包含秒，不能只拆 HH:MM。
    m = read_manifest(package)
    m["calendar"]["sessionClose"] = "15:00:30"
    write_manifest(package, m)
    assert_good(package)
    ds = open(package)
    p = ds.panel("close", end="20240102")
    assert np.isfinite(p["close"]).all()
    assert (
        ds.read(end="20240102", fields="close", as_of="2024-01-02T07:00:29Z")[
            "close"
        ].null_count()
        == 3
    )
    assert (
        ds.read(end="20240102", fields="close", as_of="2024-01-02T07:00:30Z")[
            "close"
        ].null_count()
        == 0
    )


def test_s3_3_session_open_seconds_lower_bound(package):
    from test.spec.conftest import assert_bad
    from test.spec.test_contract import change

    # 来源：与收盘时点解析对应的独立校验实现也必须保留秒精度。
    m = read_manifest(package)
    m["calendar"]["sessionOpen"] = "09:30:30"
    write_manifest(package, m)
    add_field(
        package,
        dict(name="available_at", type="datetime", knownAt="end"),
        pl.col("date").cast(pl.Datetime("us", "UTC"))
        + pl.duration(hours=1, minutes=30, seconds=30),
    )
    assert_good(package)
    change(
        package,
        lambda f: f.with_columns(pl.col("available_at") - pl.duration(microseconds=1)),
    )
    assert_bad(package, "available_at")
