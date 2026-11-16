"""§3.3/§9.3，recheck2_d2.md：依赖值和自身时点取较晚者；静态表无区间。"""

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from ashare.data.spec import open
from test.spec.conftest import assert_bad, assert_good, read_manifest, resign
from test.spec.test_contract import add_field, change, hourly
from test.spec.test_point_in_time import known


def dependency(root, mode, row_known, *, intraday=False, hour=2):
    if intraday:
        hourly(root)
    target = "available_at" if mode == "global" else "published_at"
    at = (
        pl.col("ts") + pl.duration(minutes=30)
        if intraday
        else pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(hours=hour)
    )
    add_field(
        root,
        dict(
            name=target,
            type="datetime",
            knownAt=row_known,
            unit="1",
            description="合成发布时间",
        ),
        at,
    )
    if mode == "column":
        known(root, "close", "column:published_at")
    else:
        # 排除 close 默认 end 自己挡住前视的情况，单独观察全表门槛。
        known(root, "close", "start")
    assert_good(root)
    return target


def at_view(ds, entry, instant, fields, table="bars"):
    if entry == "read":
        return ds.read(table, fields=fields, as_of=instant)
    if entry == "lazy":
        return ds._scan(table, fields=fields, as_of=instant).collect()
    from ashare.data.spec.timing import as_of_view

    raw = ds.read(table)
    return as_of_view(
        raw.lazy(),
        ds.resources[table],
        ds.manifest,
        raw.columns if fields is None else fields,
        datetime.fromisoformat(instant.replace("Z", "+00:00")),
    ).collect()


@pytest.mark.parametrize("entry", ["read", "lazy", "helper"])
@pytest.mark.parametrize("mode", ["column", "global"])
@pytest.mark.parametrize("row_known", ["start", "end"])
def test_s3_3_s9_r5_review_dependency_effective_max(package, entry, mode, row_known):
    """02:00 的值不能替代 end=07:00；只投影 close 也必须遵守依赖声明。"""
    target = dependency(package, mode, row_known)
    ds = open(package)
    for clock in ("01:59:59", "02:00:00", "06:59:59", "07:00:00", "07:00:01"):
        visible = clock >= ("02:00:00" if row_known == "start" else "07:00:00")
        instant = f"2024-01-02T{clock}Z"
        for fields in (["close"], ["close", target]):
            result = at_view(ds, entry, instant, fields)
            if mode == "global":
                assert result.height == (3 if visible else 0)
            assert result["close"].drop_nulls().to_list() == (
                [10.5, 15.5, 20.5] if visible else []
            )
            if target in fields and mode == "column" and row_known == "end":
                assert result[target].drop_nulls().len() == (3 if visible else 0)


@pytest.mark.parametrize("mode", ["column", "global"])
@pytest.mark.parametrize("row_known", ["start", "end"])
@pytest.mark.parametrize("hour", [2, 8])
def test_s9_r5_panel_matches_effective_close_view(package, mode, row_known, hour):
    """面板决策点为 07:00；未来值仍屏蔽，早于收盘的值在收盘可用。"""
    dependency(package, mode, row_known, hour=hour)
    ds = open(package)
    panel = ds.panel("close", end="20240102")
    expected = [[10.5, 15.5, 20.5]] if hour == 2 else [[np.nan] * 3]
    np.testing.assert_array_equal(panel["close"], expected)
    values = ds.read(fields="close", as_of="2024-01-02T07:00:00Z")["close"].drop_nulls()
    assert values.to_list() == ([10.5, 15.5, 20.5] if hour == 2 else [])


@pytest.mark.parametrize("mode", ["column", "global"])
@pytest.mark.parametrize("row_known", ["start", "end"])
def test_s9_r5_intraday_dependency_boundary(package, mode, row_known):
    dependency(package, mode, row_known, intraday=True)
    ds = open(package)
    for clock in ("00:29:59", "00:30:00", "00:59:59", "01:00:00"):
        visible = clock >= ("00:30:00" if row_known == "start" else "01:00:00")
        for entry in ("read", "lazy"):
            values = at_view(ds, entry, f"2024-01-02T{clock}Z", ["close"])["close"]
            assert values.drop_nulls().to_list() == (
                [10.5, 15.5, 20.5] if visible else []
            )
    panel = ds.panel("close", start="2024-01-02T00:00:00Z", end="2024-01-02T01:00:00Z")
    np.testing.assert_array_equal(panel["close"], [[10.5, 15.5, 20.5]])


def static_dependency(root, mode, row_known):
    m = read_manifest(root)
    resource = next(r for r in m["resources"] if r["kind"] == "instruments")
    target = "available_at" if mode == "global" else "published_at"
    resource["schema"]["fields"].append(
        dict(
            name=target,
            type="datetime",
            knownAt=row_known,
            unit="1",
            description="合成静态时间",
        )
    )
    if mode == "column":
        next(f for f in resource["schema"]["fields"] if f["name"] == "currency")[
            "knownAt"
        ] = f"column:{target}"
    path = root / "instruments/all.parquet"
    pl.read_parquet(path).with_columns(
        pl.lit(datetime(2024, 1, 2, 2, tzinfo=UTC)).alias(target)
    ).write_parquet(path)
    resign(root, m)
    assert_good(root)
    return target


@pytest.mark.parametrize("entry", ["read", "lazy", "helper"])
@pytest.mark.parametrize("mode", ["column", "global"])
@pytest.mark.parametrize("row_known", ["start", "end"])
def test_s3_3_s9_r5_static_default_equals_projection(package, entry, mode, row_known):
    """静态 start/end 随行可知，不虚构 date、开收盘或事件日期。"""
    target = static_dependency(package, mode, row_known)
    ds = open(package)
    for clock in ("01:59:59", "02:00:00", "07:00:00"):
        instant = f"2024-01-02T{clock}Z"
        full = at_view(ds, entry, instant, None, "instruments")
        for fields in (["currency"], ["currency", target]):
            projected = at_view(ds, entry, instant, fields, "instruments")
            assert full.select(fields).equals(projected.select(fields))
        visible = clock >= "02:00:00"
        assert full["currency"].drop_nulls().to_list() == (
            ["CNY"] * 3 if visible else []
        )
        if mode == "column":
            assert full[target].null_count() == 0
        else:
            assert full.height == (3 if visible else 0)


@pytest.mark.parametrize("mode", ["column", "global"])
def test_s3_3_r5_static_dependency_null_rejected(package, mode):
    """修静态语义时发现：无区间不能跳过时间依赖非空检查。"""
    target = static_dependency(package, mode, "end")
    change(
        package,
        lambda f: f.with_columns(
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias(target)
        ),
        "instruments/all.parquet",
    )
    report = assert_bad(package, "available_at")
    assert {c["code"] for c in report["checks"] if c["status"] == "fail"} == {
        "available_at"
    }


@pytest.mark.parametrize("mode", ["column", "global"])
def test_s3_3_s9_r5_null_time_not_replaced_by_interval_end(package, mode):
    """取 max 的边界：不能把 null 当成区间结束，让缺失公告时间漏过去。"""
    target = dependency(package, mode, "end")
    change(
        package,
        lambda f: f.with_columns(
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias(target)
        ),
    )
    assert_bad(package, "available_at")
    ds = open(package)
    for entry in ("read", "lazy"):
        assert (
            at_view(ds, entry, "2024-01-02T07:00:00Z", ["close"])["close"]
            .drop_nulls()
            .is_empty()
        )
    assert np.isnan(ds.panel("close", end="20240102")["close"]).all()


def test_s3_3_r5_checker_independent_of_reader_clocks(package, monkeypatch):
    """校验独立检查原始值/声明；02:00 早于收盘合法，不借用读取侧计算器。"""
    from ashare.data.spec import timing

    def unavailable(*args, **kwargs):
        raise AssertionError("校验器不得调用读取方的时点计算")

    monkeypatch.setattr(timing, "dependency_time", unavailable)
    monkeypatch.setattr(timing, "knowable", unavailable)
    monkeypatch.setattr(timing, "interval", unavailable)
    target = dependency(package, "column", "end")
    change(package, lambda f: f.with_columns(pl.col(target) - pl.duration(hours=1)))
    assert_bad(package, "available_at")
