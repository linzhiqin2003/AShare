"""§3.3/§9.3，recheck_d2.md P0：时间依赖只能随行可知，不推导依赖链。"""

import importlib
from datetime import UTC, datetime

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
from test.spec.test_contract import add_field
from test.spec.test_point_in_time import known

INSTANT = "2024-01-02T07:00:00Z"
ERROR = "known_at_dependency"


def referenced_package(root, mode, row_known="end"):
    target = "available_at" if mode == "global" else "published_at"
    at = pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(hours=7)
    if mode == "chain":
        add_field(
            root,
            dict(
                name="gate_at",
                type="datetime",
                knownAt="end",
                unit="1",
                description="合成次日时间",
            ),
            at + pl.duration(days=1),
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
    if mode != "global":
        known(root, "close", "column:published_at")
    assert_good(root)
    return target


def invoke(ds, entry, *, allow_unknown=False, fields="close"):
    if entry == "panel":
        return ds.panel(fields, end="20240102", allow_unknown=allow_unknown)
    if entry in {"read", "raw_read"}:
        return ds.read(
            fields=fields,
            as_of=None if entry == "raw_read" else INSTANT,
            allow_unknown=allow_unknown,
        )
    return ds._scan(
        "bars",
        fields=fields,
        as_of=None if entry == "raw_lazy" else INSTANT,
        allow_unknown=allow_unknown,
    ).collect()


@pytest.mark.parametrize(
    "mode,bad_known",
    [("direct", "end+1d"), ("chain", "column:gate_at"), ("global", "end+1d")],
)
@pytest.mark.parametrize(
    "entry", ["check", "read", "lazy", "panel", "raw_read", "raw_lazy"]
)
def test_s3_3_s9_r4_review_variants_rejected(package, mode, bad_known, entry):
    target = referenced_package(package, mode)
    if entry != "check":
        invoke(open(package), entry)
    # 完整合规基线仅改变被引用时间列的 knownAt，独立签名器更新版本。
    known(package, target, bad_known)
    if entry == "check":
        report = assert_bad(package, ERROR)
        assert {c["code"] for c in report["checks"] if c["status"] == "fail"} == {ERROR}
    else:
        for allow in (False, True):  # allow_unknown 不得绕过不支持组合的硬限制
            with pytest.raises(ValueError, match=ERROR):
                invoke(open(package), entry, allow_unknown=allow)


@pytest.mark.parametrize("mode", ["direct", "global"])
@pytest.mark.parametrize("row_known", ["start", "end"])
def test_s3_3_s9_r4_row_known_reference_works(package, mode, row_known):
    referenced_package(package, mode, row_known)
    ds = open(package)
    for entry in ("read", "lazy", "panel", "raw_read", "raw_lazy"):
        result = invoke(ds, entry)
        if entry == "panel":
            np.testing.assert_array_equal(result["close"], [[10.5, 15.5, 20.5]])
        else:
            assert result.height == (120 if entry.startswith("raw") else 3)
            assert result["close"].null_count() == 0


@pytest.mark.parametrize("mode", ["direct", "global"])
@pytest.mark.parametrize("bad_known", ["unknown", "column:self", "missing"])
def test_s3_3_s9_r4_other_dependency_declarations_rejected(package, mode, bad_known):
    target = referenced_package(package, mode)
    m = read_manifest(package)
    desc = next(f for f in m["resources"][0]["schema"]["fields"] if f["name"] == target)
    if bad_known == "missing":
        desc.pop("knownAt")
    else:
        desc["knownAt"] = (
            f"column:{target}" if bad_known == "column:self" else bad_known
        )
    write_manifest(package, m)
    assert_bad(package, ERROR)
    for entry in ("read", "lazy", "panel", "raw_read", "raw_lazy"):
        with pytest.raises(ValueError, match=ERROR):
            invoke(open(package), entry, allow_unknown=True)


@pytest.mark.parametrize("fields", [[], "open"])
def test_s9_r4_invalid_dependency_cannot_hide_behind_projection(package, fields):
    target = referenced_package(package, "direct")
    known(package, target, "end+1d")
    for entry in ("read", "lazy", "panel", "raw_read", "raw_lazy"):
        with pytest.raises(ValueError, match=ERROR):
            invoke(open(package), entry, fields=fields)


def test_s3_3_r4_checker_does_not_use_reader_guard(package, monkeypatch):
    target = referenced_package(package, "direct")
    known(package, target, "end+1d")
    dependencies = importlib.import_module("ashare.data.spec.dependencies")
    # 即使读取侧规则被删掉，独立校验器也必须抓到本缺陷。
    monkeypatch.setattr(
        dependencies, "validate_time_dependencies", lambda fields: None, raising=False
    )
    assert_bad(package, ERROR)


def test_s9_r4_direct_lazy_helper_rejects_review_example(package):
    from ashare.data.spec.timing import as_of_view

    target = referenced_package(package, "direct")
    ds = open(package)
    raw = ds.read()
    known(package, target, "end+1d")
    ds = open(package)
    with pytest.raises(ValueError, match=ERROR):
        as_of_view(
            raw.lazy(),
            ds.resources["bars"],
            ds.manifest,
            ["close", "published_at"],
            datetime(2024, 1, 2, 7, tzinfo=UTC),
        ).collect()


def test_s9_r4_reader_does_not_rely_on_checker(package, monkeypatch):
    target = referenced_package(package, "direct")
    known(package, target, "end+1d")
    checker = importlib.import_module("ashare.data.spec.check_schema")
    monkeypatch.setattr(checker, "time_dependencies", lambda *args: None)
    for entry in ("read", "lazy", "panel", "raw_read", "raw_lazy"):
        with pytest.raises(ValueError, match=ERROR):
            invoke(open(package), entry)


@pytest.mark.parametrize("row_known", ["start", "end"])
def test_s3_3_s9_r4_event_time_column_is_row_known(package, row_known):
    from test.spec.test_boundaries import events
    from test.spec.test_contract import change

    events(package)
    m = read_manifest(package)
    next(
        f for f in m["resources"][-1]["schema"]["fields"] if f["name"] == "available_at"
    )["knownAt"] = row_known
    write_manifest(package, m)
    change(
        package,
        lambda f: f.with_columns(pl.col("available_at") - pl.duration(days=1)),
        "corporate_actions/all.parquet",
    )
    assert_good(package)
    ds = open(package)
    for result in (
        ds.read("corporate_actions", as_of=INSTANT),
        ds._scan("corporate_actions", as_of=INSTANT).collect(),
    ):
        assert result.height == 1
        assert result["available_at"][0] == datetime(2024, 1, 2, 7, tzinfo=UTC)
        assert result["ex_date"][0].isoformat() == "2024-01-03"
    assert ds.read("corporate_actions", as_of="2024-01-02T06:59:59Z").is_empty()


@pytest.mark.parametrize("row_known", ["start", "end"])
def test_s3_3_s9_r4_static_available_at_is_row_known(package, row_known):
    """来源：R4 全表规则也覆盖静态品种表，不能为公告列臆造行情区间。"""
    m = read_manifest(package)
    resource = next(r for r in m["resources"] if r["kind"] == "instruments")
    resource["schema"]["fields"].append(
        dict(name="available_at", type="datetime", knownAt=row_known)
    )
    path = package / "instruments/all.parquet"
    pl.read_parquet(path).with_columns(
        pl.lit(datetime(2024, 1, 2, 7, tzinfo=UTC)).alias("available_at")
    ).write_parquet(path)
    resign(package, m)
    assert_good(package)
    ds = open(package)
    assert ds.read("instruments", as_of="2024-01-02T06:59:59Z").is_empty()
    for frame in (
        ds.read("instruments", as_of=INSTANT),
        ds._scan("instruments", as_of=INSTANT).collect(),
    ):
        assert frame.height == 3 and frame["available_at"].null_count() == 0
