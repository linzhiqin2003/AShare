"""§9.3，review_d2_r3.md P1-1/P1-2 的完整包及公开入口回归。"""

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ashare.data.spec import open
from ashare.data.spec.panel import make_panel
from ashare.data.spec.timing import as_of_view
from docs.examples.generate_datapackage import SyntheticAdapter
from test.spec.conftest import assert_bad, assert_good
from test.spec.test_contract import add_field, change, hourly
from test.spec.test_point_in_time import known


def test_s9_r3_review_off_grid_memory_reproduction():
    a = SyntheticAdapter()
    m = a.describe()
    m["calendar"] = dict(name="24x7", tz="UTC")
    r = m["resources"][0]
    r["frequency"] = "1h"
    r["schema"]["fields"][1] = dict(name="ts", type="datetime")
    r["schema"]["primaryKey"] = ["symbol", "ts"]
    f = (
        a.read("bars", "2024")
        .head(2)
        .with_columns(
            pl.Series(
                "ts",
                [
                    datetime(2024, 1, 2, 0, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 1, 30, tzinfo=UTC),
                ],
            )
        )
        .drop("date")
        .select([d["name"] for d in r["schema"]["fields"]])
    )
    ds = SimpleNamespace(
        manifest=m, resources={"bars": r}, version="v", _scan=lambda *args: f.lazy()
    )
    with pytest.raises(ValueError, match="行情时间不在面板日历/网格内"):
        make_panel(ds, ["close"], None, None, ["000001.SZ"], "bars")


@pytest.mark.parametrize("delayed", [False, True])
def test_s9_r3_off_grid_public_panel_rejected(package, delayed):
    hourly(package)
    change(
        package,
        lambda f: (
            f.filter(pl.col("symbol") == "000001.SZ")
            .head(2)
            .with_columns(
                pl.Series(
                    "ts",
                    [
                        datetime(2024, 1, 2, 0, 30, tzinfo=UTC),
                        datetime(2024, 1, 2, 1, 30, tzinfo=UTC),
                    ],
                )
            )
        ),
    )
    if delayed:
        known(package, "close", "end+1h")
    assert_good(package)
    assert open(package).read().height == 2
    with pytest.raises(ValueError, match="行情时间不在面板日历/网格内"):
        open(package).panel("close")
    # 历史回读中的不规则时间也不能伪装成允许忽略的历史行。
    with pytest.raises(ValueError, match="行情时间不在面板日历/网格内"):
        open(package).panel(
            "close", start="2024-01-02T01:00:00Z", end="2024-01-02T03:00:00Z"
        )


def test_s9_r3_history_outside_window_is_allowed(package):
    known(package, "close", "end+1d")
    full = open(package).panel("close")
    cropped = open(package).panel("close", "20240108", "20240109")
    assert cropped["close"].tobytes() == full["close"][4:6].tobytes()
    assert cropped["has_quote"].all()


def test_s9_r3_off_grid_history_is_not_discardable(package):
    """来源：P1-1，只让回读历史行错位，当前范围的行情仍正常对齐。"""
    hourly(package)
    change(
        package,
        lambda f: (
            f.filter(pl.col("symbol") == "000001.SZ")
            .head(2)
            .with_columns(
                pl.Series(
                    "ts",
                    [
                        datetime(2024, 1, 2, 0, 30, tzinfo=UTC),
                        datetime(2024, 1, 2, 2, tzinfo=UTC),
                    ],
                )
            )
        ),
    )
    known(package, "close", "end+1h")
    assert_good(package)
    with pytest.raises(ValueError, match="行情时间不在面板日历/网格内"):
        open(package).panel(
            "close", start="2024-01-02T02:00:00Z", end="2024-01-02T04:00:00Z"
        )


def unknown_dependency(package, mode):
    target = "available_at" if mode == "global" else "published_at"
    add_field(
        package,
        dict(
            name=target,
            type="datetime",
            knownAt="unknown" if mode == "chain" else "end",
            unit="1",
            description="合成时间",
        ),
        pl.col("date").cast(pl.Datetime("us", "UTC")) + pl.duration(hours=7),
    )
    if mode == "chain":
        add_field(
            package,
            dict(
                name="released_at",
                type="datetime",
                knownAt="end",
                unit="1",
                description="合成引用",
            ),
            pl.col("published_at"),
        )
        known(package, "close", "column:released_at")
    elif mode == "column":
        known(package, "close", "column:published_at")
    assert_good(package)
    known(
        package,
        "released_at" if mode == "chain" else target,
        "column:published_at" if mode == "chain" else "unknown",
    )
    return target


@pytest.mark.parametrize("mode", ["column", "global", "chain"])
def test_s9_r3_unknown_dependency_public_entries(package, mode):
    # R4 审查决定替代 R3 的显式放行：未知时间依赖现在是非法声明。
    unknown_dependency(package, mode)
    assert_bad(package, "known_at_dependency")
    ds = open(package)
    for allow in (False, True):
        with pytest.raises(ValueError, match="known_at_dependency"):
            ds.panel("close", allow_unknown=allow)
        for as_of in (None, "2024-01-02T07:00:00Z"):
            with pytest.raises(ValueError, match="known_at_dependency"):
                ds.read(fields="close", as_of=as_of, allow_unknown=allow)


def test_s9_r3_review_unknown_available_at_dependency(package):
    unknown_dependency(package, "global")
    known(package, "close", "column:available_at")
    ds = open(package)
    raw = pl.read_parquet(package / "bars/2024.parquet")
    with pytest.raises(ValueError, match="known_at_dependency"):
        as_of_view(
            raw.lazy(),
            ds.resources["bars"],
            ds.manifest,
            ["close"],
            datetime(2024, 1, 2, 7, tzinfo=UTC),
        ).collect()
    with pytest.raises(ValueError, match="known_at_dependency"):
        ds.read(fields="close", as_of="2024-01-02T07:00:00Z")


def test_s9_r3_global_unknown_covers_keys_and_masks(package):
    unknown_dependency(package, "global")
    ds = open(package)
    for allow in (False, True):
        with pytest.raises(ValueError, match="known_at_dependency"):
            ds.read(fields=[], as_of="2024-01-02T07:00:00Z", allow_unknown=allow)
        for name in ("has_quote", "can_buy"):
            with pytest.raises(ValueError, match="known_at_dependency"):
                ds.panel(name, allow_unknown=allow)


def test_s9_r3_unused_unknown_dependency_does_not_block(package):
    unknown_dependency(package, "column")
    # 没有字段引用这个 unknown 列时，它仍是合法的普通未知值。
    known(package, "close", "end")
    assert_good(package)
    ds = open(package)
    assert np.isfinite(ds.panel("open")["open"]).all()
    assert ds.read(fields="open", as_of="2024-01-02T07:00:00Z").height == 3


def test_s9_r3_unknown_value_read_opt_in(package):
    """来源：R4 保留普通未知值的显式放行，只禁止不支持的时间依赖。"""
    known(package, "close", "unknown")
    ds = open(package)
    with pytest.raises(ValueError, match="unknown"):
        ds.read(fields="close", as_of="2024-01-02T07:00:00Z")
    frame = ds.read(fields="close", as_of="2024-01-02T07:00:00Z", allow_unknown=True)
    assert frame.height == 3 and frame["close"].null_count() == 0
    assert frame.field_metadata["close"]["knownAt"] == "unknown"
    assert frame.field_metadata["close"]["unknownDependencies"] == ("close",)


def test_s9_r3_calendar_gap_rejected(package):
    # 来源：网格修复必须覆盖日历成员，不只覆盖小时对齐。
    change(
        package,
        lambda f: f.filter(pl.col("date") != pl.date(2024, 1, 3)),
        "calendar/all.parquet",
    )
    with pytest.raises(ValueError, match="行情时间不在面板日历/网格内"):
        open(package).panel("close")
