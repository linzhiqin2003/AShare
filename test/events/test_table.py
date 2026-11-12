import numpy as np
import polars as pl
import pytest

from ashare.events import align, dedupe, validate, with_timing
from test.events.reference import trade_day

DATES = np.array(
    ["20240104", "20240105", "20240108", "20240109"]
)  # 周四、周五、周一、周二


def ev(symbols, dates, timing="after_close", kind="k"):
    return pl.DataFrame(
        {
            "symbol": symbols,
            "date": dates,
            "kind": [kind] * len(symbols),
            "timing": [timing] * len(symbols),
        }
    )


def test_after_close_weekend_and_pre_open():
    a = align(
        ev(["A", "A", "A", "A"], ["20240104", "20240105", "20240106", "20240109"]),
        DATES,
        np.array(["A"]),
    )
    # 周四盘后 -> 周五；周五盘后 -> 周一；周六 -> 周一；最后一天盘后 -> 超出面板
    assert a.avail.tolist() == [1, 2, 2]
    assert a.frame["avail_date"].to_list() == ["20240105", "20240108", "20240108"]
    assert a.dropped["after_panel_end"] == 1
    b = align(
        ev(["A", "A", "A"], ["20240104", "20240106", "20240109"], "pre_open"),
        DATES,
        np.array(["A"]),
    )
    assert b.avail.tolist() == [0, 2, 3]


def test_unknown_symbol_and_before_start():
    a = align(
        ev(["A", "Z", "A"], ["20240105", "20240105", "20240101"]),
        DATES,
        np.array(["B", "A"]),
    )
    assert a.col.tolist() == [1]
    assert a.dropped == {
        "unknown_symbol": 1,
        "after_panel_end": 0,
        "before_panel_start": 1,
    }


def test_matches_plain_loop():
    rng = np.random.default_rng(3)
    dates = np.array(sorted({str(20240101 + int(x)) for x in rng.integers(0, 28, 15)}))
    raw = [str(20240101 + int(x)) for x in rng.integers(0, 28, 200)]
    timing = rng.choice(["after_close", "pre_open"], 200).tolist()
    frame = pl.DataFrame(
        {"symbol": ["A"] * 200, "date": raw, "kind": ["k"] * 200, "timing": timing}
    )
    a = align(frame, dates, np.array(["A"]))
    rows = validate(frame).iter_rows(named=True)
    expect = [
        trade_day(dates, r["date"], r["timing"]) for r in rows if r["date"] >= dates[0]
    ]
    expect = [x for x in expect if x < len(dates)]
    assert a.avail.tolist() == expect


def test_with_timing_and_validate():
    frame = pl.concat([ev(["A"], ["20240104"]), ev(["A"], ["20240104"], kind="other")])
    out = with_timing(frame, {"other": "pre_open"})
    assert align(out, DATES, np.array(["A"])).avail.tolist() == [
        1,
        0,
    ]  # 按 kind 排序：k、other
    with pytest.raises(ValueError):
        with_timing(frame, {"other": "noon"})
    with pytest.raises(ValueError):
        validate(frame.drop("timing"))
    with pytest.raises(ValueError):
        validate(frame.with_columns(pl.lit("2024-01-04").alias("date")))
    with pytest.raises(ValueError):
        align(frame, DATES[::-1], np.array(["A"]))


def test_dedupe():
    frame = pl.DataFrame(
        {"symbol": ["A", "A", "A", "B"], "date": ["20240104"] * 4, "kind": ["k"] * 4,
         "timing": ["after_close"] * 4, "x": [1.0, 3.0, 2.0, 5.0]}
    )  # fmt: skip
    assert dedupe(frame)["x"].to_list() == [1.0, 5.0]
    out = dedupe(frame, by="x", keep="max")
    assert out["x"].to_list() == [3.0, 5.0]
    assert out["n_same_day"].to_list() == [3, 1]
    assert dedupe(frame, by="x", keep="min")["x"].to_list() == [1.0, 5.0]


def test_by_year():
    a = align(ev(["A", "A"], ["20240104", "20240105"]), DATES, np.array(["A"]))
    assert a.by_year().rows() == [("2024", 2)]


def test_private_execution_days_matches_public_contract():
    """私有副本与主干 ashare.backtest.timing.execution_days 同口径（合入后换成公共函数）。"""
    from ashare.events._calendar import execution_days, first_trade_index

    assert execution_days(np.array([0, 2, 5]), 6).tolist() == [1, 3]
    assert execution_days(np.array([0, 2]), 6, lag=3).tolist() == [3, 5]
    with pytest.raises(ValueError):
        execution_days(np.array([0, 5]), 6, tail="error")
    with pytest.raises(ValueError):
        execution_days(np.array([2, 2]), 6)
    with pytest.raises(ValueError):
        execution_days(np.array([0]), 6, lag=0)
    got = first_trade_index(
        DATES,
        np.array(["20240103", "20240104", "20240104", "20240109"]),
        np.array([False, False, True, False]),
    )
    assert got.tolist() == [0, 1, 0, 4]
