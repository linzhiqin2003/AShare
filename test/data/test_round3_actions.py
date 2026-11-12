"""手工摘录复核案例的必要数字；不保存接口响应。"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ashare
from ashare.data.panel_actions import (
    choose_action,
    prior_prices,
    reconcile,
    reconciliation_check,
)
from ashare.data.panel_events import dividend_rows

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def row(stock=0.0, cash=0.0, *, copies=1):
    return dict(stk_div=stock, cash_div_tax=cash, _records=copies)


@pytest.mark.parametrize(
    "symbol,day,previous,ratio,records,expected",
    [
        (
            "688766.SH",
            "20230614",
            176.3,
            2.095 / 1.407,
            [row(0.49, 0, copies=2)],
            (1.49, 0, 2),
        ),
        (
            "002346.SZ",
            "20140321",
            21.05,
            2.05 / 1.025,
            [row(1, 0, copies=2)],
            (2, 0, 2),
        ),
        (
            "601088.SH",
            "20170710",
            22.29,
            1.53 / 1.326,
            [row(0, 2.51), row(0, 2.97)],
            (1, 2.97, 2),
        ),
        ("002157.SZ", "20231208", 2.63, 1.0, [row(1.623, 0, copies=2)], (1, 0, 4)),
        (
            "689009.SH",
            "20260807",
            42.6,
            1.064 / 1.033,
            [row(0, 12.3852)],
            (1.064 / 1.033, 0, 3),
        ),
        ("002142.SZ", "20171116", 17.58, 1.0, [row(0, 4.6)], (1, 0, 4)),
        ("600036.SH", "20130905", 10.66, 3.897 / 3.824, [], (3.897 / 3.824, 0, 3)),
        ("000887.SZ", "20070330", 3.83, 2.106 / 6.449, [], (2.106 / 6.449, 0, 3)),
        (
            "301393.SZ",
            "20261110",
            53.68,
            1.0252 / 1.0223,
            [row(0, 0.15, copies=2)],
            (1, 0.15, 2),
        ),
    ],
)
def test_named_action_cases(symbol, day, previous, ratio, records, expected):
    actual = choose_action(records, ratio, previous, tolerance=0.005)
    assert actual == expected, (symbol, day)
    theta = previous * actual[0] / (previous - actual[1])
    assert abs(theta / ratio - 1) <= 0.005


def test_economic_dedup_key_and_candidate_order():
    table = pl.DataFrame(
        dict(
            ts_code=["A"] * 3,
            ex_date=["20260105"] * 3,
            imp_ann_date=["20260101", "20260102", "20260103"],
            ann_date=["20251230"] * 3,
            div_proc=["实施"] * 3,
            stk_div=[0.1, 0.10000001, 0.1],
            cash_div_tax=[0.2, 0.20000001, 0.3],
        )
    )
    valid, stats = dividend_rows(table)
    assert len(valid) == 2 and stats["duplicate_rows_removed"] == 1
    assert sorted(valid["cash_div_tax"].to_list()) == [0.2, 0.3]
    # 全部候选与仅送转都在容差内，必须保留全部的声明现金。
    assert choose_action([row(0.1, 0.01)], 1.1, 10, tolerance=0.005) == (1.1, 0.01, 1)
    # 仅现金、仅送转、单条回退各自确实被走到。
    assert choose_action([row(0.2, 1)], 10 / 9, 10, tolerance=0.005) == (1.0, 1.0, 2)
    assert choose_action([row(0.2, 1)], 1.2, 10, tolerance=0.005) == (1.2, 0.0, 2)
    assert choose_action([row(0, 2), row(0, 3)], 10 / 7, 10, tolerance=0.005) == (
        1,
        3,
        2,
    )
    assert choose_action([row(0, 2), row(0, 3)], 1.4, 10, tolerance=0.005) == (
        1.4,
        0,
        3,
    )
    assert choose_action([row(0.5, 0)], 1.0002, 10, tolerance=0.005) == (1, 0, 4)
    assert choose_action([], 1.0002, 10, tolerance=0.005) == (1, 0, 0)


def test_suspension_shift_and_fail_closed(monkeypatch):
    dates = np.array(["20260102", "20260105", "20260106", "20260107", "20260108"])
    arrays = dict(
        close=np.array([[10.0], [np.nan], [np.nan], [9.0], [9.0]]),
        adj_factor=np.array([[1.0], [1.0], [1.0], [10 / 9], [10 / 9]]),
        is_listed=np.ones((5, 1), bool),
        split=np.zeros((5, 1)),
        cash_dividend=np.zeros((5, 1)),
        action_source=np.zeros((5, 1), np.uint8),
    )
    table = pl.DataFrame(
        dict(
            ts_code=["A"],
            ex_date=["20260105"],
            imp_ann_date=["20260102"],
            ann_date=["20260102"],
            div_proc=["实施"],
            stk_div=[0.0],
            cash_div_tax=[1.0],
        )
    )
    meta = reconcile(arrays, dates, ["A"], table)
    assert arrays["cash_dividend"][:, 0].tolist() == [0, 0, 0, 1, 0]
    assert arrays["action_source"][:, 0].tolist() == [0, 0, 0, 2, 0]
    assert meta["shifted_records"] == 1 and meta["failed"] == 0
    # 故意破坏候选输出，构建层必须拦截；显式放行仍记录失败数。
    import ashare.data.panel_actions as actions

    monkeypatch.setattr(actions, "choose_action", lambda *a, **k: (2.0, 5.0, 1))
    with pytest.raises(ValueError, match="对账失败"):
        reconcile(arrays, dates, ["A"], table)
    assert (
        reconcile(arrays, dates, ["A"], table, allow_unreconciled=True)["failed"] == 1
    )


def test_reconciliation_catches_bad_ratio_and_unbacked_event():
    close = np.array([[10.0, 10.0], [9.0, 10.0]])
    factors = np.array([[1.0, 1.0], [10 / 9, 1.0]])
    split = np.ones((2, 2))
    cash = np.zeros((2, 2))
    cash[1] = [1, 0]
    args = (prior_prices(close), factors, split, cash, np.ones((2, 2), bool))
    assert reconciliation_check(*args)[0]["failed"] == 0
    cash[1] = [2, 1]
    assert reconciliation_check(*args)[0]["failed"] == 2


def test_missing_reference_diagnostics_are_standard_json():
    import json

    dates = np.array(["20260102", "20260105"])
    arrays = dict(
        close=np.full((2, 1), np.nan),
        adj_factor=np.array([[1.0], [1.5]]),
        is_listed=np.ones((2, 1), bool),
        split=np.ones((2, 1)),
        cash_dividend=np.zeros((2, 1)),
        action_source=np.zeros((2, 1), np.uint8),
    )
    empty = pl.DataFrame(
        schema={
            "ts_code": pl.String,
            "ex_date": pl.String,
            "div_proc": pl.String,
            "imp_ann_date": pl.String,
            "stk_div": pl.Float64,
            "cash_div_tax": pl.Float64,
        }
    )
    result = reconcile(arrays, dates, ["A"], empty)
    assert result["failed"] == 0 and result["synthetic_examples"][0]["previous"] is None
    assert arrays["split"][1, 0] == 1.5
    json.dumps(result, allow_nan=False)
