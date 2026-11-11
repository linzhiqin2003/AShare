from pathlib import Path

import numpy as np
import panelbt as bt
import pytest
from panelbt.rules.ashare import (
    Board,
    infer_board,
    lot_rules,
    stamp_tax_sell,
    tradability,
)

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


def test_tradability_limits_suspension_one_price():
    p = np.array([11.0, 9.0, 10.0, 10.0, 10.0, 10.0, np.nan, 10.0])
    high = np.array([11.0, 10.0, 11.0, 11.0, 9.0, 10.0, 11.0, 11.0])
    low = np.array([10.0, 9.0, 9.0, 11.0, 9.0, 10.0, 9.0, 9.0])
    buy, sell = tradability(
        p, high, low, np.full(8, 11.0), np.full(8, 9.0), [0, 0, 1, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(buy, [0, 1, 0, 0, 1, 1, 0, 1])
    np.testing.assert_array_equal(sell, [1, 0, 0, 1, 0, 1, 0, 1])


def test_price_tolerance_boundary():
    p = np.array([11 - 0.005, 11 - 0.005 - 1e-8, 9 + 0.005, 9 + 0.005 + 1e-8])
    b, s = tradability(
        p,
        np.full(4, 11.0),
        np.full(4, 9.0),
        np.full(4, 11.0),
        np.full(4, 9.0),
        np.zeros(4, bool),
    )
    np.testing.assert_array_equal(b, [False, True, True, True])
    np.testing.assert_array_equal(s, [True, True, False, True])
    b, s = tradability([10.0], [10.0], [10.0], [np.nan], [np.nan], [False])
    assert b[0] and s[0]


@pytest.mark.parametrize(
    "date,rate",
    [
        ("2008-09-19", 0.001),
        ("2008-09-20", 0.001),
        ("2023-08-27", 0.001),
        ("2023-08-28", 0.0005),
        ("2023-08-29", 0.0005),
    ],
)
def test_tax_dates(date, rate):
    assert stamp_tax_sell([date])[0] == rate


@pytest.mark.parametrize("dates", [["2008-09-18"], ["NaT"]])
def test_tax_unsupported(dates):
    with pytest.raises(ValueError, match="dates"):
        stamp_tax_sell(dates)


def test_lots_and_code_prefixes():
    m, s = lot_rules(list(Board))
    np.testing.assert_array_equal(m, [100, 100, 200, 100])
    np.testing.assert_array_equal(s, [100, 100, 1, 1])
    np.testing.assert_array_equal(
        infer_board(
            [
                "600001",
                "000001",
                "300001",
                "301001",
                "688001",
                "689009",
                "830001",
                "870001",
                "920001",
            ]
        ),
        [0, 0, 1, 1, 2, 2, 3, 3, 3],
    )
    for bad in (["abc"], ["999999"], ["1"], ["510300"]):
        with pytest.raises(ValueError, match="codes"):
            infer_board(bad)
    with pytest.raises(ValueError, match="board"):
        lot_rules([4])
