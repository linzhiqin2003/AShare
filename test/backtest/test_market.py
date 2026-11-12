"""market_from_panel：面板 → panelbt 行情、整手和费用；与 bench 的旧手写做法逐位相同。"""

import re

import numpy as np
import panelbt
import polars as pl
import pytest
from panelbt.markets import ASHARE, Board, infer_board, lot_rules
from panelbt.markets.ashare import tradability

from ashare.backtest import (
    boards,
    execution_targets,
    market_from_panel,
    market_inputs,
)

SYMBOLS = np.array(
    ["000001.SZ", "302132.SZ", "600000.SH", "688001.SH", "920001.BJ", "T600018.SH"]
)
BOARDS = ["主板", "创业板", "主板", "科创板", "北交所", None]


class FakePanel:
    def __init__(self, arrays, dates, symbols):
        self._arrays, self.dates, self.symbols, self.version = (
            arrays,
            dates,
            symbols,
            "v",
        )

    def __getitem__(self, name):
        return self._arrays[name]

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)


class FakeData:
    """只实现 market_from_panel 用到的读取接口。"""

    def __init__(self, arrays, dates, symbols, board=BOARDS, masks=True):
        self.arrays, self.dates, self.symbols, self.board = (
            arrays,
            dates,
            symbols,
            board,
        )
        self.masks = masks
        self.requests = []

    def panel(self, names, start=None, end=None, symbols=None):
        self.requests.append(list(names))
        for name in names:
            if name in ("can_buy_open", "can_sell_open") and not self.masks:
                raise FileNotFoundError(name)
        return FakePanel({k: self.arrays[k] for k in names}, self.dates, self.symbols)

    def securities(self):
        return pl.DataFrame({"symbol": list(SYMBOLS), "board": self.board}).lazy()


def synthetic(seed=0, t=60):
    rng = np.random.default_rng(seed)
    n = len(SYMBOLS)
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.03, (t, n)), axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    a = {
        "close": close,
        "open": prev * (1 + rng.normal(0, 0.01, (t, n))),
        "vwap": (prev + close) / 2,
        "high": np.maximum(prev, close) * 1.01,
        "low": np.minimum(prev, close) * 0.99,
        "upper_limit": np.round(prev * 1.1, 2),
        "lower_limit": np.round(prev * 0.9, 2),
        "volume": rng.integers(1e4, 1e6, (t, n)).astype(float),
        "split": np.where(rng.random((t, n)) < 0.02, 1.3, np.nan),
        "cash_dividend": np.where(rng.random((t, n)) < 0.03, 0.2, np.nan),
        "is_suspended": rng.random((t, n)) < 0.05,
        "has_quote": rng.random((t, n)) > 0.02,
        "delist": np.zeros((t, n), bool),
        "adj_factor": np.ones((t, n)),
    }
    hit = rng.random((t, n)) < 0.05
    a["open"][hit] = a["upper_limit"][hit]
    a["delist"][t - 5, 5] = True
    stop = a["is_suspended"] | ~a["has_quote"]
    for name in ("open", "close", "vwap"):
        a[name][~a["has_quote"]] = np.nan
    a["can_buy_open"], a["can_sell_open"] = tradability(
        a["open"], a["high"], a["low"], a["upper_limit"], a["lower_limit"], stop
    )
    dates = np.array([f"2024{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(t)])
    return a, dates


# bench/e2e/stage_backtest.py 改用新入口之前的手写做法，原样保留作对照。
FIELDS = (
    "open",
    "close",
    "high",
    "low",
    "upper_limit",
    "lower_limit",
    "is_suspended",
    "has_quote",
    "volume",
    "split",
    "cash_dividend",
    "delist",
)
MASKS = ("can_buy_open", "can_sell_open")


def old_market_inputs(panel):
    if all(m in panel._arrays for m in MASKS):
        can_buy, can_sell = (np.asarray(panel[m]) for m in MASKS)
        p = {name: np.asarray(panel[name]) for name in FIELDS[:2] + FIELDS[8:]}
    else:
        p = {name: np.asarray(panel[name]) for name in FIELDS}
        can_buy, can_sell = tradability(
            p["open"],
            p["high"],
            p["low"],
            p["upper_limit"],
            p["lower_limit"],
            p["is_suspended"] | ~p["has_quote"],
        )
    return {
        "exec_price": p["open"],
        "close": p["close"],
        "can_buy": can_buy,
        "can_sell": can_sell,
        "volume": p["volume"],
        "split": np.where(np.isnan(p["split"]), 1.0, p["split"]),
        "cash_div": np.nan_to_num(p["cash_dividend"]),
        "delist": p["delist"],
    }


def old_config(symbols):
    out = []
    for symbol in symbols:
        code = re.sub(r"\D", "", symbol)[:6]
        if code.startswith(("300", "301")):
            out.append(Board.CHINEXT)
        elif code.startswith(("688", "689")):
            out.append(Board.STAR)
        elif code.startswith(("43", "83", "87", "88", "92")):
            out.append(Board.BSE)
        else:
            out.append(Board.MAIN)
    min_qty, step = lot_rules(np.array(out))
    return panelbt.Config(min_qty=min_qty, qty_step=step)


def assert_market_equal(a, b):
    for name in (
        "exec_price",
        "close",
        "can_buy",
        "can_sell",
        "volume",
        "split",
        "cash_div",
        "delist",
    ):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and np.array_equal(
            x, y, equal_nan=x.dtype.kind == "f"
        ), name


@pytest.mark.parametrize("masks", [True, False])
def test_matches_old_handwritten_market_bit_for_bit(masks):
    arrays, dates = synthetic()
    data = FakeData(arrays, dates, SYMBOLS, masks=masks)
    setup = market_from_panel(data)
    full = FakePanel(
        arrays if masks else {k: v for k, v in arrays.items() if k not in MASKS},
        dates,
        SYMBOLS,
    )
    old = panelbt.Market(**old_market_inputs(full))
    assert_market_equal(setup.market, old)
    signal = np.full(setup.shape, np.nan)
    signal[5::10, :5] = 0.2
    plan = execution_targets(signal)
    cfg_old = old_config(SYMBOLS)
    cfg_new = setup.config()
    np.testing.assert_array_equal(cfg_new.min_qty, cfg_old.min_qty)
    np.testing.assert_array_equal(cfg_new.qty_step, cfg_old.qty_step)
    costs = panelbt.Costs()
    a = panelbt.run(setup.market, plan, costs, cfg_new)
    b = panelbt.run(old, plan, costs, cfg_old)
    np.testing.assert_array_equal(a.nav, b.nav)
    assert a.diagnostics["trades"] > 0


def test_masks_are_used_when_present_and_rules_otherwise():
    arrays, dates = synthetic(1)
    data = FakeData(arrays, dates, SYMBOLS)
    market_from_panel(data, fields=["adj_factor"])
    assert "can_buy_open" in data.requests[-1] and "high" not in data.requests[-1]
    assert "adj_factor" in data.requests[-1]
    data = FakeData(arrays, dates, SYMBOLS, masks=False)
    market_from_panel(data)
    assert "high" in data.requests[-1]


@pytest.mark.parametrize("exec", ["close", "vwap"])
def test_other_execution_prices(exec):
    arrays, dates = synthetic(2)
    setup = market_from_panel(FakeData(arrays, dates, SYMBOLS), exec=exec)
    price = arrays[exec]
    buy, sell = tradability(
        price,
        arrays["high"],
        arrays["low"],
        arrays["upper_limit"],
        arrays["lower_limit"],
        arrays["is_suspended"] | ~arrays["has_quote"],
    )
    np.testing.assert_array_equal(setup.market.exec_price, price)
    np.testing.assert_array_equal(setup.market.can_buy, buy)
    np.testing.assert_array_equal(setup.market.can_sell, sell)
    with pytest.raises(ValueError, match="exec"):
        market_from_panel(FakeData(arrays, dates, SYMBOLS), exec="mid")


def test_missing_events_become_one_and_zero():
    arrays, dates = synthetic(3)
    m = market_from_panel(FakeData(arrays, dates, SYMBOLS)).market
    assert not np.isnan(m.split).any() and not np.isnan(m.cash_div).any()
    events = np.isfinite(arrays["split"])
    np.testing.assert_array_equal(m.split[~events], 1.0)
    np.testing.assert_array_equal(m.split[events], 1.3)
    np.testing.assert_array_equal(m.cash_div[np.isnan(arrays["cash_dividend"])], 0.0)


def test_boards_from_column_with_code_fallback():
    got = boards(pl.DataFrame({"symbol": list(SYMBOLS), "board": BOARDS}), SYMBOLS)
    assert got.tolist() == [
        Board.MAIN,
        Board.CHINEXT,
        Board.MAIN,
        Board.STAR,
        Board.BSE,
        Board.MAIN,
    ]
    # 列顺序与面板不同也能对上
    rev = pl.DataFrame({"symbol": list(SYMBOLS[::-1]), "board": BOARDS[::-1]})
    assert boards(rev, SYMBOLS).tolist() == got.tolist()


def test_boards_unknown_raise_with_names():
    table = pl.DataFrame(
        {
            "symbol": ["000001.SZ", "999999.SH", "000002.SZ"],
            "board": ["CDR", None, "主板"],
        }
    )
    with pytest.raises(ValueError, match=r"000001\.SZ.*CDR.*999999\.SH"):
        boards(table, ["000001.SZ", "999999.SH", "000002.SZ"])
    with pytest.raises(KeyError, match="证券表里没有"):
        boards(table, ["600000.SH"])


def test_infer_board_new_prefixes_and_suffixes():
    assert infer_board(["302132.SZ", "302132", "600018.SH", "920001.BJ"]).tolist() == [
        1,
        1,
        0,
        3,
    ]
    with pytest.raises(ValueError, match="T600018.SH"):
        infer_board(["T600018.SH"])


def test_setup_costs_use_panel_dates():
    arrays, dates = synthetic(4)
    setup = market_from_panel(FakeData(arrays, dates, SYMBOLS))
    c = setup.costs(commission_rate=3e-4)
    assert c.commission_rate == 3e-4
    assert len(c.tax_sell) == len(dates)
    iso = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates]
    np.testing.assert_array_equal(c.tax_sell, ASHARE.costs(iso).tax_sell)
    np.testing.assert_array_equal(c.fee_rate_buy, ASHARE.costs(iso).fee_rate_buy)


def test_panel_argument_reuses_loaded_panel():
    arrays, dates = synthetic(5)
    data = FakeData(arrays, dates, SYMBOLS)
    panel = FakePanel(arrays, dates, SYMBOLS)
    setup = market_from_panel(data, panel=panel)
    assert data.requests == [] and setup.panel is panel
    with pytest.raises(ValueError, match="start/end/symbols"):
        market_from_panel(data, start="20240101", panel=panel)


def test_market_inputs_accepts_panel_like_object():
    arrays, dates = synthetic(6)
    out = market_inputs(FakePanel(arrays, dates, SYMBOLS))
    assert set(out) == {
        "exec_price",
        "close",
        "can_buy",
        "can_sell",
        "volume",
        "split",
        "cash_div",
        "delist",
    }
