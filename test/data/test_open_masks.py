from pathlib import Path

import numpy as np
import polars as pl
from panelbt.markets import ashare as market

import ashare
import ashare.data as ad
from ashare.data import panel_trade
from ashare.data.catalog import FIELDS
from ashare.data.panel import build
from ashare.data.sources.base import FetchResult
from ashare.data.verify_panel import audit_open_masks

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)

TOL = panel_trade.PRICE_TOL


def reference(p, h, lo, up, down, stop, quote):
    return market.tradability(p, h, lo, up, down, stop | ~quote)


def edge_inputs(seed=0, shape=(300, 40)):
    rng = np.random.default_rng(seed)
    up = np.round(rng.uniform(5, 20, shape), 2)
    down = np.round(up * 0.8, 2)
    # 开盘价落在涨跌停价附近的容差两侧、恰好在边界上。
    choices = np.stack(
        [
            up,
            up - TOL,
            up - TOL - 0.001,
            up - TOL + 0.001,
            down,
            down + TOL,
            down + TOL + 0.001,
            down + TOL - 0.001,
            (up + down) / 2,
        ]
    )
    pick = rng.integers(0, len(choices), shape)
    p = np.take_along_axis(choices, pick[None], 0)[0]
    one = rng.random(shape) < 0.3
    h = np.where(one, p, p + rng.uniform(0, 1, shape))
    lo = np.where(one, p - rng.choice([0, TOL, TOL + 0.001], shape), p - 0.5)
    for array, rate in ((p, 0.05), (h, 0.03), (lo, 0.03), (up, 0.05), (down, 0.05)):
        array[rng.random(shape) < rate] = np.nan
    p[rng.random(shape) < 0.02] = 0.0
    p[rng.random(shape) < 0.01] = -1.0
    p[rng.random(shape) < 0.01] = np.inf
    stop = rng.random(shape) < 0.05
    quote = rng.random(shape) > 0.05
    return p, h, lo, up, down, stop, quote


def test_tolerance_matches_panelbt():
    assert panel_trade.PRICE_TOL == market.PRICE_TOL


def test_masks_equal_panelbt_on_edges():
    for seed in range(5):
        args = edge_inputs(seed)
        buy, sell = panel_trade.open_masks(*args)
        ref_buy, ref_sell = reference(*args)
        assert buy.dtype == bool and sell.dtype == bool
        np.testing.assert_array_equal(buy, ref_buy)
        np.testing.assert_array_equal(sell, ref_sell)
        # 用例真正覆盖了各个分支，而不是全真或全假。
        assert 0.2 < buy.mean() < 0.95 and 0.2 < sell.mean() < 0.95
        assert (buy != sell).any()


def test_hand_cases():
    assert 0.01 - 0.005 == TOL
    nan = np.nan
    rows = [
        # open, high, low, up, down, stop, quote -> buy, sell
        (11.0, 11.0, 10.5, 11.0, 9.0, False, True, False, True),  # 开盘涨停
        (10.996, 11.2, 10.5, 11.0, 9.0, False, True, False, True),  # 容差内
        (10.994, 11.0, 10.5, 11.0, 9.0, False, True, True, True),  # 容差外
        (11.0, 11.0, 11.0, 11.0, 9.0, False, True, False, True),  # 一字涨停
        (9.0, 9.0, 9.0, 11.0, 9.0, False, True, True, False),  # 一字跌停
        (9.004, 9.5, 8.0, 11.0, 9.0, False, True, True, False),  # 开盘跌停
        (10.0, 10.0, 10.0, nan, nan, False, True, True, True),  # 无涨跌停
        (10.0, 10.5, 9.5, 11.0, 9.0, True, True, False, False),  # 停牌
        (10.0, 10.5, 9.5, 11.0, 9.0, False, False, False, False),  # 无行情
        (nan, nan, nan, 11.0, 9.0, False, True, False, False),  # 未上市
        (0.0, 0.0, 0.0, 11.0, 9.0, False, True, False, False),  # 零价
        (10.0, nan, 9.5, 11.0, 9.0, False, True, True, True),  # 缺最高价
        # 最高最低恰好相差容差（浮点精确）：仍算一字板，开盘未触板也不能买。
        (0.005, 0.01, 0.005, 0.011, 0.001, False, True, False, False),
    ]
    cols = list(zip(*rows, strict=True))
    args = [np.array(c, dtype=float)[None] for c in cols[:5]]
    args += [np.array(c, dtype=bool)[None] for c in cols[5:7]]
    buy, sell = panel_trade.open_masks(*args)
    assert buy[0].tolist() == list(cols[7])
    assert sell[0].tolist() == list(cols[8])
    ref_buy, ref_sell = reference(*args)
    np.testing.assert_array_equal(buy, ref_buy)
    np.testing.assert_array_equal(sell, ref_sell)


def test_fill_blocks_and_offset(monkeypatch):
    args = edge_inputs(7, (37, 11))
    names = (
        "open",
        "high",
        "low",
        "upper_limit",
        "lower_limit",
        "is_suspended",
        "has_quote",
    )
    source = dict(zip(names, args, strict=True))
    ref_buy, ref_sell = reference(*args)
    for rows in (1, 5, 256):
        monkeypatch.setattr(panel_trade, "ROWS", rows)
        for start in (0, 3, 36):
            n = 37 - start
            buy, sell = panel_trade.fill(
                source.__getitem__,
                np.zeros((n, 11), bool),
                np.zeros((n, 11), bool),
                start,
            )
            np.testing.assert_array_equal(buy, ref_buy[start:])
            np.testing.assert_array_equal(sell, ref_sell[start:])


def limit_tables(tables):
    """000001 第 2 天开盘涨停、000003 第 4 天一字跌停（当天 ST，跌停价 9×0.95）。"""
    daily = tables["daily"].with_columns(
        pl.when(
            (pl.col("ts_code") == "000001.SZ") & (pl.col("trade_date") == "20260105")
        )
        .then(11.0)
        .otherwise(pl.col("open"))
        .alias("open"),
    )
    lock = (pl.col("ts_code") == "000003.SZ") & (pl.col("trade_date") == "20260107")
    tables["daily"] = daily.with_columns(
        *[
            pl.when(lock).then(8.55).otherwise(pl.col(c)).alias(c)
            for c in ("open", "high", "low")
        ]
    )
    return tables


def test_built_fields_match_panelbt_and_metadata(tables, make_store):
    store = make_store(limit_tables(tables))
    build(store)
    data = ad.open(store.root.parent)
    names = [
        "open",
        "high",
        "low",
        "upper_limit",
        "lower_limit",
        "is_suspended",
        "has_quote",
    ]
    p = data.panel(names + ["can_buy_open", "can_sell_open", "is_listed"])
    ref_buy, ref_sell = reference(*[np.asarray(p[n]) for n in names])
    assert p["can_buy_open"].dtype == bool and p["can_sell_open"].dtype == bool
    np.testing.assert_array_equal(p["can_buy_open"], ref_buy)
    np.testing.assert_array_equal(p["can_sell_open"], ref_sell)
    # 夹具：000001 第 2 天开盘涨停，第 3 天停牌，第 4 天开盘即跌停（前收 10、开 9）；
    # 000003 第 4 天一字跌停。
    assert p["can_buy_open"][:, 0].tolist() == [True, False, False, True]
    assert p["can_sell_open"][:, 0].tolist() == [True, True, False, False]
    assert p["can_buy_open"][:, 2].tolist() == [True, True, True, True]
    assert p["can_sell_open"][:, 2].tolist() == [True, True, True, False]
    assert not (p["can_buy_open"] & ~p["is_listed"]).any()
    for name in ("can_buy_open", "can_sell_open"):
        meta = data.field(name)
        assert meta["dtype"] == "bool" and "tradability" in meta["known_at"]
    assert FIELDS["can_buy_open"].dtype == "bool"


def test_incremental_masks_bitwise(tables, make_store):
    tables = limit_tables(tables)
    short = {
        k: (
            v.filter(pl.col("trade_date") < "20260107")
            if "trade_date" in v.columns
            else v
        )
        for k, v in tables.items()
    }
    store = make_store(short, "incremental")
    build(store, end="20260106")
    for name in (
        "daily",
        "adj_factor",
        "daily_basic",
        "stk_limit",
        "stock_st",
        "suspend_d",
    ):
        store.write(
            name,
            "20260107",
            {"trade_date": "20260107"},
            FetchResult(
                tables[name].filter(pl.col("trade_date") == "20260107"), "now", 1
            ),
        )
    assert build(store, incremental=True)["reused_rows"] == 3
    full = make_store(tables, "full")
    build(full)
    a, b = ad.open(store.root.parent), ad.open(full.root.parent)
    for name in ("can_buy_open", "can_sell_open"):
        assert a.panel([name])[name].tobytes() == b.panel([name])[name].tobytes()
    assert not b.panel(["can_sell_open"])["can_sell_open"][3, 2]


def test_verify_detects_flipped_cell(tables, make_store):
    store = make_store(limit_tables(tables))
    build(store)
    dates = ad.open(store.root.parent).panel(["close"]).dates.tolist()
    result = audit_open_masks(store, dates)
    assert result["status"] == "complete" and result["failed"] == 0
    assert result["checked"] == 12
    data = ad.open(store.root.parent)
    path = data._path / "can_sell_open.npy"
    array = np.load(path, mmap_mode="r+")
    array[1, 1] = ~array[1, 1]
    array.flush()
    del array
    result = audit_open_masks(store, dates)
    assert result["failed"] == 1
    assert result["examples"] == [{"symbol": "000002.SZ", "date": "20260105"}]
