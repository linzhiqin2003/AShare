"""可转债：时点（转股价、强赎公告、剩余规模）、派生字段、占位行、涨跌幅规则、回测行情。"""

import asyncio
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from futures_fixtures import DAYS
from test_fund_data import fake_stock_closes
from test_fund_fixtures import (
    CALL_ANN,
    CB_CHANGES,
    LAST_TRADE,
    FakeFundTushare,
    stock_price,
)

import ashare
import ashare.data as ad
from ashare.data import cb_panel
from ashare.data.cb_rules import cb_market, price_limits
from ashare.data.fund_panel import build_funds
from ashare.data.fund_sync import sync_funds
from ashare.data.store import Store

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)

FIELDS = [
    "open",
    "close",
    "volume",
    "amount",
    "has_quote",
    "is_listed",
    "delist",
    "conversion_price",
    "stock_close",
    "conversion_value",
    "conversion_premium",
    "double_low",
    "remaining_size",
    "call_announced",
    "days_to_maturity",
    "coupon_rate",
    "source_conversion_value",
]


@pytest.fixture
def cb(tmp_path, monkeypatch):
    monkeypatch.setattr(cb_panel, "stock_closes", fake_stock_closes)
    store = Store(tmp_path / "root")
    source = FakeFundTushare()
    asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1]))
    build_funds(store)
    funds = ad.open(store.root.parent).funds()
    return funds, funds.panel("cb", FIELDS), source


def col(p, code):
    return list(p.symbols).index(code)


def test_conversion_price_as_known(cb):
    _, p, _ = cb
    j = col(p, "113001.SH")
    got = np.asarray(p["conversion_price"])[:, j]
    # 直白的参考：每天取「公告次日与生效日都已到」的最后一次调整。
    for k in range(len(DAYS)):
        listed = k < 18
        want = 10.0
        for publish, effective, value in CB_CHANGES:
            if k >= max(effective, publish + 1):
                want = value
        if listed:
            assert got[k] == want, k
        else:
            assert np.isnan(got[k])
    # 第一次调整第 6 天生效、第 8 天才公告：第 6–8 天面板仍用旧价，源（按生效日）已用新价。
    assert got[6] == 10.0 and got[9] == 9.0 and got[12] == 8.5


def test_derived_fields(cb):
    _, p, _ = cb
    j = col(p, "113001.SH")
    for k in range(LAST_TRADE + 1):
        stock = stock_price("600001.SH", k)
        cp = p["conversion_price"][k, j]
        value = 100 / cp * stock
        assert p["stock_close"][k, j] == stock
        assert p["conversion_value"][k, j] == pytest.approx(value, rel=1e-15)
        premium = p["close"][k, j] / value - 1
        assert p["conversion_premium"][k, j] == pytest.approx(premium, rel=1e-12)
        assert p["double_low"][k, j] == pytest.approx(
            p["close"][k, j] + 100 * premium, rel=1e-12
        )
        maturity = (
            date(2031, 1, 1).toordinal()
            - date(int(DAYS[k][:4]), int(DAYS[k][4:6]), int(DAYS[k][6:])).toordinal()
        )
        assert p["days_to_maturity"][k, j] == maturity
        assert p["coupon_rate"][k, j] == pytest.approx(0.004)
    # 成交量手 → 张，成交额万元 → 元。
    k = 3
    assert p["volume"][k, j] == (1000 + k) * 10
    assert p["amount"][k, j] / p["volume"][k, j] == pytest.approx(
        p["close"][k, j], rel=0.02
    )


def test_call_and_remaining_size(cb):
    _, p, _ = cb
    j = col(p, "113001.SH")
    flags = np.asarray(p["call_announced"])[:, j]
    # 第 11 天收盘后公告，第 12 天起为真，摘牌（第 18 天）后为假。
    assert np.flatnonzero(flags).tolist() == list(range(CALL_ANN + 1, 18))
    size = np.asarray(p["remaining_size"])[:, j]
    assert size[4] == 1e9 and size[5] == 9e8  # 第 4 天公告，第 5 天起可知
    assert size[13] == 9e8 and size[14] == 6e8
    k = col(p, "123001.SZ")
    assert np.asarray(p["call_announced"])[:, k].sum() == 0
    assert np.asarray(p["remaining_size"])[5, k] == 5e8  # 首次公告前取发行规模


def test_placeholder_rows_and_listing(cb):
    funds, p, _ = cb
    j = col(p, "113001.SH")
    has = np.asarray(p["has_quote"])[:, j]
    assert has[: LAST_TRADE + 1].all() and not has[LAST_TRADE + 1 :].any()
    assert np.isnan(
        np.asarray(p["open"])[LAST_TRADE + 1 :, j]
    ).all()  # 占位行没有成交价
    assert (
        np.asarray(p["is_listed"])[:18, j].all()
        and not np.asarray(p["is_listed"])[18:, j].any()
    )
    assert np.flatnonzero(np.asarray(p["delist"])[:, j]).tolist() == [18]
    k = col(p, "123001.SZ")
    assert (
        not np.asarray(p["has_quote"])[:5, k].any()
        and np.asarray(p["has_quote"])[5:, k].all()
    )
    calls = funds.table("cb", "calls").collect()
    assert calls.filter(calls["status"] == "公告实施强赎")["record_date"].to_list() == [
        DAYS[17]
    ]


def test_price_limit_rules():
    dates = np.array(["20220729", "20220801", "20220802"])
    pc = np.full((3, 2), 120.0)
    first = np.zeros((3, 2), bool)
    first[1, 1] = True
    up, down = price_limits(dates, pc, first)
    assert np.isnan(up[0]).all() and np.isnan(down[0]).all()  # 新规前没有涨跌停
    assert up[1, 0] == 144.0 and down[1, 0] == 96.0
    assert up[1, 1] == 157.3 and down[1, 1] == 56.7  # 上市首日按发行价 100


def test_cb_market(cb):
    funds, _, _ = cb
    cm = cb_market(funds)
    j = list(cm.symbols).index("113001.SH")
    assert not np.asarray(cm.market.can_buy)[LAST_TRADE + 1 :, j].any()
    assert (np.asarray(cm.config().min_qty) == 10).all()
    assert np.all(np.asarray(cm.costs().tax_sell) == 0)


def test_call_flag_uses_first_implementation_notice(tmp_path, monkeypatch):
    """同一只债有两条「公告实施强赎」（重复公告）：标记从第一条的下一交易日起。"""
    import polars as pl
    from test_fund_fixtures import DAYS as D

    monkeypatch.setattr(cb_panel, "stock_closes", fake_stock_closes)
    source = FakeFundTushare()
    calls = source.tables["cb_call"]
    again = calls.filter(pl.col("is_call") == "公告实施强赎").with_columns(
        pl.lit(D[CALL_ANN + 3]).alias("ann_date")
    )
    source.tables["cb_call"] = pl.concat([calls, again])
    store = Store(tmp_path / "root")
    asyncio.run(sync_funds(source, store, D[0], D[-1]))
    build_funds(store)
    p = ad.open(store.root.parent).funds().panel("cb", ["call_announced"])
    flags = np.asarray(p["call_announced"])[:, list(p.symbols).index("113001.SH")]
    assert np.flatnonzero(flags).tolist() == list(range(CALL_ANN + 1, 18))


def run_cb(cm, weights, *, exit=True):
    import panelbt as bt

    return bt.run(
        cm.market,
        cm.targets(weights, exit=exit),
        cm.costs(commission_rate=0.0, slippage=0.0),
        cm.config(initial_cash=1e5, record_trades=True),
    )


def test_cb_redemption_and_exit(cb):
    """手算：第 1 行开盘买 113001.SH，一直持有。

    - 不清仓（exit=False）：最后交易日（第 14 行）之后估值为赎回价 100.3，第 18 行摘牌按 100.3 了结；
      最后收盘（第 14 行）远高于赎回价时，持有到摘牌只拿到赎回价。
    - 默认清仓：第 13 行（最后交易日前 1 个交易日）开盘按正常成交卖出。
    """
    funds, p, _ = cb
    cm = cb_market(funds, ["113001.SH"])
    assert cm.exit_rows.tolist() == [LAST_TRADE - 1]
    assert cm.redemption.tolist() == [100.3]
    close = np.asarray(cm.market.close)[:, 0]
    assert (close[LAST_TRADE + 1 : 18] == 100.3).all()
    can_buy = np.asarray(cm.market.can_buy)[:, 0]
    assert not can_buy[CALL_ANN + 1 :].any() and can_buy[CALL_ANN]
    weights = np.full(cm.shape, np.nan)
    weights[1] = 1.0
    held = run_cb(cm, weights, exit=False)
    qty = held.positions[17, 0]
    assert qty > 0 and held.positions[18, 0] == 0
    cash_before = held.cash[17]
    assert held.cash[18] == pytest.approx(cash_before + qty * 100.3)
    sold = run_cb(cm, weights)
    assert (
        sold.positions[LAST_TRADE - 2, 0] > 0 and sold.positions[LAST_TRADE - 1, 0] == 0
    )
    price = np.asarray(cm.market.exec_price)[LAST_TRADE - 1, 0]
    assert sold.cash[LAST_TRADE - 1] == pytest.approx(
        sold.cash[LAST_TRADE - 2] + qty * price
    )


def test_cb_apply_exit_keeps_other_targets(cb):
    funds, _, _ = cb
    cm = cb_market(funds, ["113001.SH", "123001.SZ"])
    w = np.full(cm.shape, np.nan)
    w[6] = [0.4, 0.5]
    w[16] = [0.3, 0.6]  # 清仓之后又想买回：一律为 0
    out = cm.apply_exit(w)
    row = LAST_TRADE - 1
    assert out[row].tolist() == [0.0, 0.5]  # 插入的一行：其他债保持最近一次目标
    assert out[16].tolist() == [0.0, 0.6]
    assert np.isnan(out[row + 1]).all() and out[6].tolist() == [0.4, 0.5]
    assert np.isnan(cm.apply_exit(np.full(cm.shape, np.nan))).all()  # 没有持仓不插行
    late = cb_market(funds, ["113001.SH"], exit_days=0)
    assert late.exit_rows.tolist() == [LAST_TRADE]
    early = cb_market(funds, ["113001.SH"], exit_days=10)
    assert early.exit_rows.tolist() == [CALL_ANN + 1]  # 不早于强赎公告可知


def test_cb_redemption_mark_starts_after_last_trade(cb):
    """赎回价只从最后交易日的下一行起作为估值价；最后交易日当天仍是市场收盘（否则等于提前知道赎回结果）。"""
    funds, _, _ = cb
    cm = cb_market(funds, ["113001.SH"])
    raw = np.asarray(funds.panel("cb", ["close"], symbols=["113001.SH"])["close"])[:, 0]
    close = np.asarray(cm.market.close)[:, 0]
    assert raw[LAST_TRADE] != 100.3
    assert close[LAST_TRADE] == raw[LAST_TRADE]
    assert (close[: LAST_TRADE + 1] == raw[: LAST_TRADE + 1]).all()
    assert close[LAST_TRADE + 1] == 100.3
