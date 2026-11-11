from pathlib import Path

import numpy as np
import panelbt as bt
from hypothesis import given, settings
from hypothesis import strategies as st

from .generators import case
from .test_engine import ZERO

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


@given(st.integers(0, 2**32 - 1))
@settings(max_examples=100, deadline=None)
def test_fractional_zero_cost_matches_weight_path(seed):
    rng = np.random.default_rng(seed)
    t, n = 40, 8
    p = 10 * np.exp(np.cumsum(rng.normal(0, 0.05, (t, n)), axis=0))
    weights = rng.dirichlet(np.ones(n), size=t) * rng.uniform(0.1, 0.95, (t, 1))
    days = np.arange(0, t, 3)
    w = np.full((t, n), np.nan)
    w[days] = weights[days]
    m = bt.Market(p, p, np.ones((t, n), bool), np.ones((t, n), bool))
    result = bt.run(
        m,
        bt.Targets.from_dense(w),
        ZERO,
        bt.Config(initial_cash=1e5, min_qty=200, qty_step=0, record_trades=True),
    )
    # 独立权重财富递推：调仓日只改变配置，不改变财富。
    cash = 1e5
    units = np.zeros(n)
    nav = []
    for day in range(t):
        wealth = cash + np.dot(units, p[day])
        if day in days:
            units = weights[day] * wealth / p[day]
            cash = wealth * (1 - weights[day].sum())
        nav.append((cash + np.dot(units, p[day])) / 1e5)
    np.testing.assert_allclose(result.nav, nav, rtol=1e-9, atol=1e-9)
    assert np.all(result.cash >= 0)


def test_buy_sell_masks_keep_holdings():
    rng = np.random.default_rng(11)
    p = np.exp(rng.normal(size=(20, 5))) + 10
    allowed = np.zeros((20, 5), bool)
    allowed[0] = True
    w = rng.dirichlet(np.ones(5), size=20)
    m = bt.Market(p, p, allowed, allowed)
    r = bt.run(
        m,
        bt.Targets.from_dense(w),
        ZERO,
        bt.Config(initial_cash=1e5, min_qty=0, qty_step=0, record_trades=True),
    )
    np.testing.assert_array_equal(
        r.positions, np.broadcast_to(r.positions[0], r.positions.shape)
    )


def test_random_lots_cash_accounting_and_cost_shadow():
    for seed in range(100):
        m, target, cost, cfg = case(seed + 4000)
        r = bt.run(m, target, cost, cfg)
        last = np.zeros(m.shape[1])
        previous = np.zeros(m.shape[1])
        previous_assets = cfg.initial_cash
        shadow_cash = cfg.initial_cash
        trade_days = [[] for _ in range(m.shape[0])]
        event_days = [[] for _ in range(m.shape[0])]
        for tr in r.trades:
            trade_days[tr.day].append(tr)
            if tr.qty > 0 and cfg.qty_step[tr.col] > 0:
                minimum, step = cfg.min_qty[tr.col], cfg.qty_step[tr.col]
                assert tr.qty >= minimum
                units = (tr.qty - minimum) / step
                assert abs(units - round(units)) < 1e-9
        for event in r.delist_events:
            event_days[event.day].append(event)
        for day in range(m.shape[0]):
            div = float(np.dot(previous, m.cash_div[day]))
            after_split = previous * m.split[day]
            end_prices = np.where(np.isfinite(m.close[day]), m.close[day], last)
            # 把送转后的持仓从上日收盘重估到今日收盘，零股也计入。
            pnl = float(np.dot(after_split, end_prices) - np.dot(previous, last))
            shadow_cash += div
            for event in event_days[day]:
                pnl += event.proceeds - event.qty * end_prices[event.col]
                shadow_cash += event.proceeds
            for tr in trade_days[day]:
                mid = m.exec_price[day, tr.col]
                pnl += tr.qty * (end_prices[tr.col] - mid)
                shadow_cash -= tr.qty * mid
            fees = sum(
                getattr(r, k)[day]
                for k in ("commission", "tax", "transfer_fee", "slippage_cost")
            )
            assets = r.nav[day] * cfg.initial_cash
            delta = assets - previous_assets
            np.testing.assert_allclose(
                delta,
                pnl + div - fees,
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"seed={seed},day={day}",
            )
            shadow_assets = shadow_cash + np.dot(r.positions[day], end_prices)
            # 对同一实际持仓/成交路径，扣成本财富必不高于无成本影子账本。
            assert assets <= shadow_assets + 1e-9
            np.testing.assert_allclose(
                shadow_assets - assets,
                sum(
                    np.sum(getattr(r, k)[: day + 1])
                    for k in ("commission", "tax", "transfer_fee", "slippage_cost")
                ),
                rtol=1e-9,
                atol=1e-9,
            )
            previous_assets = assets
            previous = r.positions[day].copy()
            last = end_prices
        assert np.all(r.cash >= 0)


def test_cost_monotonicity_on_common_rising_path():
    for seed in range(50):
        rng = np.random.default_rng(seed)
        p = np.cumprod(1 + rng.uniform(0, 0.03, (40, 1)), axis=0) * 10
        m = bt.Market(p, p, np.ones_like(p, bool), np.ones_like(p, bool))
        w = bt.Targets.from_dense(np.ones_like(p))
        cfg = bt.Config(initial_cash=10000, min_qty=0, qty_step=0)
        free = bt.run(m, w, ZERO, cfg)
        paid = bt.run(m, w, bt.Costs(), cfg)
        assert np.all(paid.nav <= free.nav + 1e-9)


def test_same_targets_cost_monotonicity_has_counterexample():
    # 原规格的无条件断言不成立：成本/整手改变持仓后，留存现金能躲过下跌。
    p = np.array([[10.0], [1.0]])
    m = bt.Market(p, p, np.ones_like(p, bool), np.ones_like(p, bool))
    w = bt.Targets.from_dense([[1.0], [np.nan]])
    cfg = bt.Config(initial_cash=10000, record_trades=True)
    free = bt.run(m, w, ZERO, cfg)
    paid = bt.run(m, w, bt.Costs(0, 5, 0, 0, 0, 0), cfg)
    assert paid.commission[0] == 5
    assert free.positions[0, 0] == 1000
    assert paid.positions[0, 0] == 900
    assert paid.nav[-1] > free.nav[-1]
