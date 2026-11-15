"""C4 hand-calculated counterexamples, written before production changes.

Source: DESIGN 4.17 cases 1--14 and the C4 adversarial design review.
All fixtures are synthetic. Expectations are recorded in docs/C4_NOTES.md.
"""

from dataclasses import replace

import numpy as np
import panelbt as bt
import pytest

from test.test_crypto import ZERO as FUTURES_ZERO
from test.test_crypto import scenario

ZERO = replace(FUTURES_ZERO, fee_rate_buy=0, fee_rate_sell=0)


def simulator(prices, **kwargs):
    m, _, costs, cfg = scenario(prices, np.zeros_like(prices), **kwargs)
    return bt.Simulator(m, costs, cfg, kind="shares"), (m, costs, cfg)


def markets(
    prices=(100, 100, 100, 100),
    *,
    multiplier=0.1,
    future_cash=100,
    spot_cash=100,
    future_prices=None,
    **kwargs,
):
    from ashare.backtest.crypto import CryptoMarket

    p = np.asarray(prices, float)[:, None]
    t = len(p)
    m, _, costs, cfg = scenario(
        np.asarray(future_prices if future_prices is not None else prices)[:, None],
        np.zeros((t, 1)),
        cash=future_cash,
        costs=ZERO,
        **kwargs,
    )
    m = replace(m, futures=replace(m.futures, multiplier=[multiplier]))
    sm = bt.Market(
        p, p.copy(), np.ones_like(p, bool), np.ones_like(p, bool), crypto_spot=True
    )
    sc = bt.Config(
        initial_cash=spot_cash, min_qty=0.001, qty_step=0.001, record_trades=True
    )
    out = []
    for kind, market, config in (("spot", sm, sc), ("usdm", m, cfg)):
        name = kind + ":BTCUSDT"
        close = market.close if kind == "spot" else market.futures.settlement
        signals = np.vstack(([np.nan], close[:-1]))
        out.append(
            CryptoMarket(
                market,
                np.arange(t, dtype=np.int64) * 3600000,
                (name,),
                signals,
                costs,
                config,
                {},
                {
                    "execution_symbol": np.full((t, 1), name),
                    "lifecycle": np.full((t, 1), "active"),
                    "coin_units": np.ones((t, 1)),
                    "mark_open": market.exec_price,
                },
                [
                    {
                        "instrument": name,
                        "coin": "BTC",
                        "coin_units": 1,
                        "multiplier": multiplier if kind == "usdm" else 1,
                        "contract_type": "PERPETUAL" if kind == "usdm" else "SPOT",
                    }
                ],
                [],
                kind,
                None,
            )
        )
    return tuple(out)


def test_c4_01_profit_cannot_be_transferred():
    sim, _ = simulator([[100], [150]], settlement=[[150], [150]], initial=0.2)
    sim.step([1])
    assert sim.state.wallet_balance == 100
    assert sim.state.unrealized == 50
    assert sim.state.transferable == 70
    row = sim.step(flow=-100)
    assert row.cash_flow == -70
    assert row.withdrawal_shortfall == 30
    assert row.wallet_balance == 30
    assert row.unrealized == 50


def test_c4_02_loss_reduces_transferable():
    sim, _ = simulator([[100], [80]], settlement=[[80], [80]], initial=0.375)
    sim.step([1])
    assert sim.state.wallet_balance == 100
    assert sim.state.unrealized == -20
    assert sim.state.transferable == 50


def test_c4_03_dynamic_deposit_precedes_open_check():
    for flow, expected, liquidations in ((0, 0, 1), (10, 1, 0)):
        sim, _ = simulator([[100], [90]], cash=10)
        sim.step([1])
        sim.step(flow=flow)
        assert sim.state.shares[0] == expected
        assert sim.result().coverage["liquidation_open"] == liquidations


def test_c4_04_cannot_undo_previous_close_liquidation():
    sim, _ = simulator([[100], [100]], cash=10, funding=[[10], [0]])
    sim.step([1])
    sim.step(flow=10)
    r = sim.result()
    assert r.coverage["liquidation_close"] == 1
    assert r.positions[:, 0].tolist() == [0, 0]
    assert r.cash.tolist() == [0, 10]


def test_c4_05_internal_transfer_is_not_return():
    from ashare.backtest.crypto import two_wallet

    r = two_wallet(
        *markets(),
        np.zeros((4, 1, 2)),
        initial=(100, 100),
        transfer=lambda t, s, u: (30, -20, 0, 0)[t],
    )
    np.testing.assert_array_equal(r.equity, [200] * 4)
    np.testing.assert_array_equal(r.returns, [0] * 4)
    np.testing.assert_array_equal(r.spot.cash_flow + r.usdm.cash_flow, [0] * 4)


def test_c4_06_receiver_bankrupt_neither_side_moves():
    from ashare.backtest.crypto import TwoWallet

    tw = TwoWallet(
        *markets((100, 100, 80, 80), multiplier=1, future_cash=10),
        initial=(100, 10),
        hedge="independent",
        allow_residual=True,
        transfer=lambda t, s, u: 50 if t == 3 else 0,
    )
    tw.step()
    tw.step({"BTC": (0, 1)})
    tw.step()
    tw.step()
    r = tw.result()
    assert r.spot.cash_flow[3] == r.usdm.cash_flow[3] == 0
    assert r.spot.cash[3] == 100
    assert r.transfers[-1]["reason"] == "receiver_bankrupt"


def test_c4_07_entry_add_reduce_reverse_close_reopen():
    sim, _ = simulator([[100], [120], [130], [140], [150], [160]], cash=1000)
    for q, entry, wallet, unrealized in zip(
        [1, 2, 1, -1, 0, 1],
        [100, 110, 110, 140, 0, 160],
        [1000, 1000, 1020, 1050, 1040, 1040],
        [0, 20, 20, 0, 0, 0],
        strict=True,
    ):
        sim.step([q])
        assert sim.state.entry_price[0] == entry
        assert sim.state.wallet_balance == wallet
        assert sim.state.unrealized == unrealized


def test_c4_08_fees_funding_affect_wallet_not_unrealized():
    sim, _ = simulator([[100]], settlement=[[110]], fee_rate=0.01, funding=[[2]])
    row = sim.step([1])
    assert (row.wallet_balance, row.unrealized, row.equity) == (97, 10, 107)


def test_c4_09_deficit_credit_and_bankrupt_flow_rejection():
    sim, _ = simulator([[100], [80], [80]], cash=100)
    sim.step([10])
    row = sim.step()
    assert row.bankruptcy_deficit == 100
    assert row.wallet_balance == row.unrealized == row.equity == 0
    assert sim.step(flow=50).cash_flow == 0
    assert sim.state.transferable == 0


def test_c4_10_transfer_uses_previous_settlement_current_margin():
    sim, _ = simulator([[100], [150]], cash=300, initial=[[0.1], [0.2]])
    sim.step([10])
    assert sim.state.transferable == 100


def test_c4_11_follow_actual_spot_half_fill():
    from ashare.backtest.crypto import TwoWallet

    tw = TwoWallet(
        *markets((100, 100), spot_cash=50), initial=(50, 100), hedge="follow_spot"
    )
    tw.step()
    tw.step({"BTC": (1, -1)})
    r = tw.result()
    assert r.spot.positions[-1, 0] == 0.5
    assert r.usdm.positions[-1, 0] == -5
    np.testing.assert_array_equal(r.residual_coins, [[0], [0]])


def test_c4_12_crypto_spot_shares_scaled_then_rounded():
    m = bt.Market([[110]], [[110]], [[True]], [[True]], crypto_spot=True)
    cfg = bt.Config(initial_cash=100, qty_step=0.001, min_qty=0.001, record_trades=True)
    r = bt.run(m, bt.Targets.from_dense([[1]], kind="shares"), ZERO, cfg)
    assert r.positions[0, 0] == 0.909
    assert r.cash[0] == pytest.approx(0.01)
    old = bt.run(
        replace(m, crypto_spot=False),
        bt.Targets.from_dense([[1]], kind="shares"),
        ZERO,
        cfg,
    )
    assert old.positions[0, 0] == 100 / 110


def test_c4_13_first_row_nonzero_rejected():
    from ashare.backtest.crypto import TwoWallet

    tw = TwoWallet(*markets(), initial=(100, 100))
    with pytest.raises(ValueError, match="上一|previous"):
        tw.step({"BTC": (1, -1)})
    assert tw.day == 0


@pytest.mark.parametrize("seed", range(20))
def test_c4_14_wallet_conservation_random(seed):
    rng = np.random.default_rng(seed)
    prices = rng.integers(80, 121, (30, 2)).astype(float)
    targets = rng.integers(-10, 11, (30, 2))
    m, plan, costs, cfg = scenario(
        prices,
        targets,
        cash=500,
        fee_rate=0.001,
        funding=rng.uniform(-0.5, 0.5, (30, 2)),
    )
    sim = bt.Simulator(m, costs, cfg, kind="shares")
    for i, row in enumerate(targets):
        record = sim.step(row)
        state = sim.state
        u = sum(state.shares * (prices[i] - state.entry_price))
        assert record.unrealized == pytest.approx(u, abs=1e-10)
        assert record.wallet_balance + u == pytest.approx(record.equity, abs=1e-10)
