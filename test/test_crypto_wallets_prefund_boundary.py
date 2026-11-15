"""Source: round-1 rework audit, an unused unlisted column poisoned prefunding."""

from types import SimpleNamespace

import numpy as np
import panelbt as bt
import pytest

from ashare.backtest.crypto import MarginBand, TwoWallet


def prefund_with_missing_mark(target):
    f = SimpleNamespace(
        settlement=np.array([[100.0, np.nan], [100.0, 100.0]]),
        margin_rate=0.5,
        fee_rate=0.0,
        fee_per_contract=0.0,
    )
    pair = SimpleNamespace(
        usdm_market=SimpleNamespace(futures=f, market=SimpleNamespace(shape=(2, 2))),
        hedge="follow_spot",
        _indices=[np.arange(2), np.arange(2)],
        _units=[np.ones((2, 2)), np.ones((2, 2))],
        _mult=np.array([0.1, 0.1]),
        coins=("BTC", "ETH"),
        transfer=MarginBand(0.5, 0.25, 1),
    )
    state = SimpleNamespace(
        shares=np.zeros(2),
        initial_margin=0.0,
        wallet_balance=10.0,
        unrealized=0.0,
        cash=10.0,
        bankrupt=False,
    )
    # Valid native simulator for the read-only affordability query. Missing
    # references below are injected only into the coordinator boundary inputs.
    from dataclasses import replace

    from test.test_crypto import scenario

    market, _, costs, config = scenario(
        [[100, 100]] * 2, [[0, 0]] * 2, cash=10, initial=0.5
    )
    market = replace(market, futures=replace(market.futures, multiplier=[0.1, 0.1]))
    pair._u = bt.Simulator(market, costs, config, kind="shares")
    return TwoWallet._prefund(
        pair, 1, np.array(target), -10 * np.array(target), np.ones(2), state, 0.0
    )


def test_prefund_unused_missing_mark_does_not_poison_other_coin():
    # BTC IM 25 + buffer 1 - equity 10 = 16. Flat unlisted ETH needs zero.
    assert prefund_with_missing_mark([0.5, 0]) == 16


def test_prefund_required_missing_mark_rejects_before_either_step():
    with pytest.raises(ValueError, match="结算价"):
        prefund_with_missing_mark([0.5, 0.5])


def test_prefund_independent_oracle_ignores_unheld_missing_mark():
    from dataclasses import replace

    from panelbt.reference_wallets import PairOracle

    from test.test_crypto_wallets_review import two_coin_markets

    spot, usdm = two_coin_markets(False)
    prices = usdm.futures.settlement.copy()
    prices[0, 1] = np.nan
    # Pure oracle boundary injection: public Market validation already rejects
    # missing futures settlements before a simulator can be constructed.
    usdm.market = SimpleNamespace(
        **(
            vars(usdm.market)
            | dict(
                shape=usdm.market.shape,
                futures=replace(usdm.futures, settlement=prices, margin_rate=0.5),
            )
        )
    )
    usdm.config = replace(usdm.config, initial_cash=10)
    oracle = PairOracle(
        spot, usdm, transfer=MarginBand(0.5, 0.25, 1), hedge="follow_spot"
    )
    oracle.step()
    row = oracle.step([[0.5, -0.5], [0, 0]])
    assert row["actual"] == 16
