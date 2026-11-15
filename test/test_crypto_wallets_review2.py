"""Source: Claude C4 review round 2; expectations are calculated by hand."""

import base64
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import panelbt as bt
import pytest
from panelbt.reference_wallets import PairOracle

from ashare.backtest.crypto import MarginBand, TwoWallet
from test.test_crypto_wallets import markets
from test.test_crypto_wallets_review import legacy_fixture, two_coin_markets


@pytest.mark.parametrize("resume", [False, True])
def test_review2_exit_retries_blocked_future_only_until_both_legs_close(resume):
    spot, usdm = two_coin_markets()
    buy = usdm.market.can_buy.copy()
    buy[25, 0] = False
    funding = usdm.futures.funding.copy()
    funding[26, 0] = 1
    usdm.market = replace(
        usdm.market, can_buy=buy, futures=replace(usdm.futures, funding=funding)
    )
    kwargs = dict(trade_hour=0, hedge="follow_spot", residual_limit=100)
    tw = TwoWallet(spot, usdm, **kwargs)
    for t in range(26):
        tw.step({"BTC": (0.5, -0.5), "ETH": (0.5, -0.5)} if t == 24 else None)
    if resume:
        tw = TwoWallet.restore(tw.snapshot(), spot, usdm, **kwargs)
    tw.step()
    r = tw.result()
    # BTC spot exits at 25, its blocked short must exit at 26 before funding.
    # ETH's blocked opening at 24 must never be retried off schedule.
    np.testing.assert_array_equal(r.spot.positions[24:27, 0], [0.5, 0, 0])
    np.testing.assert_array_equal(r.usdm.positions[24:27, 0], [-5, -5, 0])
    np.testing.assert_array_equal(r.usdm.positions[24:27, 1], [0, 0, 0])
    assert r.equity[-1] == 300  # Constant prices, zero fees and no funding held.


@pytest.mark.parametrize("engine", ["native", "oracle"])
def test_review2_zero_buffer_prefund_credits_seven_contracts_hand_expected(engine):
    spot, usdm = markets(
        (10, 10, 11), spot_cash=100, future_cash=1.05, multiplier=0.1, initial=0.5
    )
    policy = MarginBand(0.5, 0.25, 1, buffer=0)
    # Hand calculation: .7 BTC / .1 BTC per contract = 7 contracts;
    # IM = 7 * .1 * 10 * .5 = 3.5; transfer is mathematically 2.45.
    # Spot +.7 and futures -.7 earn +.7 and -.7 when price rises by 1.
    # IEEE addition must really credit the computed IM, without a cash epsilon.
    if engine == "native":
        tw = TwoWallet(
            spot, usdm, hedge="follow_spot", transfer=policy, residual_limit=10
        )
        tw.step()
        tw.step({"BTC": (0.7, -0.7)})
        tw.step()
        r = tw.result()
        positions, equity = r.usdm.positions[:, 0], r.equity
        assert r.transfers[0]["actual"] <= 100
        np.testing.assert_array_equal(r.spot.cash_flow + r.usdm.cash_flow, [0] * 3)
    else:
        oracle = PairOracle(spot, usdm, hedge="follow_spot", transfer=policy)
        records = [oracle.step(t) for t in (None, [[0.7, -0.7]], None)]
        positions = [r["usdm"]["positions"][0] for r in records]
        equity = [r["equity"] for r in records]
    np.testing.assert_array_equal(positions, [0, -7, -7])
    np.testing.assert_allclose(equity, [101.05] * 3, rtol=0, atol=2e-14)


def test_review2_zero_buffer_rounding_topup_never_exceeds_spot_cash():
    # The mathematical transfer is 2.45, but that exact available float cannot
    # credit the seventh contract. Capacity remains strict, so report failure.
    tw = TwoWallet(
        *markets((10, 10), spot_cash=2.45, future_cash=1.05, initial=0.5),
        hedge="independent",
        transfer=MarginBand(0.5, 0.25, 1, buffer=0),
        allow_residual=True,
    )
    tw.step()
    tw.step({"BTC": (0, -0.7)})
    r = tw.result()
    assert r.transfers[0]["actual"] == 2.45
    assert r.transfers[0]["requested"] > 2.45
    assert r.transfers[0]["rescue_failed"]
    assert r.transfers[0]["reason"] == "capacity"
    assert r.spot.cash[-1] == 0
    assert r.usdm.positions[-1, 0] == -6


@dataclass
class Counter:
    n: int = 0
    size: int = 0

    def __call__(self, t, s, u):
        self.n += 1
        return self.size

    def config(self):
        return {"size": self.size}

    def state(self):
        return {"n": self.n}

    def restore(self, state):
        self.n = state["n"]


def test_review2_restore_advanced_policy_checks_config_not_mutable_state():
    spot, usdm = markets()
    policy = Counter()
    tw = TwoWallet(spot, usdm, transfer=policy)
    tw.step()
    blob = tw.snapshot()
    policy.n = 9
    restored = TwoWallet.restore(blob, spot, usdm, transfer=policy)
    assert restored.transfer.n == 1
    assert policy.n == 9
    restored.step()
    assert restored.transfer.n == 2
    assert policy.n == 9
    with pytest.raises(ValueError, match="配置"):
        TwoWallet.restore(blob, spot, usdm, transfer=Counter(size=1))


@pytest.mark.parametrize("kind", ["state", "result"])
@pytest.mark.parametrize("display", ["repr", "asdict"])
def test_review2_legacy_dataclass_display_omits_unavailable_fields(
    kind, display, capsys
):
    market, costs, config = legacy_fixture()
    blob = base64.b64decode(
        (Path(__file__).parent / "fixtures/c4_legacy_snapshot.txt").read_text()
    )
    sim = bt.Simulator.restore(blob, market, costs, config, kind="shares")
    if kind == "result":
        while sim.day < sim.n_days:
            sim.step()
    value = sim.state if kind == "state" else sim.result()
    shown = repr(value) if display == "repr" else asdict(value)
    missing = (
        "wallet_balance",
        "unrealized",
        "transferable",
        "entry_price",
        "notional",
        "initial_margin",
    )
    for name in missing:
        assert (name + "=" if display == "repr" else name) not in shown
        with pytest.raises(ValueError, match="ledger"):
            getattr(value, name)
    assert "cash" in shown and "equity" in shown
    print(value)
    assert "cash=" in capsys.readouterr().out
    # Filtering legacy metadata must not change the regular dataclass schema.
    normal = bt.Simulator(market, costs, config, kind="shares")
    assert "wallet_balance" in asdict(normal.state)
