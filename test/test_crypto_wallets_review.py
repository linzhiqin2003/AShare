"""Regressions from Claude C4 review round 1, with hand-computed expectations."""

import base64
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import panelbt as bt
import pytest
from panelbt.reference_wallets import WalletOracle

from ashare.backtest.crypto import MarginBand, TwoWallet
from test.test_crypto import scenario
from test.test_crypto_wallets import ZERO, markets


@pytest.mark.parametrize("engine", ["native", "oracle"])
def test_review_p1_1_decimal_reduction_native_and_independent_oracle(engine):
    p = [[100], [100], [200]]
    m = bt.Market(p, p, [[True]] * 3, [[True]] * 3, crypto_spot=True)
    cfg = bt.Config(initial_cash=1000, min_qty=0.1, qty_step=0.1, record_trades=True)
    targets = [[1], [0.8], [0.8]]
    if engine == "native":
        r = bt.run(m, bt.Targets.from_dense(targets, kind="shares"), ZERO, cfg)
        np.testing.assert_allclose(r.positions[:, 0], [1, 0.8, 0.8], atol=1e-15, rtol=0)
        assert r.equity[-1] == 1080  # 900 + sale 20 + remaining .8 * 200.
        return
    oracle = WalletOracle(m, ZERO, cfg)
    for target in targets:
        oracle.step(target)
    assert oracle.position[0] == pytest.approx(0.8, abs=1e-15)
    assert oracle.money + oracle.position[0] * 200 == 1080


def test_review_p1_1_decimal_capacity_full_close_after_split():
    m = bt.Market(
        [[100], [100]],
        [[100], [100]],
        [[True]] * 2,
        [[True]] * 2,
        crypto_spot=True,
        split=[[1], [1.5]],
        volume=[[100], [15]],
    )
    cfg = bt.Config(
        initial_cash=1000,
        min_qty=0.1,
        qty_step=0.1,
        max_participation=0.01,
        record_trades=True,
    )
    r = bt.run(m, bt.Targets.from_dense([[0.1], [0]], kind="shares"), ZERO, cfg)
    assert r.positions[-1, 0] == 0
    assert r.cash[-1] == 1005


@pytest.mark.parametrize("hedge", ["follow_spot", "independent"])
def test_review_p1_2_decimal_coin_contract_integer_boundary(hedge):
    tw = TwoWallet(*markets((100, 100, 200)), hedge=hedge, allow_residual=True)
    tw.step()
    tw.step({"BTC": (0.3, -0.3)})
    tw.step()
    r = tw.result()
    assert r.usdm.positions[1, 0] == -3
    np.testing.assert_allclose(r.equity, [200, 200, 200], atol=1e-12, rtol=0)


def two_coin_markets(with_exit=True, *, step=0.001):
    p = np.full((27, 2), 100.0)
    sell = np.ones_like(p, bool)
    sell[24, 1] = False
    fund = np.zeros_like(p)
    fund[25, 1] = 1
    fm, _, _, _ = scenario(p, np.zeros_like(p), sell=sell, funding=fund)
    s, u = markets([100] * 27, spot_cash=200)
    s.market = bt.Market(
        p, p, np.ones_like(p, bool), np.ones_like(p, bool), crypto_spot=True
    )
    s.config = replace(s.config, min_qty=step, qty_step=step)
    u.market = replace(fm, futures=replace(fm.futures, multiplier=[0.1, 0.1]))
    for m in (s, u):
        m.symbols = tuple(m.kind + ":" + c + "USDT" for c in ("BTC", "ETH"))
        m.profiles = [
            dict(m.profiles[0], coin=c, instrument=m.symbols[j])
            for j, c in enumerate(("BTC", "ETH"))
        ]
        m.signal_prices = np.vstack([np.full((1, 2), np.nan), p[:-1]])
        m.arrays = dict(
            coin_units=np.ones_like(p),
            lifecycle=np.full(p.shape, "active"),
            execution_symbol=np.tile(m.symbols, (27, 1)),
        )
    if with_exit:
        s.events = [
            dict(
                event_id="exit",
                instrument=s.symbols[0],
                kind="delist",
                known_at=25 * 3600000,
                effective_at=26 * 3600000,
                status="active",
            )
        ]
    return s, u


def test_review_p1_3_daily_delist_only_rehedges_affected_coin():
    tw = TwoWallet(
        *two_coin_markets(), trade_hour=0, hedge="follow_spot", residual_limit=100
    )
    for t in range(27):
        tw.step({"BTC": (0.5, -0.5), "ETH": (0.5, -0.5)} if t == 24 else None)
    r = tw.result()
    np.testing.assert_array_equal(r.usdm.positions[24:26, 1], [0, 0])
    np.testing.assert_array_equal(r.usdm.positions[24:26, 0], [-5, 0])
    assert r.equity[-1] == 300  # ETH stays flat; no unplanned funding income.


@dataclass(frozen=True)
class ConfiguredTarget:
    size: float

    def __call__(self, t, s, u):
        return {"BTC": (self.size, -self.size)} if t else {"BTC": (0, 0)}

    def state(self):
        return {}

    def restore(self, state):
        assert state == {}

    def config(self):
        return {"size": self.size}


def test_review_p2_4_snapshot_checks_fixed_strategy_config():
    s, u = markets((100, 100, 100), funding=[[0], [0], [1]])
    a = TwoWallet(s, u, ConfiguredTarget(0.5), hedge="follow_spot")
    a.step()
    with pytest.raises(ValueError, match="参数|配置|fingerprint"):
        TwoWallet.restore(a.snapshot(), s, u, ConfiguredTarget(1), hedge="follow_spot")


def legacy_fixture():
    m, _, cost, cfg = scenario([[100], [100], [120]], [[1], [1], [1]])
    return m, cost, cfg


def test_review_p2_5_old_liquidation_snapshot_legacy_path():
    m, cost, cfg = legacy_fixture()
    blob = base64.b64decode(
        (Path(__file__).parent / "fixtures/c4_legacy_snapshot.txt").read_text()
    )
    sim = bt.Simulator.restore(blob, m, cost, cfg, kind="shares")
    assert sim.snapshot() == blob
    assert sim.state.shares[0] == 1
    for name in ("wallet_balance", "unrealized", "transferable", "entry_price"):
        with pytest.raises(ValueError, match="ledger|账本"):
            getattr(sim.state, name)
    with pytest.raises(ValueError, match="ledger|账本"):
        sim.step(flow=0)
    assert sim.snapshot() == blob
    sim.step([1])
    sim.step([1])
    np.testing.assert_array_equal(sim.result().equity, [100, 100, 120])
    with pytest.raises(ValueError, match="ledger|账本"):
        _ = sim.result().wallet_balance


def test_review_p2_6_prefund_rebalance_target_before_spot_execution():
    tw = TwoWallet(
        *markets(
            (100, 100, 120, 100),
            spot_cash=125,
            future_cash=50,
            multiplier=0.001,
            initial=0.5,
        ),
        hedge="follow_spot",
        transfer=MarginBand(0.5, 0.25, 1),
        allow_residual=True,
    )
    for q in (None, (1, -1), (0.5, -0.5), (1, -1)):
        tw.step(None if q is None else {"BTC": q})
    assert tw.state[0].shares[0] == 1
    assert tw.state[1].shares[0] == -1000
    assert any(e["actual"] > 0 for e in tw.transfers)
    assert tw.result().diagnostics["valid"]


def test_review_quantity_tolerance_never_relaxes_cash_or_true_fraction():
    m = bt.Market([[100]], [[100]], [[True]], [[True]], crypto_spot=True)
    cfg = bt.Config(
        initial_cash=29.999999999, min_qty=0.1, qty_step=0.1, record_trades=True
    )
    r = bt.run(m, bt.Targets.from_dense([[0.3]], kind="shares"), ZERO, cfg)
    assert r.cash[0] >= 0
    assert r.positions[0, 0] == 0.2
    tw = TwoWallet(*markets(), hedge="independent", allow_residual=True)
    tw.step()
    tw.step({"BTC": (0.299999, -0.299999)})
    assert tw.state[1].shares[0] == -2


def test_review_prefund_hand_amounts_off_switch_and_cash_cap():
    for enabled, expected in ((True, -1000), (False, -800)):
        tw = TwoWallet(
            *markets(
                (100, 100, 120, 100),
                spot_cash=125,
                future_cash=50,
                multiplier=0.001,
                initial=0.5,
            ),
            hedge="follow_spot",
            transfer=MarginBand(0.5, 0.25, 1, pre_fund=enabled),
            allow_residual=True,
        )
        for q in (None, (1, -1), (0.5, -0.5), (1, -1)):
            tw.step(None if q is None else {"BTC": q})
        assert tw.state[1].shares[0] == expected
        if enabled:
            np.testing.assert_allclose(
                [x["requested"] for x in tw.transfers], [2, 30.4], atol=1e-12, rtol=0
            )
        else:
            assert tw.transfers == []
    s, u = markets((100, 100), spot_cash=2, future_cash=10, initial=0.5, fee_rate=0.01)
    tw = TwoWallet(
        s,
        u,
        transfer=MarginBand(0.5, 0.25, 1),
        hedge="independent",
        allow_residual=True,
    )
    tw.step()
    tw.step({"BTC": (0, -1)})
    # Target IM 50 + fees 1 + buffer 2 - equity 10 = 43; cash limits it to 2.
    assert tw.transfers[0]["requested"] == 43
    assert tw.transfers[0]["actual"] == 2
    assert tw.transfers[0]["rescue_failed"]


def test_review_snapshot_requires_fixed_config_protocol():
    class NoConfig(ConfiguredTarget):
        config = None

    tw = TwoWallet(*markets(), NoConfig(0.5))
    with pytest.raises(ValueError, match="config"):
        tw.snapshot()


def test_review_residual_classifies_min_notional_underfill_separately():
    s, u = markets((100, 100, 100), min_notional=[[50], [50], [50]])
    tw = TwoWallet(s, u, hedge="follow_spot", allow_residual=True)
    tw.step()
    tw.step({"BTC": (0.5, -0.5)})
    tw.step({"BTC": (0.6, -0.6)})
    r = tw.result()
    assert r.usdm.positions[-1, 0] == -5
    assert abs(r.rounding_residual_notional[-1, 0]) < 1e-12
    assert r.underfill_residual_notional[-1, 0] == pytest.approx(10)


def test_review_legacy_missing_ledger_does_not_report_false_zero_margin():
    # Source: rework fallback audit. A restored held contract cannot advertise
    # zero notional/IM via placeholder values in the unavailable-wallet state.
    m, cost, cfg = legacy_fixture()
    blob = base64.b64decode(
        (Path(__file__).parent / "fixtures/c4_legacy_snapshot.txt").read_text()
    )
    sim = bt.Simulator.restore(blob, m, cost, cfg, kind="shares")
    for field in ("notional", "initial_margin"):
        with pytest.raises(ValueError, match="ledger"):
            getattr(sim.state, field)
