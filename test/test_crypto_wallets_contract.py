"""C4 contract regressions; each expectation comes from the design review."""

from dataclasses import replace

import numpy as np
import panelbt as bt
import pytest

from ashare.backtest.crypto import MarginBand, TwoWallet, daily_targets, two_wallet
from ashare.backtest.crypto_run import InvalidCryptoResult
from test.test_crypto_wallets import ZERO, markets, simulator


def test_dynamic_flow_conflicts_even_zero_and_rejects_nonfinite_atomically():
    sim, _ = simulator(
        [[100], [100]], config=bt.Config(initial_cash=100, cash_flow=[0, 1])
    )
    old = sim.snapshot()
    with pytest.raises(ValueError, match="cash_flow"):
        sim.step(flow=0)
    assert sim.snapshot() == old
    sim, _ = simulator([[100], [100]])
    old = sim.snapshot()
    for f in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            sim.step(flow=f)
        assert sim.snapshot() == old


def test_wallet_static_flow_keeps_old_limits_and_timing():
    sim, _ = simulator(
        [[100], [150]],
        settlement=[[150], [150]],
        initial=0.2,
        config=bt.Config(initial_cash=100, cash_flow=[0, -100]),
    )
    row = sim.step([1])
    assert row.wallet_balance == 100
    # Existing cash-flow policy may withdraw settled profit; the new limit applies only to flow=.
    assert sim.state.wallet_balance == 0
    assert sim.state.cash == 50
    assert sim.state.equity == 150
    assert sim.state.unrealized == 50
    sim.step()
    assert sim.result().wallet_balance.tolist() == [100, 0]


def test_wallet_expiry_realizes_original_entry_and_fee():
    sim, args = simulator(
        [[100], [110], [99]],
        settlement=[[105], [115], [99]],
        expiry=[2],
        delivery_fee_rate=0.01,
    )
    for row in ([2], None, None):
        sim.step(row)
    r = sim.result()
    np.testing.assert_allclose(r.wallet_balance, [100, 100, 96.02], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r.unrealized, [10, 30, 0], rtol=0, atol=1e-12)
    assert sim.state.entry_price[0] == 0
    batch = bt.run_batch(
        args[0],
        [bt.Targets.from_dense([[2], [np.nan], [np.nan]], kind="shares")],
        args[1],
        args[2],
    )
    np.testing.assert_array_equal(batch.wallet_balance[0], r.wallet_balance)


def test_wallet_entry_uses_effective_price_not_raw_cost():
    sim, _ = simulator([[100], [120]], costs=replace(ZERO, slippage=0.01))
    sim.step([2])
    assert sim.state.entry_price[0] == 101
    row = sim.step([1])
    assert sim.state.entry_price[0] == 101
    assert row.wallet_balance == pytest.approx(117.8)
    assert row.unrealized == 19


def test_dynamic_flow_and_entry_snapshot_each_boundary():
    sim, (m, cost, cfg) = simulator([[100], [120], [90], [110]], fee_rate=0.001)
    plan = ([1], [2], [-1], [0])
    flows = (0, 10, -3, 0)
    for target, flow in zip(plan, flows, strict=True):
        sim.step(target, flow=flow)
        blob = sim.snapshot()
        restored = bt.Simulator.restore(blob, m, cost, cfg, kind="shares")
        assert restored.snapshot() == blob
        assert restored.state.wallet_balance == sim.state.wallet_balance
        np.testing.assert_array_equal(restored.state.entry_price, sim.state.entry_price)
        sim = restored
    r = sim.result()
    np.testing.assert_allclose(
        r.wallet_balance + r.unrealized, r.equity, atol=1e-12, rtol=0
    )


def test_margin_band_uses_previous_settlement_equity():
    sim, _ = simulator([[100], [140], [160]], cash=50)
    sim.step([-1])
    assert MarginBand(0.5, 0.25, 1)(1, None, sim.state) == 0
    sim.step()
    # W=50 U=-40 nominal=140 -> request .5*140-10=60.
    assert MarginBand(0.5, 0.25, 1)(2, None, sim.state) == 60
    other, _ = simulator([[100], [100]], cash=200)
    other.step([1])
    assert MarginBand(0.5, 0.25, 1)(1, None, other.state) == -150
    for values in ((0.5, 0.5, 1), (0.5, 0, 0.5), (np.nan, 0, 1)):
        with pytest.raises(ValueError):
            MarginBand(*values)


def test_same_reference_and_thousand_coin_units():
    spot, usdm = markets((100, 100), future_prices=(110, 110), multiplier=0.1)
    tw = TwoWallet(spot, usdm, initial=(500, 500), kind="weights", allow_residual=True)
    tw.step()
    tw.step({"BTC": (0.4, -0.4)})
    r = tw.result()
    assert r.spot.positions[-1, 0] == 4
    assert r.usdm.positions[-1, 0] == -40
    assert r.residual_coins[-1, 0] == 0
    spot, usdm = markets((0.1, 0.1), future_prices=(100, 100), multiplier=0.001)
    usdm.arrays["coin_units"][:] = 1000
    tw = TwoWallet(spot, usdm, initial=(500, 500))
    tw.step()
    tw.step({"BTC": (100, -100)})
    r = tw.result()
    assert r.usdm.positions[-1, 0] == -100
    assert r.residual_coins[-1, 0] == 0


def test_residual_default_rejects_and_explicit_allow_marks_invalid():
    spot, usdm = markets((100, 100))
    plan = np.array([[[0, 0]], [[1, 0]]])
    with pytest.raises(InvalidCryptoResult) as exc:
        two_wallet(spot, usdm, plan, initial=(200, 100))
    assert exc.value.result.diagnostics["residual_max"] == 100
    r = two_wallet(spot, usdm, plan, initial=(200, 100), allow_residual=True)
    assert not r.diagnostics["valid"]


def test_decisions_are_frozen_and_half_row_failure_invalidates(monkeypatch):
    seen = []

    def transfer(t, s, u):
        seen.append((s, u))
        return 10

    tw = TwoWallet(*markets((100, 100)), transfer=transfer)
    tw.step()
    tw.step({"BTC": (0.5, -0.5)})
    assert seen[0][0].shares[0] == 0
    assert seen[1][0].shares[0] == 0
    assert seen[1][0].cash == 90
    tw = TwoWallet(*markets())

    def failure(*args, **kwargs):
        raise ValueError("injected second step failure")

    monkeypatch.setattr(tw._u, "step", failure)
    with pytest.raises(ValueError, match="injected"):
        tw.step()
    for operation in (tw.snapshot, tw.result, tw.step, lambda: tw.state):
        with pytest.raises(RuntimeError, match="失效"):
            operation()


def test_actual_flow_mismatch_invalidates(monkeypatch):
    tw = TwoWallet(*markets(), transfer=lambda *args: 10)
    original = tw._u.step
    monkeypatch.setattr(
        tw._u,
        "step",
        lambda target, flow: original(target, flow=flow)._replace(cash_flow=0),
    )
    with pytest.raises(RuntimeError, match="实际划转"):
        tw.step()
    with pytest.raises(RuntimeError, match="失效"):
        tw.snapshot()


def test_pair_snapshot_fingerprint_and_custom_protocol():
    spot, usdm = markets((100, 100, 130, 80))
    plan = np.array([[[0, 0]], [[0.5, -0.5]], [[0.5, -0.5]], [[0, 0]]])
    kw = dict(
        initial=(100, 100), transfer=MarginBand(0.5, 0.25, 1), hedge="follow_spot"
    )
    tw = TwoWallet(spot, usdm, plan, **kw)
    tw.step()
    tw.step()
    blob = tw.snapshot()
    restored = TwoWallet.restore(blob, spot, usdm, plan, **kw)
    a, b = tw.run(), restored.run()
    for field in ("equity", "returns", "residual_coins", "residual_notional"):
        assert getattr(a, field).tobytes() == getattr(b, field).tobytes()
    assert a.transfers == b.transfers
    with pytest.raises(ValueError, match="指纹"):
        TwoWallet.restore(blob, spot, usdm, plan, **(kw | {"hedge": "independent"}))
    with pytest.raises(ValueError, match="校验"):
        TwoWallet.restore(blob[:-1], spot, usdm, plan, **kw)
    tw = TwoWallet(spot, usdm, transfer=lambda *args: 0)
    with pytest.raises(ValueError, match="state"):
        tw.snapshot()


def test_daily_target_shift_once_and_transfers_every_hour():
    spot, usdm = markets([100] * 49)
    plan = (daily_targets(([0], [[0.5]])), daily_targets(([0], [[-0.5]])))
    calls = []
    r = two_wallet(
        spot,
        usdm,
        plan,
        initial=(100, 100),
        transfer=lambda t, *_: calls.append(t) or 0,
    )
    assert calls == list(range(49))
    assert r.spot.positions[:24].sum() == 0
    assert np.all(r.spot.positions[24:] == 0.5)
    assert len(r.daily()) == 2


def test_new_listing_missing_previous_price_rejected():
    spot, usdm = markets()
    spot.signal_prices[2] = np.nan
    tw = TwoWallet(spot, usdm)
    tw.step()
    tw.step()
    with pytest.raises(ValueError, match="上一"):
        tw.step({"BTC": (1, -1)})
    assert tw.day == 2


def test_c4_oracle_seed5_full_withdrawal_does_not_create_bankruptcy():
    # Source: independent corpus seed 5. W-cash=2.84e-14 after expiry.
    # Withdrawing W must not turn an empty solvent account into negative cash.
    from test.generators_wallets import case

    spot, usdm, plan, policy, hedge = case(5)
    r = two_wallet(
        spot,
        usdm,
        plan,
        transfer=policy,
        hedge=hedge,
        allow_residual=True,
        allow_tier_exceeded=True,
    )
    assert r.usdm.bankruptcy_events == []
    assert r.usdm.cash[10] == 0
    assert r.usdm.cash_flow[11] > 0


def test_example_cli_does_not_import_bench_profile_as_stdlib():
    # Source: first real example invocation, bench/profile.py shadowed profile.
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    p = subprocess.run(
        [sys.executable, "bench/crypto_wallets_example.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p.returncode == 0, p.stderr


def test_crypto_spot_full_close_keeps_odd_lot_after_volume_limit():
    # Source: C4 full-close contract audit. Rounding capacity before checking
    # a complete close incorrectly strands 0.05 of a 0.15 position.
    m = bt.Market(
        [[10], [10]],
        [[10], [10]],
        [[True], [True]],
        [[True], [True]],
        volume=[[10], [1.5]],
        split=[[1], [1.5]],
        crypto_spot=True,
    )
    cfg = bt.Config(
        initial_cash=100,
        min_qty=0.1,
        qty_step=0.1,
        max_participation=0.1,
        record_trades=True,
    )
    r = bt.run(m, bt.Targets.from_dense([[0.1], [0]], kind="shares"), ZERO, cfg)
    assert r.positions[-1, 0] == 0


def test_delisted_stranded_spot_keeps_finite_residual_and_snapshot():
    # Source: C4 final boundary review; signal prices disappear after delisting.
    spot, usdm = markets()
    mask = spot.market.can_sell.copy()
    mask[2:] = False
    spot.market = replace(spot.market, can_sell=mask)
    spot.arrays["lifecycle"][2:] = "ended"
    spot.signal_prices[3:] = np.nan
    spot.events.append(
        dict(
            event_id="exit",
            instrument="spot:BTCUSDT",
            kind="delist",
            known_at=2 * 3600000,
            effective_at=2 * 3600000,
            status="active",
        )
    )
    tw = TwoWallet(spot, usdm, allow_residual=True)
    tw.step()
    tw.step({"BTC": (1, -0.5)})
    tw.step()
    tw.step()
    assert tw.snapshot().startswith(b"TWALLET1")
    result = tw.result()
    assert result.residual_notional[-1, 0] == 50
    assert result.diagnostics["spot"]["unrealized"][0]["quantity"] == 1


def test_function_protocol_restore_does_not_mutate_callers_strategy():
    # Source: C4 final snapshot review; deepcopy(function) aliases its closure.
    counter = {"count": 0}

    def transfer(t, s, u):
        counter["count"] += 1
        return 0

    transfer.state = lambda: dict(counter)
    transfer.restore = lambda state: counter.update(state)
    transfer.config = lambda: {}
    spot, usdm = markets()
    tw = TwoWallet(spot, usdm, transfer=transfer)
    tw.step()
    blob = tw.snapshot()
    restored = TwoWallet.restore(blob, spot, usdm, transfer=transfer)
    assert counter["count"] == 1
    restored.step()
    assert counter["count"] == 1
    assert restored.transfer.state() == {"count": 2}


def test_daily_blocked_targets_do_not_retry_and_exit_overrides_schedule():
    spot, usdm = markets([100] * 49)
    mask = spot.market.can_buy.copy()
    mask[24] = False
    spot.market = replace(spot.market, can_buy=mask)
    plan = (
        daily_targets(([0, 24 * 3600000], [[0.5], [0.5]])),
        daily_targets(([0, 24 * 3600000], [[-0.5], [-0.5]])),
    )
    r = two_wallet(spot, usdm, plan, hedge="follow_spot")
    assert np.all(r.spot.positions[:48] == 0)
    assert r.spot.positions[48, 0] == 0.5
    spot, usdm = markets([100] * 49)
    spot.events.append(
        dict(
            event_id="exit",
            instrument="spot:BTCUSDT",
            kind="delist",
            known_at=25 * 3600000,
            effective_at=30 * 3600000,
            status="active",
        )
    )
    r = two_wallet(spot, usdm, plan, hedge="follow_spot")
    assert r.spot.positions[24, 0] == 0.5
    assert np.all(r.spot.positions[25:] == 0)
    assert np.all(r.usdm.positions[25:] == 0)


def test_crypto_spot_marker_belongs_only_to_market():
    # Source: C4 final diff review; the market flag was also added to Targets.
    from dataclasses import fields

    assert "crypto_spot" in {f.name for f in fields(bt.Market)}
    assert "crypto_spot" not in {f.name for f in fields(bt.Targets)}


@pytest.mark.parametrize("record", [False, True])
def test_empty_wallet_batch_keeps_enabled_output_shape(record):
    # Source: C4 final output-schema review; an empty batch lost enabled W/U.
    _, (market, costs, cfg) = simulator([[100], [100]])
    result = bt.run_batch(market, [], costs, replace(cfg, record_trades=record))
    assert result.wallet_balance is not None
    assert result.unrealized is not None
    assert result.wallet_balance.shape == (0, 2)
    assert result.unrealized.shape == (0, 2)


def test_combined_diagnostics_preserve_single_wallet_fields():
    # Source: C4 final public-schema review; merged diagnostics omitted C3 fields.
    spot, usdm = markets((100, 100, 80), multiplier=1, future_cash=10)
    plan = np.zeros((3, 1, 2))
    plan[1, 0, 1] = 1
    r = two_wallet(spot, usdm, plan, allow_residual=True)
    diag = r.diagnostics["usdm"]
    assert diag["bankruptcy_deficit"] == 10
    assert diag["clearance_fee"] == 0
    assert diag["liquidation_events"][0]["time"] == 2 * 3600000
    for side in ("spot", "usdm"):
        d = r.diagnostics[side]
        assert d["periods_per_year"] == 8760
        assert d["daily_periods_per_year"] == 365
        assert d["rolls"] == []
        assert d["roll_overlap_events"] == 0
        assert "carry_blocked" in d


@pytest.mark.parametrize(
    "minimum,target,expected",
    [
        (0.07, 0.07, 0.07),
        (0.07, 0.08, 0.08),
        (0.07, 0.29, 0.29),
        (0.075, 0.08, 0.08),
        (0.075, 0.074, 0),
    ],
)
def test_decimal_spot_lattice_does_not_raise_minimum_or_lose_step(
    minimum, target, expected
):
    # Source: C4 final decimal boundary review; ceil(.07/.01) incorrectly gives 8.
    market = bt.Market([[1.0]], [[1.0]], [[True]], [[True]], crypto_spot=True)
    r = bt.run(
        market,
        bt.Targets.from_dense([[target]], kind="shares"),
        ZERO,
        bt.Config(initial_cash=1, min_qty=minimum, qty_step=0.01, record_trades=True),
    )
    assert r.positions[0, 0] == pytest.approx(expected, rel=0, abs=1e-16)
