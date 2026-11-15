"""Source: review round 3; exact rational budgets and strict snapshot schema."""

import hashlib
import json
import math
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import panelbt as bt
import pytest
from panelbt.reference_wallets import PairOracle

from ashare.backtest.crypto import MarginBand, TwoWallet
from test.test_crypto import scenario
from test.test_crypto_wallets import markets
from test.test_crypto_wallets_review import two_coin_markets


@pytest.mark.parametrize("engine", ["native", "oracle"])
def test_review3_prefund_fee_subtraction_hand_expected(engine):
    s, u = markets(
        (10, 10, 11),
        spot_cash=100,
        future_cash=0.15,
        multiplier=0.1,
        initial=0.5,
        fee_rate=0.0004,
    )
    transfer = MarginBand(0.5, 0.25, 1, buffer=0)
    if engine == "native":
        tw = TwoWallet(s, u, hedge="follow_spot", transfer=transfer, residual_limit=10)
        tw.step()
        tw.step({"BTC": (0.1, -0.1)})
        tw.step()
        r = tw.result()
        positions, equity = r.usdm.positions[:, 0], r.equity
    else:
        tw = PairOracle(s, u, hedge="follow_spot", transfer=transfer)
        rows = [tw.step(t) for t in (None, [[0.1, -0.1]], None)]
        positions = [r["usdm"]["positions"][0] for r in rows]
        equity = [r["equity"] for r in rows]
    # One contract needs .5 IM and .0004 fee; hedge P&L cancels at price 11.
    np.testing.assert_array_equal(positions, [0, -1, -1])
    np.testing.assert_allclose(equity, [100.15, 100.1496, 100.1496], rtol=0, atol=3e-14)


@pytest.mark.parametrize("engine", ["native", "oracle"])
def test_review3_fraction_prefund_random_prices_fees_multipliers_targets(engine):
    rng = np.random.default_rng(34017)
    covered = set()
    for seed in range(2048):
        price = (
            F(int(rng.integers(1, 100000)), 100)
            if seed % 2
            else F(int(rng.choice([1, 2, 5, 10, 20, 100])))
        )
        multiplier = (F(1, 1000), F(1, 100), F(1, 10), F(1), F(10))[
            int(rng.integers(5))
        ]
        quantity = int(rng.integers(1, 41)) if seed % 2 else int(rng.integers(1, 5))
        bp = int(rng.integers(11))
        fee = F(bp, 10000)
        margin = (F(1, 10), F(1, 5), F(1, 2))[int(rng.integers(3))]
        nominal = quantity * multiplier * price
        required = nominal * (margin + fee)
        initial = float(nominal * margin * F(int(rng.integers(1, 8)), 10))
        available = float(4 * nominal + 100)
        exact_flow = required - F.from_float(initial)
        theoretical = min(
            quantity,
            (F.from_float(initial) + F.from_float(available))
            // (multiplier * price * (margin + fee)),
        )
        assert theoretical == quantity and exact_flow > 0
        s, u = markets(
            (float(price),) * 2,
            spot_cash=available,
            future_cash=initial,
            multiplier=float(multiplier),
            initial=float(margin),
            fee_rate=float(fee),
        )
        policy = MarginBand(0.5, 0.25, 1, buffer=0)
        coins = float(quantity * multiplier)
        if engine == "native":
            tw = TwoWallet(
                s, u, hedge="follow_spot", transfer=policy, allow_residual=True
            )
            tw.step()
            sr, ur = tw.step({"BTC": (coins, -coins)})
            actual = tw.state[1].shares[0]
            flow = ur.cash_flow
            assert sr.cash_flow == -ur.cash_flow
        else:
            tw = PairOracle(s, u, hedge="follow_spot", transfer=policy)
            tw.step()
            row = tw.step([[coins, -coins]])
            actual = row["usdm"]["positions"][0]
            flow = row["actual"]
            assert row["spot"]["flow"] == -row["usdm"]["flow"]
        case = dict(
            seed=seed,
            price=str(price),
            multiplier=str(multiplier),
            quantity=quantity,
            fee_bp=bp,
            margin=str(margin),
            initial=initial,
            exact_required=str(required),
            exact_flow=str(exact_flow),
            actual=actual,
            flow=flow,
        )
        assert actual == -theoretical, json.dumps(case, sort_keys=True)
        assert 0 <= flow <= available, case
        # Exact rational amount, compared only within the binary64 rounding
        # neighborhood; the integer-fill and cash-cap assertions are exact.
        assert abs(F.from_float(float(flow)) - exact_flow) <= 16 * F.from_float(
            math.ulp(float(required))
        ), case
        covered.add(bp)
    assert covered == set(range(11))


@pytest.mark.parametrize("engine", ["native", "oracle"])
def test_review3_zero_request_steps_credited_cash_without_subnormal_loop(engine):
    s, u = markets((10, 10), future_cash=0.5004, initial=0.5, fee_rate=0.0004)
    policy = MarginBand(0.5, 0.25, 1, buffer=0)
    if engine == "native":
        tw = TwoWallet(s, u, hedge="follow_spot", transfer=policy, residual_limit=10)
        tw.step()
        tw.step({"BTC": (0.1, -0.1)})
        assert tw.state[1].shares[0] == -1
        flow = tw.transfers[0]["actual"]
    else:
        tw = PairOracle(s, u, hedge="follow_spot", transfer=policy)
        tw.step()
        row = tw.step([[0.1, -0.1]])
        assert row["usdm"]["positions"][0] == -1
        flow = row["actual"]
    assert flow == math.ulp(0.5004)


def test_review3_rust_preview_matches_fill_on_adjacent_float_budgets():
    # The new read-only predicate must match both basket and sequential gates,
    # including existing holdings, reductions, reversals and per-contract fees.
    rng = np.random.default_rng(34018)
    covered = set()
    for seed in range(128):
        p = rng.integers(1, 10000, 2) / 100
        mult = rng.choice([0.001, 0.1, 1], 2)
        rate = rng.choice([0.1, 0.2, 0.5], 2)
        fee = rng.integers(0, 11, 2) / 10000
        per = rng.choice([0, 0.001, 0.01], 2)
        before = np.array([2, -2]) if seed % 2 else np.zeros(2)
        goal = rng.integers(1, 9, 2) * rng.choice([-1, 1], 2)
        required = sum(
            abs(goal) * mult * p * rate + abs(goal - before) * (mult * p * fee + per)
        )
        for direction in (-math.inf, math.inf):
            m, _, costs, config = scenario(np.tile(p, (2, 1)), [[0, 0]] * 2, cash=10000)
            m = replace(
                m,
                futures=replace(
                    m.futures,
                    multiplier=mult,
                    margin_rate=rate[None],
                    fee_rate=fee[None],
                    fee_per_contract=per[None],
                ),
            )
            sim = bt.Simulator(m, costs, config, kind="shares")
            sim.step(before)
            state = sim.state
            flow = max(
                math.nextafter(required - state.cash, direction), -state.transferable
            )
            blob = sim.snapshot()
            feasible = sim.opening_feasible(goal, p, flow=flow)
            assert sim.snapshot() == blob
            sim.step(goal, flow=flow)
            filled = np.array_equal(sim.state.shares, goal)
            assert feasible == filled, (seed, flow, feasible, sim.state.shares, goal)
            covered.add(feasible)
    assert covered == {True, False}


def rewrite_blob(blob, change):
    data = json.loads(blob[40:])
    change(data)
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return b"TWALLET1" + hashlib.sha256(payload).digest() + payload


FIELDS = (
    "version",
    "fingerprint",
    "day",
    "spot",
    "usdm",
    "transfers",
    "ratio",
    "exit_pending",
    "equity",
    "residual",
    "residual_value",
    "rounding_value",
    "underfill_value",
    "transfer_state",
    "target_state",
)


@pytest.mark.parametrize("field", FIELDS)
def test_review3_snapshot_requires_every_schema_field_even_at_day_zero(field):
    s, u = markets()
    tw = TwoWallet(s, u)
    missing = rewrite_blob(tw.snapshot(), lambda data: data.pop(field))
    with pytest.raises(ValueError, match="schema|字段"):
        TwoWallet.restore(missing, s, u)


def test_review3_cancelled_delist_snapshot_preserves_exit_or_rejects_missing_field():
    s, u = two_coin_markets()
    s.events.append(dict(s.events[0], known_at=26 * 3600000, status="cancelled"))
    buy = u.market.can_buy.copy()
    buy[25, 0] = False
    funding = u.futures.funding.copy()
    funding[26, 0] = 1
    u.market = replace(
        u.market, can_buy=buy, futures=replace(u.futures, funding=funding)
    )
    kwargs = dict(trade_hour=0, hedge="follow_spot", residual_limit=100)
    tw = TwoWallet(s, u, **kwargs)
    for t in range(26):
        tw.step({"BTC": (0.5, -0.5), "ETH": (0.5, -0.5)} if t == 24 else None)
    blob = tw.snapshot()
    restored = TwoWallet.restore(blob, s, u, **kwargs)
    restored.step()
    assert restored.result().usdm.positions[-1, 0] == 0
    assert restored.result().equity[-1] == 300
    with pytest.raises(ValueError, match="schema|字段"):
        TwoWallet.restore(
            rewrite_blob(blob, lambda d: d.pop("exit_pending")), s, u, **kwargs
        )


def test_review3_snapshot_rejects_old_initial_state_fingerprint():
    s, u = markets()
    tw = TwoWallet(s, u, transfer=MarginBand(0.5, 0.25, 1))
    tw._strategy_initial["initial_state"] = tw.transfer.state()
    tw._fingerprint = tw._digest()
    with pytest.raises(ValueError, match="指纹"):
        TwoWallet.restore(tw.snapshot(), s, u, transfer=MarginBand(0.5, 0.25, 1))
