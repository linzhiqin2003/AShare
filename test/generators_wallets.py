"""Synthetic C4 corpus: independent input generation and branch accounting."""

from dataclasses import replace

import numpy as np
from panelbt.reference_wallets import PairOracle

from ashare.backtest.crypto import MarginBand, TwoWallet
from test.test_crypto_wallets import markets

BRANCHES = (
    "spot_to_usdm",
    "usdm_to_spot",
    "truncated",
    "rejected",
    "rescue_success",
    "rescue_failure",
    "bankrupt_deposit_rejected",
    "reversal",
    "delivery",
    "delisting",
    "residual",
)


def case(seed):
    rng = np.random.default_rng(seed)
    n = 12
    prices = rng.integers(75, 126, n).astype(float)
    op = prices + rng.integers(-5, 6, n)
    high = op + rng.integers(0, 11, n)
    low = op - rng.integers(0, 11, n)
    spot, usdm = markets(
        prices,
        future_prices=op,
        multiplier=0.25,
        spot_cash=float(rng.integers(30, 401)),
        future_cash=float(rng.integers(15, 201)),
        high=high[:, None],
        low=low[:, None],
        settlement=rng.integers(low, high + 1)[:, None],
        funding=rng.choice([-0.5, 0, 0.5, 2], (n, 1)),
        fee_rate=0.001,
        clearance=0.01,
        expiry=[int(rng.integers(5, 13))],
        initial=rng.choice([0.1, 0.2, 0.3], (n, 1)),
        qty_step=rng.integers(1, 3, (n, 1)),
        min_qty=np.ones((n, 1)),
        volume=rng.integers(1, 31, (n, 1)),
        buy=rng.random((n, 1)) > 0.1,
        sell=rng.random((n, 1)) > 0.1,
    )
    spot.config = replace(spot.config, qty_step=0.25, min_qty=0.25, carry_blocked=False)
    usdm.config = replace(usdm.config, carry_blocked=False, max_participation=0.5)
    spot.costs = replace(spot.costs, fee_rate_buy=0.001, fee_rate_sell=0.001)
    plan = np.zeros((n, 1, 2))
    plan[1:, 0, 0] = rng.integers(0, 21, n - 1) * 0.25
    plan[1:, 0, 1] = rng.integers(-20, 21, n - 1) * 0.25
    hedge = "follow_spot" if seed % 3 == 0 else "independent"
    if hedge == "follow_spot":
        plan[:, 0, 1] = -plan[:, 0, 0]
    requests = rng.choice([-100, -40, 0, 30, 100], n)
    policy = (
        MarginBand(0.5, 0.25, 1, pre_fund=False)
        if seed % 2 == 0
        else lambda t, *_: requests[t]
    )
    usdm.profiles[0]["contract_type"] = "CURRENT_QUARTER" if seed % 2 else "PERPETUAL"
    return spot, usdm, plan, policy, hedge


def compare(seed):
    spot, usdm, plan, policy, hedge = case(seed)
    tw = TwoWallet(
        spot,
        usdm,
        transfer=policy,
        hedge=hedge,
        allow_residual=True,
        allow_tier_exceeded=True,
    )
    oracle = PairOracle(spot, usdm, transfer=policy, hedge=hedge)
    branches = dict.fromkeys(BRANCHES, 0)
    for t, target in enumerate(plan):
        before = oracle.usdm.state()
        ss = oracle.spot.state()
        request = float(policy(t, ss, before))
        r0 = oracle.step(target)
        sr, ur = tw.step(target)
        for actual, expected in ((sr, r0["spot"]), (ur, r0["usdm"])):
            for field, key in (
                ("cash_flow", "flow"),
                ("equity", "equity"),
                ("cash", "cash"),
            ):
                np.testing.assert_allclose(
                    getattr(actual, field),
                    expected[key],
                    rtol=1e-12,
                    atol=1e-10,
                    err_msg=f"seed={seed}, row={t}, {field}",
                )
        state = tw.state
        np.testing.assert_allclose(
            state[0].shares,
            r0["spot"]["positions"],
            rtol=0,
            atol=1e-12,
            err_msg=f"spot {seed}/{t}",
        )
        np.testing.assert_allclose(
            state[1].shares,
            r0["usdm"]["positions"],
            rtol=0,
            atol=1e-12,
            err_msg=f"usdm {seed}/{t}",
        )
        for field, key in (
            ("wallet_balance", "wallet_balance"),
            ("unrealized", "unrealized"),
            ("entry_price", "entry"),
        ):
            np.testing.assert_allclose(
                getattr(state[1], field),
                r0["usdm"][key],
                rtol=1e-12,
                atol=1e-10,
                err_msg=f"seed={seed}, row={t}, {field}",
            )
        np.testing.assert_allclose(
            ur.wallet_balance + ur.unrealized, ur.equity, rtol=1e-12, atol=1e-10
        )
        assert sr.cash_flow + ur.cash_flow == 0
        a = r0["actual"]
        branches["spot_to_usdm"] += a > 0
        branches["usdm_to_spot"] += a < 0
        branches["truncated"] += abs(a) < abs(request)
        branches["rejected"] += bool(request and not a)
        branches["bankrupt_deposit_rejected"] += bool(
            request > 0 and before.bankrupt and not a
        )
        branches["reversal"] += bool(np.any(before.shares * state[1].shares < 0))
        expiry = bool(r0["usdm"]["expiries"])
        branches["delivery"] += expiry and seed % 2 == 1
        branches["delisting"] += expiry and seed % 2 == 0
        branches["residual"] += bool(np.any(np.abs(r0["residual"]) > 1e-12))
        # A genuine opening rescue: the previous wallet would fail at this open.
        q = before.shares * usdm.futures.multiplier
        op = usdm.futures.liquidation.mark_open[t]
        e = before.wallet_balance + sum(q * (op - before.entry_price))
        mm = sum(np.abs(q) * op * 0.01)
        needed = bool(np.any(q) and e <= mm + sum(np.abs(q) * op) * 1e-12)
        branches["rescue_success"] += bool(
            request > 0
            and a > 0
            and needed
            and "open" not in r0["usdm"]["liquidations"]
        )
        branches["rescue_failure"] += bool(
            request > a and request > 0 and r0["usdm"]["liquidations"]
        )
    result = tw.result()
    np.testing.assert_allclose(
        result.equity, [r["equity"] for r in oracle.records], rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        result.residual_coins,
        [r["residual"] for r in oracle.records],
        rtol=0,
        atol=1e-12,
    )
    return {k: int(v) for k, v in branches.items()}


def corpus(seeds=500):
    counts = dict.fromkeys(BRANCHES, 0)
    for seed in range(seeds):
        for key, value in compare(seed).items():
            counts[key] += value
    assert all(counts.values()), counts
    return counts


def decimal_daily_corpus(seeds=120):
    """Multi-coin daily cases; independent oracle decides exits and prefunding."""
    import panelbt as bt

    from test.test_crypto import scenario
    from test.test_crypto_wallets import ZERO

    counts = dict.fromkeys(
        (
            "multiple_coins",
            "step_0.1",
            "step_0.001",
            "daily_trade_hour",
            "spot_delist",
            "unrelated_blocked_not_retried",
            "prefunded_target",
        ),
        0,
    )
    for seed in range(seeds):
        rng = np.random.default_rng(88000 + seed)
        trade_hour = seed % 6
        days, cols = 77 + trade_hour, 2 + seed % 2
        first = 24 + trade_hour
        step = 0.1 if seed % 2 else 0.001
        p = np.broadcast_to(
            rng.integers(90, 111, cols).astype(float), (days, cols)
        ).copy()
        sell = np.ones_like(p, bool)
        sell[first, 1] = False
        funding = rng.choice([0, 0.01, -0.01], p.shape)
        fm, _, _, fc = scenario(
            p,
            np.zeros_like(p),
            sell=sell,
            funding=funding,
            initial=0.5,
            cash=20,
            costs=ZERO,
            fee_rate=0.001,
        )
        s, u = markets([100] * days, spot_cash=1000)
        s.market = bt.Market(
            p, p, np.ones_like(p, bool), np.ones_like(p, bool), crypto_spot=True
        )
        s.config = replace(s.config, min_qty=step, qty_step=step)
        u.market = replace(
            fm, futures=replace(fm.futures, multiplier=np.full(cols, 0.1))
        )
        u.config = fc
        for m in (s, u):
            coins = ("BTC", "ETH", "SOL")[:cols]
            m.symbols = tuple(m.kind + ":" + c + "USDT" for c in coins)
            m.profiles = [
                dict(m.profiles[0], coin=c, instrument=m.symbols[j])
                for j, c in enumerate(coins)
            ]
            m.signal_prices = np.vstack([np.full((1, cols), np.nan), p[:-1]])
            m.arrays = dict(
                coin_units=np.ones_like(p),
                lifecycle=np.full(p.shape, "active"),
                execution_symbol=np.tile(m.symbols, (days, 1)),
            )
        s.events = [
            dict(
                event_id="exit",
                instrument=s.symbols[0],
                kind="delist",
                status="active",
                known_at=int(s.times[first + 1]),
                effective_at=int(s.times[first + 2]),
            )
        ]
        policy = MarginBand(0.5, 0.25, 1)
        tw = TwoWallet(
            s,
            u,
            transfer=policy,
            hedge="follow_spot",
            trade_hour=trade_hour,
            allow_residual=True,
        )
        oracle = PairOracle(
            s, u, transfer=policy, hedge="follow_spot", trade_hour=trade_hour
        )
        for t in range(days):
            # Include deliberately irrelevant off-schedule targets.
            quantities = rng.choice([0.3, 0.8, 1.0, 1.1], cols)
            if t == first:
                quantities[:] = 0.3
            target = (
                np.stack([quantities, -quantities], axis=-1) if t >= first else None
            )
            expected = oracle.step(target)
            sr, ur = tw.step(target)
            np.testing.assert_allclose(
                [sr.equity, ur.equity, sr.cash_flow, ur.cash_flow],
                [
                    expected["spot"]["equity"],
                    expected["usdm"]["equity"],
                    -expected["actual"],
                    expected["actual"],
                ],
                rtol=1e-12,
                atol=1e-10,
                err_msg=f"decimal seed={seed}, row={t}",
            )
            for state, side in zip(tw.state, ("spot", "usdm"), strict=True):
                np.testing.assert_allclose(
                    state.shares, expected[side]["positions"], atol=1e-12, rtol=0
                )
            if t == first + 1:
                assert tw.state[1].shares[1] == 0
                assert tw.state[0].shares[0] == tw.state[1].shares[0] == 0
                counts["spot_delist"] += 1
                counts["unrelated_blocked_not_retried"] += 1
        result = tw.result()
        np.testing.assert_allclose(
            result.equity, [x["equity"] for x in oracle.records], atol=1e-10, rtol=1e-12
        )
        np.testing.assert_allclose(
            result.residual_coins,
            [x["residual"] for x in oracle.records],
            atol=1e-12,
            rtol=0,
        )
        counts["prefunded_target"] += sum(e["actual"] > 0 for e in result.transfers)
        counts["multiple_coins"] += cols > 1
        counts["step_0.1" if step == 0.1 else "step_0.001"] += 1
        counts["daily_trade_hour"] += 1
    assert all(counts.values()), counts
    return {k: int(v) for k, v in counts.items()}
