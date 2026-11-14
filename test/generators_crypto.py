"""C1 branch-directed and random synthetic hourly contract panels."""

from dataclasses import replace

import numpy as np
import panelbt as bt

from .test_crypto import ZERO, scenario

BRANCHES = (
    "funding_paid",
    "funding_received",
    "liquidation_open",
    "liquidation_post_trade",
    "liquidation_intraday",
    "liquidation_close",
    "liquidation_single",
    "liquidation_multi",
    "liquidation_mixed",
    "liquidation_lambda_one",
    "liquidation_nondecreasing",
    "clearance_capped",
    "bankruptcy_deficit",
    "futures_reopen",
    "futures_expiry",
    "futures_rule_scaled",
    "futures_min_notional",
    "futures_reduce_exempt",
    "futures_full_close",
    "reversal",
    "tier_cap_exceeded",
)

# A phase count alone cannot establish that interactions were exercised.
# These keys require evidence on one and the same liquidation row.
COMBINATIONS = tuple(
    f"{phase}.{feature}"
    for phase in ("post_trade", "close")
    for feature in (
        "multi_mixed_fees",
        "expiry_suppressed" if phase == "post_trade" else "expiry_released",
        "funding_suppressed" if phase == "post_trade" else "funding_debited",
        "pending_and_reopen",
        "all_interactions",
    )
)


def combination_counts(inputs, result):
    """Evidence from fills/events on a shared row, never generator labels."""
    from collections import Counter

    counts = Counter()
    # Only classify a phase when the result uniquely identifies the episode.
    phases = ("open", "post_trade", "intraday", "close")
    if sum(result.coverage["liquidation_" + p] for p in phases) != 1:
        return counts
    phase = next(p for p in phases if result.coverage["liquidation_" + p])
    if phase not in ("post_trade", "close"):
        return counts
    events = result.liquidation_events
    if not events:
        return counts
    day = events[0].day
    assert all(e.day == day for e in events)
    mixed = any(e.qty > 0 for e in events) and any(e.qty < 0 for e in events)
    fees = result.commission[day] > 0 and result.clearance_fee[day] > 0
    if len({e.col for e in events}) < 2 or not mixed or not fees:
        return counts
    counts[f"{phase}.multi_mixed_fees"] += 1
    m, _, _, _ = inputs
    if phase == "post_trade":
        if any(m.futures.expiry[e.col] == day for e in events) and not any(
            e.day == day for e in result.expiry_events
        ):
            counts[f"{phase}.expiry_suppressed"] += 1
        if (
            m.futures.funding is not None
            and any(-e.qty * m.futures.funding[day, e.col] > 0 for e in events)
            and result.funding[day] == 0
        ):
            counts[f"{phase}.funding_suppressed"] += 1
    else:
        if any(e.day == day and e.qty != 0 for e in result.expiry_events):
            counts[f"{phase}.expiry_released"] += 1
        if result.funding[day] > 0:
            counts[f"{phase}.funding_debited"] += 1
    if (
        result.n_blocked_buy[day] + result.n_blocked_sell[day] > 0
        and day + 2 < m.shape[0]
        and not result.positions[day + 1].any()
        and result.positions[day + 2].any()
        and result.coverage["futures_reopen"] > 0
    ):
        counts[f"{phase}.pending_and_reopen"] += 1
    if sum(counts.values()) == 4:
        counts[f"{phase}.all_interactions"] += 1
    return counts


def directed_cases():
    yield scenario([[100], [100], [100]], [[1], [-1], [0]], funding=[[2], [3], [0]])
    yield scenario([[100], [80]], [[1], [0]], cash=10, clearance=0.01)
    yield scenario(
        [[100], [98]], [[2], [1]], cash=3, marks=[[100], [100]], initial=0.01
    )
    yield scenario([[100]], [[1]], cash=10, low=[[80]], clearance=0.1)
    yield scenario([[100, 100]], [[1, 1]], cash=20, low=[[80, 80]])
    yield scenario(
        [[100, 100]], [[1, -1]], cash=20, high=[[100, 120]], low=[[80, 100]], maint=0.02
    )
    yield scenario([[100]], [[1]], cash=10, low=[[90 / 0.99]])
    yield scenario([[100]], [[1]])
    yield scenario([[100]], [[1]], cash=1.1, initial=0.01, funding=[[0.2]])
    yield scenario([[100], [50]], [[1], [1]], cash=10, low=[[80], [50]], initial=0.001)
    # Cash delivery is also the contract-delisting settlement primitive.
    yield scenario(
        [[100], [101]],
        [[1], [1]],
        settlement=[[99], [99]],
        expiry=[0],
        delivery_fee_rate=0.001,
    )
    yield scenario(
        [[100]], [[100]], cash=57, qty_step=[[2]], min_qty=[[3]], min_notional=[[0]]
    )
    yield scenario(
        [[100]],
        [[10]],
        cash=1000,
        qty_step=[[3]],
        min_qty=[[3]],
        min_notional=[[400]],
        volume=[[5]],
        config=bt.Config(initial_cash=1000, max_participation=1, record_trades=True),
    )
    yield scenario([[100], [100]], [[5], [3]], cash=1000, min_notional=[[0], [1000]])
    yield scenario(
        [[100], [100]], [[5], [-1]], cash=1000, qty_step=[[1], [3]], min_qty=[[1], [3]]
    )
    yield scenario([[100]], [[1]], high=[[200]], low=[[99]], cap=150)
    for phase in ("post_trade", "close"):
        for side in (1, -1):
            yield combination_case(phase, side)


def combination_case(phase, side):
    """R1: four-column account, expiry/funding boundary, blocked target, reopen.

    Flat marks isolate the phase; reverse all directions to exercise both signs.
    Col 2 expires at row 1; col 3 is blocked on rows 0/1, available thereafter.
    """
    prices = np.full((5, 4), 100.0)
    targets = (
        np.array(
            [
                [2, -1, 1, 1],
                [1 if phase == "post_trade" else 2, -1, 1, 1],
                [np.nan] * 4,
                [1, 0, 0, 0],
                [np.nan] * 4,
            ]
        )
        * side
    )
    if phase == "post_trade":
        prices[1, 0] -= 3 * side
    masks = np.ones((5, 4), bool)
    masks[:2, 3] = False
    funding = np.zeros((5, 4))
    funding[1] = np.array([1.5, -0.5, 99, 0]) * side
    return scenario(
        prices,
        targets,
        cash=6,
        marks=np.full((5, 4), 100.0),
        funding=funding,
        initial=0.001,
        fee_rate=0.001,
        delivery_fee_rate=0.001,
        clearance=0.001,
        expiry=[5, 5, 1, 5],
        buy=masks,
        sell=masks,
    )


def case(seed):
    rng = np.random.default_rng(seed)
    t, n = int(rng.integers(4, 18)), int(rng.integers(1, 6))
    o = rng.uniform(65, 135, (t, n))
    h, lows = o * rng.uniform(1, 1.3, (t, n)), o * rng.uniform(0.7, 1, (t, n))
    settle = rng.uniform(lows, h)
    p = o * rng.uniform(0.995, 1.005, (t, n))
    targets = rng.integers(-15, 16, (t, n)).astype(float)
    targets[rng.random(t) < 0.25] = np.nan
    args = scenario(
        p,
        targets,
        cash=float(rng.uniform(80, 800)),
        marks=o,
        high=h,
        low=lows,
        settlement=settle,
        funding=rng.uniform(-2, 2, (t, n)),
        maint=0.02,
        initial=float(rng.choice([0.05, 0.1, 0.2])),
        clearance=float(rng.choice([0, 0.01, 0.1])),
        cap=float(rng.choice([500, 1e30])),
        expiry=rng.integers(t // 2, t + 1, n),
        qty_step=rng.integers(1, 5, (t, n)),
        min_qty=rng.integers(1, 7, (t, n)),
        min_notional=rng.choice([0, 50, 300, 1000], (t, n)),
        fee_rate=0.0005,
        volume=rng.uniform(5, 40, (t, n)),
        buy=rng.random((t, n)) > 0.12,
        sell=rng.random((t, n)) > 0.12,
        costs=replace(ZERO, slippage=0.0002, impact_k=np.full((t, n), 0.0001)),
    )
    m, plan, costs, cfg = args
    cfg = replace(cfg, max_participation=0.5, carry_blocked=seed % 2 == 0)
    if seed % 3 == 0:
        plan = bt.Targets.from_dense(targets / 4, kind="weights")
    return m, plan, costs, cfg


def mixed_case(seed):
    """Optional funding, maintenance and order units beside real spot holdings."""
    from .generators_futures import rich_case

    m, plan, costs, cfg = rich_case(seed)
    rng = np.random.default_rng(seed)
    m = replace(
        m,
        futures=replace(
            m.futures,
            funding=rng.uniform(-2, 2, m.shape),
            maint_rate=0.01,
            qty_step=rng.integers(1, 4, m.shape),
            min_qty=rng.integers(1, 5, m.shape),
        ),
    )
    return m, plan, costs, cfg
