"""Kernel against the structurally independent mixed spot/futures reference."""

from collections import Counter
from dataclasses import replace

import numpy as np
import panelbt as bt
from panelbt.reference import run as reference
from panelbt.reference_futures import MixedEngine

from .comparison import assert_results_equal
from .generators_futures import rich_case

SEEDS = range(2000)
# Ulp-level roundoff: kernel and reference reach the same cash by a different
# floating-point path (the legacy cash account's fractional-share
# affordability bisection in some seeds, operand association in the
# futures/margin path in others, e.g. seed 26941, a margin account). One side
# keeps about 1e-13 of residual cash where the other has 0.0, and the next
# withdrawal flips `coverage["cash_flow"]` by one. Every other field must match.
#
# Which seeds hit it depends on the platform: the generator's np.exp / log /
# dirichlet pick SIMD and libm paths by CPU, so inputs differ in the last bit.
# On macOS arm64 it was seeds 2 and 1280; on GitHub's Linux runners seed 151
# failed in two of four runs (2026-11-14; 1-ulp jitter of exp alone flips it
# in 172 of 300 trials). So any seed may differ by one in that counter, but at
# most ROUNDOFF_LIMIT seeds: a systematic counting bug hits far more.
ROUNDOFF_LIMIT = 8
EVENTS = (
    "futures_open",
    "futures_close",
    "margin_call",
    "futures_margin_call",
    "liquidation",
    "futures_liquidation",
    "futures_expiry",
    "expired_target",
    "bankruptcy",
    "futures_rounded_to_zero",
    "futures_margin_scaled",
    "margin_scaled",
    "reversal",
    "invalid_quote",
)


def test_rich_corpus_matches_reference_and_exercises_every_event():
    totals, failures, roundoff = Counter(), [], []
    for seed in SEEDS:
        market, targets, costs, config = rich_case(seed)
        engine = MixedEngine(market, targets, costs, config)
        expected = engine.run()
        actual = bt.run(market, targets, costs, config)
        if abs(actual.coverage["cash_flow"] - expected.coverage["cash_flow"]) == 1:
            # Only the cash_flow counter may differ, and only by one.
            roundoff.append(seed)
            actual.coverage = {
                **actual.coverage,
                "cash_flow": expected.coverage["cash_flow"],
            }
        try:
            assert_results_equal(actual, expected)
        except AssertionError as exc:
            # 记下差在哪：2026-11-14 第一次在 Linux CI 上只报了「seed 151 不同」，看不出原因
            failures.append((seed, " ".join(str(exc).split())[:600]))
        totals.update({k: int(expected.coverage[k]) for k in EVENTS})
        totals.update(engine.stats)
        totals["cash_account"] += int(bool(engine.classic))
        totals["cash_account_supervised"] += int(
            bool(engine.classic) and engine.book.held_ever
        )
        totals["financing"] += int(config.margin.allow_financing)
        totals["shares"] += int(targets.kind == "shares")
    assert not failures, "kernel differs from reference:\n" + "\n".join(
        f"seed {seed}: {why}" for seed, why in failures
    )
    assert len(roundoff) <= ROUNDOFF_LIMIT, ("cash_flow off by one", roundoff)
    missing = [
        k
        for k in (
            *EVENTS,
            "reduction_debt",
            "expired_target_forced",
            "cash_account",
            "cash_account_supervised",
            "financing",
            "shares",
        )
        if totals[k] == 0
    ]
    assert not missing, (missing, totals)
    assert totals["cash_account"] >= 0.3 * len(SEEDS)


def _without_future_targets(seed):
    """The same case with every futures target removed."""
    market, targets, costs, config = rich_case(seed)
    dense = targets.to_dense(*market.shape)
    future = np.asarray(market.futures.is_future)
    scheduled = ~np.all(np.isnan(dense), axis=1)
    dense[:, future] = np.where(scheduled[:, None], 0.0, np.nan)
    return market, bt.Targets.from_dense(dense, kind=targets.kind), costs, config


def _spot_only(market, targets, costs):
    keep = np.flatnonzero(~np.asarray(market.futures.is_future))

    def cut(a):
        return None if a is None else np.asarray(a)[:, keep]

    spot = bt.Market(
        cut(market.exec_price),
        cut(market.close),
        cut(market.can_buy),
        cut(market.can_sell),
        volume=cut(market.volume),
        split=cut(market.split),
        cash_div=cut(market.cash_div),
        delist=cut(market.delist),
        can_short=cut(market.can_short),
    )
    plan = bt.Targets.from_dense(
        targets.to_dense(*market.shape)[:, keep], kind=targets.kind
    )
    slip = costs.slippage if np.ndim(costs.slippage) == 0 else cut(costs.slippage)
    return spot, plan, replace(costs, slippage=slip, impact_k=cut(costs.impact_k)), keep


def test_untraded_futures_columns_leave_spot_bitwise_unchanged():
    for seed in range(200):
        market, targets, costs, config = _without_future_targets(seed)
        spot, plan, spot_costs, keep = _spot_only(market, targets, costs)
        for runner in (reference, bt.run):
            mixed, alone = (
                runner(market, targets, costs, config),
                runner(spot, plan, spot_costs, config),
            )
            for name in ("cash", "equity", "long_value", "short_value", "margin_call"):
                np.testing.assert_array_equal(
                    getattr(mixed, name), getattr(alone, name), err_msg=f"{seed} {name}"
                )
            np.testing.assert_array_equal(mixed.positions[:, keep], alone.positions)
            assert mixed.diagnostics == alone.diagnostics, seed
