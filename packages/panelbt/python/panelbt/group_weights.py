"""Fused grouping and weight research with explicitly aligned execution rows."""

import numpy as np

from . import _core
from ._readonly import contiguous, readonly


def run_group_weights(
    factor,
    price,
    close,
    adjustment,
    logcap,
    universes,
    *,
    signal_start,
    execution_start,
    industry=None,
    industry_zero_missing=False,
    vectors=(),
    can_buy=None,
    can_sell=None,
    n_groups=10,
    neutralize=None,
    decay=0,
    weighted=False,
    profile=False,
    rebalance=1,
    record_holdings=False,
):
    """Run all groups, long/short and optional normalized weights in one call.

    Price panels contain the full history. Factor, logcap, group labels, vectors
    and pool masks contain only rows from ``signal_start``. Factor row i executes
    at the explicitly supplied ``execution_start + i``; no shift is inferred.
    Prices are adjusted and forward-filled; PnL uses the next price row. Exposure
    uses signal-day logcap and 20/60/240-row close history. Industry labels <0
    are missing; ``industry_zero_missing`` additionally excludes zero. Logcap
    is supplied by the caller to retain its math library.

    This is pure weight accounting, with the same ledger as run_weights_batch.
    Results contain (books, active rows) pnl/turnover, (4, active rows) exposures
    in size/reversal20/reversal60/momentum order, first_live, and bad_rows.
    The last signal is validated even if its execution lies beyond the history.
    Missing exposures are NaN; statistical window masking is the caller's job.
    Inputs must not be mutated while the GIL is released.

    Rebalance on signal rows 0, N, 2N, ...; intervening execution rows hold
    actual weights unchanged with zero turnover. Optional main-pool long/short
    holdings contain sparse post-trade rows (days, offsets, columns, weights).
    """
    modes = {None: 0, "industry": 1, "cap": 2, "cap+industry": 3}
    if neutralize not in modes:
        raise ValueError(f"unknown neutralization: {neutralize!r}")
    if int(decay) != decay or decay < 0:
        raise ValueError("decay must be a nonnegative integer")
    if (
        isinstance(rebalance, (bool, np.bool_))
        or not isinstance(rebalance, (int, np.integer))
        or rebalance < 1
    ):
        raise ValueError("rebalance must be a positive integer")

    def array(value, dtype):
        return contiguous(value, dtype)

    def prices(value):
        x = np.asarray(value)
        return array(x, x.dtype if x.dtype in (np.float32, np.float64) else np.float64)

    args = (
        array(factor, np.float64),
        array(logcap, np.float64),
        np.empty((0, 0), np.int64)
        if industry is None
        else array(
            industry, np.uint32 if np.asarray(industry).dtype == np.uint32 else np.int64
        ),
        [array(v, np.float64) for v in vectors],
        [array(u, bool) for u in universes],
        prices(price),
        prices(close),
        prices(adjustment),
        np.empty((0, 0), bool) if can_buy is None else array(can_buy, bool),
        np.empty((0, 0), bool) if can_sell is None else array(can_sell, bool),
        (
            signal_start,
            execution_start,
            n_groups,
            modes[neutralize],
            int(decay),
            bool(weighted),
            bool(profile),
            bool(industry_zero_missing),
            int(rebalance),
            bool(record_holdings),
        ),
    )
    borrowed = [*args[:3], *args[3], *args[4], *args[5:10]]
    with readonly(borrowed):
        return _core.run_group_weights(*args)
