"""Fused grouping and weight research with explicitly aligned execution rows."""

import numpy as np

from . import _core
from ._readonly import contiguous, readonly


class GroupExposureCache:
    """Share means only for identical risk buffers, pool masks and history rows.

    Scope this cache to one task. All bound arrays must remain unchanged for its
    entire lifetime. References are retained so buffer addresses cannot be reused.
    Different signals and rebalance schedules may use the same cache concurrently.
    """

    def __init__(self, logcap, universes, close, adjustment, signal_start):
        self._arrays = [logcap, *universes, close, adjustment]
        self._key = self._identity(self._arrays, signal_start)
        self._native = _core.ExposureCache(len(logcap), len(universes))

    @staticmethod
    def _identity(arrays, signal_start):
        return (
            signal_start,
            tuple(
                (a.__array_interface__["data"][0], a.shape, a.strides, a.dtype.str)
                for a in arrays
            ),
        )

    def _check(self, logcap, universes, close, adjustment, signal_start):
        if (
            self._identity([logcap, *universes, close, adjustment], signal_start)
            != self._key
        ):
            raise ValueError(
                "exposure cache requires identical risk buffers, pools and signal window"
            )
        return self._native


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
    exposure_cache=None,
    _variants=None,
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
    if exposure_cache is not None or _variants is not None:
        cache = (
            None
            if exposure_cache is None
            else exposure_cache._check(args[1], args[4], args[6], args[7], signal_start)
        )
        args += (cache, _variants)
    with readonly(borrowed):
        return _core.run_group_weights(*args)


def run_group_weights_batch(*args, variants, **kwargs):
    """Share daily targets/prices across independent weight ledgers in one call.

    ``variants`` contains dictionaries with rebalance, weighted and record_holdings
    (defaults: 1, False, False). Other arguments match run_group_weights; ledger
    options must be supplied only in variants. Results follow variant order.
    Working target buffers remain bounded to 64 days, regardless of history size.
    """
    names = {"rebalance", "weighted", "record_holdings"}
    if names.intersection(kwargs):
        raise ValueError("supply ledger options in variants only")
    settings = []
    for variant in variants:
        if set(variant) - names:
            raise ValueError("unknown grouped weight variant option")
        n = variant.get("rebalance", 1)
        if (
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 1
        ):
            raise ValueError("rebalance must be a positive integer")
        settings.append(
            (
                int(n),
                bool(variant.get("weighted", False)),
                bool(variant.get("record_holdings", False)),
            )
        )
    return run_group_weights(*args, **kwargs, _variants=settings)
