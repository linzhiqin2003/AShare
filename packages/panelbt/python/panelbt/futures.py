"""Pure NumPy helpers for hedging and scheduled index-futures rolls."""

from __future__ import annotations

import warnings

import numpy as np

from .models import Targets


class HedgeRoundingWarning(UserWarning):
    """A positive hedge exposure rounded to zero whole contracts."""


def hedge_contracts(
    stock_value, future_price, multiplier, *, beta=1.0, hedge_ratio=1.0
):
    """Signed short hedge in whole contracts, rounded toward zero.

    This matches the kernel's weight-to-contract rule, so the hedge never
    exceeds the requested exposure. A positive exposure smaller than one
    contract returns 0 and emits `HedgeRoundingWarning`.
    """
    value, price, mult, exposure, ratio = np.broadcast_arrays(
        np.asarray(stock_value, float),
        np.asarray(future_price, float),
        np.asarray(multiplier, float),
        np.asarray(beta, float),
        np.asarray(hedge_ratio, float),
    )
    if (
        not all(np.isfinite(x).all() for x in (value, price, mult, exposure, ratio))
        or np.any(value < 0)
        or np.any(price <= 0)
        or np.any(mult <= 0)
        or np.any(exposure < 0)
        or np.any(ratio < 0)
    ):
        raise ValueError(
            "hedge inputs require finite nonnegative exposure and positive price/multiplier"
        )
    unsigned = value * exposure * ratio / (price * mult)
    contracts = np.floor(unsigned)
    lost = (unsigned > 0) & (contracts == 0)
    if np.any(lost):
        warnings.warn(
            f"{int(np.count_nonzero(lost))} hedge exposure(s) below one contract "
            "rounded to 0 contracts",
            HedgeRoundingWarning,
            stacklevel=2,
        )
    result = -contracts.astype(np.int64)
    return int(result) if result.ndim == 0 else result


def roll_targets(expiry_rows, periods: int, roll_days: int, weight: float) -> Targets:
    """Move weight to the next contract on expiry minus N trading rows.

    Contracts are ordered by ascending expiry. Only the first row and roll rows
    are scheduled; other rows remain unscheduled. The schedule only moves to
    later contracts: when none expires more than `roll_days` rows ahead, the
    latest unexpired contract is held until its expiry, then the plan closes it.
    """
    expiry = np.asarray(expiry_rows)
    if (
        expiry.ndim != 1
        or expiry.dtype.kind not in "iu"
        or len(expiry) == 0
        or np.any(np.diff(expiry) <= 0)
        or np.any(expiry < 0)
        or not isinstance(periods, int)
        or periods <= 0
        or not isinstance(roll_days, int)
        or roll_days < 0
        or not np.isfinite(weight)
    ):
        raise ValueError(
            "roll schedule requires ordered nonnegative expiries, positive T and nonnegative N"
        )
    dates = []
    columns = []
    weights = []
    pointers = [0]
    previous = None
    for day in range(periods):
        eligible = np.flatnonzero(expiry > day + roll_days)
        if len(eligible):
            active = int(eligible[0])
        else:
            # No contract is far enough away: hold the latest unexpired one.
            alive = np.flatnonzero(expiry >= day)
            active = int(alive[-1]) if len(alive) else None
        if previous is not None and active is not None and expiry[previous] >= day:
            # Never roll back to an earlier contract.
            active = max(active, previous)
        if active == previous and day != 0:
            continue
        dates.append(day)
        if active is not None and weight != 0:
            columns.append(active)
            weights.append(weight)
        pointers.append(len(columns))
        previous = active
    return Targets(dates, pointers, columns, weights)
