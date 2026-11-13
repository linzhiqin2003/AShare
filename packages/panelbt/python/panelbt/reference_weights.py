"""Frozen NumPy pure-weight simulator, used as a bitwise oracle."""

import numpy as np

_TOL = 1e-12


def _advance(target, prev, buy, sell):
    """Freeze blocked positions and rescale each side to its remaining budget."""
    budget_long = float(target[target > 0].sum())
    budget_short = float(-target[target < 0].sum())
    frozen = np.zeros(target.shape, dtype=bool)

    while True:
        free = ~frozen
        used_long = float(prev[frozen & (prev > 0)].sum())
        used_short = float(-prev[frozen & (prev < 0)].sum())
        free_long = float(target[free & (target > 0)].sum())
        free_short = float(-target[free & (target < 0)].sum())
        scale_long = (
            max(budget_long - used_long, 0.0) / free_long if free_long > 0 else 0.0
        )
        scale_short = (
            max(budget_short - used_short, 0.0) / free_short if free_short > 0 else 0.0
        )

        cand = np.where(frozen, prev, 0.0)
        up = free & (target > 0)
        down = free & (target < 0)
        cand[up] = target[up] * scale_long
        cand[down] = target[down] * scale_short

        violate = free & (
            ((cand > prev + _TOL) & ~buy) | ((cand < prev - _TOL) & ~sell)
        )
        if not violate.any():
            return cand
        frozen |= violate


def simulate(target, fwd_ret, can_buy=None, can_sell=None, *, keep_weights=True):
    """Frozen NumPy reference: NaNs are zero, limits apply to signed changes."""
    tgt = np.asarray(target, dtype=np.float64)
    ret = np.asarray(fwd_ret, dtype=np.float64)
    if tgt.ndim != 2 or ret.shape != tgt.shape:
        raise ValueError("target and fwd_ret must be matching two-dimensional arrays")
    for name, flags in (("can_buy", can_buy), ("can_sell", can_sell)):
        if flags is not None and np.shape(flags) != tgt.shape:
            raise ValueError(f"{name} must match target")
    n_rows, n_cols = tgt.shape
    pnl = np.zeros(n_rows, dtype=np.float64)
    turnover = np.zeros(n_rows, dtype=np.float64)
    held = np.zeros(tgt.shape) if keep_weights else None
    limited = can_buy is not None or can_sell is not None
    buy = None if can_buy is None else np.asarray(can_buy, dtype=bool)
    sell = None if can_sell is None else np.asarray(can_sell, dtype=bool)
    prev = np.zeros(n_cols, dtype=np.float64)
    # Compact columns in ascending order before every floating reduction.
    live = np.zeros(0, dtype=np.int64)
    for t in range(n_rows):
        row = tgt[t]
        wanted = np.flatnonzero((row != 0.0) & ~np.isnan(row))
        cols = np.union1d(live, wanted) if live.size else wanted
        if cols.size == 0:
            continue
        before = prev[cols]
        want = np.nan_to_num(row[cols])
        if limited:
            ok_buy = np.ones(cols.size, dtype=bool) if buy is None else buy[t, cols]
            ok_sell = np.ones(cols.size, dtype=bool) if sell is None else sell[t, cols]
            cur = _advance(want, before, ok_buy, ok_sell)
        else:
            cur = want
        if keep_weights:
            held[t, cols] = cur
        pnl[t] = float(np.dot(cur, np.nan_to_num(ret[t, cols])))
        turnover[t] = float(np.abs(cur - before).sum())
        prev[cols] = cur
        live = cols[cur != 0.0]
    return {"weights": held, "pnl": pnl, "turnover": turnover}
