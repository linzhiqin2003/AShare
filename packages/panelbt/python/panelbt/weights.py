"""Pure weight accounting; targets are already aligned to execution days."""

from dataclasses import dataclass

import numpy as np

from . import _core


@dataclass(frozen=True)
class GroupWeightTargets:
    """Compact equal-weight targets sharing a (T, N) integer label array.

    Nonmembers have zero weight; -1 denotes missing labels. Each selected group
    receives its budget divided by its member count. If both groups match, the
    short side is subtracted from the long side. No date shift is performed.
    """

    groups: np.ndarray
    long_group: int
    short_group: int = -1
    long_budget: float = 1.0
    short_budget: float = 0.0


@dataclass
class WeightResult:
    """Daily holding-period PnL, two-sided turnover, and optional held weights."""

    weights: np.ndarray | None
    pnl: np.ndarray
    turnover: np.ndarray


def run_weights_batch(
    fwd_ret, targets, can_buy=None, can_sell=None, *, keep_weights=True
):
    """Simulate dense or grouped target plans together while releasing the GIL.

    Unlike ``run``, this mode has no cash, quantities, costs, lot rounding, or
    price drift. Restrictions freeze previous weights; free weights use the
    remaining long/short budgets. NaNs in targets and returns are zero; infinities
    use NumPy ``nan_to_num`` conventions. Inputs must not be mutated during a run.
    ``keep_weights`` is a bool or one bool per plan. Empty batches are supported.
    """
    ret = np.ascontiguousarray(fwd_ret, dtype=np.float64)
    if ret.ndim != 2:
        raise ValueError("fwd_ret must be two-dimensional")
    flags = []
    for name, value in (("can_buy", can_buy), ("can_sell", can_sell)):
        flag = (
            np.empty((0, 0), bool)
            if value is None
            else np.ascontiguousarray(value, dtype=bool)
        )
        if value is not None and flag.shape != ret.shape:
            raise ValueError(f"{name} must match fwd_ret")
        flags.append(flag)
    targets = list(targets)
    keep = (
        [bool(keep_weights)] * len(targets)
        if isinstance(keep_weights, (bool, np.bool_))
        else list(keep_weights)
    )
    if len(keep) != len(targets):
        raise ValueError("keep_weights must have one entry per target")
    empty_float, empty_int = np.empty((0, 0)), np.empty((0, 0), np.int64)
    plans = []
    for target, save in zip(targets, keep, strict=True):
        if isinstance(target, GroupWeightTargets):
            groups = np.asarray(target.groups)
            if groups.dtype.kind not in "iu" or groups.shape != ret.shape:
                raise ValueError(
                    "group labels must be an integer array matching fwd_ret"
                )
            if target.long_group < 0 or target.short_group < -1:
                raise ValueError("invalid group selection")
            plans.append(
                (
                    empty_float,
                    np.ascontiguousarray(groups, dtype=np.int64),
                    target.long_group,
                    target.short_group,
                    target.long_budget,
                    target.short_budget,
                    bool(save),
                )
            )
        else:
            dense = np.ascontiguousarray(target, dtype=np.float64)
            if dense.shape != ret.shape:
                raise ValueError("targets must match fwd_ret")
            plans.append((dense, empty_int, -1, -1, 0.0, 0.0, bool(save)))
    return [WeightResult(**r) for r in _core.run_weights_batch(ret, *flags, plans)]


def run_weights(target, fwd_ret, can_buy=None, can_sell=None, *, keep_weights=True):
    """Run one pure weight plan; see :func:`run_weights_batch` for accounting."""
    return run_weights_batch(
        fwd_ret, [target], can_buy, can_sell, keep_weights=keep_weights
    )[0]
