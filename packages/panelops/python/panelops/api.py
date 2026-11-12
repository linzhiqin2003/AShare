"""Strict public array interface and benchmark-selected dispatch."""

from dataclasses import dataclass
from functools import partial

import numpy as np

from . import _core, _numpy

_THREADS = 0
# Updated only from measured comparisons; native remains directly benchmarkable.
BACKENDS = {}
_EMPTY = np.empty((0, 0), dtype=np.float64)
_MASK = np.empty((0, 0), dtype=bool)
_GROUPS = np.empty((0, 0), dtype=np.int64)
TS_OPS = tuple(
    (
        "delay delta sum mean std min max rank quantile argmin argmax product "
        "corr cov decay_linear ema skew kurt slope residual regression_slope regression_residual "
        "var zscore rsquare median mad count sumif ffill idxmin idxmax"
    ).split()
)
CS_OPS = (
    "rank",
    "zscore",
    "winsor_quantile",
    "winsor_mad",
    "group_demean",
    "neutralize",
    "scale",
    "demean",
)


def set_num_threads(threads=0):
    """Select CPU count (0) or an explicit positive Rayon thread count."""
    global _THREADS
    _THREADS = integer(threads, "threads", 0)


def get_num_threads():
    """Return the configured count; zero means available CPU count."""
    return _THREADS


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def scalar(value, name, lower, upper):
    if not np.isscalar(value) or not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return float(value)


def panel(value, name="x", dtype=np.float64, shape=None):
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise TypeError(f"{name} must be a NumPy {np.dtype(dtype)} array")
    if value.ndim != 2 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous two-dimensional array")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}")
    return value


def optional(value, x, name, dtype=np.float64):
    if value is None:
        return _MASK if dtype is bool else _GROUPS if dtype is np.int64 else _EMPTY
    return panel(value, name, dtype, x.shape)


def _masked(x, mask):
    return np.where(np.isfinite(x) & (True if mask is None else mask), x, np.nan)


def _numpy_input(x, mask):
    """NumPy kernels already exclude nonfinites; avoid redundant copies."""
    return x if mask is None else np.where(mask, x, np.nan)


def _ts(
    op,
    x,
    window=20,
    *,
    min_periods=None,
    y=None,
    mask=None,
    ddof=1,
    q=0.5,
    normalize=False,
    alpha=None,
    periods=1,
    backend=None,
    layout=None,
):
    panel(x)
    yy, mm = optional(y, x, "y"), optional(mask, x, "mask", bool)
    window = integer(window, "window")
    mp = window if min_periods is None else integer(min_periods, "min_periods")
    if mp > window:
        raise ValueError("min_periods must not exceed window")
    ddof, periods = integer(ddof, "ddof", 0), integer(periods, "periods", 0)
    q = scalar(q, "q", 0, 1)
    alpha = (
        2 / (window + 1)
        if alpha is None
        else scalar(alpha, "alpha", np.nextafter(0.0, 1.0), 1)
    )
    if (
        op in ("corr", "cov", "regression_slope", "regression_residual", "sumif")
        and y is None
    ):
        raise ValueError(f"{op} requires y")
    selected = backend or BACKENDS.get("ts_" + op, "rust")
    if selected == "numpy":
        if op not in ("delay", "delta"):
            raise ValueError("NumPy dispatch is not configured for this operator")
        a = _masked(x, mask)
        result = np.full_like(a, np.nan)
        if periods < len(x):
            result[periods:] = (
                a[: len(x) - periods]
                if op == "delay"
                else a[periods:] - a[: len(x) - periods]
            )
        if mask is not None:
            result[~mask] = np.nan
        return result
    if selected != "rust":
        raise ValueError("backend must be rust or numpy")
    return _core.ts(
        x,
        yy,
        mm,
        op,
        window,
        mp,
        ddof,
        q,
        normalize,
        alpha,
        periods,
        _THREADS,
        0 if layout is None else integer(layout, "layout", 0),
    )


def ts_delay(x, periods=1, *, mask=None, backend=None):
    """Lag by nonnegative trading rows, preserving the output-date mask."""
    return _ts("delay", x, periods=periods, mask=mask, backend=backend)


def ts_delta(x, periods=1, *, mask=None, backend=None):
    """Subtract the lagged observation from today's observation."""
    return _ts("delta", x, periods=periods, mask=mask, backend=backend)


def ts_sum(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing sum; see SPEC.md for validity and degeneracy rules."""
    return _ts("sum", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_mean(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing mean; see SPEC.md for validity and degeneracy rules."""
    return _ts("mean", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_std(x, window=20, *, min_periods=None, mask=None, ddof=1, backend=None):
    """Trailing std; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "std", x, window, min_periods=min_periods, mask=mask, backend=backend, ddof=ddof
    )


def ts_min(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing min; see SPEC.md for validity and degeneracy rules."""
    return _ts("min", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_max(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing max; see SPEC.md for validity and degeneracy rules."""
    return _ts("max", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_rank(
    x, window=20, *, min_periods=None, mask=None, normalize=False, backend=None
):
    """Trailing rank; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "rank",
        x,
        window,
        min_periods=min_periods,
        mask=mask,
        backend=backend,
        normalize=normalize,
    )


def ts_quantile(x, window=20, *, min_periods=None, mask=None, q=0.5, backend=None):
    """Trailing quantile; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "quantile", x, window, min_periods=min_periods, mask=mask, backend=backend, q=q
    )


def ts_argmin(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing argmin; see SPEC.md for validity and degeneracy rules."""
    return _ts("argmin", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_argmax(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing argmax; see SPEC.md for validity and degeneracy rules."""
    return _ts("argmax", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_product(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing product; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "product", x, window, min_periods=min_periods, mask=mask, backend=backend
    )


def ts_corr(x, y, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing corr; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "corr", x, window, min_periods=min_periods, mask=mask, backend=backend, y=y
    )


def ts_cov(x, y, window=20, *, min_periods=None, mask=None, ddof=1, backend=None):
    """Trailing cov; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "cov",
        x,
        window,
        min_periods=min_periods,
        mask=mask,
        backend=backend,
        y=y,
        ddof=ddof,
    )


def ts_decay_linear(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing decay_linear; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "decay_linear", x, window, min_periods=min_periods, mask=mask, backend=backend
    )


def ts_ema(x, window=20, *, min_periods=None, mask=None, alpha=None, backend=None):
    """Trailing ema; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "ema",
        x,
        window,
        min_periods=min_periods,
        mask=mask,
        backend=backend,
        alpha=alpha,
    )


def ts_skew(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing skew; see SPEC.md for validity and degeneracy rules."""
    return _ts("skew", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_kurt(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing kurt; see SPEC.md for validity and degeneracy rules."""
    return _ts("kurt", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_slope(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing slope; see SPEC.md for validity and degeneracy rules."""
    return _ts("slope", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_residual(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing residual; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "residual", x, window, min_periods=min_periods, mask=mask, backend=backend
    )


def ts_regression_slope(x, y, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing regression_slope; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "regression_slope",
        x,
        window,
        min_periods=min_periods,
        mask=mask,
        backend=backend,
        y=y,
    )


def ts_regression_residual(
    x, y, window=20, *, min_periods=None, mask=None, backend=None
):
    """Trailing regression_residual; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "regression_residual",
        x,
        window,
        min_periods=min_periods,
        mask=mask,
        backend=backend,
        y=y,
    )


def ts_var(x, window=20, *, min_periods=None, mask=None, ddof=1, backend=None):
    """Trailing var; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "var", x, window, min_periods=min_periods, mask=mask, backend=backend, ddof=ddof
    )


def ts_zscore(x, window=20, *, min_periods=None, mask=None, ddof=1, backend=None):
    """Trailing zscore; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "zscore",
        x,
        window,
        min_periods=min_periods,
        mask=mask,
        backend=backend,
        ddof=ddof,
    )


def ts_rsquare(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing rsquare; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "rsquare", x, window, min_periods=min_periods, mask=mask, backend=backend
    )


def ts_median(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing median; see SPEC.md for validity and degeneracy rules."""
    return _ts("median", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_mad(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing mad; see SPEC.md for validity and degeneracy rules."""
    return _ts("mad", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_count(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing count; see SPEC.md for validity and degeneracy rules."""
    return _ts("count", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_sumif(x, condition, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing sumif; see SPEC.md for validity and degeneracy rules."""
    return _ts(
        "sumif",
        x,
        window,
        min_periods=min_periods,
        mask=mask,
        backend=backend,
        y=condition,
    )


def ts_idxmin(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing idxmin; see SPEC.md for validity and degeneracy rules."""
    return _ts("idxmin", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_idxmax(x, window=20, *, min_periods=None, mask=None, backend=None):
    """Trailing idxmax; see SPEC.md for validity and degeneracy rules."""
    return _ts("idxmax", x, window, min_periods=min_periods, mask=mask, backend=backend)


def ts_ffill(x, *, limit=None, mask=None, backend=None):
    """Forward-fill valid observations, optionally limiting their age in rows."""
    periods = 0 if limit is None else integer(limit, "limit")
    return _ts("ffill", x, 1, periods=periods, mask=mask, backend=backend)


def _cs(
    op,
    x,
    *,
    mask=None,
    normalize=False,
    ddof=0,
    lower=0.01,
    upper=0.99,
    scale=3.0,
    groups=None,
    exposures=None,
    weights=None,
    rcond=1e-12,
    backend=None,
):
    panel(x)
    mm, gg, ww = (
        optional(mask, x, "mask", bool),
        optional(groups, x, "groups", np.int64),
        optional(weights, x, "weights"),
    )
    ee = (
        []
        if exposures is None
        else [panel(e, "exposure", shape=x.shape) for e in exposures]
    )
    if op == "group_demean" and groups is None:
        raise ValueError("group_demean requires groups")
    ddof = integer(ddof, "ddof", 0)
    lower, upper = scalar(lower, "lower", 0, 1), scalar(upper, "upper", lower, 1)
    scale, rcond = (
        scalar(scale, "scale", 0, np.finfo(float).max),
        scalar(rcond, "rcond", np.nextafter(0.0, 1.0), 1),
    )
    selected = backend or BACKENDS.get("cs_" + op, "rust")
    if selected == "numpy":
        return _numpy.cs(
            _numpy_input(x, mask),
            op,
            normalize,
            ddof,
            lower,
            upper,
            scale,
            groups,
            ee,
            weights,
            rcond,
        )
    if selected != "rust":
        raise ValueError("backend must be rust or numpy")
    return _core.cs(
        x, mm, op, gg, ee, ww, normalize, ddof, lower, upper, scale, rcond, _THREADS
    )


for _op in CS_OPS:
    _function = partial(_cs, _op)
    _function.__name__ = "cs_" + _op
    _function.__doc__ = f"Cross-sectional {_op}; see SPEC.md."
    globals()[_function.__name__] = _function


def ic(x, returns, *, mask=None, rank=False, backend=None):
    """Daily correlation of already aligned panels; no implicit time shift."""
    panel(x)
    panel(returns, "returns", shape=x.shape)
    mm = optional(mask, x, "mask", bool)
    selected = backend or BACKENDS.get("rank_ic" if rank else "ic", "rust")
    if selected == "numpy":
        return _numpy.ic(_numpy_input(x, mask), returns, rank)
    if selected != "rust":
        raise ValueError("backend must be rust or numpy")
    return _core.ic(x, returns, mm, rank, _THREADS)


def rank_ic(x, returns, **kwargs):
    """Daily Spearman correlation, ranking on pairwise complete observations."""
    return ic(x, returns, rank=True, **kwargs)


def rank_ic_many(scores, returns, *, mask=None, backend=None):
    """Daily Spearman correlation of several score panels against one return panel.

    Returns are ranked once per day and reused for every score; each output equals
    ``rank_ic(score, returns, mask=mask)`` bit for bit. ``scores`` may be any iterable
    of panels and is consumed one panel at a time, so a generator keeps only the
    current panel alive. Returns a list of 1-D arrays in input order.
    """
    panel(returns, "returns")
    mm = optional(mask, returns, "mask", bool)
    selected = backend or BACKENDS.get("rank_ic", "rust")
    if selected not in ("rust", "numpy"):
        raise ValueError("backend must be rust or numpy")
    order = None
    out = []
    for score in scores:
        panel(score)
        panel(returns, "returns", shape=score.shape)
        if selected == "numpy":
            out.append(ic(score, returns, mask=mask, rank=True, backend="numpy"))
            continue
        if order is None:
            order = _core.return_order(returns, mm, _THREADS)
        out.append(_core.rank_ic_ordered(score, order, _THREADS))
    return out


def factor_autocorr(x, periods=1, *, rank=True, mask=None, backend=None):
    """Lagged daily cross-sectional correlation on the common valid universe."""
    panel(x)
    periods = integer(periods, "periods", 0)
    mm = optional(mask, x, "mask", bool)
    selected = backend or BACKENDS.get("factor_autocorr", "rust")
    if selected == "rust":
        return _core.autocorr(x, mm, periods, rank, _THREADS)
    if selected != "numpy":
        raise ValueError("backend must be rust or numpy")
    lag = ts_delay(x, periods, mask=mask, backend="numpy")
    return ic(x, lag, mask=mask, rank=rank, backend="numpy")


def group_returns(scores, returns, q=5, *, weights=None, mask=None, backend=None):
    """Ordinal quantile returns; missing returns excluded after score grouping."""
    panel(scores)
    panel(returns, "returns", shape=scores.shape)
    q = integer(q, "q")
    mm, ww = optional(mask, scores, "mask", bool), optional(weights, scores, "weights")
    selected = backend or BACKENDS.get("group_returns", "rust")
    if selected == "numpy":
        return _numpy.group_returns(_numpy_input(scores, mask), returns, q, weights)
    if selected != "rust":
        raise ValueError("backend must be rust or numpy")
    return _core.grouped(scores, returns, mm, ww, q, 1, False, _THREADS)


def group_turnover(scores, q=5, periods=1, *, mask=None, backend=None):
    """Fraction of today's group absent from the same lagged group."""
    panel(scores)
    q, periods = integer(q, "q"), integer(periods, "periods", 0)
    mm = optional(mask, scores, "mask", bool)
    selected = backend or BACKENDS.get("group_turnover", "rust")
    if selected == "numpy":
        return _numpy.group_turnover(_numpy_input(scores, mask), q, periods)
    if selected != "rust":
        raise ValueError("backend must be rust or numpy")
    return _core.grouped(scores, _EMPTY, mm, _EMPTY, q, periods, True, _THREADS)


@dataclass(frozen=True)
class SparseTargets:
    """CSR target arrays, independent of any backtest engine."""

    days: np.ndarray
    indptr: np.ndarray
    cols: np.ndarray
    weights: np.ndarray


def targets(scores, *, k=None, fraction=None, days=None, mask=None, backend=None):
    """Select top scores into equal-weight sparse target rows."""
    panel(scores)
    mm = optional(mask, scores, "mask", bool)
    if k is not None and fraction is not None:
        raise ValueError("provide k or fraction, not both")
    k = 0 if k is None else integer(k, "k")
    fraction = (
        0.2
        if fraction is None
        else scalar(fraction, "fraction", np.nextafter(0.0, 1.0), 1)
    )
    if days is None:
        days = np.arange(len(scores), dtype=np.int64)
    if not isinstance(days, np.ndarray) or days.dtype != np.int64:
        raise TypeError("days must be an int64 array")
    if (
        days.ndim != 1
        or not days.flags.c_contiguous
        or np.any(days < 0)
        or np.any(days >= len(scores))
        or np.any(np.diff(days) <= 0)
    ):
        raise ValueError(
            "days must be contiguous, strictly increasing, valid row indices"
        )
    selected = backend or BACKENDS.get("targets", "rust")
    if selected == "numpy":
        ptr, cols, weights = _numpy.targets(
            _numpy_input(scores, mask), days, k, fraction
        )
    elif selected == "rust":
        ptr, cols, weights = _core.targets(scores, mm, days, k, fraction, _THREADS)
    else:
        raise ValueError("backend must be rust or numpy")
    return SparseTargets(days.copy(), ptr, cols, weights)


ELEMENTWISE_ARITY = {
    "gt": 2,
    "ge": 2,
    "lt": 2,
    "le": 2,
    "eq": 2,
    "ne": 2,
    "and_": 2,
    "or_": 2,
    "not_": 1,
    "where": 3,
    "maximum": 2,
    "minimum": 2,
    "sign": 1,
    "power": 2,
    "signed_power": 2,
    "sqrt": 1,
    "exp": 1,
    "clip": 1,
    "is_nan": 1,
    "fill_nan": 2,
    "add": 2,
    "subtract": 2,
    "multiply": 2,
    "divide": 2,
    "log": 1,
    "abs": 1,
    "negate": 1,
}


def elementwise(op, *args, mask=None, lower=-1.0, upper=1.0, backend=None):
    """Fused float64 elementwise operation, with scalar broadcasting; see SPEC."""
    if op not in ELEMENTWISE_ARITY or len(args) != ELEMENTWISE_ARITY[op]:
        raise ValueError("invalid elementwise operation or arity")
    arrays = [a for a in args if isinstance(a, np.ndarray)]
    shape = (
        arrays[0].shape
        if arrays
        else mask.shape
        if isinstance(mask, np.ndarray)
        else (1, 1)
    )
    for a in arrays:
        panel(a, shape=shape)
    for a in args:
        if not isinstance(a, np.ndarray) and (
            isinstance(a, (bool, np.bool_))
            or not isinstance(a, (int, float, np.integer, np.floating))
        ):
            raise TypeError("operands must be float64 panels or numeric scalars")
    if mask is not None:
        panel(mask, "mask", bool, shape)
    lower = scalar(lower, "lower", -np.finfo(float).max, np.finfo(float).max)
    upper = scalar(upper, "upper", lower, np.finfo(float).max)
    selected = backend or BACKENDS.get("ew_" + op, "rust")
    if selected == "numpy" or not arrays:
        if selected not in ("numpy", "rust"):
            raise ValueError("backend must be rust or numpy")
        result = _numpy.elementwise(op, *args, mask=mask, lower=lower, upper=upper)
        if not arrays and mask is not None:
            return np.where(mask, result, np.nan)
        return result
    if selected != "rust":
        raise ValueError("backend must be rust or numpy")
    args = [*args, *([0.0] * (3 - len(args)))]
    flags = [not isinstance(a, np.ndarray) for a in args]
    values = [float(a) if flag else 0.0 for a, flag in zip(args, flags, strict=True)]
    inputs = [_EMPTY if flag else a for a, flag in zip(args, flags, strict=True)]
    return _core.elementwise(
        *inputs,
        _MASK if mask is None else mask,
        op,
        values,
        flags,
        *shape,
        lower,
        upper,
        _THREADS,
    )
