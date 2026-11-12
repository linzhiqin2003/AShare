"""Vectorized baselines for operators added in the second review round."""

import numpy as np
from baselines import bottleneck_ts, numpy_ts, pandas_ts, polars_long_ts, polars_ts

ELEMENTS = (
    "gt ge lt le eq ne and_ or_ not_ where maximum minimum sign power "
    "signed_power sqrt exp clip is_nan fill_nan"
).split()
PAIRED = {"corr", "cov", "regression_slope", "regression_residual", "sumif"}
NEW_TS = {
    "rsquare",
    "var",
    "zscore",
    "median",
    "mad",
    "count",
    "sumif",
    "ffill",
    "idxmin",
    "idxmax",
}


def numpy_extra(x, op, window, min_periods, y=None):
    if op == "var":
        return numpy_ts(x, "std", window, min_periods) ** 2
    if op == "zscore":
        return (x - numpy_ts(x, "mean", window, min_periods)) / numpy_ts(
            x, "std", window, min_periods
        )
    if op == "rsquare":
        time = np.broadcast_to(np.arange(len(x), dtype=float)[:, None], x.shape)
        return numpy_ts(x, "corr", window, min_periods, time) ** 2
    if op == "median":
        return numpy_ts(x, "quantile", window, min_periods)
    if op in ("idxmin", "idxmax"):
        return window - numpy_ts(x, op.replace("idx", "arg"), window, min_periods)
    if op in ("count", "sumif"):
        valid = np.isfinite(x)
        if op == "sumif":
            valid &= np.isfinite(y)
        values = np.where(
            valid,
            (x != 0).astype(float) if op == "count" else np.where(y != 0, x, 0.0),
            np.nan,
        )
        return numpy_ts(values, "sum", window, min_periods)
    if op == "ffill":
        indices = np.where(np.isfinite(x), np.arange(len(x))[:, None], -1)
        np.maximum.accumulate(indices, axis=0, out=indices)
        result = x[indices, np.arange(x.shape[1])]
        result[indices < 0] = np.nan
        return result
    if op == "mad":
        w = np.lib.stride_tricks.sliding_window_view(
            np.pad(x, ((window - 1, 0), (0, 0)), constant_values=np.nan), window, axis=0
        )
        result = np.empty_like(x)
        chunk = max(1, 4_000_000 // (window * x.shape[1]))
        for i in range(0, len(x), chunk):
            v = w[i : i + chunk]
            center = np.nanmedian(v, axis=-1, keepdims=True)
            a = np.nanmedian(abs(v - center), axis=-1)
            a[np.isfinite(v).sum(-1) < min_periods] = np.nan
            result[i : i + chunk] = a
        return result
    raise NotImplementedError(op)


def pandas_extra(x, op, window, min_periods, y=None):
    import pandas as pd

    a = pd.DataFrame(x)
    r = a.rolling(window, min_periods=min_periods)
    if op == "var":
        return r.var(ddof=1).to_numpy()
    if op == "zscore":
        return ((a - r.mean()) / r.std(ddof=1)).to_numpy()
    if op == "rsquare":
        b = pd.DataFrame(
            np.broadcast_to(np.arange(len(x), dtype=float)[:, None], x.shape)
        )
        return r.corr(b).to_numpy() ** 2
    if op == "median":
        return r.median().to_numpy()
    if op in ("count", "sumif"):
        valid = np.isfinite(x)
        if op == "sumif":
            valid &= np.isfinite(y)
        data = (x != 0).astype(float) if op == "count" else np.where(y != 0, x, 0.0)
        return (
            pd.DataFrame(np.where(valid, data, np.nan))
            .rolling(window, min_periods=min_periods)
            .sum()
            .to_numpy()
        )
    if op == "ffill":
        return a.ffill().to_numpy()
    if op in ("idxmin", "idxmax"):
        fn = np.nanargmin if op == "idxmin" else np.nanargmax
        return window - r.apply(lambda v: fn(v[::-1]), raw=True).to_numpy()
    if op == "mad":
        return r.apply(
            lambda v: np.nanmedian(abs(v - np.nanmedian(v))), raw=True
        ).to_numpy()
    raise NotImplementedError(op)


def bottleneck_extra(x, op, window, min_periods, y=None):
    if window > len(x):
        padding = window - len(x)
        padded = np.pad(x, ((padding, 0), (0, 0)), constant_values=np.nan)
        other = (
            None
            if y is None
            else np.pad(y, ((padding, 0), (0, 0)), constant_values=np.nan)
        )
        return bottleneck_extra(padded, op, window, min_periods, other)[padding:]
    import bottleneck as bn

    kw = dict(window=window, min_count=min_periods, axis=0)
    if op == "var":
        return bn.move_var(x, ddof=1, **kw)
    if op == "zscore":
        return (x - bn.move_mean(x, **kw)) / bn.move_std(x, ddof=1, **kw)
    if op == "rsquare":
        time = np.broadcast_to(np.arange(len(x), dtype=float)[:, None], x.shape)
        return bottleneck_ts(x, "corr", window, min_periods, time) ** 2
    if op == "median":
        return bn.move_median(x, **kw)
    if op in ("idxmin", "idxmax"):
        return window - getattr(bn, "move_" + op.replace("idx", "arg"))(x, **kw)
    if op in ("count", "sumif"):
        valid = np.isfinite(x)
        if op == "sumif":
            valid &= np.isfinite(y)
        data = (x != 0).astype(float) if op == "count" else np.where(y != 0, x, 0.0)
        return bn.move_sum(np.where(valid, data, np.nan), **kw)
    raise NotImplementedError(op)


def polars_extra(x, op, window, min_periods, y=None, long=False):
    import polars as pl

    t, n = x.shape
    kw = dict(window_size=window, min_samples=min_periods)
    if long:
        data = {"x": x.T.ravel()}
        if y is not None:
            data["y"] = y.T.ravel()
        data["symbol"] = np.repeat(np.arange(n), t)
        data["time"] = np.tile(np.arange(t, dtype=float), n)
        frame = pl.DataFrame(data).with_columns(pl.col("x").fill_nan(None))
        if y is not None:
            frame = frame.with_columns(pl.col("y").fill_nan(None))
    else:
        frame = pl.DataFrame(x, orient="row").with_columns(pl.all().fill_nan(None))

    def expr(a, b, clock):
        if op == "var":
            return a.rolling_var(ddof=1, **kw)
        if op == "zscore":
            return (a - a.rolling_mean(**kw)) / a.rolling_std(ddof=1, **kw)
        if op == "rsquare":
            return pl.rolling_corr(a, clock, **kw) ** 2
        if op == "median":
            return a.rolling_median(**kw)
        if op == "count":
            return (a != 0).cast(pl.Float64).rolling_sum(**kw)
        if op == "sumif":
            return (
                pl.when(a.is_not_null() & b.is_not_null())
                .then(pl.when(b != 0).then(a).otherwise(0.0))
                .otherwise(None)
                .rolling_sum(**kw)
            )
        if op == "ffill":
            return a.forward_fill()
        raise NotImplementedError(op)

    if long:
        out = frame.select(
            expr(pl.col("x"), pl.col("y"), pl.col("time"))
            .over("symbol")
            .alias("result")
        )
        return out["result"].to_numpy().reshape(n, t).T.copy()
    if y is not None:
        frame = frame.hstack(
            pl.DataFrame({f"y{i}": y[:, i] for i in range(n)})
        ).with_columns(pl.col("^y.*$").fill_nan(None))
    expressions = [
        expr(
            pl.col(f"column_{i}"),
            pl.col(f"y{i}"),
            pl.int_range(0, pl.len()).cast(pl.Float64),
        ).alias(f"column_{i}")
        for i in range(n)
    ]
    return frame.select(expressions).to_numpy()


def ts(x, op, method, window=20, min_periods=10, y=None):
    if op in ("delay", "delta"):
        if method == "numpy":
            out = np.full_like(x, np.nan)
            out[window:] = x[:-window] if op == "delay" else x[window:] - x[:-window]
            return out
    if op not in NEW_TS:
        f = {
            "numpy": numpy_ts,
            "bottleneck": bottleneck_ts,
            "pandas": pandas_ts,
            "polars": polars_ts,
            "polars_long": polars_long_ts,
        }[method]
        result = f(x, op, window=window, min_periods=min_periods, y=y)
        if min_periods <= 2 and op in ("regression_slope", "regression_residual"):
            # Exactly two pairs determine a line. Prefix/library moments can lose
            # precision for nearly equal predictors; use the direct closed form.
            valid = np.isfinite(x) & np.isfinite(y)
            prefix = np.vstack(
                (np.zeros((1, x.shape[1]), dtype=int), valid.cumsum(axis=0))
            )
            count = prefix[1:] - prefix[np.maximum(0, np.arange(len(x)) + 1 - window)]
            result = result.copy()
            for t, j in zip(*np.nonzero(count == 2), strict=True):
                start = max(0, t - window + 1)
                mask = valid[start : t + 1, j]
                a = x[start : t + 1, j][mask]
                b = y[start : t + 1, j][mask]
                slope = (a[1] - a[0]) / (b[1] - b[0]) if b[1] != b[0] else np.nan
                result[t, j] = (
                    slope
                    if op == "regression_slope"
                    else 0.0
                    if valid[t, j] and np.isfinite(slope)
                    else np.nan
                )
        return result
    if method.startswith("polars"):
        return polars_extra(
            x, op, window, min_periods, y, long=method.endswith("_long")
        )
    return {
        "numpy": numpy_extra,
        "bottleneck": bottleneck_extra,
        "pandas": pandas_extra,
    }[method](x, op, window, min_periods, y)


def numpy_element(x, y, op, z=None):
    """Ufunc outputs are cleaned in place, avoiding a second full result copy."""
    unary = {"sign": np.sign, "sqrt": np.sqrt, "exp": np.exp, "not_": lambda a: a == 0}
    binary = {
        "gt": np.greater,
        "ge": np.greater_equal,
        "lt": np.less,
        "le": np.less_equal,
        "eq": np.equal,
        "ne": np.not_equal,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "power": np.power,
        "and_": lambda a, b: np.logical_and(a != 0, b != 0),
        "or_": lambda a, b: np.logical_or(a != 0, b != 0),
    }
    with np.errstate(all="ignore"):
        if op == "is_nan":
            return (~np.isfinite(x)).astype(float)
        if op == "where":
            out = np.where(x != 0, y, z)
            valid = np.isfinite(x)
        elif op == "fill_nan":
            out = np.where(np.isfinite(x), x, y)
            valid = np.ones(x.shape, dtype=bool)
        elif op == "clip":
            out = np.clip(x, -1.0, 1.0)
            valid = np.isfinite(x)
        elif op == "signed_power":
            out = np.abs(x)
            np.power(out, y, out=out)
            out *= np.sign(x)
            valid = np.isfinite(x)
            valid &= np.isfinite(y)
        elif op in unary:
            out = np.asarray(unary[op](x), dtype=float)
            valid = np.isfinite(x)
        else:
            out = np.asarray(binary[op](x, y), dtype=float)
            valid = np.isfinite(x)
            valid &= np.isfinite(y)
        valid &= np.isfinite(out)
        np.copyto(out, np.nan, where=~valid)
        return out


def element(x, y, op, method, z=None):
    if method == "numpy":
        return numpy_element(x, y, op, z)
    if method == "pandas":
        # pandas dispatches these array operations to NumPy; retain its frame overhead.
        import pandas as pd

        a, b = pd.DataFrame(x), pd.DataFrame(y)
        result = element(a.to_numpy(), b.to_numpy(), op, "numpy", z)
        return pd.DataFrame(result).to_numpy()
    if not method.startswith("polars"):
        raise NotImplementedError(op)
    import polars as pl

    a, b = pl.col("x"), pl.col("y")
    frame = pl.DataFrame(
        {"x": x.ravel(), "y": y.ravel(), **({"z": z.ravel()} if op == "where" else {})}
    ).with_columns(pl.all().fill_nan(None))
    expression = {
        "gt": lambda: a > b,
        "ge": lambda: a >= b,
        "lt": lambda: a < b,
        "le": lambda: a <= b,
        "eq": lambda: a == b,
        "ne": lambda: a != b,
        "and_": lambda: (a != 0) & (b != 0),
        "or_": lambda: (a != 0) | (b != 0),
        "not_": lambda: a == 0,
        "where": lambda: pl.when(a.is_not_null()).then(
            pl.when(a != 0).then(b).otherwise(pl.col("z"))
        ),
        "maximum": lambda: pl.when(a.is_not_null() & b.is_not_null()).then(
            pl.max_horizontal(a, b)
        ),
        "minimum": lambda: pl.when(a.is_not_null() & b.is_not_null()).then(
            pl.min_horizontal(a, b)
        ),
        "sign": lambda: a.sign(),
        "power": lambda: a.pow(b),
        "signed_power": lambda: a.sign() * a.abs().pow(b),
        "sqrt": lambda: a.sqrt(),
        "exp": lambda: a.exp(),
        "clip": lambda: a.clip(-1.0, 1.0),
        "is_nan": lambda: a.is_null(),
        "fill_nan": lambda: a.fill_null(b),
    }[op]()
    if op in ("and_", "or_"):
        expression = (
            pl.when(a.is_not_null() & b.is_not_null()).then(expression).otherwise(None)
        )
    return frame.select(expression.cast(pl.Float64)).to_numpy().reshape(x.shape)
