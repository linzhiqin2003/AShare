"""Independent vectorized competitors; no calls into the native implementation."""

import warnings

import numpy as np

TS = (
    "sum mean std min max rank quantile argmin argmax product corr cov "
    "decay_linear ema skew kurt slope residual regression_slope regression_residual"
).split()


def numpy_ts(
    x,
    op,
    window=20,
    min_periods=None,
    y=None,
    ddof=1,
    q=0.5,
    normalize=False,
    alpha=None,
):
    """Chunked window reductions bound temporary storage, with cumulative sums."""
    mp = window if min_periods is None else min_periods
    t, n = x.shape
    out = np.full_like(x, np.nan)
    if not t or not n:
        return out
    if op in ("sum", "mean"):
        valid = np.isfinite(x)
        s = np.vstack((np.zeros((1, n)), np.cumsum(np.where(valid, x, 0), axis=0)))
        c = np.vstack((np.zeros((1, n), dtype=np.int64), valid.cumsum(axis=0)))
        starts = np.maximum(0, np.arange(t) + 1 - window)
        sums, counts = s[1:] - s[starts], c[1:] - c[starts]
        with np.errstate(all="ignore"):
            return np.where(
                counts >= mp, sums if op == "sum" else sums / counts, np.nan
            )
    if op in (
        "std",
        "corr",
        "cov",
        "slope",
        "residual",
        "regression_slope",
        "regression_residual",
    ):
        return numpy_moments(x, op, window, mp, y, ddof)
    if op in ("skew", "kurt") and window >= 20:
        return numpy_higher(x, op, window, mp)
    if op in ("product", "decay_linear"):
        return numpy_simple(x, op, window, mp)
    if op == "ema":
        a = 2 / (window + 1) if alpha is None else alpha
        state = np.full(n, np.nan)
        count = np.zeros(n, dtype=int)
        for i, row in enumerate(x):
            valid = np.isfinite(row)
            state = np.where(
                valid, np.where(count == 0, row, a * row + (1 - a) * state), state
            )
            count += valid
            out[i] = np.where(count >= mp, state, np.nan)
        return out
    padded = np.pad(x, ((window - 1, 0), (0, 0)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    yw = None
    if y is not None:
        yp = np.pad(y, ((window - 1, 0), (0, 0)), constant_values=np.nan)
        yw = np.lib.stride_tricks.sliding_window_view(yp, window, axis=0)
    # Aim at <= 32 MiB per temporary, several temporaries coexist.
    chunk = max(1, 4_000_000 // (n * window))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for start in range(0, t, chunk):
            v = windows[start : start + chunk]
            valid = np.isfinite(v)
            if yw is not None:
                valid = valid & np.isfinite(yw[start : start + chunk])
            v = np.where(valid, v, np.nan)
            count = valid.sum(-1)
            if op == "std":
                z = np.nanstd(v, axis=-1, ddof=ddof)
            elif op in ("min", "max"):
                z = getattr(np, "nan" + op)(v, axis=-1)
            elif op == "quantile":
                z = window_quantile(v, count, q)
            elif op == "rank":
                last = v[..., -1, None]
                z = (v < last).sum(-1) + ((v == last).sum(-1) + 1) / 2
                if normalize:
                    z = np.where(count == 1, 0.5, (z - 1) / (count - 1))
                z = np.where(np.isfinite(last[..., 0]), z, np.nan)
            elif op in ("argmin", "argmax"):
                a = np.where(valid, v, np.inf if op == "argmin" else -np.inf)
                z = getattr(np, op)(a[..., ::-1], axis=-1).astype(float)
            elif op == "product":
                z = np.nanprod(v, axis=-1)
                z = np.where((v == 0).any(-1), 0, z)
            elif op == "decay_linear":
                w = np.arange(1, window + 1)
                z = np.nansum(v * w, axis=-1) / (valid * w).sum(-1)
            elif op in ("skew", "kurt"):
                d = v - np.nanmean(v, axis=-1, keepdims=True)
                variance = np.nanmean(d * d, axis=-1)
                z = np.nanmean(
                    d * d * d if op == "skew" else (d * d) * (d * d), axis=-1
                ) / variance ** (1.5 if op == "skew" else 2)
                if op == "kurt":
                    z -= 3
                z = np.where(count >= (3 if op == "skew" else 4), z, np.nan)
            else:
                b = (
                    np.broadcast_to(np.arange(window), v.shape)
                    if yw is None
                    else yw[start : start + chunk]
                )
                b = np.where(valid, b, np.nan)
                mean_v, mean_b = np.nanmean(v, -1), np.nanmean(b, -1)
                dv, db = v - mean_v[..., None], b - mean_b[..., None]
                cross, vb = np.nansum(dv * db, -1), np.nansum(db * db, -1)
                if op == "cov":
                    z = cross / (count - ddof)
                    z = np.where(count > ddof, z, np.nan)
                elif op == "corr":
                    z = cross / np.sqrt(vb * np.nansum(dv * dv, -1))
                else:
                    z = cross / vb
                    if op.endswith("residual"):
                        z = v[..., -1] - mean_v - z * (b[..., -1] - mean_b)
                z = np.where(count >= 2, z, np.nan)
            out[start : start + chunk] = np.where(count >= mp, z, np.nan)
    return out


def pandas_ts(x, op, window=20, min_periods=None, y=None, ddof=1, **kwargs):
    import pandas as pd

    mp = window if min_periods is None else min_periods
    frame = pd.DataFrame(x)
    if op in ("delay", "delta"):
        lag = frame.shift(window)
        return (lag if op == "delay" else frame - lag).to_numpy()
    if op in (
        "product",
        "decay_linear",
        "slope",
        "residual",
        "regression_slope",
        "regression_residual",
    ):
        return pandas_composed(x, op, window, mp, y)
    r = frame.rolling(window, min_periods=mp)
    if op in ("sum", "mean", "std", "min", "max"):
        return getattr(r, op)(**({"ddof": ddof} if op == "std" else {})).to_numpy()
    if op in ("corr", "cov"):
        return getattr(r, op)(
            pd.DataFrame(y), **({"ddof": ddof} if op == "cov" else {})
        ).to_numpy()
    if op in ("skew", "kurt"):
        count = r.count().to_numpy()
        raw = getattr(r, op)().to_numpy()
        with np.errstate(all="ignore"):
            if op == "skew":
                result = raw * (count - 2) / np.sqrt(count * (count - 1))
            else:
                result = (raw * (count - 2) * (count - 3) / (count - 1) - 6) / (
                    count + 1
                )
        return np.where(r.var(ddof=0).to_numpy() > 0, result, np.nan)
    if op == "quantile":
        return r.quantile(kwargs.get("q", 0.5)).to_numpy()
    if op == "rank":
        z = r.rank(method="average").to_numpy()
        if kwargs.get("normalize", False):
            c = r.count().to_numpy()
            z = np.where(c == 1, 0.5, (z - 1) / (c - 1))
        return z
    if op == "ema":
        return (
            frame.ewm(
                alpha=kwargs.get("alpha") or 2 / (window + 1),
                adjust=False,
                ignore_na=True,
                min_periods=mp,
            )
            .mean()
            .to_numpy()
        )
    raise NotImplementedError(op)


def bottleneck_ts(x, op, window=20, min_periods=None, ddof=1, **kwargs):
    import bottleneck as bn

    mp = window if min_periods is None else min_periods
    if window > len(x):
        padded = np.pad(x, ((window - len(x), 0), (0, 0)), constant_values=np.nan)
        if kwargs.get("y") is not None:
            kwargs = dict(
                kwargs,
                y=np.pad(
                    kwargs["y"], ((window - len(x), 0), (0, 0)), constant_values=np.nan
                ),
            )
        return bottleneck_ts(padded, op, window, mp, ddof, **kwargs)[-len(x) :]
    mapping = dict(
        sum="sum",
        mean="mean",
        std="std",
        min="min",
        max="max",
        rank="rank",
        argmin="argmin",
        argmax="argmax",
    )
    if op == "quantile" and kwargs.get("q", 0.5) == 0.5:
        return bn.move_median(x, window, min_count=mp, axis=0)
    composed = (
        "corr",
        "cov",
        "slope",
        "residual",
        "regression_slope",
        "regression_residual",
        "decay_linear",
        "product",
        "skew",
        "kurt",
    )
    if op in composed:
        return bottleneck_composed(x, op, window, mp, ddof, kwargs.get("y"))
    if op not in mapping:
        raise NotImplementedError(op)
    z = getattr(bn, "move_" + mapping[op])(
        x, window, min_count=mp, axis=0, **({"ddof": ddof} if op == "std" else {})
    )
    if op == "rank":
        c = bn.move_sum(np.isfinite(x).astype(float), window, min_count=1, axis=0)
        z = (z + 1) / 2
        if not kwargs.get("normalize", False):
            z = z * (c - 1) + 1
    return z


def polars_ts(x, op, window=20, min_periods=None, y=None, ddof=1, **kwargs):
    import polars as pl

    mp = window if min_periods is None else min_periods
    f = pl.DataFrame(x, orient="row").with_columns(pl.all().fill_nan(None))
    a = pl.all()
    if op in ("delay", "delta"):
        return f.select(
            a.shift(window) if op == "delay" else a - a.shift(window)
        ).to_numpy()
    if op in (
        "product",
        "decay_linear",
        "slope",
        "residual",
        "regression_slope",
        "regression_residual",
    ):
        columns = f.columns
        if y is not None:
            f = f.hstack(
                pl.DataFrame(
                    y, schema=[f"y{i}" for i in range(x.shape[1])], orient="row"
                ).with_columns(pl.all().fill_nan(None))
            )
        time = pl.int_range(0, pl.len()).cast(pl.Float64)
        return f.select(
            [
                polars_composed_expression(
                    pl.col(c),
                    None if y is None else pl.col(f"y{i}"),
                    time,
                    op,
                    window,
                    mp,
                ).alias(c)
                for i, c in enumerate(columns)
            ]
        ).to_numpy()
    if op in ("rank", "skew", "kurt"):
        method = "kurtosis" if op == "kurt" else op
        args = {"window_size": window, "min_samples": mp}
        if op in ("skew", "kurt"):
            args.update(bias=True, min_samples=max(mp, 3 if op == "skew" else 4))
        return f.select(getattr(a, "rolling_" + method)(**args)).to_numpy()
    if op in ("sum", "mean", "min", "max", "std", "quantile"):
        args = {"window_size": window, "min_samples": mp}
        if op == "std":
            args["ddof"] = ddof
        if op == "quantile":
            args.update(quantile=kwargs.get("q", 0.5), interpolation="linear")
        return f.select(getattr(a, "rolling_" + op)(**args)).to_numpy()
    if op == "ema":
        return f.select(
            a.ewm_mean(
                alpha=kwargs.get("alpha") or 2 / (window + 1),
                adjust=False,
                ignore_nulls=True,
                min_samples=mp,
            ).forward_fill()
        ).to_numpy()
    if op in ("cov", "corr"):
        fy = pl.DataFrame(
            y, schema=[f"y{i}" for i in range(x.shape[1])], orient="row"
        ).with_columns(pl.all().fill_nan(None))
        f = f.hstack(fy)
        return f.select(
            [
                getattr(pl, "rolling_" + op)(
                    pl.col(c),
                    pl.col(f"y{i}"),
                    window_size=window,
                    min_samples=mp,
                    **({"ddof": ddof} if op == "cov" else {}),
                ).alias(c)
                for i, c in enumerate(f.columns[: x.shape[1]])
            ]
        ).to_numpy()
    raise NotImplementedError(op)


def numpy_moments(x, op, window, mp, y=None, ddof=1):
    """Blocked centered prefix moments: O(T*N), bounded temporary memory."""
    out = np.full_like(x, np.nan)
    n = x.shape[1]
    with np.errstate(all="ignore"):
        for start in range(0, len(x), 128):
            end = min(start + 128, len(x))
            left = max(0, start - window + 1)
            a = x[left:end]
            b = (
                np.broadcast_to(np.arange(left, end)[:, None], a.shape)
                if y is None
                else y[left:end]
            )
            valid = np.isfinite(a) & np.isfinite(b)
            first = valid.argmax(0)
            ox = np.where(valid.any(0), a[first, np.arange(n)], 0)
            oy = np.where(valid.any(0), b[first, np.arange(n)], 0)
            a = np.where(valid, a - ox, 0)
            b = np.where(valid, b - oy, 0)
            rows = np.arange(start, end) - left + 1
            begins = np.maximum(0, rows - window)

            def rolling(values, rows=rows, begins=begins):
                prefix = np.vstack((np.zeros((1, n)), np.cumsum(values, axis=0)))
                return prefix[rows] - prefix[begins]

            count = rolling(valid)
            sx, sy = rolling(a), rolling(b)
            xx = np.maximum(0, rolling(a * a) - sx * sx / count)
            if op == "std":
                z = np.where(count > ddof, np.sqrt(xx / (count - ddof)), np.nan)
            else:
                yy = np.maximum(0, rolling(b * b) - sy * sy / count)
                xy = rolling(a * b) - sx * sy / count
                if op == "cov":
                    z = np.where(count > ddof, xy / (count - ddof), np.nan)
                elif op == "corr":
                    z = np.where(count >= 2, xy / np.sqrt(xx * yy), np.nan)
                else:
                    z = np.where(count >= 2, xy / yy, np.nan)
                    if op.endswith("residual"):
                        z = a[rows - 1] - sx / count - z * (b[rows - 1] - sy / count)
                        z = np.where(valid[rows - 1], z, np.nan)
            out[start:end] = np.where(count >= mp, z, np.nan)
    return out


def polars_long_ts(x, op, window=20, min_periods=None, y=None, ddof=1, **kwargs):
    """Long-form grouped rolling competitor, including conversion cost."""
    import polars as pl

    mp = window if min_periods is None else min_periods
    t, n = x.shape
    data = {"symbol": np.repeat(np.arange(n), t), "x": x.T.ravel()}
    if op in (
        "product",
        "decay_linear",
        "slope",
        "residual",
        "regression_slope",
        "regression_residual",
    ):
        data["time"] = np.tile(np.arange(t, dtype=float), n)
    if y is not None:
        data["y"] = y.T.ravel()
    f = pl.DataFrame(data).with_columns(pl.col("x").fill_nan(None))
    a = pl.col("x")
    args = dict(window_size=window, min_samples=mp)
    if op in ("delay", "delta"):
        expression = a.shift(window) if op == "delay" else a - a.shift(window)
    elif op in (
        "product",
        "decay_linear",
        "slope",
        "residual",
        "regression_slope",
        "regression_residual",
    ):
        if y is not None:
            f = f.with_columns(pl.col("y").fill_nan(None))
        expression = polars_composed_expression(
            a, None if y is None else pl.col("y"), pl.col("time"), op, window, mp
        )
    elif op in ("sum", "mean", "min", "max", "std", "quantile", "rank", "skew", "kurt"):
        method = "kurtosis" if op == "kurt" else op
        if op == "std":
            args["ddof"] = ddof
        if op == "quantile":
            args.update(quantile=kwargs.get("q", 0.5), interpolation="linear")
        if op in ("skew", "kurt"):
            args.update(bias=True, min_samples=max(mp, 3 if op == "skew" else 4))
        expression = getattr(a, "rolling_" + method)(**args)
    elif op in ("corr", "cov"):
        f = f.with_columns(pl.col("y").fill_nan(None))
        expression = getattr(pl, "rolling_" + op)(
            a, pl.col("y"), **args, **({"ddof": ddof} if op == "cov" else {})
        )
    elif op == "ema":
        expression = a.ewm_mean(
            alpha=kwargs.get("alpha") or 2 / (window + 1),
            adjust=False,
            ignore_nulls=True,
            min_samples=mp,
        ).forward_fill()
    else:
        raise NotImplementedError(op)
    return np.ascontiguousarray(
        f.select(expression.over("symbol")).to_numpy().reshape(n, t).T
    )


def partition_quantile(values, count, q):
    """Partition each valid-count cohort at only its two interpolation indices."""
    width = values.shape[-1]
    flat = np.ascontiguousarray(values).reshape(-1, width)
    counts = count.ravel()
    result = np.full(len(counts), np.nan)
    for size in np.unique(counts):
        if not size:
            continue
        ids = np.flatnonzero(counts == size)
        position = (size - 1) * q
        low, high = int(position), int(np.ceil(position))
        selected = np.partition(flat[ids], (low, high), axis=-1)
        result[ids] = selected[:, low] + (selected[:, high] - selected[:, low]) * (
            position - low
        )
    return result.reshape(count.shape)


def bottleneck_composed(x, op, window, mp, ddof, y):
    """Compose moving primitives rather than hiding a Python window callback."""
    import bottleneck as bn

    with np.errstate(all="ignore"):
        valid = np.isfinite(x)
        if y is not None:
            valid &= np.isfinite(y)
        a = np.where(valid, x, np.nan)
        count = bn.move_sum(valid.astype(float), window, min_count=1, axis=0)

        def total(v):
            return bn.move_sum(v, window, min_count=mp, axis=0)

        def average(v):
            return bn.move_mean(v, window, min_count=mp, axis=0)

        if op == "product":
            zeros = total(np.where(valid, (a == 0).astype(float), np.nan))
            negatives = total(np.where(valid, (a < 0).astype(float), np.nan))
            logs = np.log(np.where(a == 0, 1, abs(a)))
            return np.where(
                zeros > 0, 0, np.exp(total(logs)) * np.where(negatives % 2, -1, 1)
            )
        if op == "decay_linear":
            time = np.arange(len(x), dtype=float)[:, None]
            weighted = total(a * time) - (time - window) * total(a)
            denom = total(np.where(valid, time, np.nan)) - (time - window) * count
            return weighted / denom
        if op in ("skew", "kurt"):
            # Fixed origin reduces cancellation; benchmark inputs are unit scale.
            origin = np.where(
                valid.any(0), a[valid.argmax(0), np.arange(x.shape[1])], 0
            )
            d = a - origin
            m1, m2, m3 = average(d), average(d * d), average(d * d * d)
            variance = m2 - m1 * m1
            if op == "skew":
                result = (m3 - 3 * m1 * m2 + 2 * m1**3) / variance**1.5
                minimum = 3
            else:
                m4 = average((d * d) * (d * d))
                result = (m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1**4) / (
                    variance * variance
                ) - 3
                minimum = 4
            return np.where((count >= minimum) & (variance > 0), result, np.nan)
        b = np.where(valid, np.arange(len(x))[:, None] if y is None else y, np.nan)
        ma, mb = average(a), average(b)
        cross = average(a * b) - ma * mb
        if op == "cov":
            return np.where(count > ddof, cross * count / (count - ddof), np.nan)
        std_b = bn.move_std(b, window, min_count=mp, ddof=0, axis=0)
        if op == "corr":
            return cross / (
                bn.move_std(a, window, min_count=mp, ddof=0, axis=0) * std_b
            )
        slope = cross / (std_b * std_b)
        result = a - ma - slope * (b - mb) if op.endswith("residual") else slope
        return np.where(count >= 2, result, np.nan)


def numpy_simple(x, op, window, mp):
    """Blocked prefix formulas for decay and guarded log products."""
    out = np.full_like(x, np.nan)
    n = x.shape[1]
    with np.errstate(all="ignore"):
        for start in range(0, len(x), 128):
            end = min(len(x), start + 128)
            left = max(0, start - window + 1)
            a = x[left:end]
            valid = np.isfinite(a)
            rows = np.arange(start, end) - left + 1
            begins = np.maximum(0, rows - window)

            def rolling(v, rows=rows, begins=begins):
                prefix = np.vstack((np.zeros((1, n)), np.cumsum(v, axis=0)))
                return prefix[rows] - prefix[begins]

            count = rolling(valid)
            if op == "decay_linear":
                time = np.arange(len(a))[:, None]
                shifted = (rows - 1 - window)[:, None]
                total = rolling(np.where(valid, a, 0))
                weighted = rolling(np.where(valid, a * time, 0)) - shifted * total
                denom = rolling(valid * time) - shifted * count
                z = weighted / denom
            else:
                zeros = rolling(valid & (a == 0))
                negatives = rolling(valid & (a < 0))
                logs = np.log(np.where(valid & (a != 0), abs(a), 1))
                log_total, magnitude = rolling(logs), rolling(abs(logs))
                z = np.where(
                    zeros > 0, 0, np.exp(log_total) * np.where(negatives % 2, -1, 1)
                )
                risky = (magnitude > 700) & (zeros == 0) & (count >= mp)
                for row, col in zip(*np.nonzero(risky), strict=True):
                    t = start + row
                    v = x[max(0, t - window + 1) : t + 1, col]
                    z[row, col] = np.prod(v[np.isfinite(v)])
            out[start:end] = np.where(count >= mp, z, np.nan)
    return out


def numpy_higher(x, op, window, mp):
    """Centered blocked prefix high moments with a cancellation fallback."""
    out = np.full_like(x, np.nan)
    degree = 3 if op == "skew" else 4
    n = x.shape[1]
    with np.errstate(all="ignore"):
        for start in range(0, len(x), 128):
            end = min(len(x), start + 128)
            left = max(0, start - window + 1)
            values = x[left:end]
            valid = np.isfinite(values)
            first = valid.argmax(0)
            origin = np.where(valid.any(0), values[first, np.arange(n)], 0)
            d = np.where(valid, values - origin, 0)
            rows = np.arange(start, end) - left + 1
            begins = np.maximum(0, rows - window)

            def rolling(v, rows=rows, begins=begins):
                prefix = np.vstack((np.zeros((1, n)), np.cumsum(v, axis=0)))
                return prefix[rows] - prefix[begins]

            count = rolling(valid)
            m1 = rolling(d) / count
            m2 = rolling(d * d) / count
            m3 = rolling(d * d * d) / count
            variance = np.maximum(0, m2 - m1 * m1)
            if degree == 3:
                numerator = m3 - 3 * m1 * m2 + 2 * m1 * m1 * m1
                magnitude = abs(m3) + abs(3 * m1 * m2) + abs(2 * m1 * m1 * m1)
                denom = variance * np.sqrt(variance)
            else:
                m4 = rolling((d * d) * (d * d)) / count
                numerator = m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1**4
                magnitude = (
                    abs(m4) + abs(4 * m1 * m3) + abs(6 * m1 * m1 * m2) + abs(3 * m1**4)
                )
                denom = variance * variance
            result = numerator / denom - (3 if degree == 4 else 0)
            risky = (
                (variance <= 0)
                | (64 * np.finfo(float).eps * magnitude > 1e-10 * denom)
                | (abs(origin) * np.finfo(float).eps > np.sqrt(variance) * 1e-12)
            )
            eligible = (count >= mp) & (count >= degree)
            for row, col in zip(*np.nonzero(risky & eligible), strict=True):
                t = start + row
                v = x[max(0, t - window + 1) : t + 1, col]
                v = v[np.isfinite(v)]
                centered = v - v.mean()
                var = np.mean(centered * centered)
                result[row, col] = (
                    np.mean(centered**degree) / var ** (degree / 2)
                    - (3 if degree == 4 else 0)
                    if var > 0
                    else np.nan
                )
            out[start:end] = np.where(eligible, result, np.nan)
    return out


def pandas_composed(x, op, window, mp, y):
    """Vectorized rolling compositions, without per-window Python callbacks."""
    import pandas as pd

    valid = np.isfinite(x)
    if y is not None:
        valid &= np.isfinite(y)
    a = pd.DataFrame(np.where(valid, x, np.nan))
    r = a.rolling(window, min_periods=mp)
    count = pd.DataFrame(valid.astype(float)).rolling(window, min_periods=1).sum()
    with np.errstate(all="ignore"):
        if op == "product":
            zeros = (
                pd.DataFrame(np.where(valid, x == 0, np.nan))
                .rolling(window, min_periods=mp)
                .sum()
            )
            negatives = (
                pd.DataFrame(np.where(valid, x < 0, np.nan))
                .rolling(window, min_periods=mp)
                .sum()
            )
            logs = np.log(a.abs().where(a.ne(0), 1))
            result = np.exp(logs.rolling(window, min_periods=mp).sum())
            return np.where(zeros > 0, 0, result * np.where(negatives % 2, -1, 1))
        time = np.broadcast_to(np.arange(len(x))[:, None], x.shape)
        b = pd.DataFrame(np.where(valid, time if y is None else y, np.nan))
        if op == "decay_linear":
            weighted = (a * time).rolling(window, min_periods=mp).sum() - (
                time - window
            ) * r.sum()
            denom = b.rolling(window, min_periods=mp).sum() - (time - window) * count
            return (weighted / denom).to_numpy()
        br = b.rolling(window, min_periods=mp)
        slope = r.cov(b, ddof=0) / br.var(ddof=0)
        result = (
            a - r.mean() - slope * (b - br.mean()) if op.endswith("residual") else slope
        )
        return np.where(count >= 2, result, np.nan)


def polars_composed_expression(a, b, time, op, window, mp):
    """Build one native rolling expression for either wide or grouped data."""
    import polars as pl

    if b is not None:
        valid = a.is_not_null() & b.is_not_null()
        a, b = pl.when(valid).then(a), pl.when(valid).then(b)
    count = a.is_not_null().cast(pl.Float64).rolling_sum(window, min_samples=1)
    args = dict(window_size=window, min_samples=mp)
    if op == "product":
        zeros = (a == 0).cast(pl.Float64).rolling_sum(**args)
        negatives = (a < 0).cast(pl.Float64).rolling_sum(**args)
        logs = pl.when(a == 0).then(1).otherwise(a.abs()).log()
        return (
            pl.when(zeros > 0)
            .then(0)
            .otherwise(
                logs.rolling_sum(**args).exp()
                * pl.when(negatives % 2 == 0).then(1).otherwise(-1)
            )
        )
    if b is None:
        b = pl.when(a.is_not_null()).then(time)
    if op == "decay_linear":
        numerator = (a * time).rolling_sum(**args) - (time - window) * a.rolling_sum(
            **args
        )
        denominator = b.rolling_sum(**args) - (time - window) * count
        return numerator / denominator
    variance = b.rolling_var(ddof=0, **args)
    slope = pl.rolling_cov(a, b, ddof=0, **args) / variance
    result = (
        a - a.rolling_mean(**args) - slope * (b - b.rolling_mean(**args))
        if op.endswith("residual")
        else slope
    )
    return pl.when((count >= 2) & (variance > 0)).then(result)


def sorted_quantile(values, count, q):
    ordered = np.sort(values, axis=-1)
    position = np.maximum(count - 1, 0) * q
    low = np.take_along_axis(ordered, position.astype(int)[..., None], axis=-1)[..., 0]
    high = np.take_along_axis(
        ordered, np.ceil(position).astype(int)[..., None], axis=-1
    )[..., 0]
    return low + (high - low) * (position - np.floor(position))


def window_quantile(values, count, q):
    """Measured crossover: SIMD sorting wins short windows; partition wins long."""
    implementation = sorted_quantile if values.shape[-1] <= 60 else partition_quantile
    return implementation(values, count, q)
