"""Vectorized NumPy candidates used in benchmark-based dispatch."""

import numpy as np


def rank(x, normalize=False):
    valid = np.isfinite(x)
    a = np.where(valid, x, np.inf)
    order = np.argsort(a, axis=1, kind="stable")
    sorted_x = np.take_along_axis(a, order, axis=1)
    n = x.shape[1]
    if n == 0:
        return x.copy()
    positions = np.broadcast_to(np.arange(n), x.shape)
    start = np.maximum.accumulate(
        np.where(
            np.concatenate(
                (np.ones((len(x), 1), bool), sorted_x[:, 1:] != sorted_x[:, :-1]), 1
            ),
            positions,
            0,
        ),
        axis=1,
    )
    end = np.minimum.accumulate(
        np.where(
            np.concatenate(
                (sorted_x[:, :-1] != sorted_x[:, 1:], np.ones((len(x), 1), bool)), 1
            ),
            positions,
            n,
        )[:, ::-1],
        axis=1,
    )[:, ::-1]
    result = np.empty_like(x)
    np.put_along_axis(result, order, 1 + (start + end) / 2, axis=1)
    if normalize:
        count = valid.sum(1, keepdims=True)
        with np.errstate(all="ignore"):
            result = np.where(count == 1, 0.5, (result - 1) / (count - 1))
    return np.where(valid, result, np.nan)


def cs(
    x,
    op,
    normalize=False,
    ddof=0,
    lower=0.01,
    upper=0.99,
    scale=3.0,
    groups=None,
    exposures=(),
    weights=None,
    rcond=1e-12,
):
    import warnings

    if x.size == 0:
        return x.copy()
    a = np.where(np.isfinite(x), x, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if op == "rank":
            return rank(a, normalize)
        if op in ("zscore", "demean"):
            first = np.argmax(np.isfinite(a), axis=1)
            d = a - a[np.arange(len(a)), first, None]
            d = d - np.nanmean(d, axis=1, keepdims=True)
            if op == "demean":
                return d
            count = np.isfinite(d).sum(1, keepdims=True)
            return d / np.sqrt(np.nansum(d * d, axis=1, keepdims=True) / (count - ddof))
        if op == "scale":
            return a / np.nansum(np.abs(a), axis=1, keepdims=True)
        if op == "winsor_quantile":
            limits = np.nanquantile(a, [lower, upper], axis=1)
            return np.clip(a, limits[0, :, None], limits[1, :, None])
        if op == "winsor_mad":
            med = np.nanmedian(a, 1, keepdims=True)
            mad = np.nanmedian(abs(a - med), 1, keepdims=True)
            return np.clip(a, med - scale * mad, med + scale * mad)
    out = np.full_like(a, np.nan)
    for t, row in enumerate(a):
        good = np.isfinite(row)
        if groups is not None:
            good &= groups[t] >= 0
        for e in exposures:
            good &= np.isfinite(e[t])
        if weights is not None:
            good &= np.isfinite(weights[t]) & (weights[t] > 0)
        ids = np.flatnonzero(good)
        if not len(ids):
            continue
        if op == "group_demean":
            _, inv = np.unique(groups[t, ids], return_inverse=True)
            means = np.bincount(inv, weights=row[ids]) / np.bincount(inv)
            out[t, ids] = row[ids] - means[inv]
        elif op == "neutralize":
            design = [np.ones(len(ids))]
            if groups is not None:
                _, inv = np.unique(groups[t, ids], return_inverse=True)
                design.extend(np.eye(inv.max() + 1)[inv].T)
            design.extend(e[t, ids] for e in exposures)
            matrix = np.array(design).T
            w = np.ones(len(ids)) if weights is None else np.sqrt(weights[t, ids])
            aw = matrix * w[:, None]
            norms = np.linalg.norm(aw, axis=0)
            norms[norms == 0] = 1
            beta = np.linalg.lstsq(aw / norms, row[ids] * w, rcond=rcond)[0] / norms
            out[t, ids] = row[ids] - matrix @ beta
        else:
            raise ValueError(op)
    return out


def ic(x, y, rank_values=False):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    if rank_values:
        x, y = rank(x), rank(y)
    if x.shape[1] == 0:
        return np.full(len(x), np.nan)
    first = np.argmax(valid, axis=1)
    x = x - x[np.arange(len(x)), first, None]
    y = y - y[np.arange(len(y)), first, None]
    count = valid.sum(1, keepdims=True)
    with np.errstate(all="ignore"):
        dx = np.where(valid, x - np.nansum(x, 1, keepdims=True) / count, 0)
        dy = np.where(valid, y - np.nansum(y, 1, keepdims=True) / count, 0)
        return (dx * dy).sum(1) / np.sqrt((dx * dx).sum(1) * (dy * dy).sum(1))


def labels(scores, q):
    valid = np.isfinite(scores)
    order = np.argsort(np.where(valid, scores, np.inf), axis=1, kind="stable")
    ordinal = np.empty_like(order)
    np.put_along_axis(
        ordinal, order, np.broadcast_to(np.arange(scores.shape[1]), scores.shape), 1
    )
    count = np.maximum(valid.sum(1, keepdims=True), 1)
    return np.where(valid, ordinal * q // count, -1)


def group_returns(scores, returns, q=5, weights=None):
    g = labels(scores, q)
    valid = (g >= 0) & np.isfinite(returns)
    w = np.ones_like(scores) if weights is None else weights
    valid &= np.isfinite(w) & (w > 0)
    indices = (np.arange(len(scores))[:, None] * q + g)[valid]
    sums = np.bincount(
        indices, weights=(returns[valid] * w[valid]), minlength=len(scores) * q
    )
    denom = np.bincount(indices, weights=w[valid], minlength=len(scores) * q)
    with np.errstate(all="ignore"):
        return (sums / denom).reshape(len(scores), q)


def group_turnover(scores, q=5, periods=1):
    g = labels(scores, q)
    out = np.full((len(scores), q), np.nan)
    for i in range(q):
        now = g[periods:] == i
        old = g[:-periods] == i if periods else g == i
        with np.errstate(all="ignore"):
            out[periods:, i] = 1 - (now & old).sum(1) / now.sum(1)
    return out


def targets(scores, days, k, fraction):
    cols, weights, indptr = [], [], [0]
    for t in days:
        row = scores[t]
        ids = np.flatnonzero(np.isfinite(row))
        count = min(k, len(ids)) if k else int(np.ceil(len(ids) * fraction))
        if count:
            # Partition finds the cutoff; explicit tie handling preserves column order.
            cutoff = np.partition(row[ids], len(ids) - count)[len(ids) - count]
            selected = np.sort(
                np.concatenate(
                    (
                        ids[row[ids] > cutoff],
                        ids[row[ids] == cutoff][: count - np.sum(row[ids] > cutoff)],
                    )
                )
            )
            cols.extend(selected)
            weights.extend([1 / count] * count)
        indptr.append(len(cols))
    return (
        np.asarray(indptr, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(weights, dtype=float),
    )


def elementwise(op, *args, mask=None, lower=-1.0, upper=1.0):
    """Independent NumPy implementation, retaining scalar broadcasts."""
    operations = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
        "log": np.log,
        "abs": np.abs,
        "negate": np.negative,
        "gt": np.greater,
        "ge": np.greater_equal,
        "lt": np.less,
        "le": np.less_equal,
        "eq": np.equal,
        "ne": np.not_equal,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "sign": np.sign,
        "power": np.power,
        "sqrt": np.sqrt,
        "exp": np.exp,
    }
    with np.errstate(all="ignore"):
        if op == "where":
            result = np.where(
                np.isfinite(args[0]), np.where(args[0] != 0, args[1], args[2]), np.nan
            )
        elif op == "is_nan":
            result = np.asarray(~np.isfinite(args[0]), dtype=np.float64)
        elif op == "fill_nan":
            result = np.where(np.isfinite(args[0]), args[0], args[1])
        elif op == "clip":
            result = np.clip(args[0], lower, upper)
        elif op == "signed_power":
            result = np.sign(args[0]) * np.power(np.abs(args[0]), args[1])
        elif op in ("and_", "or_", "not_"):
            if op == "not_":
                result = args[0] == 0
            else:
                result = (np.logical_and if op == "and_" else np.logical_or)(
                    args[0] != 0, args[1] != 0
                )
            valid = np.isfinite(args[0])
            for a in args[1:]:
                valid = valid & np.isfinite(a)
            result = np.where(valid, result, np.nan)
        else:
            result = operations[op](*args)
            if op in ("gt", "ge", "lt", "le", "eq", "ne"):
                result = np.where(
                    np.isfinite(args[0]) & np.isfinite(args[1]), result, np.nan
                )
        if op not in ("where", "is_nan", "fill_nan"):
            valid = np.isfinite(args[0])
            for arg in args[1:]:
                valid = valid & np.isfinite(arg)
            result = np.where(valid, result, np.nan)
        result = np.asarray(result, dtype=np.float64)
        if result.ndim == 0:
            return float(result) if np.isfinite(result) else float("nan")
        result[~np.isfinite(result)] = np.nan
        if mask is not None:
            result[~mask] = np.nan
        return result
