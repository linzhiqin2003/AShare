"""Slow executable specification, independent of the production kernels."""

import math

import numpy as np


def clean(x, mask=None):
    a = np.array(x, copy=True)
    a[~np.isfinite(a)] = np.nan
    if mask is not None:
        a[~mask] = np.nan
    return a


def ranks(v, normalize=False):
    r = np.array([1 + np.sum(v < a) + (np.sum(v == a) - 1) / 2 for a in v])
    return (
        np.full_like(r, 0.5)
        if normalize and len(v) == 1
        else ((r - 1) / (len(v) - 1) if normalize else r)
    )


def centered(v):
    """Direct shifted two-pass centering; exact constants have zero deviations."""
    d = v - v[0]
    return d - math.fsum(d) / len(d)


def ts(
    x,
    op,
    window=20,
    min_periods=None,
    y=None,
    mask=None,
    ddof=1,
    q=0.5,
    normalize=False,
    alpha=None,
    periods=1,
    limit=None,
    rows=None,
    ties="latest",
):
    """Evaluate cells directly; optional rows limits audit work, not input history."""
    a, b = clean(x, mask), None if y is None else clean(y, mask)
    out = np.full_like(a, np.nan)
    mp = window if min_periods is None else min_periods
    for j in range(a.shape[1]):
        state, seen, last = np.nan, 0, 0
        for t in range(len(a)) if rows is None or op in ("ema", "ffill") else rows:
            if op in ("delay", "delta"):
                if t >= periods:
                    out[t, j] = (
                        a[t - periods, j]
                        if op == "delay"
                        else a[t, j] - a[t - periods, j]
                    )
                continue
            if op == "ffill":
                if np.isfinite(a[t, j]):
                    state, last = a[t, j], t
                if limit is None or t - last <= limit:
                    out[t, j] = state
                continue
            if op == "ema":
                if np.isfinite(a[t, j]):
                    al = 2 / (window + 1) if alpha is None else alpha
                    state = a[t, j] if not seen else al * a[t, j] + (1 - al) * state
                    seen += 1
                if seen >= mp:
                    out[t, j] = state
                continue
            start = max(0, t - window + 1)
            idx = np.arange(start, t + 1)
            valid = np.isfinite(a[idx, j])
            if b is not None:
                valid &= np.isfinite(b[idx, j])
            idx = idx[valid]
            v = a[idx, j]
            c = len(v)
            if c < mp:
                continue
            mean = math.fsum(v) / c
            d = centered(v)
            var = math.fsum(d * d)
            if op == "sum":
                z = math.fsum(v)
            elif op == "mean":
                z = mean
            elif op in ("std", "var", "zscore"):
                variance = var / (c - ddof) if c > ddof else np.nan
                z = variance if op == "var" else np.sqrt(variance)
                if op == "zscore":
                    z = d[-1] / z if idx[-1] == t and z > 0 else np.nan
            elif op == "count":
                z = np.count_nonzero(v)
            elif op == "sumif":
                z = math.fsum(v[b[idx, j] != 0])
            elif op in ("min", "max"):
                z = getattr(np, op)(v)
            elif op in ("argmin", "argmax", "idxmin", "idxmax"):
                extreme = np.min(v) if op in ("argmin", "idxmin") else np.max(v)
                tied = np.flatnonzero(v == extreme)
                z = t - idx[tied[0] if ties == "earliest" else tied[-1]]
                if op.startswith("idx"):
                    z = window - z
            elif op in ("quantile", "median", "mad"):
                z = np.quantile(v, q if op == "quantile" else 0.5)
                if op == "mad":
                    z = np.median(np.abs(v - z))
            elif op == "rank":
                z = (
                    1 + np.sum(v < v[-1]) + (np.sum(v == v[-1]) - 1) / 2
                    if idx[-1] == t
                    else np.nan
                )
                if normalize:
                    z = (
                        (0.5 if c == 1 else (z - 1) / (c - 1))
                        if idx[-1] == t
                        else np.nan
                    )
            elif op == "product":
                z = v[0] if window == 1 else 0.0 if np.any(v == 0) else np.prod(v)
            elif op == "decay_linear":
                weights = idx - (t - window + 1) + 1
                z = math.fsum(v * weights) / sum(weights)
            elif op in ("skew", "kurt"):
                power = 3 if op == "skew" else 4
                z = (
                    (math.fsum(d**power) / c) / (var / c) ** (power / 2)
                    if c >= power and var > 0
                    else np.nan
                )
                if op == "kurt":
                    z -= 3
            else:
                predictor = idx.astype(float) if b is None else b[idx, j]
                dp = centered(predictor)
                pv, cross = math.fsum(dp * dp), math.fsum(d * dp)
                if op == "cov":
                    z = cross / (c - ddof) if c > ddof else np.nan
                elif op in ("corr", "rsquare"):
                    z = (
                        cross / math.sqrt(var) / math.sqrt(pv)
                        if c >= 2 and var > 0 and pv > 0
                        else np.nan
                    )
                    if op == "rsquare":
                        z = z * z
                else:
                    z = cross / pv if c >= 2 and pv > 0 else np.nan
                    if op.endswith("residual"):
                        z = (
                            (0.0 if c == 2 and pv > 0 else d[-1] - z * dp[-1])
                            if idx[-1] == t
                            else np.nan
                        )
            out[t, j] = z
    if mask is not None:
        out[~mask] = np.nan
    return out


def cs(
    x,
    op,
    mask=None,
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
    """Direct per-date cross-sectional specification."""
    a = clean(x, mask)
    out = np.full_like(a, np.nan)
    for t, row in enumerate(a):
        valid = np.isfinite(row)
        if groups is not None:
            valid &= groups[t] >= 0
        for exposure in exposures:
            valid &= np.isfinite(exposure[t])
        if weights is not None:
            valid &= np.isfinite(weights[t]) & (weights[t] > 0)
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        v = row[ids]
        if op == "rank":
            z = ranks(v, normalize)
        elif op in ("zscore", "demean", "scale"):
            d = centered(v)
            sd = (
                math.sqrt(math.fsum(d * d) / (len(v) - ddof))
                if len(v) > ddof
                else np.nan
            )
            if op == "demean":
                z = d
            elif op == "scale":
                absolute = math.fsum(np.abs(v))
                z = v / absolute if absolute > 0 else np.full_like(v, np.nan)
            else:
                z = d / sd if sd > 0 else np.full_like(v, np.nan)
        elif op == "winsor_quantile":
            z = np.clip(v, np.quantile(v, lower), np.quantile(v, upper))
        elif op == "winsor_mad":
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            z = np.clip(v, med - scale * mad, med + scale * mad)
        elif op == "group_demean":
            z = np.array(
                [
                    v[k] - np.mean(v[groups[t, ids] == groups[t, j]])
                    for k, j in enumerate(ids)
                ]
            )
        elif op == "neutralize":
            design = [np.ones(len(v))]
            if groups is not None:
                design.extend(
                    (groups[t, ids] == g).astype(float)
                    for g in np.unique(groups[t, ids])
                )
            design.extend(e[t, ids] for e in exposures)
            design = np.array(design).T
            w = np.ones(len(v)) if weights is None else np.sqrt(weights[t, ids])
            weighted = design * w[:, None]
            norms = np.linalg.norm(weighted, axis=0)
            norms[norms == 0] = 1
            coef = np.linalg.lstsq(weighted / norms, v * w, rcond=rcond)[0] / norms
            z = v - design @ coef
        else:
            raise ValueError(op)
        out[t, ids] = z
    return out


def ic(x, y, mask=None, rank=False):
    a, b = clean(x, mask), clean(y, mask)
    out = np.full(len(a), np.nan)
    for t in range(len(a)):
        good = np.isfinite(a[t]) & np.isfinite(b[t])
        v, w = a[t, good], b[t, good]
        if len(v) < 2:
            continue
        if rank:
            v, w = ranks(v), ranks(w)
        v, w = centered(v), centered(w)
        vx, vy = math.fsum(v * v), math.fsum(w * w)
        if vx > 0 and vy > 0:
            out[t] = math.fsum(v * w) / math.sqrt(vx) / math.sqrt(vy)
    return out


def memberships(row, q):
    ids = sorted(np.flatnonzero(np.isfinite(row)), key=lambda j: (row[j], j))
    out = [set() for _ in range(q)]
    for i, j in enumerate(ids):
        out[i * q // len(ids)].add(j)
    return out


def group_returns(scores, returns, q=5, weights=None, mask=None):
    a, b = clean(scores, mask), clean(returns, mask)
    out = np.full((len(a), q), np.nan)
    for t in range(len(a)):
        for g, members in enumerate(memberships(a[t], q)):
            ids = sorted(
                j
                for j in members
                if np.isfinite(b[t, j])
                and (
                    weights is None
                    or (np.isfinite(weights[t, j]) and weights[t, j] > 0)
                )
            )
            if ids:
                w = np.ones(len(ids)) if weights is None else weights[t, ids]
                out[t, g] = np.average(b[t, ids], weights=w)
    return out


def group_turnover(scores, q=5, periods=1, mask=None):
    a = clean(scores, mask)
    out = np.full((len(a), q), np.nan)
    for t in range(periods, len(a)):
        now, old = memberships(a[t], q), memberships(a[t - periods], q)
        for g in range(q):
            if now[g]:
                out[t, g] = 1 - len(now[g] & old[g]) / len(now[g])
    return out


def targets(scores, k=None, fraction=0.2, days=None, mask=None):
    a = clean(scores, mask)
    days = np.arange(len(a)) if days is None else days
    cols, weights, indptr = [], [], [0]
    for t in days:
        ids = sorted(np.flatnonzero(np.isfinite(a[t])), key=lambda j: (-a[t, j], j))
        count = min(k, len(ids)) if k is not None else math.ceil(len(ids) * fraction)
        selected = sorted(ids[:count])
        cols.extend(selected)
        if count:
            weights.extend([1 / count] * count)
        indptr.append(len(cols))
    return (
        np.asarray(days, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def rank_ic(x, y, mask=None):
    """Spearman reference, ranking only pairwise-complete observations."""
    return ic(x, y, mask=mask, rank=True)


def factor_autocorr(x, periods=1, mask=None, rank=True):
    """Explicit lag followed by pairwise cross-sectional correlation."""
    lag = ts(x, "delay", periods=periods, mask=mask)
    return ic(x, lag, mask=mask, rank=rank)


def elementwise(op, *args, mask=None, lower=-1.0, upper=1.0):
    """Direct scalar specification; it does not call an optimized backend."""
    import math
    import operator

    operations = {
        "gt": operator.gt,
        "ge": operator.ge,
        "lt": operator.lt,
        "le": operator.le,
        "eq": operator.eq,
        "ne": operator.ne,
        "maximum": max,
        "minimum": min,
        "power": math.pow,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
        "log": math.log,
        "abs": abs,
        "negate": operator.neg,
    }
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
    out = np.empty(arrays[0].shape, dtype=float)
    for index in np.ndindex(out.shape):
        if mask is not None and not mask[index]:
            out[index] = np.nan
            continue
        values = [float(a[index]) for a in arrays]
        a = values[0]
        try:
            if op == "is_nan":
                z = float(not math.isfinite(a))
            elif op == "fill_nan":
                z = a if math.isfinite(a) else values[1]
            elif op == "where":
                z = values[1] if a != 0 else values[2]
                z = z if math.isfinite(a) else math.nan
            elif not all(math.isfinite(v) for v in values):
                z = math.nan
            elif op == "and_":
                z = float(a != 0 and values[1] != 0)
            elif op == "or_":
                z = float(a != 0 or values[1] != 0)
            elif op == "not_":
                z = float(a == 0)
            elif op == "sign":
                z = float((a > 0) - (a < 0))
            elif op == "signed_power":
                z = ((a > 0) - (a < 0)) * math.pow(abs(a), values[1])
            elif op == "clip":
                z = min(upper, max(lower, a))
            else:
                z = operations[op](*values)
        except (ValueError, ZeroDivisionError, OverflowError):
            z = math.nan
        out[index] = z if math.isfinite(z) else math.nan
    return float(out) if out.ndim == 0 else out
