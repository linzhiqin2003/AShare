"""Native pandas, Polars and bottleneck cross-sectional competitors."""

import numpy as np


def pandas_op(x, op, y=None, groups=None):
    import pandas as pd

    a = pd.DataFrame(x)
    if op == "cs_rank":
        return a.rank(axis=1).to_numpy()
    if op == "cs_zscore":
        return (
            a.sub(a.mean(axis=1), axis=0).div(a.std(axis=1, ddof=0), axis=0).to_numpy()
        )
    if op == "cs_winsor_quantile":
        return a.clip(
            lower=a.quantile(0.01, axis=1), upper=a.quantile(0.99, axis=1), axis=0
        ).to_numpy()
    if op == "cs_winsor_mad":
        med = a.median(axis=1)
        mad = a.sub(med, axis=0).abs().median(axis=1)
        return a.clip(lower=med - 3 * mad, upper=med + 3 * mad, axis=0).to_numpy()
    if op in ("ic", "rank_ic", "factor_autocorr"):
        b = a.shift(1) if op == "factor_autocorr" else pd.DataFrame(y)
        valid = a.notna() & b.notna()
        a, b = a.where(valid), b.where(valid)
        if op != "ic":
            a, b = a.rank(axis=1), b.rank(axis=1)
        return a.corrwith(b, axis=1).to_numpy()
    if op == "targets":
        count = np.ceil(a.count(axis=1) * 0.2)
        selected = a.rank(axis=1, ascending=False, method="first").le(count, axis=0)
        rows, cols = np.nonzero(selected.to_numpy())
        counts = np.bincount(rows, minlength=len(x))
        return np.r_[0, counts.cumsum()], cols, 1 / counts[rows]
    labels = None
    if op != "cs_group_demean":
        labels = (
            (a.rank(axis=1, method="first") - 1)
            .mul(5)
            .div(a.count(axis=1), axis=0)
            .apply(np.floor)
        )
    if op == "group_turnover":
        out = []
        for i in range(5):
            now = labels == i
            denom = now.sum(axis=1)
            out.append(1 - (now & now.shift(1, fill_value=False)).sum(axis=1) / denom)
        result = pd.concat(out, axis=1).to_numpy()
        result[0] = np.nan
        return result
    frame = pd.DataFrame(
        {
            "day": np.repeat(np.arange(len(x)), x.shape[1]),
            "value": x.ravel(),
            "group": groups.ravel()
            if op == "cs_group_demean"
            else labels.to_numpy().ravel(),
        }
    )
    if op == "cs_group_demean":
        frame.loc[frame["group"] < 0, "value"] = np.nan
        z = frame["value"] - frame.groupby(["day", "group"])["value"].transform("mean")
        return z.to_numpy().reshape(x.shape)
    if op == "group_returns":
        frame["value"] = y.ravel()
        return (
            frame.groupby(["day", "group"])["value"]
            .mean()
            .unstack()
            .reindex(index=range(len(x)), columns=range(5))
            .to_numpy()
        )
    raise NotImplementedError(op)


def polars_op(x, op, y=None, groups=None):
    import polars as pl

    t, n = x.shape
    data = {"day": np.repeat(np.arange(t), n), "x": x.ravel()}
    if y is not None:
        data["y"] = y.ravel()
    if groups is not None:
        data["group"] = groups.ravel()
    frame = pl.DataFrame(data).with_columns(pl.col("x").fill_nan(None))
    a = pl.col("x")
    if op == "cs_rank":
        result = frame.select(a.rank("average").over("day"))
    elif op == "cs_zscore":
        result = frame.select((a - a.mean().over("day")) / a.std(ddof=0).over("day"))
    elif op == "cs_winsor_quantile":
        result = frame.select(
            a.clip(
                a.quantile(0.01, interpolation="linear").over("day"),
                a.quantile(0.99, interpolation="linear").over("day"),
            )
        )
    elif op == "cs_winsor_mad":
        frame = frame.with_columns(a.median().over("day").alias("med"))
        frame = frame.with_columns(
            (a - pl.col("med")).abs().median().over("day").alias("mad")
        )
        result = frame.select(
            a.clip(pl.col("med") - 3 * pl.col("mad"), pl.col("med") + 3 * pl.col("mad"))
        )
    elif op == "cs_group_demean":
        frame = frame.with_columns(
            pl.when(pl.col("group") >= 0).then(a).otherwise(None).alias("x")
        )
        result = frame.select(a - a.mean().over(["day", "group"]))
    elif op in ("ic", "rank_ic", "factor_autocorr"):
        if op == "factor_autocorr":
            frame = frame.with_columns(a.shift(n).alias("y"))
        frame = frame.with_columns(pl.col("y").fill_nan(None))
        paired = a.is_not_null() & pl.col("y").is_not_null()
        frame = frame.with_columns(
            pl.when(paired).then(pl.col(c)).otherwise(None).alias(c) for c in ["x", "y"]
        )
        if op != "ic":
            frame = frame.with_columns(
                pl.col(c).rank("average").over("day").alias(c) for c in ["x", "y"]
            )
        return (
            frame.group_by("day")
            .agg(pl.corr("x", "y").alias("value"))
            .sort("day")["value"]
            .to_numpy()
        )
    elif op in ("group_returns", "group_turnover", "targets"):
        descending = op == "targets"
        frame = frame.with_columns(
            a.rank("ordinal", descending=descending).over("day").alias("rank"),
            a.count().over("day").alias("count"),
        )
        if op == "targets":
            selected = (
                frame.with_row_index("index")
                .filter(pl.col("rank") <= (pl.col("count") * 0.2).ceil())
                .select("index")
            )
            indices = selected["index"].to_numpy()
            rows, cols = indices // n, indices % n
            counts = np.bincount(rows, minlength=t)
            return np.r_[0, counts.cumsum()], cols, 1 / counts[rows]
        frame = frame.with_columns(
            ((pl.col("rank") - 1) * 5 / pl.col("count")).floor().alias("bucket")
        )
        if op == "group_returns":
            frame = frame.with_columns(pl.col("y").fill_nan(None))
            frame = (
                frame.group_by(["day", "bucket"])
                .agg(pl.col("y").mean().alias("value"))
                .filter(pl.col("bucket").is_not_null())
            )
        else:
            frame = frame.with_columns(pl.col("bucket").shift(n).alias("previous"))
            frame = (
                frame.group_by(["day", "bucket"])
                .agg(
                    (
                        1
                        - (pl.col("bucket") == pl.col("previous"))
                        .fill_null(False)
                        .mean()
                    ).alias("value")
                )
                .filter(pl.col("bucket").is_not_null())
            )
        out = np.full((t, 5), np.nan)
        out[frame["day"].to_numpy(), frame["bucket"].to_numpy().astype(int)] = frame[
            "value"
        ].to_numpy()
        if op == "group_turnover":
            out[0] = np.nan
        return out
    else:
        raise NotImplementedError(op)
    return result.to_numpy().reshape(t, n)


def bottleneck_op(x, op, y=None):
    import bottleneck as bn

    if op == "cs_rank":
        return bn.nanrankdata(x, axis=1)
    if op == "cs_zscore":
        return (x - bn.nanmean(x, axis=1)[:, None]) / bn.nanstd(x, axis=1)[:, None]
    if op == "cs_winsor_mad":
        med = bn.nanmedian(x, axis=1)[:, None]
        mad = bn.nanmedian(abs(x - med), axis=1)[:, None]
        return np.clip(x, med - 3 * mad, med + 3 * mad)
    if op in ("ic", "rank_ic"):
        good = np.isfinite(x) & np.isfinite(y)
        a, b = np.where(good, x, np.nan), np.where(good, y, np.nan)
        if op == "rank_ic":
            a, b = bn.nanrankdata(a, axis=1), bn.nanrankdata(b, axis=1)
        a = a - bn.nanmean(a, axis=1)[:, None]
        b = b - bn.nanmean(b, axis=1)[:, None]
        return bn.nansum(a * b, axis=1) / np.sqrt(
            bn.nansum(a * a, axis=1) * bn.nansum(b * b, axis=1)
        )
    raise NotImplementedError(op)
