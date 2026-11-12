"""Whole-panel Polars fixed-size Array expressions; no per-day Python loop."""

import numpy as np
import polars as pl
from common import FEE


def prepare(market, target):
    p = market.exec_price
    t, n = p.shape
    days = target.days
    start = days[np.searchsorted(days, np.arange(t), side="right") - 1]
    w = target.to_dense(t, n)[start]
    old = np.vstack((np.zeros((1, n)), w[:-1]))
    old_start = np.r_[0, start[:-1]]
    # Fixed-size arrays keep the matrix layout without thousands of expression
    # columns or a costly long-table group_by/join. Polars owns the arithmetic.
    frame = pl.DataFrame(
        {
            "p": p,
            "prior": np.vstack((p[0], p[:-1])),
            "base": p[old_start],
            "old": old,
            "target": w,
            "rebalance": np.isin(np.arange(t), days),
        }
    ).lazy()
    return frame


def execute(frame):
    frame = frame.with_columns(
        (pl.col("old") * pl.col("p") / pl.col("base")).alias("today"),
        (pl.col("old") * pl.col("prior") / pl.col("base")).alias("yesterday"),
    ).with_columns(
        (1 - pl.col("old").arr.sum() + pl.col("today").arr.sum()).alias("gross"),
        (1 - pl.col("old").arr.sum() + pl.col("yesterday").arr.sum()).alias("previous"),
    )
    turnover = (
        (pl.col("target") - pl.col("today") / pl.col("gross"))
        .arr.eval(pl.element().abs())
        .arr.sum()
    )
    fee = pl.when(pl.col("rebalance")).then(turnover * FEE).otherwise(0)
    return (
        frame.select(
            (pl.col("gross") / pl.col("previous") * (1 - fee)).cum_prod().alias("nav")
        )
        .collect()["nav"]
        .to_numpy()
    )


def run(market, target):
    return execute(prepare(market, target))
