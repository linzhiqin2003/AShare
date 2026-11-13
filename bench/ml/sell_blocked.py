"""「卖出日卖不出」两种标签口径在真实数据上差多少：默认 price（按当天成交价算）对比
carry（顺延到第一个可卖日，最多 10 天）。

只统计 2012 年起、默认股票池内、ret5 标签。结果写进 bench/ml/results/sell_blocked.json。
用法：.venv/bin/python bench/ml/sell_blocked.py
"""

from __future__ import annotations

import sys

import numpy as np
from mlbench import DATA, log, save

import ashare.data as ad
import ashare.ml as ml
from ashare.ml import labels as L

START = "20120101"
MAX_CARRY = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def main():
    data = ad.open(root=DATA)
    price = ml.label("ret5")
    carry = ml.label("ret5", sell_blocked="carry", max_carry=MAX_CARRY)
    fields = sorted(set(price.fields()) | set(ml.walk.UNIVERSE_FIELDS))
    panel = data.panel(fields)
    universe = ml.default_universe(panel)
    a = L.compute(panel, price, universe)
    b = L.compute(panel, carry, universe)
    dates = np.asarray(panel.dates).astype(str)
    first = int(np.searchsorted(dates, START))
    last = len(dates) - carry.span  # 两种口径都有完整窗口的信号日
    rows = slice(first, last)
    _, can_sell = L.tradable(panel, "open")
    hold = price.lag + price.horizon
    blocked = np.zeros_like(universe)
    blocked[: len(dates) - hold] = ~can_sell[hold:]
    ra, rb = a.raw[rows], b.raw[rows]
    base = np.isfinite(ra)  # price 口径有标签的样本
    hit = base & blocked[rows]  # 其中卖出日不可卖
    both = hit & np.isfinite(rb)
    diff = rb[both] - ra[both]
    signal = np.arange(first, last)[:, None]
    days = (b.end[rows] - (signal + hold))[both]  # 顺延了几天
    whole = np.where(both, rb - ra, 0.0)[base]
    out = {
        "samples": int(base.sum()),
        "blocked_share": float(hit.sum() / base.sum()),
        "carried_and_sold": int(both.sum()),
        "unsold_after_carry": int((hit & ~np.isfinite(rb)).sum()),
        "mean_diff_blocked": float(diff.mean()),
        "median_diff_blocked": float(np.median(diff)),
        "carry_days_median": float(np.median(days)),
        "carry_days_p90": float(np.percentile(days, 90)),
        "mean_diff_all_bp": float(whole.mean() * 1e4),
        "price_nan_but_carry_finite": int((~base & np.isfinite(rb)).sum()),
        "start": START,
        "max_carry": MAX_CARRY,
    }
    log("sell_blocked", out)
    save("sell_blocked" if MAX_CARRY == 10 else f"sell_blocked_{MAX_CARRY}", out)


if __name__ == "__main__":
    main()
