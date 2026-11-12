"""事件模块的性能基线：全市场全历史面板上几十万条事件的对齐、目标生成和事件研究。

    python -m ashare.events.bench --root /path/to/data --repeat 3

真实事件（业绩预告全部、业绩快报、分红两种进度日）约 25 万条；另把它们复制成
约 100 万条（日期不变、证券随机换）测放大后的耗时。每项重复 ``repeat`` 次取中位数。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from . import extract
from .market import load
from .study import event_study, group_benchmark, size_industry_groups
from .table import align
from .targets import to_targets


def _median(fn, repeat):
    times, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def _events(ds):
    cols = ["symbol", "date", "kind", "timing"]
    frames = [
        extract.forecasts(ds).select(*cols, pl.col("change_mid").alias("value")),
        extract.expresses(ds).select(*cols, pl.col("yoy").alias("value")),
        extract.dividends(ds, "resolution").select(
            *cols, pl.col("cash_per_share").alias("value")
        ),
        extract.dividends(ds, "report").select(
            *cols, pl.col("cash_per_share").alias("value")
        ),
    ]
    return pl.concat(frames)


def run(ds, repeat=3):
    out = {}
    out["load_seconds"], m = _median(lambda: load(ds), 1)
    t, n = m.returns.shape
    out["panel"] = [t, n]
    out["extract_seconds"], real = _median(lambda: _events(ds), 1)
    rng = np.random.default_rng(0)
    big = pl.concat(
        [real]
        + [
            real.with_columns(
                pl.Series("symbol", rng.choice(m.symbols, real.height).astype(str))
            )
            for _ in range(3)
        ]
    )
    out["group_benchmark_seconds"], gb = _median(
        lambda: group_benchmark(
            m.returns, size_industry_groups(m["industry"], m["total_market_cap"])
        ),
        repeat,
    )
    for label, ev in (("real", real), ("x4", big)):
        res = {"events": ev.height}
        res["align"], aligned = _median(
            lambda ev=ev: align(ev, m.dates, m.symbols), repeat
        )
        res["aligned"] = len(aligned)
        res["study_build"], st = _median(
            lambda a=aligned: event_study(
                a, m.returns, m.market_return, m.dates, m.symbols, window=(-20, 60)
            ),
            repeat,
        )
        res["study_test_1000_boot"], _ = _median(lambda st=st: st.test(1, 20), repeat)
        for name, kw in (
            ("targets_no_cap", {}),
            ("targets_cap50", {"max_positions": 50}),
            (
                "targets_value_cap50",
                {"max_positions": 50, "weight": "value", "value": "value"},
            ),
        ):
            a = aligned
            if kw.get("weight") == "value":
                a = align(ev.filter(pl.col("value") > 0), m.dates, m.symbols)
            res[name], plan = _median(
                lambda a=a, kw=kw: to_targets(
                    a, m.dates, m.symbols, m["can_buy_open"], holding=20, **kw
                ),
                repeat,
            )
            res[name + "_rows"] = len(plan.targets.days)
        out[label] = res
    return out


def main(argv=None):
    import os

    import ashare.data as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv)
    res = run(ad.open(root=args.root), args.repeat)
    res["loadavg"] = os.getloadavg()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
