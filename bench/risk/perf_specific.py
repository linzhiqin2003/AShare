"""P-006 小基准：每天三个特质方差，从头算（含把历史拼成 T×N 数组）对分块累加。

合成残差 1300×5923（缺失 10%），窗口 1008 天，在同一进程里两种做法按日交替计时。
"""

import json
import os
import time

import numpy as np

from ashare.risk.fundamental import FundamentalConfig
from ashare.risk.specific import SpecificMoments, direct_variances


def main(t=1300, n=5923, measured=120, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, (t, n))
    x[rng.random((t, n)) < 0.1] = np.nan
    c = FundamentalConfig()
    moments = SpecificMoments(n, c)
    history = []
    old, new = [], []
    for day in range(t):
        moments.push(day, x[day])
        history.append(x[day])
        if len(history) > c.history:
            history.pop(0)
        if day < t - measured:
            continue
        start = time.perf_counter()
        rs = np.asarray(history)
        direct = direct_variances(rs, c)
        np.isfinite(rs).sum(axis=0)
        old.append(time.perf_counter() - start)
        start = time.perf_counter()
        got = moments.variances(day)
        new.append(time.perf_counter() - start)
        assert all(
            np.max(np.abs(a - b) / np.where(b != 0, b, 1)) <= 1e-12
            for a, b in zip(got[:3], direct, strict=True)
        )
    report = {
        name: {"min_ms": min(v) * 1e3, "median_ms": float(np.median(v)) * 1e3}
        for name, v in (("direct", old), ("blocks", new))
    }
    report["load"] = os.getloadavg()[0]
    report["fallback_columns"] = moments.fallback_columns
    print(json.dumps(report))


if __name__ == "__main__":
    main()
