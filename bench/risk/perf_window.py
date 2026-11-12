"""P-005 小基准：从头算模式每天取 1008×5923 残差窗口——把列表拼成新数组 对 环形缓冲视图。

两种做法在同一进程里按日交替，各 200 天；视图之后都过一遍 `portopt.ew_variance` 的入口检查，
计入它对新数组的首次触碰（缺页）代价。
"""

import json
import os
import time

import numpy as np
import portopt

from ashare.risk.window import ResidualWindow


def main(t=1300, n=5923, history=1008, measured=200):
    rng = np.random.default_rng(1)
    rows = rng.normal(0, 0.02, (t, n))
    ring, listed = ResidualWindow(n, history), []
    old, new = [], []
    for day in range(t):
        ring.push(day, rows[day])
        listed.append(rows[day].copy())
        if len(listed) > history:
            listed.pop(0)
        if day < t - measured:
            continue
        start = time.perf_counter()
        a = np.asarray(listed)
        portopt.ew_variance(a[-2:], 84.0)  # 入口检查只看最后两行，突出取窗口本身
        old.append(time.perf_counter() - start)
        start = time.perf_counter()
        b = ring.window(day - history + 1, day)
        portopt.ew_variance(b[-2:], 84.0)
        new.append(time.perf_counter() - start)
        assert np.array_equal(a, b)
    print(
        json.dumps(
            {
                "stack_ms": {
                    "min": min(old) * 1e3,
                    "median": float(np.median(old)) * 1e3,
                },
                "view_ms": {
                    "min": min(new) * 1e3,
                    "median": float(np.median(new)) * 1e3,
                },
                "load": os.getloadavg()[0],
            }
        )
    )


if __name__ == "__main__":
    main()
