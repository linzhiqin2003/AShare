"""测试用的合成面板：上市前后空行、停牌、并列价格、复权因子跳变、行业缺失、上市首日。

外部因子包的测试也可以用：``from ashare.factor.formula.testing import synthetic_source``。
"""

import numpy as np

from .inputs import PublicSource


def synthetic_source(t=160, n=14, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.array([f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(t)])
    symbols = np.array([f"{600000 + j}.SH" for j in range(n)])
    listed = np.ones((t, n), bool)
    listed[:30, 1] = False  # 中途上市
    listed[:75, 2] = False
    listed[120:, 3] = False  # 退市
    listed[:, 4] = False  # 从未上市
    quote = listed.copy()
    for j in range(n):  # 随机停牌，含连续多日
        for start in rng.integers(0, t, 3):
            quote[start : start + rng.integers(1, 9), j] = False
    steps = rng.normal(0, 0.02, (t, n))
    steps[:, 5] = 0.0  # 长期不动的价格：常数窗口
    steps[60:90, 6] = 0.0
    close = 10 * np.exp(np.cumsum(steps, axis=0))
    close = np.round(close, 2)  # 两位小数制造并列
    close[:, 7] = np.round(close[:, 7], 0)
    high = np.round(close * (1 + rng.uniform(0, 0.03, (t, n))), 2)
    low = np.round(close * (1 - rng.uniform(0, 0.03, (t, n))), 2)
    open_ = np.round(low + (high - low) * rng.uniform(0, 1, (t, n)), 2)
    volume = np.round(rng.lognormal(12, 1, (t, n)), -2)
    volume[:, 8] = 1e6  # 常数成交量
    amount = volume * (low + high) / 2
    adj = np.cumprod(np.where(rng.uniform(0, 1, (t, n)) < 0.02, 1.1, 1.0), axis=0)
    previous = np.vstack([np.full((1, n), np.nan), close[:-1] * adj[:-1] / adj[1:]])
    # 上市天数（上市当日为 1）：第 1、2 只在样本内上市，第 10 只样本第一天上市，
    # 其余在样本开始前早已上市。上市首日的前收盘是发行价。
    age = np.cumsum(listed, axis=0).astype(np.uint16)
    early = listed[0] & (np.arange(n) != 10)
    age[:, early] += 299
    age = np.where(listed, age, 0).astype(np.uint16)
    first = age == 1
    previous = np.where(first, np.round(close * 0.7, 2), previous)
    industry = rng.choice([801010, 801030, 801040], (t, n)).astype(np.uint32)
    industry[:, 9] = 0
    cap = close * 1e8
    arrays = {}
    for name, values in [
        ("open", open_),
        ("high", high),
        ("low", low),
        ("close", close),
        ("volume", volume),
        ("amount", amount),
        ("previous_close", previous),
        ("total_market_cap", cap),
    ]:
        arrays[name] = np.where(quote, values, np.nan)
    arrays["vwap"] = np.where(quote, amount / volume, np.nan)
    arrays["adj_factor"] = np.where(listed, adj, np.nan)
    arrays["is_listed"] = listed
    arrays["listing_age"] = age
    arrays["industry"] = np.where(listed, industry, 0).astype(np.uint32)
    bench = 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, t)))
    bench_open = bench * (1 + rng.normal(0, 0.003, t))
    bench[5] = np.nan
    return PublicSource(
        dates, symbols, arrays, f"synthetic-{seed}", "TEST", bench_open, bench
    )
