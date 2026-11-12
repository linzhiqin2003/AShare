"""合成行情上的小回测，供报告测试用。"""

from __future__ import annotations

import numpy as np
import panelbt

from ashare.report import Benchmark, build_tearsheet


def business_days(start: str, n: int) -> list[str]:
    days = np.busday_offset(np.datetime64(start), np.arange(n), roll="forward")
    return [str(d) for d in days]


def make_case(
    T: int = 300,
    N: int = 8,
    *,
    seed: int = 0,
    every: int = 21,
    pick: int = 4,
    start: str = "2020-01-01",
    blocked: bool = True,
    empty_rows: tuple[int, ...] = (),
    pick_by_row=None,
    record: bool = True,
):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.015, size=(T, N))
    close = 10 * np.cumprod(1 + r, axis=0)
    open_ = close * (1 + rng.normal(0, 0.002, size=(T, N)))
    can_buy = np.ones((T, N), bool)
    can_sell = np.ones((T, N), bool)
    if blocked:
        can_buy[rng.random((T, N)) < 0.05] = False
        can_sell[rng.random((T, N)) < 0.05] = False
    market = panelbt.Market(open_, close, can_buy, can_sell)
    rows = list(range(1, T, every))
    days, ptr, cols, w = [], [0], [], []
    for k, d in enumerate(rows):
        if d in empty_rows:
            chosen = []
        else:
            n = pick_by_row(k) if pick_by_row else pick
            chosen = sorted(rng.choice(N, size=n, replace=False).tolist())
        days.append(d)
        cols.extend(chosen)
        w.extend([0.98 / max(1, len(chosen))] * len(chosen))
        ptr.append(len(cols))
    targets = panelbt.Targets(
        np.array(days), np.array(ptr), np.array(cols), np.array(w)
    )
    costs = panelbt.Costs(
        commission_rate=3e-4,
        commission_min=5.0,
        tax_sell=5e-4,
        fee_rate_buy=0.0,
        fee_rate_sell=0.0,
        slippage=5e-4,
    )
    config = panelbt.Config(
        initial_cash=1e6,
        min_qty=100,
        qty_step=100,
        record_trades=record,
        dividend_tax_rate=0.1,
    )
    result = panelbt.run(market, targets, costs, config)
    dates = business_days(start, T)
    symbols = [f"{600000 + i:06d}.SH" for i in range(N)]
    bench = 1000 * np.cumprod(1 + rng.normal(0.0002, 0.012, size=T))
    industry = np.tile(np.array([1, 1, 2, 2, 3, 3, 4, 0] * (N // 8 + 1))[:N], (T, 1))
    return dict(
        result=result,
        market=market,
        targets=targets,
        dates=dates,
        symbols=symbols,
        bench=bench,
        industry=industry,
        costs=costs,
        config=config,
    )


def build(case, *, benchmark=True, industry=True, **kw):
    args = dict(
        names={s: f"股票{i}" for i, s in enumerate(case["symbols"])},
        costs=case["costs"],
        config=case["config"],
    )
    if benchmark is True:
        args["benchmark"] = Benchmark(case["bench"], "基准指数", "price", "000000.SH")
    elif benchmark:
        args["benchmark"] = benchmark
    if industry:
        args["industry"] = case["industry"]
        args["industry_names"] = {1: "银行", 2: "电子", 3: "医药生物", 4: "汽车"}
    args.update(kw)
    return build_tearsheet(
        case["result"],
        case["market"],
        case["targets"],
        case["dates"],
        case["symbols"],
        **args,
    )
