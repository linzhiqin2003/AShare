"""把配置结果交给 `panelbt` 回测，并接到 `ashare.report`。

只有收益率表时用 `returns_market`：价格由收益连乘得到，成交价取**成交日收盘价**（信号日 t−1
收盘后决定、t 日收盘成交，比开盘成交多等半天，偏保守），份额不取整，不设涨跌停。
有真实行情时用 `ashare.data.fund_rules.fund_market` 等函数给出的 `Market`、`Config`、`Costs`。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Backtest(NamedTuple):
    result: object
    market: object
    targets: object
    config: object
    costs: object


def returns_market(returns):
    """T×N 日收益 → `panelbt.Market`（价格从 1 起连乘；收益缺失的日子不可交易）。"""
    import panelbt as bt

    r = np.asarray(
        returns.to_numpy(dtype=float) if hasattr(returns, "to_numpy") else returns,
        dtype=float,
    )
    valid = np.isfinite(r)
    price = np.cumprod(1.0 + np.where(valid, r, 0.0), axis=0)
    # 第一个有效收益之前的价格没有意义：不可交易、收盘价缺失。
    started = np.maximum.accumulate(valid, axis=0)
    started[0] = started[0] | valid[0]
    close = np.where(started, price, np.nan)
    tradable = valid & started
    return bt.Market(close, close, tradable, tradable)


def run(allocation, market=None, *, returns=None, costs=None, config=None):
    """回测一份配置。没给 market 时由 returns 建（`returns_market`），此时默认零费用、份额不取整。

    返回 `Backtest(result, market, targets, config, costs)`。
    """
    import panelbt as bt

    if market is None:
        if returns is None:
            raise ValueError("需要 market 或 returns")
        market = returns_market(returns)
        costs = costs or bt.Costs(
            commission_rate=0.0,
            commission_min=None,
            fee_rate_buy=0.0,
            fee_rate_sell=0.0,
            slippage=0.0,
        )
        n = market.shape[1]
        config = config or bt.Config(
            min_qty=np.zeros(n), qty_step=np.zeros(n), record_trades=True
        )
    costs = costs if costs is not None else bt.Costs()
    config = config if config is not None else bt.Config()
    if tuple(market.shape) != (allocation.n_rows, len(allocation.symbols)):
        raise ValueError(
            f"market 形状 {market.shape} 与配置 ({allocation.n_rows}, {len(allocation.symbols)}) 不一致"
        )
    if (
        allocation.weights.size
        and allocation.weights.sum(axis=1).max() > 1 + 1e-9
        and not config.margin.allow_financing
    ):
        raise ValueError(
            "权重和超过 1（加了杠杆）：现金账户不能执行，需要允许融资的保证金账户"
        )
    targets = allocation.targets()
    result = bt.run(market, targets, costs, config)
    return Backtest(result, market, targets, config, costs)


def tearsheet(
    allocation,
    market=None,
    *,
    returns=None,
    costs=None,
    config=None,
    title="资产配置",
    **kwargs,
):
    """回测并生成 `ashare.report.Tearsheet`（`print(ts)` 看摘要，`ts.save_json(path)` 存档）。"""
    from ashare.report import Tearsheet, build_tearsheet

    if config is not None and not config.record_trades:
        from dataclasses import replace

        config = replace(config, record_trades=True)
    bt_ = run(allocation, market, returns=returns, costs=costs, config=config)
    dates = [
        d if "-" in d else f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in allocation.all_dates
    ]
    data = build_tearsheet(
        bt_.result,
        bt_.market,
        bt_.targets,
        dates,
        allocation.symbols,
        title=title,
        costs=bt_.costs,
        config=bt_.config,
        **kwargs,
    )
    return Tearsheet(data)
