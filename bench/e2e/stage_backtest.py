"""阶段 6：回测——真实面板行情与公司行为、完整 A 股规则；单次与 200 组批量，带检查与 validate=False。

信号在收盘算出，下一交易日开盘价成交（错位由 ashare.backtest.execution_targets 做，panelbt 不移位）；
行情与整手规则走 ashare.backtest 的正式入口。印花税
2008-09-19 之后用 panelbt 的 stamp_tax_sell，之前按卖出 0.3%/0.1% 的历史税率近似。
"""

import numpy as np
import panelbt
import panelops
from common import digest, measure
from panelbt.markets.ashare import lot_rules, stamp_tax_sell

import ashare.data
from ashare.backtest import boards, execution_targets, market_inputs, required_fields
from ashare.factor import Evaluator, default_registry

REBALANCE_EVERY = 21
BATCH = 200


def panel_fields(data):
    """开盘成交需要的字段（面板有 can_buy_open/can_sell_open 时直接用）。"""
    masks = {"can_buy_open", "can_sell_open"} <= set(data.fields())
    return list(required_fields("open", masks=masks))


def costs(dates):
    """剖析固定用的近似费率（不随费率表改动，保证与历史基线可比）。

    2008-09-19 前按卖出单边 0.3%/0.1% 近似，忽略买方印花税和历史过户费；
    真实历史费率见 ashare.backtest.BacktestMarket.costs / panelbt.markets.ASHARE。
    """
    days = np.asarray(
        [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates], dtype="datetime64[D]"
    )
    tax = np.where(days < np.datetime64("2008-04-24"), 0.003, 0.001)
    late = days >= np.datetime64("2008-09-19")
    tax[late] = stamp_tax_sell(days[late])
    return panelbt.Costs(tax_sell=tax)


def config(data, symbols):
    """整手规则按证券表的板块列（ashare.backtest.boards）。"""
    data = ashare.data.open(data) if not hasattr(data, "securities") else data
    min_qty, step = lot_rules(boards(data.securities(), symbols))
    return panelbt.Config(min_qty=min_qty, qty_step=step)


def plans(score, t, validate=True, count=1):
    """count 组：调仓相位 0..9 × 持仓比例 20 档；成交日由 ashare.backtest 错位一天。"""
    out = []
    fractions = np.linspace(0.02, 0.4, 20)
    for i in range(count):
        phase, fraction = i // 20 % 10, fractions[i % 20] if count > 1 else 0.1
        days = np.arange(252 + phase, t - 1, REBALANCE_EVERY, dtype=np.int64)
        sparse = panelops.targets(score, fraction=float(fraction), days=days)
        out.append(execution_targets(sparse, t=t, tail="error", validate=validate))
    return out


def run(ctx):
    panelops.set_num_threads(ctx["threads"])
    data = ashare.data.open(ctx["data"])
    panel = data.panel(panel_fields(data) + ["adj_factor"])
    t, n = panel.shape
    stats, inputs = measure(lambda: market_inputs(panel), 1)
    cases = [{"case": "adapter_inputs", "scale": {"days": t, "symbols": n}, **stats}]
    cost, cfg = costs(panel.dates), config(data, panel.symbols)
    score = np.ascontiguousarray(
        Evaluator().evaluate(default_registry()["reversal_20"].expression, panel)
    )
    scale = {"days": t, "symbols": n, "rebalance_every": REBALANCE_EVERY}
    for validate in (True, False):
        label = "validate" if validate else "novalidate"
        stats, market = measure(
            lambda v=validate: panelbt.Market(**inputs, validate=v), ctx["repeat"](3)
        )
        cases.append({"case": f"market_build_{label}", "scale": scale, **stats})
        single = plans(score, t, validate)[0]
        stats, result = measure(
            lambda m=market, s=single: panelbt.run(m, s, cost, cfg), ctx["repeat"](5)
        )
        cases.append(
            {
                "case": f"run_single_{label}",
                "scale": {**scale, "rebalances": len(single.days)},
                **stats,
                "summary": {
                    "nav": digest(result.nav),
                    "final_nav": float(result.nav[-1]),
                },
            }
        )
        stats, batch = measure(lambda v=validate: plans(score, t, v, BATCH), 1)
        cases.append(
            {"case": f"targets_build_{BATCH}_{label}", "scale": scale, **stats}
        )
        stats, result = measure(
            lambda m=market, b=batch: panelbt.run_batch(
                m, b, cost, cfg, threads=ctx["threads"]
            ),
            ctx["repeat"](2),
        )
        cases.append(
            {
                "case": f"run_batch_{BATCH}_{label}",
                "scale": {**scale, "strategies": BATCH},
                **stats,
                "summary": {"nav": digest(result.nav)},
            }
        )
        del market, batch
    return cases
