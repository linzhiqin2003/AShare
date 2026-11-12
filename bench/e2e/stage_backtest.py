"""阶段 6：回测——真实面板行情与公司行为、完整 A 股规则；单次与 200 组批量，带检查与 validate=False。

信号在收盘算出，下一交易日开盘价成交（移位在这里做，panelbt 不移位）。印花税
2008-09-19 之后用 panelbt 的 stamp_tax_sell，之前按卖出 0.3%/0.1% 的历史税率近似。
"""

import re

import numpy as np
import panelbt
import panelops
from common import digest, measure
from panelbt.markets.ashare import Board, lot_rules, stamp_tax_sell, tradability

import ashare.data
from ashare.factor import Evaluator, default_registry

FIELDS = (
    "open",
    "close",
    "high",
    "low",
    "upper_limit",
    "lower_limit",
    "is_suspended",
    "has_quote",
    "volume",
    "split",
    "cash_dividend",
    "delist",
)
REBALANCE_EVERY = 21
BATCH = 200


# 数据层建面板时已算好的开盘可买、可卖掩码（P-016）；旧面板没有时现算，口径相同。
MASKS = ("can_buy_open", "can_sell_open")


def panel_fields(data):
    return list(FIELDS) + [m for m in MASKS if m in data.fields()]


def market_inputs(panel):
    if all(m in panel._arrays for m in MASKS):
        can_buy, can_sell = (np.asarray(panel[m]) for m in MASKS)
        p = {name: np.asarray(panel[name]) for name in FIELDS[:2] + FIELDS[8:]}
    else:
        p = {name: np.asarray(panel[name]) for name in FIELDS}
        can_buy, can_sell = tradability(
            p["open"],
            p["high"],
            p["low"],
            p["upper_limit"],
            p["lower_limit"],
            p["is_suspended"] | ~p["has_quote"],
        )
    return {
        "exec_price": p["open"],
        "close": p["close"],
        "can_buy": can_buy,
        "can_sell": can_sell,
        "volume": p["volume"],
        "split": np.where(np.isnan(p["split"]), 1.0, p["split"]),
        "cash_div": np.nan_to_num(p["cash_dividend"]),
        "delist": p["delist"],
    }


def costs(dates):
    days = np.asarray(
        [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates], dtype="datetime64[D]"
    )
    tax = np.where(days < np.datetime64("2008-04-24"), 0.003, 0.001)
    late = days >= np.datetime64("2008-09-19")
    tax[late] = stamp_tax_sell(days[late])
    return panelbt.Costs(tax_sell=tax)


def config(symbols):
    boards = []
    for symbol in symbols:
        code = re.sub(r"\D", "", symbol)[:6]
        if code.startswith(("300", "301")):
            boards.append(Board.CHINEXT)
        elif code.startswith(("688", "689")):
            boards.append(Board.STAR)
        elif code.startswith(("43", "83", "87", "88", "92")):
            boards.append(Board.BSE)
        else:
            boards.append(Board.MAIN)
    min_qty, step = lot_rules(np.array(boards))
    return panelbt.Config(min_qty=min_qty, qty_step=step)


def plans(score, t, validate=True, count=1):
    """count 组：调仓相位 0..9 × 持仓比例 20 档；成交日 = 信号日 + 1。"""
    out = []
    fractions = np.linspace(0.02, 0.4, 20)
    for i in range(count):
        phase, fraction = i // 20 % 10, fractions[i % 20] if count > 1 else 0.1
        days = np.arange(252 + phase, t - 1, REBALANCE_EVERY, dtype=np.int64)
        sparse = panelops.targets(score, fraction=float(fraction), days=days)
        out.append(
            panelbt.Targets(
                sparse.days + 1,
                sparse.indptr,
                sparse.cols,
                sparse.weights,
                validate=validate,
            )
        )
    return out


def run(ctx):
    panelops.set_num_threads(ctx["threads"])
    data = ashare.data.open(ctx["data"])
    panel = data.panel(panel_fields(data) + ["adj_factor"])
    t, n = panel.shape
    stats, inputs = measure(lambda: market_inputs(panel), 1)
    cases = [{"case": "adapter_inputs", "scale": {"days": t, "symbols": n}, **stats}]
    cost, cfg = costs(panel.dates), config(panel.symbols)
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
