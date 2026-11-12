"""在真实面板上核对：ashare 的正式入口与改动前 bench 里的手写做法逐位相同。

pytest 不许读真实数据目录，所以这一项单独跑：

    uv run --no-sync python bench/e2e/usability_equivalence.py [数据根目录] [--risk-days N]

下面的 old_* 函数原样冻结自 2026-11-13 主干的 bench/e2e/stage_backtest.py 与
risk_adapter.py（改用新入口之前）。任何一项不逐位相同就以非零状态退出。
"""

import argparse
import re
import sys
from dataclasses import fields as dataclass_fields

import numpy as np
import panelbt
import panelops
from panelbt.markets.ashare import Board, lot_rules, tradability

import ashare.data
from ashare.backtest import execution_targets, market_from_panel
from ashare.factor import Evaluator, default_registry

OLD_FIELDS = (
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
MASKS = ("can_buy_open", "can_sell_open")


def old_market_inputs(panel):
    if all(m in panel._arrays for m in MASKS):
        can_buy, can_sell = (np.asarray(panel[m]) for m in MASKS)
        p = {name: np.asarray(panel[name]) for name in OLD_FIELDS[:2] + OLD_FIELDS[8:]}
    else:
        p = {name: np.asarray(panel[name]) for name in OLD_FIELDS}
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


def old_config(symbols):
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


def old_plans(score, t, count):
    out = []
    fractions = np.linspace(0.02, 0.4, 20)
    for i in range(count):
        phase, fraction = i // 20 % 10, fractions[i % 20] if count > 1 else 0.1
        days = np.arange(252 + phase, t - 1, 21, dtype=np.int64)
        sparse = panelops.targets(score, fraction=float(fraction), days=days)
        out.append(
            panelbt.Targets(sparse.days + 1, sparse.indptr, sparse.cols, sparse.weights)
        )
    return out


def old_risk(root, days=None):
    from ashare.risk import FundamentalConfig, MemoryPanel

    names = {
        "close": "close",
        "adj_factor": "adj_factor",
        "total_mv": "total_market_cap",
        "circ_mv": "float_market_cap",
        "turnover_rate": "turnover",
        "pb": "pb",
        "pe_ttm": "pe_ttm",
        "is_suspended": "is_suspended",
        "is_listed": "is_listed",
    }
    data = ashare.data.open(root)
    panel = data.panel([*names.values(), "industry"])
    rows = slice(None if days is None else len(panel.dates) - days, None)
    fields = {}
    for target, name in names.items():
        value = panel[name][rows]
        fields[target] = (
            value.astype(bool)
            if target.startswith("is_")
            else np.asarray(value, dtype=np.float64)
        )
    raw = np.asarray(panel["industry"][rows], dtype=np.int64)
    codes = np.unique(raw[raw > 0])
    lookup = np.full(int(codes.max()) + 1, -1, dtype=np.int64)
    lookup[codes] = np.arange(len(codes))
    fields["industry"] = lookup[raw]
    source = MemoryPanel(panel.dates[rows], tuple(panel.symbols), panel.version, fields)
    return source, FundamentalConfig(industry_codes=tuple(range(len(codes))))


def equal(name, a, b, failures):
    a, b = np.asarray(a), np.asarray(b)
    same = a.dtype == b.dtype and a.shape == b.shape
    same = same and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    print(f"  {'OK ' if same else 'DIFF'} {name} {a.dtype}{a.shape}")
    if not same:
        failures.append(name)


def check_backtest(root, failures):
    print("回测行情、整手、错位：")
    data = ashare.data.open(root)
    names = [*OLD_FIELDS, *MASKS, "adj_factor"]
    panel = data.panel(names)
    old = panelbt.Market(**old_market_inputs(panel))
    setup = market_from_panel(data, fields=["adj_factor"])
    for f in (
        "exec_price",
        "close",
        "can_buy",
        "can_sell",
        "volume",
        "split",
        "cash_div",
        "delist",
    ):
        equal(f"Market.{f}", getattr(setup.market, f), getattr(old, f), failures)
    cfg_old, cfg_new = old_config(panel.symbols), setup.config()
    equal("Config.min_qty", cfg_new.min_qty, cfg_old.min_qty, failures)
    equal("Config.qty_step", cfg_new.qty_step, cfg_old.qty_step, failures)
    t = panel.shape[0]
    score = np.ascontiguousarray(
        Evaluator().evaluate(default_registry()["reversal_20"].expression, panel)
    )
    olds = old_plans(score, t, 20)
    news = []
    fractions = np.linspace(0.02, 0.4, 20)
    for i in range(20):
        days = np.arange(252, t - 1, 21, dtype=np.int64)
        sparse = panelops.targets(score, fraction=float(fractions[i]), days=days)
        news.append(execution_targets(sparse, t=t, tail="error"))
    for k in ("days", "indptr", "cols", "weights"):
        equal(
            f"Targets.{k}（20 组）",
            np.concatenate([getattr(x, k) for x in news]),
            np.concatenate([getattr(x, k) for x in olds]),
            failures,
        )
    costs = panelbt.Costs(tax_sell=0.001)
    a = panelbt.run_batch(setup.market, news, costs, cfg_new)
    b = panelbt.run_batch(old, olds, costs, cfg_old)
    equal("run_batch.nav（20 组）", a.nav, b.nav, failures)
    print(f"  期末净值 {np.round(a.nav[:3, -1], 6).tolist()} …")


def check_risk(root, days, failures):
    from ashare.risk import from_panel, model_series

    print(f"风险模型（最近 {days} 个交易日，全市场）：")
    old_source, old_config = old_risk(root, days)
    data = ashare.data.open(root)
    start = data.panel(["close"]).dates[-days]
    source, config = from_panel(data, start=start)
    for f in old_source.fields:
        equal(f"source[{f}]", source[f], old_source[f], failures)
    for f in dataclass_fields(old_config):
        if f.name == "industry_names":
            continue
        if getattr(config, f.name) != getattr(old_config, f.name):
            print(f"  DIFF config.{f.name}")
            failures.append(f"config.{f.name}")
    print(f"  行业因子名：{config.industry_names}")
    count = 0
    for new, old in zip(
        model_series(source, config=config),
        model_series(old_source, config=old_config),
        strict=True,
    ):
        count += 1
        for k in ("B", "F", "D"):
            if not np.array_equal(getattr(new, k), getattr(old, k)):
                print(f"  DIFF {new.date} {k}")
                failures.append(f"{new.date}.{k}")
        renamed = [
            n if not n.startswith("industry_") else "industry_*" for n in new.factors
        ]
        original = [
            n if not n.startswith("industry_") else "industry_*" for n in old.factors
        ]
        if renamed != original:
            failures.append(f"{new.date}.factors")
    print(f"  {count} 个日模型 B/F/D 逐位比较完毕；最后一天因子 {new.factors[1:3]} …")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=None)
    parser.add_argument("--risk-days", type=int, default=300)
    parser.add_argument(
        "--skip", choices=("backtest", "risk"), action="append", default=[]
    )
    args = parser.parse_args()
    failures = []
    if "backtest" not in args.skip:
        check_backtest(args.root, failures)
    if "risk" not in args.skip:
        check_risk(args.root, args.risk_days, failures)
    print("全部逐位相同" if not failures else f"不一致：{failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
