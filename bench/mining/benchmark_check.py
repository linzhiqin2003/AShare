"""真实数据上的基准口径检查：固定随机分数的多头端，相对新旧两种基准的超额信息比率。

固定随机分数（每只股票一个随机数，整段不变）没有任何预测力，多头端（前 10%、5 日调仓）
相对「同口径」基准的超额应当接近 0。分别在两个区间、零成本与默认 A 股费用下，算：

- universe：默认基准，同一套 `panelbt` 成交机制持有信号日股票池全体等权；
- close：旧基准，股票池等权、收盘到收盘、不扣成本。

另给「前 100%」组合（持有整个股票池，就是新基准本身）相对旧基准的超额，用来量出旧基准的
结构偏差。默认费用分两种账户规模跑：默认（`CAPITAL`）与 1 亿元（最低佣金让持有全体的基准
多扣成本）。只输出汇总数字（均值、标准差、个数），不含任何个股或行情数值。

    uv run --no-sync python bench/mining/benchmark_check.py --root <数据根目录> \\
        --out bench/mining/results/benchmark_check.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import panelbt

ZERO = panelbt.Costs(
    commission_rate=0.0, commission_min=None, fee_rate_buy=0.0, fee_rate_sell=0.0,
    slippage=0.0,
)  # fmt: skip
PERIODS = {"2016-2021": ("20160101", "20211231"), "2022-2025": ("20220101", "20251231")}


def _load():
    out = subprocess.run(["uptime"], capture_output=True, text=True).stdout
    return out.split("load averages:")[-1].strip()


def period(data, start, end, n_random, seed):
    from ashare.backtest import market_from_panel
    from ashare.factor.evaluate import FactorEvaluator

    panel = data.panel(list(FactorEvaluator.required_fields()), start=start, end=end)
    market = market_from_panel(data, panel=panel, validate=False)
    t, n = len(panel.dates), len(panel.symbols)
    rng = np.random.default_rng(seed)
    factors = {f"r{k}": np.repeat(rng.standard_normal((1, n)), t, axis=0)
               for k in range(n_random)}  # fmt: skip
    out = {"rows": t, "symbols": n}
    for cost_name, costs, capital in (("zero", ZERO, None), ("default", None, None),
                                      ("default_1e8", None, 1e8)):  # fmt: skip
        cap = {} if capital is None else {"capital": capital}
        for mode in ("universe", "close"):
            fe = FactorEvaluator(panel, horizons=(5,), benchmark=mode, market=market,
                                 costs=costs, size_buckets=1, **cap)  # fmt: skip
            reps = fe.evaluate_many(factors)
            irs = [r.metric("all", "long.information_ratio") for r in reps.values()]
            ex = [r.metric("all", "long.excess_return") for r in reps.values()]
            out[f"{cost_name}/{mode}"] = {
                "ir_mean": float(np.mean(irs)), "ir_sd": float(np.std(irs, ddof=1)),
                "excess_mean": float(np.mean(ex)), "count": len(irs),
            }  # fmt: skip
            if mode == "close":
                full = FactorEvaluator(panel, horizons=(5,), benchmark=mode, market=market,
                                       costs=costs, size_buckets=1, top_fraction=1.0,
                                       **cap)  # fmt: skip
                ones = np.ones((t, n))
                r = full.evaluate(ones)
                out[f"{cost_name}/all_vs_close"] = {
                    "ir": r.metric("all", "long.information_ratio"),
                    "excess": r.metric("all", "long.excess_return"),
                    "benchmark_cost": r.metric("all", "long.cost"),
                }
            del fe, reps
    return out


def main(argv=None):
    import ashare.data as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", type=int, default=15)
    args = ap.parse_args(argv)
    data = ad.open(root=args.root)
    result = {"settings": {"top_fraction": 0.1, "rebalance": 5, "random": args.random},
              "load_before": _load()}  # fmt: skip
    started = time.time()
    for k, (name, (a, b)) in enumerate(PERIODS.items()):
        result[name] = period(data, a, b, args.random, 100 + k)
        print(name, json.dumps(result[name], ensure_ascii=False), flush=True)
    result["seconds"] = time.time() - started
    result["load_after"] = _load()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
