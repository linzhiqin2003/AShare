"""只替换模拟的隔离基准：其余计算直接使用冻结 main 的源代码。"""

import inspect
from types import FunctionType

import numpy as np
import panelbt

from bench.groupbt.oracle.groupbt import api

VALIDATION = api.VALIDATION


def batch(groups, built, fwd, buy, sell, options):
    n = options["n_groups"]
    labels = np.full(groups.shape, -1, dtype=np.int64)
    labels[1:] = groups[:-1]
    plans = [panelbt.GroupWeightTargets(labels, g) for g in range(n)]
    plans.append(panelbt.GroupWeightTargets(labels, 0, n - 1, 0.5, 0.5))
    if options["weights_portfolio"]:
        plans.append(api._shift(built["weights"]))
    result = panelbt.run_weights_batch(
        fwd, plans, buy, sell, keep_weights=[i == n for i in range(len(plans))]
    )
    return [vars(r) for r in result]


# Clone the frozen function's globals instead of mutating the oracle module.
# Only this explicit block is substituted, making the stage reproducible without
# storing a second full copy of the production pipeline.
_source = inspect.getsource(api._one)
_begin = _source.index("    def run(target, keep=False):")
_end = _source.index("    # 持仓在成交日", _begin)
_source = (
    _source[:_begin]
    + """    runs = batch(groups, built, fwd, can_buy, can_sell, options)
    group_runs, long_short = runs[:n], runs[n]
    pnl = {"long_short": long_short["pnl"], "top": group_runs[0]["pnl"]}
    turnover = {"long_short": long_short["turnover"], "top": group_runs[0]["turnover"]}
    if options["weights_portfolio"]:
        pnl["weights"], turnover["weights"] = runs[-1]["pnl"], runs[-1]["turnover"]
"""
    + _source[_end:]
)
_globals = {**vars(api), "batch": batch}
exec(compile(_source, __file__, "exec"), _globals)
run = FunctionType(api.run.__code__, _globals, "run", api.run.__defaults__)
run.__kwdefaults__ = api.run.__kwdefaults__
