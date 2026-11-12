"""bt's documented WeighTarget/Rebalance pipeline with fractional positions.

bt allocates fees within each child's capital allocation, so its post-cost
weights differ slightly from panelbt's pre-cost equity sizing. This difference
is measured, and a zero-fee control verifies timing and target alignment.
"""

import bt
import numpy as np
import pandas as pd
from common import FEE, INITIAL


def prepare(market, target, task=1):
    p = market.exec_price
    t, n = p.shape
    index = pd.date_range("2000-01-01", periods=t)
    columns = [str(i) for i in range(n)]
    data = pd.DataFrame(p, index=index, columns=columns)
    weights = pd.DataFrame(
        target.to_dense(t, n)[target.days], index=index[target.days], columns=columns
    )
    return data, weights, task


def execute(prepared):
    data, weights, task = prepared
    algos = (
        [bt.algos.WeighTarget(weights), bt.algos.Rebalance()]
        if task == 1
        else [LotRebalance(weights)]
    )
    strategy = bt.Strategy("weights", algos)
    test = bt.Backtest(
        strategy,
        data,
        initial_capital=INITIAL * (100 if task == 2 else 1),
        commissions=lambda q, p: abs(q) * p * FEE,
        integer_positions=False,
        progress_bar=False,
    )
    test.run()
    return np.asarray(test.strategy.values.iloc[1:]) / (
        INITIAL * (100 if task == 2 else 1)
    )


class LotRebalance(bt.Algo):
    """Native bt quantity transactions after sizing the shared cash basket."""

    def __init__(self, weights):
        self.rows = {date: row.to_numpy() for date, row in weights.iterrows()}
        self.columns = list(weights.columns)

    def __call__(self, target):
        if target.now not in self.rows:
            return True
        weights = self.rows[target.now]
        price = target.universe.loc[target.now, self.columns].to_numpy()
        held = np.array(
            [
                target.children[c].position if c in target.children else 0.0
                for c in self.columns
            ]
        )
        desired = np.floor(weights * target.value / price / 100) * 100
        delta = desired - held
        available = target.capital + np.sum(np.maximum(-delta, 0) * price * (1 - FEE))
        need = np.sum(np.maximum(delta, 0) * price * (1 + FEE))
        if need > available:
            delta = np.where(
                delta > 0, np.floor(delta * (available / need) / 100) * 100, delta
            )
        for i in np.r_[np.flatnonzero(delta < 0), np.flatnonzero(delta > 0)]:
            target.transact(float(delta[i]), child=self.columns[i], update=False)
        return True


def run(market, target, task=1):
    return execute(prepare(market, target, task))
