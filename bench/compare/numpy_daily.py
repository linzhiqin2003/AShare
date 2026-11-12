"""Step-one daily full/sparse NumPy algorithms, with benchmark fee parameters."""

import numpy as np
from common import FEE, INITIAL


def run(market, target, task, sparse=True):
    p, close = market.exec_price, market.close
    t, n = p.shape
    shares = np.zeros(n)
    weights = np.zeros(n)
    initial = INITIAL * (100 if task == 2 else 1)
    cash = initial
    held = np.empty(0, dtype=np.int64)
    all_cols = np.arange(n)
    nav = np.empty(t)
    cursor = 0
    for day in range(t):
        if cursor < len(target.days) and target.days[cursor] == day:
            value = cash + (
                np.dot(shares[held], p[day, held]) if sparse else np.dot(shares, p[day])
            )
            weights.fill(0)
            begin, end = target.indptr[cursor : cursor + 2]
            cols = target.cols[begin:end]
            weights[cols] = target.weights[begin:end]
            active = np.union1d(held, cols) if sparse else all_cols
            price = p[day, active]
            q = shares[active].copy()
            desired = weights[active] * value / price
            if task == 2:
                desired = np.floor(desired / 100) * 100
            sold = np.maximum(q - desired, 0)
            cash += np.sum(sold * price * (1 - FEE))
            q -= sold
            buy = np.maximum(desired - q, 0)
            need = buy * price * (1 + FEE)
            if need.sum() > cash:
                buy *= cash / need.sum()
                if task == 2:
                    buy = np.floor(buy / 100) * 100
                need = buy * price * (1 + FEE)
            cash -= need.sum()
            q += buy
            shares[active] = q
            held = active[q > 0]
            cursor += 1
        nav[day] = (
            cash
            + (
                np.dot(shares[held], close[day, held])
                if sparse
                else np.dot(shares, close[day])
            )
        ) / initial
    return nav
