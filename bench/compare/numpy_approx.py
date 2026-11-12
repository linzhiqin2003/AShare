"""Whole-matrix drifted weights with post-cost proportional reallocation.

Within each rebalance interval quantities are frozen. Fees are estimated from
frictionless drift and deducted multiplicatively; retained cash from earlier
fees is ignored. This is an explicit approximation to panelbt's pre-cost target.
"""

import numpy as np
from common import FEE


def prepare(market, target):
    p = market.exec_price
    t, n = p.shape
    days = target.days
    start = days[np.searchsorted(days, np.arange(t), side="right") - 1]
    w = target.to_dense(t, n)[start]
    old_w = np.vstack((np.zeros((1, n)), w[:-1]))
    old_start = np.r_[0, start[:-1]]
    return p, days, w, old_w, old_start


def execute(prepared):
    p, days, w, old_w, old_start = prepared
    t = len(p)
    today = old_w * p / p[old_start]
    yesterday = old_w * np.vstack((p[0], p[:-1])) / p[old_start]
    gross = 1 - old_w.sum(axis=1) + today.sum(axis=1)
    previous = 1 - old_w.sum(axis=1) + yesterday.sum(axis=1)
    turnover = np.abs(w - today / gross[:, None]).sum(axis=1)
    costs = np.zeros(t)
    costs[days] = turnover[days] * FEE
    return np.cumprod(gross / previous * (1 - costs))


def run(market, target):
    return execute(prepare(market, target))
