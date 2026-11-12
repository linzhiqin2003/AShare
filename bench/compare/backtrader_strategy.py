"""Preloaded/runonce Backtrader with same-bar open execution and no observers.

A seed bar enables cheat-on-open on the first measured day. Target quantities
use the current open equity snapshot. Fractional sizes are explicit; orders are
submitted in sell-before-buy order. No target or price is shifted in the result.
"""

import backtrader as bt
import numpy as np
import pandas as pd
from common import FEE, INITIAL


class Strategy(bt.Strategy):
    params = (("target", None), ("prices", None), ("task", 1), ("initial", INITIAL))

    def __init__(self):
        self.rows = {int(day): r for r, day in enumerate(self.p.target.days)}
        self.nav = []

    def prenext_open(self):
        self.next_open()

    def nextstart_open(self):
        self.next_open()

    def next_open(self):
        day = len(self.data) - 2
        if day not in self.rows:
            return
        row = self.rows[day]
        target = self.p.target
        a, b = target.indptr[row : row + 2]
        desired = np.zeros(len(self.datas))
        desired[target.cols[a:b]] = target.weights[a:b]
        held = np.array([self.getposition(d).size for d in self.datas])
        prices = self.p.prices[day]
        value = self.broker.getcash() + np.dot(held, prices)
        desired = desired * value / prices
        if self.p.task == 2:
            desired = np.floor(desired / 100) * 100
        delta = desired - held
        if self.p.task == 2:
            available = self.broker.getcash() + np.sum(
                np.maximum(-delta, 0) * prices * (1 - FEE)
            )
            need = np.sum(np.maximum(delta, 0) * prices * (1 + FEE))
            if need > available:
                delta = np.where(
                    delta > 0, np.floor(delta * (available / need) / 100) * 100, delta
                )
        for i in np.flatnonzero(delta < 0):
            self.sell(data=self.datas[i], size=-delta[i])
        for i in np.flatnonzero(delta > 0):
            self.buy(data=self.datas[i], size=delta[i])

    def next(self):
        if len(self.data) > 1:
            self.nav.append(self.broker.getvalue() / self.p.initial)


def prepare(market, target, task=1):
    p = market.exec_price
    t, n = p.shape
    dates = pd.date_range("2000-01-01", periods=t + 1)
    frames = []
    for i in range(n):
        price = np.r_[p[0, i], p[:, i]]
        frames.append(
            pd.DataFrame(
                dict(
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=np.full(t + 1, 1e12),
                    openinterest=np.zeros(t + 1),
                ),
                index=dates,
            )
        )
    return p, target, task, frames


def execute(prepared):
    p, target, task, frames = prepared
    cerebro = bt.Cerebro(cheat_on_open=True, stdstats=False)
    initial = INITIAL * (100 if task == 2 else 1)
    cerebro.broker.setcash(initial)
    cerebro.broker.setcommission(commission=FEE)
    cerebro.broker.set_checksubmit(False)
    for frame in frames:
        cerebro.adddata(bt.feeds.PandasDirectData(dataname=frame))
    cerebro.addstrategy(Strategy, target=target, prices=p, task=task, initial=initial)
    strategy = cerebro.run(runonce=True, preload=True, exactbars=False)[0]
    return np.asarray(strategy.nav)


def run(market, target, task=1):
    return execute(prepare(market, target, task))
