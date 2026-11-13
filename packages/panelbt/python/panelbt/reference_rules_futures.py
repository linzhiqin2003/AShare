"""Scalar reference for exit rules in a mixed spot and futures account.

Combines `reference_rules.RulesEngine` (exit rules) with
`reference_futures.MixedEngine` (the futures book and the mixed daily
sequence). Written from SPEC.md ("Exit rules in mixed panels"):

- rules act on spot columns only; futures columns are never evaluated, have
  no cost, entry row or peak, and their ordinary targets are never blocked;
- the rule phase runs where it runs without futures: before the rebalance
  for `fill_time` "close"/"average", after the openings for "open";
- in a margin account the own-cash trim of a rule reduction excludes the
  futures collateral held at that moment (last settlement, or the execution
  price of contracts traded earlier in the row);
- limit targets are rejected with futures columns.
"""

from __future__ import annotations

import math

from .reference_futures import MixedEngine
from .reference_rules import RulesEngine


def run(market, targets, costs=None, config=None):
    return MixedRulesEngine(market, targets, costs, config).run()


class MixedRulesEngine(RulesEngine, MixedEngine):
    def __init__(self, market, targets, costs=None, config=None):
        if targets.limits is not None and any(math.isfinite(x) for x in targets.limits):
            raise ValueError("limit targets are not supported with futures columns")
        self.today = 0
        super().__init__(market, targets, costs, config)

    # ------------------------------------------------- spot-only rule state
    def own_cash(self, cash, long, short):
        # Collateral of the contracts held right now is not own cash.
        return super().own_cash(cash, long, short) - self.collateral(self.today)

    def marked(self):
        long = short = 0.0
        for i in self.spot:
            x = self.quantities[i] * self.marks[i]
            if x > 0:
                long += x
            else:
                short -= x
        return self.cash, long, short

    def intraday(self, day):
        """`RulesEngine.intraday` over spot columns (no limit orders here)."""
        market = self.market
        for i in self.spot:
            current = self.quantities[i]
            if current == 0:
                self.due[i] = None
                continue
            if self.due[i] is not None:
                self.branches["due_retry"] += 1
                reason, raw = self.due[i], float(market.exec_price[day, i])
            elif self.fresh[i]:
                continue
            else:
                hit = self.trigger(i, day)
                if hit is None:
                    continue
                reason, raw = hit
                self.rule_counts[reason] += 1
            self.exited[i] = True
            self.block[i] = day
            self.reduce(i, -current, raw, reason, day)
            if self.quantities[i] != 0:
                self.due[i] = reason
                self.rule_counts["rule_deferred"] += 1
            else:
                self.due[i] = None

    # ------------------------------------------------------- daily sequence
    def day(self, day):
        self.today = day
        self.row_scaled = (self.counts["cash_scaled"], self.coverage["margin_scaled"])
        self.limit_scaled = [False, False]
        flow = self.cash_flow(day)
        self.corporate_actions(day)
        d = self.begin(day)  # rules first when fill_time is close/average
        if d.forced:
            self.stats["expired_target_forced"] += self.count_expired(d, d.commands)
            self.liquidation_plan(d)
        self.d = d
        order = self.command_order(d.commands)
        for i in order:
            if self.is_future[i]:
                self.future_column(i, d)
        for i in order:
            if not self.is_future[i]:
                self.spot_column(i, d)
        self.openings(d)
        if self.rules.fill_time == "open" and not d.forced:
            self.intraday(day)
        long, short = self.close_spot(day)
        high, low = self.market.high, self.market.low
        for i in self.spot:
            q = self.quantities[i]
            if q == 0 or self.fresh[i]:
                continue
            if q > 0 and math.isfinite(high[day, i]):
                self.peak[i] = max(self.peak[i], float(high[day, i]))
            elif q < 0 and math.isfinite(low[day, i]):
                self.peak[i] = min(self.peak[i], float(low[day, i]))
        self.settle(day)
        self.expire(day)
        self.finish(d, flow, long, short)
        self.check_invariants(day, long, short)
        for i in range(self.n):
            self.bought[i] = self.used[i] = 0.0
            self.fresh[i] = self.exited[i] = False
        self.limit_today = set()
        self.limit_orders = []


__all__ = ["MixedRulesEngine", "run"]
