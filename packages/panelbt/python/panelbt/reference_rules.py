"""Scalar reference for declarative exit rules and limit targets.

Written from SPEC.md ("Declarative exit rules and limit targets"), not from
the Rust kernel. Spot account rules come unchanged from `reference.Engine`;
this module adds position bookkeeping (cost, entry row, peak), the intraday
phase (rule exits, then limit orders) and the re-entry block, and places the
intraday phase before (fill_time close/average) or after (open) the ordinary
rebalance.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import numpy as np

from .models import RULE_COUNTERS, TradeReason
from .reference import Engine

REASONS = ("stop_loss", "take_profit", "trailing_stop", "max_hold")


def run(market, targets, costs=None, config=None):
    return RulesEngine(market, targets, costs, config).run()


def percent(a, day, i):
    if a is None:
        return math.nan
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return float(a)
    if a.ndim == 1:
        return float(a[i])
    return float(a[day, i])


class RulesEngine(Engine):
    def __init__(self, market, targets, costs=None, config=None):
        super().__init__(market, targets, costs, config)
        n = self.n
        self.rules = self.config.rules
        self.cost = [0.0] * n
        self.entry = [-1] * n
        self.peak = [0.0] * n
        self.bought = [0.0] * n
        self.used = [0.0] * n
        self.fresh = [False] * n
        self.exited = [False] * n
        self.due = [None] * n
        self.block = [None] * n
        self.limit_orders = []  # [col, target, limit], ascending columns
        self.limit_today = set()
        self.rule_counts = dict.fromkeys(RULE_COUNTERS, 0)
        # Which spec branches a case reached (read by the coverage tests).
        self.branches = Counter()
        self.override = None  # (col, raw price) of a rule or limit fill
        self.reason = "rebalance"
        self.entering_fresh = self.rules.fill_time != "open"

    # ------------------------------------------------------- price override
    def quote(self, q, i, day):
        if self.override is None or self.override[0] != i:
            return super().quote(q, i, day)
        market = self.market
        saved = market.exec_price
        row = np.array(saved[day], dtype=float)
        row[i] = self.override[1]
        patched = np.array(saved, dtype=float)
        patched[day] = row
        object.__setattr__(market, "exec_price", patched)
        try:
            return super().quote(q, i, day)
        finally:
            object.__setattr__(market, "exec_price", saved)

    def raw(self, i, day):
        if self.override is not None and self.override[0] == i:
            return self.override[1]
        return float(self.market.exec_price[day, i])

    def execute(self, i, q, day, rebalance, forced):
        before = self.quantities[i]
        raw = self.raw(i, day)
        trades = None if self.trades is None else len(self.trades)
        super().execute(i, q, day, rebalance, forced)
        after = self.quantities[i]
        if trades is not None and self.reason != "rebalance":
            self.trades[-1] = replace(
                self.trades[-1], reason=TradeReason[self.reason.upper()]
            )
        # Position bookkeeping (raw prices).
        if after == 0:
            self.cost[i], self.entry[i], self.peak[i] = 0.0, -1, 0.0
            self.due[i] = None  # a due exit ends when the position is gone
        elif before == 0 or (before > 0) != (after > 0):
            self.cost[i], self.entry[i], self.peak[i] = raw, day, raw
            self.fresh[i] = self.entering_fresh
        elif abs(after) > abs(before):
            self.cost[i] = (abs(before) * self.cost[i] + abs(q) * raw) / abs(after)
        self.used[i] += abs(q)
        if q > 0 and after > 0:
            self.bought[i] += min(q, after)

    # ---------------------------------------------------- corporate actions
    def corporate_actions(self, day):
        market = self.market
        held = [self.quantities[i] != 0 for i in range(self.n)]
        super().corporate_actions(day)
        for i in range(self.n):
            if not held[i]:
                continue
            if self.quantities[i] == 0:  # delisted
                self.cost[i], self.entry[i], self.peak[i] = 0.0, -1, 0.0
                self.due[i] = None
                continue
            div = float(market.cash_div[day, i]) if market.cash_div is not None else 0.0
            if self.rules.dividend_adjusts_cost and div != 0:
                self.branches["dividend_adjusted"] += 1
                self.cost[i] = max(self.cost[i] - div, 0.0)
                self.peak[i] = max(self.peak[i] - div, 0.0)
            factor = float(market.split[day, i]) if market.split is not None else 1.0
            if factor != 1:
                self.cost[i] /= factor
                self.peak[i] /= factor

    # ------------------------------------------------------------ re-entry
    def blocks(self, i, day):
        if self.due[i] is not None:
            return True
        if self.block[i] is None:
            return False
        k = self.rules.cooldown
        return k is None or day <= self.block[i] + k

    # --------------------------------------------------------- daily phase
    def begin(self, day):
        targets = self.targets
        row = self.cursor < len(targets.days) and targets.days[self.cursor] == day
        if row:
            if self.rules.cooldown is None:
                for i in range(self.n):
                    if self.block[i] is not None and self.block[i] < day:
                        self.block[i] = None
            self.limit_orders = []
            limits = targets.limits
            if limits is not None:
                for j in range(
                    targets.indptr[self.cursor], targets.indptr[self.cursor + 1]
                ):
                    if math.isfinite(limits[j]):
                        i = int(targets.cols[j])
                        self.limit_orders.append(
                            [i, float(targets.weights[j]), float(limits[j])]
                        )
                        self.limit_today.add(i)
        forced = self.supervised() and (
            (self.call and self.margin.margin_call == "liquidate") or self.bankrupt
        )
        if self.rules.fill_time != "open" and not forced:
            self.intraday(day)
        return super().begin(day)

    def spot_column(self, i, d):
        if not d.forced and (
            self.exited[i] or self.due[i] is not None or i in self.limit_today
        ):
            return
        self.ordinary_column(i, d)

    def ordinary_column(self, i, d):
        """`Engine.spot_column` with the re-entry block on openings."""
        market, config, margin, v = self.market, self.config, self.margin, self.v
        data, coverage = self.data, self.coverage
        quantities, closing, opening, quote = (
            self.quantities,
            self.closing,
            self.opening,
            self.quote,
        )
        day, forced, value = d.day, d.forced, d.value
        if self.dead[i]:
            return
        w = d.commands[i]
        p = float(market.exec_price[day, i])
        mark = p if math.isfinite(p) else self.marks[i]
        if forced or self.targets.kind == "shares":
            target = w
            coverage["shares_target"] += int(not forced)
        else:
            x = w * max(value, 0.0) * (1 - config.cash_buffer) / mark if mark > 0 else 0
            target = (
                x
                if abs(x) < v[6][i] and quantities[i] * x > 0
                else math.copysign(opening(abs(x), i), x)
            )
        if abs(target - quantities[i]) <= 1e-9 * max(
            abs(target), abs(quantities[i]), 1
        ):
            target = quantities[i]
        if (
            not forced
            and self.targets.kind == "weights"
            and w != 0
            and quantities[i] != 0
            and value > 0
            and abs(w - quantities[i] * mark / value) < config.trade_band
        ):
            coverage["trade_band"] += 1
            return
        cap = self.cap(i, day)
        current = quantities[i]
        reduction = (
            d.planned[i]
            if forced
            else (
                -closing(current - max(target, 0), current, i)
                if current > 0 and target < current
                else closing(min(target, 0) - current, -current, i)
                if current < 0 and target > current
                else 0.0
            )
        )
        blocked_buy = blocked_sell = False
        used = 0.0
        if reduction != 0:
            allowed = math.isfinite(p) and (
                market.can_buy[day, i] if reduction > 0 else market.can_sell[day, i]
            )
            q = (
                math.copysign(
                    closing(min(abs(reduction), cap), abs(current), i), reduction
                )
                if allowed
                else 0.0
            )
            if q != reduction:
                blocked_buy = reduction > 0
                blocked_sell = reduction < 0
            if (
                q != 0
                and quote(q, i, day)[0] > 0
                and math.isfinite(quote(q, i, day)[0])
                and not self.classic
                and not margin.allow_financing
                and self.own(d.account) >= -1e-10 * config.initial_cash
                and self.own(self.project(d.account, i, q, day)) < 0
            ):
                requested_quantity = q
                lo, hi = 0.0, abs(q)
                for _ in range(64):
                    mid = (lo + hi) / 2
                    x = math.copysign(closing(mid, abs(current), i), q)
                    if self.own(self.project(d.account, i, x, day)) >= 0:
                        lo = mid
                    else:
                        hi = mid
                q = math.copysign(closing(lo, abs(current), i), q)
                if abs(q - requested_quantity) > 1e-9 * max(
                    abs(q), abs(requested_quantity), 1
                ):
                    data["n_reduction_clipped"][day] += 1
                    coverage["reduction_clipped"] += 1
            if q != 0:
                price, a, c, tax, fee, _, _ = quote(q, i, day)
                candidate = self.project(d.account, i, q, day)
                if (
                    price > 0
                    and math.isfinite(price)
                    and (
                        not self.classic
                        # Legacy check while cash is nonnegative; with debt from a
                        # futures loss a sale need only cover its own fees.
                        or (
                            self.cash + a >= c + tax + fee
                            if self.cash >= 0
                            else a >= c + tax + fee
                        )
                    )
                ):
                    self.execute(i, q, day, d.rebalance, forced)
                    d.account = candidate
                    used = abs(q)
                elif not math.isfinite(price) or price <= 0:
                    data["n_invalid_quote"][day] += 1
                    coverage["invalid_quote"] += 1
            if forced and abs(reduction) - used > 1e-9 * max(
                abs(reduction), abs(current), 1
            ):
                d.unfilled = True
        after = quantities[i]
        wanted = (
            opening(target - after, i)
            if target > 0 and after >= 0 and target > after and target > current
            else -opening(after - target, i)
            if target < 0 and after <= 0 and target < after and target < current
            else 0.0
        )
        opens = (
            abs(wanted) > 1e-9 * max(abs(target), abs(after), 1)
            and not forced
            and not self.bankrupt
        )
        if opens and self.blocks(i, day):
            self.branches[
                "reentry_cooldown"
                if self.rules.cooldown is not None
                else "reentry_until_row"
            ] += 1
            # Re-entry after a rule exit is blocked: no order, no carry.
            self.rule_counts["reentry_blocked"] += 1
            opens = False
        if opens:
            allowed = math.isfinite(p) and (
                market.can_buy[day, i]
                if wanted > 0
                else market.can_sell[day, i]
                and margin.allow_short
                and (market.can_short is None or market.can_short[day, i])
            )
            q = (
                math.copysign(opening(min(abs(wanted), max(cap - used, 0)), i), wanted)
                if allowed
                else 0.0
            )
            if q != 0:
                price = quote(q, i, day)[0]
                if not math.isfinite(price) or price <= 0:
                    q = 0.0
                    data["n_invalid_quote"][day] += 1
                    coverage["invalid_quote"] += 1
            if wanted != q:
                blocked_buy |= wanted > 0
                blocked_sell |= wanted < 0
            if q != 0:
                coverage["reversal"] += int(current * q < 0 and used > 0)
                d.orders.append((i, q))
        if mark == 0 and w != 0 and target >= current:
            blocked_buy |= w > 0
            blocked_sell |= w < 0
        self.blocked(i, w, d, blocked_buy, blocked_sell)

    def day(self, day):
        flow = self.cash_flow(day)
        self.corporate_actions(day)
        d = self.begin(day)
        if d.forced:
            self.liquidation_plan(d)
        self.d = d
        for i in self.command_order(d.commands):
            self.spot_column(i, d)
        if d.orders:
            if self.classic:
                self.cash_basket(d, d.orders)
            else:
                self.margin_basket(d, d.orders)
        if self.rules.fill_time == "open" and not d.forced:
            self.intraday(day)
        long, short = self.close_spot(day)
        high = self.market.high
        low = self.market.low
        for i in range(self.n):
            q = self.quantities[i]
            if q == 0 or self.fresh[i] or high is None:
                continue
            if q > 0 and math.isfinite(high[day, i]):
                self.peak[i] = max(self.peak[i], float(high[day, i]))
            elif q < 0 and math.isfinite(low[day, i]):
                self.peak[i] = min(self.peak[i], float(low[day, i]))
        self.finish(d, flow, long, short)
        for i in range(self.n):
            self.bought[i] = self.used[i] = 0.0
            self.fresh[i] = self.exited[i] = False
        self.limit_today = set()
        if self.rules.limit_valid == "day":
            self.limit_orders = []
        elif self.limit_orders:
            self.branches["limit_carried"] += 1

    # ------------------------------------------------------------ intraday
    def own_cash(self, cash, long, short):
        return cash - (0 if self.margin.short_proceeds_usable else short)

    def marked(self):
        long = short = 0.0
        for i in range(self.n):
            x = self.quantities[i] * self.marks[i]
            if x > 0:
                long += x
            else:
                short -= x
        return self.cash, long, short

    def project_at(self, a, i, q, day, raw):
        cash, long, short = a
        old = self.quantities[i]
        self.override = (i, raw)
        try:
            change = self.movement(q, i, day)
        finally:
            self.override = None
        return (
            cash + change,
            long + (max(old + q, 0) - max(old, 0)) * raw,
            short - (min(old + q, 0) - min(old, 0)) * raw,
        )

    def lots(self, i):
        step = self.step(i)
        return float(self.v[6][i]), step

    def fill_at(self, i, q, day, raw, reason):
        self.override, self.reason = (i, raw), reason
        try:
            self.execute(i, q, day, True, False)  # rule fills are not carry trades
        finally:
            self.override, self.reason = None, "rebalance"

    def reduce(self, i, request, raw, reason, day):
        market = self.market
        current = self.quantities[i]
        if not (math.isfinite(raw) and raw > 0):
            return 0.0
        if not (market.can_buy[day, i] if request > 0 else market.can_sell[day, i]):
            self.branches["reduce_mask_blocked"] += 1
            return 0.0
        avail = abs(request)
        if self.rules.sell_lock and current > 0:
            avail = min(avail, max(current - self.bought[i], 0.0))
            if avail < abs(request):
                self.branches["reduce_sell_lock"] += 1
        cap = max(self.cap(i, day) - self.used[i], 0.0)
        if cap < avail:
            self.branches["reduce_capped"] += 1
        q = math.copysign(self.closing(min(avail, cap), abs(current), i), request)
        if q == 0:
            return 0.0
        self.override = (i, raw)
        try:
            price, amount, comm, tax, fee, _, _ = self.quote(q, i, day)
        finally:
            self.override = None
        if not (math.isfinite(price) and price > 0):
            return 0.0
        if self.classic:
            ok = (
                self.cash + amount >= comm + tax + fee
                if self.cash >= 0
                else amount >= comm + tax + fee
            )
            if not ok:
                self.branches["reduce_fee_blocked"] += 1
                return 0.0
        elif not self.margin.allow_financing:
            a = self.marked()
            if (
                self.own_cash(*a) >= -1e-10 * self.config.initial_cash
                and self.own_cash(*self.project_at(a, i, q, day, raw)) < 0
            ):
                lo, hi = 0.0, abs(q)
                for _ in range(64):
                    mid = (lo + hi) / 2
                    x = math.copysign(self.closing(mid, abs(current), i), q)
                    if self.own_cash(*self.project_at(a, i, x, day, raw)) >= 0:
                        lo = mid
                    else:
                        hi = mid
                q = math.copysign(self.closing(lo, abs(current), i), q)
                self.branches["reduce_cash_trimmed"] += 1
                if q == 0:
                    return 0.0
        self.fill_at(i, q, day, raw, reason)
        return q

    def trigger(self, i, day):
        market, rules = self.market, self.rules
        current = self.quantities[i]
        hold = (
            rules.max_hold is not None
            and self.entry[i] >= 0
            and day - self.entry[i] >= rules.max_hold
        )
        exec_price = float(market.exec_price[day, i])
        if hold and rules.fill_time == "open":
            self.branches["hold_at_open"] += 1
            return "max_hold", exec_price
        o, h, low = (
            float(market.open[day, i]),
            float(market.high[day, i]),
            float(market.low[day, i]),
        )
        if math.isfinite(o) and math.isfinite(h) and math.isfinite(low):
            sl = percent(rules.stop_loss, day, i)
            tr = percent(rules.trailing_stop, day, i)
            tp = percent(rules.take_profit, day, i)
            cost, peak = self.cost[i], self.peak[i]
            if current > 0:
                lines = []
                if not math.isnan(sl):
                    lines.append((cost * (1 - sl), 0, "stop_loss"))
                if not math.isnan(tr):
                    lines.append((peak * (1 - tr), 1, "trailing_stop"))
                # The highest stop line is hit first on the way down.
                stop, why = (-math.inf, None)
                for value, _rank, name in lines:
                    if value > stop:
                        stop, why = value, name
                target = cost * (1 + tp) if not math.isnan(tp) else math.inf
                if why == "trailing_stop" and not math.isnan(sl):
                    self.branches["trailing_above_stop"] += 1
                if why is not None and o <= stop:
                    self.branches["gap_stop"] += 1
                    return why, o
                if o >= target:
                    self.branches["gap_target"] += 1
                    return "take_profit", o
                stop_hit = why is not None and low <= stop
                target_hit = h >= target
            else:
                lines = []
                if not math.isnan(sl):
                    lines.append((cost * (1 + sl), "stop_loss"))
                if not math.isnan(tr):
                    lines.append((peak * (1 + tr), "trailing_stop"))
                stop, why = (math.inf, None)
                for value, name in lines:
                    if value < stop:
                        stop, why = value, name
                target = cost * (1 - tp) if not math.isnan(tp) and tp < 1 else -math.inf
                self.branches["short_evaluated"] += 1
                if why is not None and o >= stop:
                    self.branches["gap_stop_short"] += 1
                    return why, o
                if o <= target:
                    self.branches["gap_target_short"] += 1
                    return "take_profit", o
                stop_hit = why is not None and h >= stop
                target_hit = low <= target
            if stop_hit and target_hit:
                self.branches[f"both_hit_{rules.both_hit}"] += 1
                if rules.both_hit == "target":
                    return "take_profit", target
                return why, stop
            if stop_hit:
                return why, stop
            if target_hit:
                return "take_profit", target
        if hold:
            return "max_hold", exec_price
        return None

    def intraday(self, day):
        market = self.market
        for i in range(self.n):
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
        if self.limit_orders:
            self.limits(day)

    def limits(self, day):
        market, config = self.market, self.config
        equity = self.cash + sum(
            self.quantities[i] * self.marks[i] for i in range(self.n)
        )
        keep, plan = [], []
        for i, w, limit in self.limit_orders:
            if self.dead[i]:
                continue
            if self.exited[i] or self.due[i] is not None:
                self.branches["limit_after_exit"] += 1
                keep.append([i, w, limit])
                continue
            minimum, step = self.lots(i)
            current = self.quantities[i]
            if self.targets.kind == "shares":
                target = w
            else:
                x = w * max(equity, 0.0) * (1 - config.cash_buffer) / limit
                target = (
                    x
                    if abs(x) < minimum and current * x > 0
                    else math.copysign(self.opening(abs(x), i), x)
                )
            if abs(target - current) <= 1e-9 * max(abs(target), abs(current), 1):
                target = current
            if target == current:
                continue
            o, h, low = (
                float(market.open[day, i]),
                float(market.high[day, i]),
                float(market.low[day, i]),
            )
            valid = math.isfinite(o) and math.isfinite(h) and math.isfinite(low)
            touch = self.rules.limit_fill == "touch"
            if target > current:
                touched = valid and (low < limit or (touch and low == limit))
                raw = min(o, limit)
            else:
                touched = valid and (h > limit or (touch and h == limit))
                raw = max(o, limit)
            if valid and (low == limit if target > current else h == limit):
                self.branches[
                    "limit_touch_only_" + ("filled" if touched else "skipped")
                ] += 1
            keep.append([i, w, limit])
            if not touched:
                self.rule_counts["limit_unfilled"] += 1
                continue
            if raw == o:
                self.branches["limit_at_open"] += 1
            plan.append(
                {
                    "col": i,
                    "target": target,
                    "raw": raw,
                    "short": False,
                    "scaled": False,
                }
            )
        self.limit_orders = keep
        for p in plan:
            i, target, raw = p["col"], p["target"], p["raw"]
            current = self.quantities[i]
            if current > 0 and target < current:
                request = -self.closing(current - max(target, 0), current, i)
            elif current < 0 and target > current:
                request = self.closing(min(target, 0) - current, -current, i)
            else:
                request = 0.0
            if request != 0:
                q = self.reduce(i, request, raw, "limit", day)
                if q != 0:
                    self.rule_counts["limit_filled"] += 1
                p["short"] = q != request
        plan.sort(key=lambda p: (p["target"] > self.quantities[p["col"]], p["col"]))
        self.entering_fresh = True
        for p in plan:
            i, target, raw = p["col"], p["target"], p["raw"]
            after = self.quantities[i]
            minimum, step = self.lots(i)
            if target > 0 and after >= 0 and target > after:
                opening = self.opening(target - after, i)
            elif target < 0 and after <= 0 and target < after:
                opening = -self.opening(after - target, i)
            else:
                opening = 0.0
            if abs(opening) <= 1e-9 * max(abs(target), abs(after), 1):
                continue
            if self.bankrupt:
                p["short"] = True
                continue
            if self.blocks(i, day):
                self.rule_counts["reentry_blocked"] += 1
                p["short"] = True
                continue
            if opening > 0:
                allowed = bool(market.can_buy[day, i])
            else:
                allowed = bool(
                    market.can_sell[day, i]
                    and self.margin.allow_short
                    and (market.can_short is None or market.can_short[day, i])
                )
            cap = max(self.cap(i, day) - self.used[i], 0.0)
            q = (
                math.copysign(self.opening(min(abs(opening), cap), i), opening)
                if allowed
                else 0.0
            )
            if q != 0:
                self.override = (i, raw)
                try:
                    price = self.quote(q, i, day)[0]
                finally:
                    self.override = None
                if not (math.isfinite(price) and price > 0):
                    q = 0.0
            # Short of the opening for a mask, the cap, a block or a bad quote;
            # a cut to what the account affords is counted like an ordinary
            # purchase (cash_scaled / margin_scaled) and completes the order.
            if q != opening:
                p["short"] = True
            if q != 0:
                if self.classic:

                    def cost(y, i=i, raw=raw):
                        if y == 0:
                            return 0.0
                        self.override = (i, raw)
                        try:
                            return -self.movement(y, i, day)
                        finally:
                            self.override = None

                    if cost(q) > self.cash:
                        self.branches["limit_cash_scaled"] += 1
                        self.counts["cash_scaled"] += 1
                        p["scaled"] = True
                        lo, hi = 0.0, q
                        for _ in range(64):
                            mid = (lo + hi) / 2
                            if cost(mid) <= self.cash:
                                lo = mid
                            else:
                                hi = mid
                        q = self.opening(lo, i)
                        while q > 0 and cost(q) > self.cash:
                            q = (
                                math.nextafter(q, -math.inf)
                                if step == 0
                                else self.opening(q - step, i)
                            )
                else:
                    a = self.marked()

                    def ok(y, a=a, i=i, raw=raw):
                        cash, long, short = self.project_at(a, i, y, day, raw)
                        return self.feasible((cash, long, short, 0.0), day)

                    if not ok(q):
                        self.branches["limit_margin_scaled"] += 1
                        self.coverage["margin_scaled"] += 1
                        p["scaled"] = True
                        lo, hi = 0.0, abs(q)
                        for _ in range(64):
                            mid = (lo + hi) / 2
                            if ok(math.copysign(self.opening(mid, i), q)):
                                lo = mid
                            else:
                                hi = mid
                        q = math.copysign(self.opening(lo, i), q)
            if abs(q) > 1e-9 * max(abs(target), abs(after), 1):
                self.fill_at(i, q, day, raw, "limit")
                self.rule_counts["limit_filled"] += 1
        self.entering_fresh = self.rules.fill_time != "open"
        for p in plan:
            if p["short"]:
                self.rule_counts["limit_blocked"] += 1
        done = {
            p["col"]
            for p in plan
            if (p["scaled"] and not p["short"])
            or abs(p["target"] - self.quantities[p["col"]])
            <= 1e-9 * max(abs(p["target"]), abs(self.quantities[p["col"]]), 1)
        }
        self.limit_orders = [o for o in self.limit_orders if o[0] not in done]

    def result(self):
        if self.rules.limit_valid == "until_next" and self.limit_orders:
            self.branches["limit_until_next_left"] += 1
        r = super().result()
        r.diagnostics = {**r.diagnostics, **self.rule_counts}
        return r
