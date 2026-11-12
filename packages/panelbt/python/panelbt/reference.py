"""Deliberately scalar executable specification, independent of Rust execution.

`Engine` holds the spot account rules. Its daily sequence is a series of
methods so that `reference_futures` can reuse every spot rule unchanged and
insert the futures book between them. An account snapshot is the tuple
`(cash, long_value, short_value, future_collateral)`; the last element is
always 0.0 for a spot-only market, which leaves every spot comparison
bit-for-bit identical to the original single-function implementation.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from .models import (
    COUNTERS,
    EXTRA_COUNTERS,
    METRICS,
    Config,
    Costs,
    DelistEvent,
    LiquidationEvent,
    Result,
    Trade,
)


def run(market, targets, costs=None, config=None):
    if market.futures is not None:
        from .reference_futures import run as run_futures

        return run_futures(market, targets, costs, config)
    return Engine(market, targets, costs, config).run()


def expand(value, size):
    a = np.asarray(value, dtype=float)
    return np.full(size, float(a)) if a.ndim == 0 else a.copy()


def rate(a, day, col):
    a = np.asarray(a)
    return (
        float(a)
        if a.ndim == 0
        else float(a[day])
        if a.ndim == 1
        else float(a[day, col])
    )


class Engine:
    """Spot account: positions, cash and the daily execution rules."""

    def __init__(self, market, targets, costs=None, config=None):
        costs, config = costs or Costs(), config or Config()
        self.market, self.targets, self.costs, self.config = (
            market,
            targets,
            costs,
            config,
        )
        self.margin = margin = config.margin
        self.t, self.n = t, n = market.shape
        targets.validate_shape(t, n)
        self.validate_signs()
        v = [
            expand(getattr(costs, key), t)
            for key in (
                "tax_sell",
                "tax_buy",
                "fee_rate_sell",
                "fee_rate_buy",
                "fee_per_share_sell",
                "fee_per_share_buy",
            )
        ]
        v += [expand(config.min_qty, n), expand(config.qty_step, n)]
        v += [
            expand(getattr(config, key), t)
            for key in ("dividend_tax_rate", "cash_rate", "cash_flow")
        ]
        v += [
            expand(getattr(margin, key), t)
            for key in (
                "long_initial",
                "short_initial",
                "long_maintenance",
                "short_maintenance",
                "financing_rate",
            )
        ]
        self.v = v
        # Spot columns; the futures engine removes contract columns from this list.
        self.spot = list(range(n))
        self.quantities, self.marks = [0.0] * n, [0.0] * n
        # Last valid close adjusted for corporate actions (0 before any close).
        # `marks` may instead hold an execution-price fallback while held.
        self.closes = [0.0] * n
        self.dead = [False] * n
        self.cash = config.initial_cash
        self.data = {
            key: np.zeros(
                t,
                dtype=np.int64
                if key.startswith("n_")
                else bool
                if key == "margin_call"
                else float,
            )
            for key in METRICS
        }
        self.counts = dict.fromkeys(COUNTERS, 0)
        self.coverage = dict.fromkeys(EXTRA_COUNTERS, 0)
        self.contribution = np.zeros(n)
        self.positions = np.zeros((t, n)) if config.record_trades else None
        self.trades = [] if config.record_trades else None
        self.events, self.liquidations, self.bankruptcies = [], [], []
        self.expiries = []
        self.pending = {}
        self.cursor = 0
        self.previous, self.unit = self.cash, 1.0
        self.call = self.bankrupt = False
        self.classic = (
            not margin.allow_short
            and not margin.allow_financing
            and np.all(np.asarray(margin.long_initial) == 1)
            and np.all(np.asarray(margin.long_maintenance) == 0)
            and np.all(np.asarray(margin.short_maintenance) == 0)
        )

    # ----------------------------------------------------------- input checks
    def validate_signs(self):
        if not self.margin.allow_short and np.any(self.targets.weights < 0):
            raise ValueError("negative targets require allow_short")

    # ------------------------------------------------------------ fill rules
    def step(self, i):
        return 0.0 if self.targets.kind == "shares" else float(self.v[7][i])

    def opening(self, q, i):
        v = self.v
        d = self.step(i)
        if d == 0:
            return max(q, 0.0)
        return (
            0.0 if q < v[6][i] else float(v[6][i] + math.floor((q - v[6][i]) / d) * d)
        )

    def closing(self, q, pos, i):
        v, step = self.v, self.step
        return (
            min(q, pos)
            if q >= pos or step(i) == 0
            else 0.0
            if q < v[6][i]
            else math.floor(q / step(i)) * step(i)
        )

    def quote(self, q, i, day):
        market, costs, v = self.market, self.costs, self.v
        if q == 0:
            return (0.0,) * 7
        raw = float(market.exec_price[day, i])
        slip = rate(costs.slippage, day, i)
        slipped = raw * (1 + slip) if q > 0 else raw * (1 - slip)
        impact = 0.0
        if costs.impact_k is not None:
            volume = float(market.volume[day, i])
            impact = (
                float(costs.impact_k[day, i])
                * (abs(q) / volume) ** costs.impact_exponent
                if math.isfinite(volume) and volume > 0
                else math.inf
            )
        price = slipped * (1 + impact) if q > 0 else slipped * (1 - impact)
        if not math.isfinite(price) or price <= 0:
            return (price, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        amount = abs(q) * price
        comm = (
            max(
                amount * costs.commission_rate + abs(q) * costs.commission_per_share,
                (costs.commission_min or 0.0),
            )
            if amount > 0
            else 0.0
        )
        if costs.commission_max_rate is not None:
            comm = min(comm, costs.commission_max_rate * amount)
        d = 1 if q > 0 else 0
        return (
            price,
            amount,
            comm,
            amount * v[d][day],
            amount * v[2 + d][day] + abs(q) * v[4 + d][day],
            abs(q) * raw * slip,
            abs(q) * abs(price - slipped),
        )

    def movement(self, q, i, day):
        _, a, c, tax, fee, _, _ = self.quote(q, i, day)
        return -(a + c + tax + fee) if q > 0 else a - c - tax - fee

    # ------------------------------------------------------ account snapshots
    def collateral(self, day):
        """Pretrade future collateral; a spot account has none."""
        return 0.0

    def balance(self, day):
        long = short = 0.0
        for i in self.spot:
            p = float(self.market.exec_price[day, i])
            x = self.quantities[i] * (p if math.isfinite(p) else self.marks[i])
            if x > 0:
                long += x
            else:
                short -= x
        return self.cash, long, short, self.collateral(day)

    def project(self, a, i, q, day):
        c, long, s, fc = a
        p = float(self.market.exec_price[day, i])
        old = self.quantities[i]
        return (
            c + self.movement(q, i, day),
            long + (max(old + q, 0) - max(old, 0)) * p,
            s - (min(old + q, 0) - min(old, 0)) * p,
            fc,
        )

    def own(self, a):
        """Own cash available to the account (restricted proceeds and collateral excluded)."""
        return a[0] - (0 if self.margin.short_proceeds_usable else a[2]) - a[3]

    def feasible(self, a, day):
        c, long, s, fc = a
        v = self.v
        return v[11][day] * long + v[12][day] * s + fc <= c + long - s and (
            self.margin.allow_financing or self.own(a) >= 0
        )

    def supervised(self):
        """Whether maintenance calls, liquidation and insolvency are enforced."""
        return not self.classic

    # -------------------------------------------------------------- execution
    def execute(self, i, q, day, rebalance, forced):
        data, counts, coverage, costs, v = (
            self.data,
            self.counts,
            self.coverage,
            self.costs,
            self.v,
        )
        price, amount, comm, tax, fee, slip, impact = self.quote(q, i, day)
        before = self.quantities[i]
        change = self.movement(q, i, day)
        self.cash += change
        self.contribution[i] += change
        self.quantities[i] += q
        for key, x in (
            ("turnover", amount),
            ("commission", comm),
            ("tax", tax),
            ("other_fee", fee),
            ("slippage_cost", slip),
            ("impact_cost", impact),
        ):
            data[key][day] += x
        counts["trades"] += 1
        counts["min_commission"] += int(
            comm > amount * costs.commission_rate + abs(q) * costs.commission_per_share
        )
        counts["carry_trade"] += int(not rebalance)
        coverage["impact"] += int(impact > 0)
        coverage["per_share_fee"] += int(
            costs.commission_per_share > 0 or v[5 if q > 0 else 4][day] > 0
        )
        coverage["commission_cap"] += int(
            costs.commission_max_rate is not None
            and comm
            < max(
                amount * costs.commission_rate + abs(q) * costs.commission_per_share,
                (costs.commission_min or 0.0),
            )
        )
        coverage["open_short"] += int(self.quantities[i] < min(before, 0))
        coverage["cover_short"] += int(before < 0 and q > 0)
        if forced:
            coverage["liquidation"] += 1
            self.liquidations.append(LiquidationEvent(day, i, q))
        if self.trades is not None:
            self.trades.append(Trade(day, i, q, price, comm, tax, fee, slip, impact))

    # --------------------------------------------------------- daily sequence
    def run(self):
        for day in range(self.t):
            self.day(day)
        return self.result()

    def day(self, day):
        flow = self.cash_flow(day)
        self.corporate_actions(day)
        d = self.begin(day)
        if d.forced:
            self.liquidation_plan(d)
        for i in self.command_order(d.commands):
            self.spot_column(i, d)
        if d.orders:
            if self.classic:
                self.cash_basket(d, d.orders)
            else:
                self.margin_basket(d, d.orders)
        long, short = self.close_spot(day)
        self.finish(d, flow, long, short)

    def cash_flow(self, day):
        quantities, marks, data, coverage = (
            self.quantities,
            self.marks,
            self.data,
            self.coverage,
        )
        short = sum(-quantities[i] * marks[i] for i in self.spot if quantities[i] < 0)
        available = max(
            self.cash
            - (0 if self.margin.short_proceeds_usable else short)
            - self.collateral(day),
            0,
        )
        requested = float(self.v[10][day])
        flow = max(requested, -available)
        self.cash += flow
        data["cash_flow"][day] = flow
        data["withdrawal_shortfall"][day] = flow - requested
        coverage["cash_flow"] += int(flow != 0)
        coverage["withdrawal_shortfall"] += int(flow > requested)
        return flow

    def corporate_actions(self, day):
        market, config, v = self.market, self.config, self.v
        quantities, marks = self.quantities, self.marks
        data, counts, coverage = self.data, self.counts, self.coverage
        for i in self.spot:
            if quantities[i] != 0:
                div = (
                    float(market.cash_div[day, i])
                    if market.cash_div is not None
                    else 0.0
                )
                gross = quantities[i] * div
                tax = max(gross, 0) * v[8][day]
                self.cash += gross - tax
                self.contribution[i] += gross - tax
                data["dividend_gross"][day] += gross
                data["dividend_tax"][day] += tax
                counts["dividend"] += int(div > 0)
                coverage["dividend_tax"] += int(tax > 0)
                coverage["short_dividend"] += int(gross < 0)
                factor = (
                    float(market.split[day, i]) if market.split is not None else 1.0
                )
                quantities[i] *= factor
                counts["split"] += int(factor != 1)
            if marks[i] > 0:
                div_price = (
                    float(market.cash_div[day, i])
                    if market.cash_div is not None
                    else 0.0
                )
                factor_price = (
                    float(market.split[day, i]) if market.split is not None else 1.0
                )
                marks[i] = max(marks[i] - div_price, 0) / factor_price
            if self.closes[i] > 0:
                div_price = (
                    float(market.cash_div[day, i])
                    if market.cash_div is not None
                    else 0.0
                )
                factor_price = (
                    float(market.split[day, i]) if market.split is not None else 1.0
                )
                self.closes[i] = max(self.closes[i] - div_price, 0) / factor_price
            if market.delist is not None and market.delist[day, i]:
                self.dead[i] = True
                if quantities[i] != 0:
                    price = marks[i] * config.delist_recovery
                    proceeds = quantities[i] * price
                    self.cash += proceeds
                    self.contribution[i] += proceeds
                    self.events.append(
                        DelistEvent(day, i, quantities[i], price, proceeds)
                    )
                    quantities[i] = 0.0
                    counts["delist"] += 1

    def begin(self, day):
        """Resolve the day's commands: scheduled row, carried orders or liquidation."""
        targets = self.targets
        rebalance = self.cursor < len(targets.days) and targets.days[self.cursor] == day
        forced = self.supervised() and (
            (self.call and self.margin.margin_call == "liquidate") or self.bankrupt
        )
        if rebalance:
            self.pending = {}
            commands = {i: 0.0 for i in range(self.n) if self.quantities[i] != 0}
            for j in range(
                targets.indptr[self.cursor], targets.indptr[self.cursor + 1]
            ):
                commands[int(targets.cols[j])] = float(targets.weights[j])
            self.cursor += 1
        else:
            commands = dict(self.pending) if self.config.carry_blocked else {}
            self.pending = {}
        account = self.balance(day)
        value = account[0] + account[1] - account[2]
        return SimpleNamespace(
            day=day,
            rebalance=rebalance,
            forced=forced,
            commands=commands,
            requested=commands,
            account=account,
            value=value,
            planned={},
            orders=[],
            unfilled=False,
        )

    def liquidation_target(self, i, w, value, day, current):
        """Unrounded holdings requested by a target on a liquidation row."""
        p = float(self.market.exec_price[day, i])
        mark = p if math.isfinite(p) else self.marks[i]
        return (
            w
            if self.targets.kind == "shares"
            else w * max(value, 0) * (1 - self.config.cash_buffer) / mark
            if mark > 0
            else current
        )

    def liquidation_plan(self, d):
        day, quantities = d.day, self.quantities
        self.pending = {}
        requested = d.commands.copy()
        d.requested = requested
        account, value = d.account, d.value

        def liquidated(keep):
            a = account
            for i in range(self.n):
                current = quantities[i]
                if current == 0:
                    continue
                q = -math.copysign(
                    self.closing(abs(current) * (1 - keep), abs(current), i), current
                )
                if q != 0 and math.isfinite(self.market.exec_price[day, i]):
                    price = self.quote(q, i, day)[0]
                    if price > 0 and math.isfinite(price):
                        a = self.project(a, i, q, day)
            return a

        if self.bankrupt or value <= 0 or not self.feasible(liquidated(0), day):
            keep = 0.0
        elif self.feasible(liquidated(1), day):
            keep = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(64):
                mid = (lo + hi) / 2
                if self.feasible(liquidated(mid), day):
                    lo = mid
                else:
                    hi = mid
            keep = lo
        commands = {}
        for i, current in enumerate(quantities):
            if current == 0:
                continue
            amount = self.closing(abs(current) * (1 - keep), abs(current), i)
            if d.rebalance and i in requested:
                desired = self.liquidation_target(i, requested[i], value, day, current)
                reduction = (
                    current - max(desired, 0)
                    if current > 0
                    else min(desired, 0) - current
                )
                amount = max(amount, self.closing(reduction, abs(current), i))
            d.planned[i] = -math.copysign(amount, current)
            commands[i] = current + d.planned[i]
        d.commands = commands

    def command_order(self, commands):
        if self.classic:
            return sorted(commands)
        return sorted(commands, key=lambda i: (self.quantities[i] < 0, i))

    def cap(self, i, day):
        cap = math.inf
        market, config = self.market, self.config
        if market.volume is not None and config.max_participation > 0:
            volume = float(market.volume[day, i])
            cap = (
                volume * config.max_participation
                if math.isfinite(volume) and volume > 0
                else 0.0
            )
            if self.step(i) > 0:
                cap = math.floor(cap / self.step(i)) * self.step(i)
        return cap

    def spot_column(self, i, d):
        """Size one spot command, execute its reduction and queue its opening."""
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
        if (
            abs(wanted) > 1e-9 * max(abs(target), abs(after), 1)
            and not forced
            and not self.bankrupt
        ):
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

    def blocked(self, i, w, d, blocked_buy, blocked_sell):
        for flag, name in ((blocked_buy, "buy"), (blocked_sell, "sell")):
            if flag:
                self.data[f"n_blocked_{name}"][d.day] += 1
                self.counts[f"blocked_{name}"] += 1
        if (blocked_buy or blocked_sell) and self.config.carry_blocked and not d.forced:
            self.pending[i] = w

    def cash_basket(self, d, orders, reserved=0.0):
        """Legacy proportional-then-affordable long-only basket.

        `reserved` is cash already pledged elsewhere (future collateral).
        """
        day, quantities, v = d.day, self.quantities, self.v
        movement, opening, step = self.movement, self.opening, self.step
        total = sum(-movement(q, i, day) for i, q in orders)
        cash = self.cash - reserved
        scale = cash / total if total > cash else 1.0
        self.counts["cash_scaled"] += int(total > cash)
        for i, desired in orders:
            cash = self.cash - reserved
            q = opening(desired * scale, i) if scale < 1 else desired
            if q > 0 and -movement(q, i, day) > cash:
                if step(i) == 0:
                    lo, hi = 0.0, q
                    for _ in range(80):
                        mid = (lo + hi) / 2
                        if -movement(mid, i, day) <= cash:
                            lo = mid
                        else:
                            hi = mid
                    q = lo
                elif -movement(float(v[6][i]), i, day) > cash:
                    q = 0.0
                else:
                    lo, hi = 0, math.floor((q - v[6][i]) / step(i))
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if -movement(float(v[6][i] + mid * step(i)), i, day) <= cash:
                            lo = mid
                        else:
                            hi = mid - 1
                    q = float(v[6][i] + lo * step(i))
            if q > 1e-9 * max(abs(quantities[i] + desired), abs(quantities[i]), 1):
                self.execute(i, q, day, d.rebalance, False)

    def order_key(self, order):
        return (order[1] > 0, order[0])

    def margin_basket(self, d, orders, feasible=None):
        """Scale all openings by a common factor, then check each fill prefix.

        `feasible` defaults to the account's initial-margin and own-cash test.
        """
        day, quantities = d.day, self.quantities
        orders.sort(key=self.order_key)
        account = d.account
        if feasible is None:
            feasible = self.feasible

        def trial(scale):
            a = account
            for i, q in orders:
                x = math.copysign(self.opening(abs(q) * scale, i), q)
                if x != 0:
                    a = self.project(a, i, x, day)
            return a

        scale = 1.0
        if not feasible(trial(1.0), day):
            self.basket_scaled(orders)
            lo, hi = 0.0, 1.0
            for _ in range(64):
                mid = (lo + hi) / 2
                if feasible(trial(mid), day):
                    lo = mid
                else:
                    hi = mid
            scale = lo
        for i, desired in orders:
            q = math.copysign(self.opening(abs(desired) * scale, i), desired)
            if q != 0 and not feasible(self.project(account, i, q, day), day):
                lo, hi = 0.0, abs(q)
                for _ in range(64):
                    mid = (lo + hi) / 2
                    x = math.copysign(self.opening(mid, i), q)
                    if feasible(self.project(account, i, x, day), day):
                        lo = mid
                    else:
                        hi = mid
                q = math.copysign(self.opening(lo, i), q)
            if abs(q) > 1e-9 * max(abs(quantities[i] + desired), abs(quantities[i]), 1):
                account = self.project(account, i, q, day)
                self.execute(i, q, day, d.rebalance, False)
        d.account = account

    def basket_scaled(self, orders):
        self.coverage["margin_scaled"] += 1

    def close_spot(self, day):
        """Mark spot holdings at the close and charge borrow fees."""
        market, margin = self.market, self.margin
        quantities, marks, data = self.quantities, self.marks, self.data
        long = short = 0.0
        for i in self.spot:
            p = float(market.close[day, i])
            if math.isfinite(p):
                marks[i] = self.closes[i] = p
            elif quantities[i] == 0:
                # A flat column forgets any execution-price fallback, so a
                # later reentry is valued like a new position.
                marks[i] = self.closes[i]
            elif (
                not self.classic
                and marks[i] == 0
                and quantities[i] != 0
                and math.isfinite(market.exec_price[day, i])
            ):
                marks[i] = float(market.exec_price[day, i])
            value_i = quantities[i] * marks[i]
            if value_i > 0:
                long += value_i
            else:
                short -= value_i
                fee = -value_i * rate(margin.borrow_rate, day, i)
                self.cash -= fee
                self.contribution[i] -= fee
                data["borrow_cost"][day] += fee
                self.coverage["borrow_cost"] += int(fee > 0)
        return long, short

    def closing_collateral(self, day):
        """Future collateral at today's settlement; a spot account has none."""
        return 0.0

    def held(self):
        return any(self.quantities)

    def finish(self, d, flow, long, short):
        """Interest, equity, returns, maintenance status and bankruptcy."""
        day, v, config, data, coverage = (
            d.day,
            self.v,
            self.config,
            self.data,
            self.coverage,
        )
        fc = self.closing_collateral(day)
        own = self.own((self.cash, long, short, fc))
        interest = max(own, 0) * v[9][day]
        financing = max(-own, 0) * v[15][day]
        self.cash += interest - financing
        coverage["cash_interest"] += int(interest > 1e-10)
        coverage["financing_interest"] += int(financing > 1e-10)
        cash = self.cash
        equity = cash + long - short
        previous = self.previous
        denominator = previous + flow
        ret = (
            equity / denominator - 1
            if denominator
            > 8
            * np.finfo(float).eps
            * max(config.initial_cash, abs(previous), abs(flow))
            else 0
        )
        self.unit *= 1 + ret
        required = v[11][day] * long + v[12][day] * short + fc
        tolerance = 1e-10 * max(abs(equity), long, short, 1)
        self.call = self.supervised() and (
            v[13][day] * long + v[14][day] * short + fc > equity + tolerance
            or d.unfilled
        )
        coverage["margin_call"] += int(self.call)
        self.margin_called(self.call)
        valued = all(self.marks[i] > 0 for i in self.spot if self.quantities[i] != 0)
        if (
            self.supervised()
            and equity <= 0
            and not self.bankrupt
            and valued
            and (self.held() or cash < 0)
        ):
            self.bankrupt = True
            self.bankruptcies.append(day)
            coverage["bankruptcy"] += 1
            coverage["genuine_bankruptcy"] += int(short > 0 or own < -tolerance)
        for key, x in (
            ("nav", equity / config.initial_cash),
            ("cash", cash),
            ("cash_interest", interest),
            ("returns", ret),
            ("unit_nav", self.unit),
            ("long_value", long),
            ("short_value", short),
            ("financing_interest", financing),
            (
                "margin_usage",
                required / equity if equity > 0 else math.inf if required > 0 else 0,
            ),
            ("margin_call", self.call),
            ("equity", equity),
        ):
            data[key][day] = x
        data["turnover"][day] = (
            data["turnover"][day] * 0.5 / d.value if d.value > 0 else 0.0
        )
        self.previous = equity
        if self.positions is not None:
            self.positions[day] = self.quantities

    def margin_called(self, call):
        """Hook for instrument-specific call diagnostics."""

    def final_marks(self):
        values = np.zeros(self.n)
        for i in self.spot:
            values[i] = self.quantities[i] * self.marks[i]
        return values

    def result(self):
        self.contribution += self.final_marks()
        return Result(
            **self.data,
            positions=self.positions,
            trades=self.trades,
            delist_events=self.events,
            diagnostics=self.counts,
            coverage=self.coverage,
            contribution=self.contribution,
            liquidation_events=self.liquidations,
            bankruptcy_events=self.bankruptcies,
            expiry_events=self.expiries,
        )
