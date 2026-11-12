"""Scalar reference for mixed spot and daily-settled futures accounts.

Written from SPEC.md ("Futures: instrument columns, settlement and expiry"),
not from the Rust kernel. Spot rules are reused from `reference.Engine`
unchanged; this module adds the futures book and the mixed daily sequence:

    cash flow -> corporate actions -> commands (expired targets, liquidation
    plan) -> reductions (futures, then spot) -> openings (futures first) ->
    close: spot marks and borrow fees -> futures variation settlement ->
    expiry delivery -> interest -> maintenance and insolvency.

Account invariants checked at the end of every row:

    equity            = cash + spot_long - spot_short
    available cash    = cash - restricted_spot_short - future_collateral
    future collateral = sum |contracts| * multiplier * settlement * margin_rate
    every open contract's settlement basis equals today's settlement
    an expired contract holds no position; contract counts are integers
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .models import ExpiryEvent, LiquidationEvent, Trade
from .reference import Engine, rate


def run(market, targets, costs=None, config=None):
    return MixedEngine(market, targets, costs, config).run()


def per_contract(a, day, col):
    """Read a validated (1,1), (1,N) or T×N futures rate."""
    return float(a[day if a.shape[0] > 1 else 0, col if a.shape[1] > 1 else 0])


class FuturesBook:
    """Settlement basis and collateral marks of open contracts.

    Contract counts live in the engine's shared position vector so positions
    and liquidation events are recorded uniformly with spot columns.
    """

    def __init__(self, futures, n):
        self.f = futures
        self.columns = [i for i in range(n) if bool(futures.is_future[i])]
        self.multiplier = [float(x) for x in futures.multiplier]
        self.expiry = [int(x) for x in futures.expiry]
        # Last settlement or same-day execution price of the open position.
        self.basis = [0.0] * n
        # Price used for the column's collateral until the next close.
        self.collateral_mark = [0.0] * n
        self.held_ever = False

    def margin_rate(self, day, i):
        return per_contract(self.f.margin_rate, day, i)

    def collateral(self, contracts, day, i, mark):
        return abs(contracts) * self.multiplier[i] * mark * self.margin_rate(day, i)

    def expired(self, day, i):
        return day > self.expiry[i]


class MixedEngine(Engine):
    def __init__(self, market, targets, costs=None, config=None):
        self.book = None
        super().__init__(market, targets, costs, config)
        self.book = FuturesBook(market.futures, self.n)
        self.is_future = [bool(x) for x in market.futures.is_future]
        self.spot = [i for i in range(self.n) if not self.is_future[i]]
        self.classic_futures_basket = False
        # Reference-only event tallies for corpus coverage (not part of Result).
        self.stats = Counter()

    # ----------------------------------------------------------- input checks
    def validate_signs(self):
        fut = np.asarray(self.market.futures.is_future, dtype=bool)
        cols, w = self.targets.cols, self.targets.weights
        spot = ~fut[cols]
        if not self.margin.allow_short and np.any(w[spot] < 0):
            raise ValueError("negative spot targets require allow_short")
        if self.targets.kind == "shares" and np.any(w[~spot] != np.trunc(w[~spot])):
            raise ValueError("futures shares: target contracts must be integers")

    # ------------------------------------------- per-instrument fill rules
    def opening(self, q, i):
        if not self.is_future[i]:
            return super().opening(q, i)
        # Whole contracts, rounded toward zero.
        return float(math.floor(q)) if q >= 1 else 0.0

    def closing(self, q, pos, i):
        if not self.is_future[i]:
            return super().closing(q, pos, i)
        return min(q, pos) if q >= pos else 0.0 if q < 1 else float(math.floor(q))

    def cap(self, i, day):
        if not self.is_future[i]:
            return super().cap(i, day)
        market, config = self.market, self.config
        if market.volume is None or config.max_participation <= 0:
            return math.inf
        volume = float(market.volume[day, i])
        if not (math.isfinite(volume) and volume > 0):
            return 0.0
        return float(math.floor(volume * config.max_participation))

    def quote(self, q, i, day):
        """(price, notional, commission, tax, other_fee, slippage, impact)."""
        if not self.is_future[i]:
            return super().quote(q, i, day)
        if q == 0:
            return (0.0,) * 7
        market, costs, f = self.market, self.costs, self.market.futures
        mult = self.book.multiplier[i]
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
        n = abs(q)
        commission = n * (
            mult * price * per_contract(f.fee_rate, day, i)
            + per_contract(f.fee_per_contract, day, i)
        )
        return (
            price,
            n * mult * price,
            commission,
            0.0,
            0.0,
            n * raw * slip * mult,
            n * abs(price - slipped) * mult,
        )

    def realized(self, q, i, price):
        """Variation realized by closing contracts at `price` against their basis."""
        current = self.quantities[i]
        if current == 0 or current * q >= 0:
            return 0.0
        closed = min(abs(q), abs(current))
        return (
            math.copysign(closed, current)
            * self.book.multiplier[i]
            * (price - self.book.basis[i])
        )

    def movement(self, q, i, day):
        if not self.is_future[i]:
            return super().movement(q, i, day)
        price, _, commission, _, _, _, _ = self.quote(q, i, day)
        return self.realized(q, i, price) - commission

    # ------------------------------------------------------ account snapshots
    def collateral(self, day):
        """Collateral of open contracts at their current collateral marks."""
        if self.book is None:
            return 0.0
        total = 0.0
        for i in self.book.columns:
            if self.quantities[i] != 0:
                total += self.book.collateral(
                    self.quantities[i], day, i, self.book.collateral_mark[i]
                )
        return total

    def project(self, a, i, q, day):
        if not self.is_future[i]:
            return super().project(a, i, q, day)
        c, long, s, fc = a
        book, current = self.book, self.quantities[i]
        price = self.quote(q, i, day)[0]
        before = book.collateral(current, day, i, book.collateral_mark[i])
        after = book.collateral(current + q, day, i, price)
        return (c + self.movement(q, i, day), long, s, fc - before + after)

    def supervised(self):
        # A cash account is supervised only after it has held a contract.
        return not self.classic or (self.book is not None and self.book.held_ever)

    # -------------------------------------------------------------- execution
    def execute(self, i, q, day, rebalance, forced):
        if not self.is_future[i]:
            return super().execute(i, q, day, rebalance, forced)
        book, data, coverage = self.book, self.data, self.coverage
        price, notional, commission, _, _, slip, impact = self.quote(q, i, day)
        current = self.quantities[i]
        pnl = self.realized(q, i, price)
        change = pnl - commission
        self.cash += change
        self.contribution[i] += change
        data["futures_pnl"][day] += pnl
        after = current + q
        if current * after > 0 and abs(after) > abs(current):
            # Adding contracts: quantity-weighted basis.
            book.basis[i] = (current * book.basis[i] + q * price) / after
        elif current * after <= 0 and after != 0:
            # Opening from flat or reversing: the new side starts here.
            book.basis[i] = price
        self.quantities[i] = after
        book.collateral_mark[i] = price
        book.held_ever |= after != 0
        for key, x in (
            ("turnover", notional),
            ("commission", commission),
            ("slippage_cost", slip),
            ("impact_cost", impact),
        ):
            data[key][day] += x
        self.counts["trades"] += 1
        self.counts["carry_trade"] += int(not rebalance)
        coverage["impact"] += int(impact > 0)
        reducing = current != 0 and current * q < 0
        coverage["futures_close"] += int(reducing)
        coverage["futures_open"] += int(not reducing)
        if forced:
            coverage["liquidation"] += 1
            coverage["futures_liquidation"] += 1
            self.liquidations.append(LiquidationEvent(day, i, q))
        if self.trades is not None:
            self.trades.append(
                Trade(day, i, q, price, commission, 0.0, 0.0, slip, impact)
            )

    # --------------------------------------------------------- daily sequence
    def day(self, day):
        flow = self.cash_flow(day)
        self.corporate_actions(day)
        d = self.begin(day)
        if d.forced:
            # The requested row is replaced by the plan; its expired targets count.
            self.stats["expired_target_forced"] += self.count_expired(d, d.commands)
            self.liquidation_plan(d)
        order = self.command_order(d.commands)
        # Reductions: futures first, then spot long sales, then spot covers.
        for i in order:
            if self.is_future[i]:
                self.future_column(i, d)
        for i in order:
            if not self.is_future[i]:
                self.spot_column(i, d)
        self.openings(d)
        long, short = self.close_spot(day)
        self.settle(day)
        self.expire(day)
        self.finish(d, flow, long, short)
        self.check_invariants(day, long, short)

    def command_order(self, commands):
        # Futures first; within each instrument class the spot risk order
        # (long sales before short covers in a margin account) is kept.
        futures = super().command_order({i: 0 for i in commands if self.is_future[i]})
        spot = super().command_order({i: 0 for i in commands if not self.is_future[i]})
        return futures + spot

    def count_expired(self, d, commands):
        counted = 0
        for i, w in commands.items():
            if self.is_future[i] and self.book.expired(d.day, i) and w != 0:
                self.data["n_expired_target"][d.day] += 1
                self.coverage["expired_target"] += 1
                counted += 1
        return counted

    def liquidation_target(self, i, w, value, day, current):
        if not self.is_future[i]:
            return super().liquidation_target(i, w, value, day, current)
        if self.targets.kind == "shares":
            return w
        mark = self.execution_mark(i, day)
        # Unrounded contracts, exactly as spot uses unrounded shares.
        return (
            w
            * max(value, 0)
            * (1 - self.config.cash_buffer)
            / (mark * self.book.multiplier[i])
            if mark > 0
            else current
        )

    def execution_mark(self, i, day):
        """Execution price, else the last settlement of the column."""
        p = float(self.market.exec_price[day, i])
        if math.isfinite(p):
            return p
        if self.quantities[i] != 0:
            return self.book.collateral_mark[i]
        return float(self.market.futures.settlement[day - 1, i]) if day > 0 else 0.0

    def future_column(self, i, d):
        """Size one contract command, execute its reduction, queue its opening."""
        market, config, data, coverage = (
            self.market,
            self.config,
            self.data,
            self.coverage,
        )
        book, day, forced, value = self.book, d.day, d.forced, d.value
        w = d.commands[i]
        if book.expired(day, i):
            if w != 0 and not forced:
                data["n_expired_target"][day] += 1
                coverage["expired_target"] += 1
            return
        mult = book.multiplier[i]
        p = float(market.exec_price[day, i])
        mark = self.execution_mark(i, day)
        current = self.quantities[i]
        x = 0.0
        if forced or self.targets.kind == "shares":
            target = w
            coverage["shares_target"] += int(not forced)
        else:
            x = (
                w * max(value, 0.0) * (1 - config.cash_buffer) / (mark * mult)
                if mark > 0
                else 0.0
            )
            target = math.copysign(math.floor(abs(x)), x)
        if abs(target - current) <= 1e-9 * max(abs(target), abs(current), 1):
            target = current
        if (
            not forced
            and self.targets.kind == "weights"
            and w != 0
            and current != 0
            and value > 0
            and abs(w - current * mult * mark / value) < config.trade_band
        ):
            coverage["trade_band"] += 1
            return
        if not forced and self.targets.kind == "weights" and x != 0 and target == 0:
            # Genuine rounding of a nonzero size; banded orders are not sized,
            # and E <= 0 or a missing mark gives an exact zero, not a rounding.
            coverage["futures_rounded_to_zero"] += 1
        cap = self.cap(i, day)
        if forced:
            reduction = d.planned[i]
        elif current > 0 and target < current:
            reduction = -self.closing(current - max(target, 0), current, i)
        elif current < 0 and target > current:
            reduction = self.closing(min(target, 0) - current, -current, i)
        else:
            reduction = 0.0
        blocked_buy = blocked_sell = False
        used = 0.0
        if reduction != 0:
            allowed = math.isfinite(p) and (
                market.can_buy[day, i] if reduction > 0 else market.can_sell[day, i]
            )
            q = (
                math.copysign(
                    self.closing(min(abs(reduction), cap), abs(current), i), reduction
                )
                if allowed
                else 0.0
            )
            if q != reduction:
                blocked_buy = reduction > 0
                blocked_sell = reduction < 0
            # Reductions only release collateral: never trimmed by own cash.
            if q != 0:
                price = self.quote(q, i, day)[0]
                if price > 0 and math.isfinite(price):
                    before = self.own(d.account)
                    d.account = self.project(d.account, i, q, day)
                    self.execute(i, q, day, d.rebalance, forced)
                    used = abs(q)
                    # Debt created by a reduction is retained, never clipped.
                    self.stats["reduction_debt"] += int(
                        before >= 0 and self.own(d.account) < 0
                    )
                else:
                    data["n_invalid_quote"][day] += 1
                    coverage["invalid_quote"] += 1
            if forced and abs(reduction) - used > 1e-9 * max(
                abs(reduction), abs(current), 1
            ):
                d.unfilled = True
        after = self.quantities[i]
        if target > 0 and after >= 0 and target > after and target > current:
            wanted = self.opening(target - after, i)
        elif target < 0 and after <= 0 and target < after and target < current:
            wanted = -self.opening(after - target, i)
        else:
            wanted = 0.0
        if wanted != 0 and not forced and not self.bankrupt:
            # Future shorts borrow nothing: no allow_short or can_short check.
            allowed = math.isfinite(p) and (
                market.can_buy[day, i] if wanted > 0 else market.can_sell[day, i]
            )
            q = (
                math.copysign(
                    self.opening(min(abs(wanted), max(cap - used, 0)), i), wanted
                )
                if allowed
                else 0.0
            )
            if q != 0:
                price = self.quote(q, i, day)[0]
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

    def order_key(self, order):
        i, q = order
        return (not self.is_future[i], q > 0, i)

    def openings(self, d):
        futures = [o for o in d.orders if self.is_future[o[0]]]
        spot = [o for o in d.orders if not self.is_future[o[0]]]
        if not self.classic:
            if d.orders:
                self.margin_basket(d, d.orders)
            return
        # Cash specialization: contracts first, scaled so collateral <= cash,
        # then the legacy spot basket funded by cash minus collateral.
        if futures:
            self.classic_futures_basket = True
            self.margin_basket(d, futures, lambda a, day: self.own(a) >= 0)
            self.classic_futures_basket = False
        if spot:
            self.cash_basket(d, spot, reserved=self.collateral(d.day))

    def basket_scaled(self, orders):
        if not self.classic_futures_basket:
            self.coverage["margin_scaled"] += 1
        if any(self.is_future[i] for i, _ in orders):
            self.coverage["futures_margin_scaled"] += 1

    # ------------------------------------------------------------------ close
    def settle(self, day):
        """Daily variation: contracts earn settlement minus basis in cash."""
        settlement = self.market.futures.settlement
        for i in self.book.columns:
            q = self.quantities[i]
            if q == 0:
                continue
            price = float(settlement[day, i])
            variation = q * self.book.multiplier[i] * (price - self.book.basis[i])
            self.cash += variation
            self.contribution[i] += variation
            self.data["futures_pnl"][day] += variation
            self.book.basis[i] = self.book.collateral_mark[i] = price

    def expire(self, day):
        """Cash delivery at the expiry row's delivery settlement price."""
        f = self.market.futures
        for i in self.book.columns:
            q = self.quantities[i]
            if self.book.expiry[i] != day or q == 0:
                continue
            price = float(f.settlement[day, i])
            fee = abs(q) * (
                self.book.multiplier[i]
                * price
                * per_contract(f.delivery_fee_rate, day, i)
                + per_contract(f.delivery_fee_per_contract, day, i)
            )
            self.cash -= fee
            self.contribution[i] -= fee
            self.data["delivery_fee"][day] += fee
            self.data["other_fee"][day] += fee
            self.expiries.append(ExpiryEvent(day, i, q, price, fee))
            self.coverage["futures_expiry"] += 1
            self.quantities[i] = 0.0

    def closing_collateral(self, day):
        fc = self.collateral(day)
        self.data["futures_margin"][day] = fc
        return fc

    def margin_called(self, call):
        self.coverage["futures_margin_call"] += int(
            call and any(self.quantities[i] != 0 for i in self.book.columns)
        )

    def check_invariants(self, day, long, short):
        data, book = self.data, self.book
        settlement = self.market.futures.settlement
        assert data["equity"][day] == data["cash"][day] + long - short
        fc = 0.0
        for i in book.columns:
            q = self.quantities[i]
            assert q == math.floor(q), (day, i, q)
            if q != 0:
                assert not book.expired(day, i) and book.expiry[i] != day, (day, i)
                assert book.basis[i] == float(settlement[day, i]), (day, i)
                fc += book.collateral(q, day, i, float(settlement[day, i]))
        assert math.isclose(
            fc, data["futures_margin"][day], rel_tol=1e-12, abs_tol=1e-9
        )
        available = (
            self.cash
            - (0 if self.margin.short_proceeds_usable else short)
            - data["futures_margin"][day]
        )
        assert available == self.own(
            (self.cash, long, short, data["futures_margin"][day])
        )
