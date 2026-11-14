"""Independent scalar oracle for DESIGN 4.17's hourly futures contract.

No kernel functions or execution helpers are used. Candidate baskets are
simulated on copies of a ledger, then their full marked balance is recomputed.
This is deliberately slow and intended for adversarial small panels.
"""

import math

import numpy as np

from .models import (
    COUNTERS,
    CRYPTO_COUNTERS,
    CRYPTO_METRICS,
    EXTRA_COUNTERS,
    METRICS,
    Config,
    Costs,
    ExpiryEvent,
    LiquidationEvent,
    Result,
    Trade,
)


def scalar(value, day, col=0):
    a = np.asarray(value)
    if a.ndim == 0:
        return float(a)
    if a.ndim == 1:
        return float(a[day])
    return float(a[day if a.shape[0] > 1 else 0, col if a.shape[1] > 1 else 0])


class CryptoEngine:
    def __init__(self, market, targets, costs=None, config=None):
        self.m, self.f = market, market.futures
        self.l = self.f.liquidation
        self.plan = targets
        self.cfg, self.costs = config or Config(), costs or Costs()
        self.t, self.n = market.shape
        self.q = [0.0] * self.n
        self.b = [0.0] * self.n
        self.cash = self.previous = self.cfg.initial_cash
        self.unit = 1.0
        self.bankrupt = False
        self.pending = {}
        self.data = {
            k: np.zeros(
                self.t,
                dtype=np.int64
                if k.startswith("n_")
                else bool
                if k == "margin_call"
                else float,
            )
            for k in (*METRICS, *CRYPTO_METRICS)
        }
        self.counters = dict.fromkeys((*COUNTERS, *EXTRA_COUNTERS, *CRYPTO_COUNTERS), 0)
        self.contribution = np.zeros(self.n)
        self.positions = np.zeros((self.t, self.n)) if self.cfg.record_trades else None
        self.trades = [] if self.cfg.record_trades else None
        self.liquidations, self.bankruptcies, self.expiries = [], [], []
        self.seen_liquidation = False
        self.day = 0

    def count(self, name, amount=1):
        self.counters[name] += amount

    def add(self, name, value):
        self.data[name][self.day] += value

    def held(self):
        return [i for i, q in enumerate(self.q) if q]

    def quote(self, i, q, raw=None):
        d = self.day
        raw = float(self.m.exec_price[d, i]) if raw is None else raw
        slip = scalar(self.costs.slippage, d, i)
        slipped = raw * (1 + math.copysign(slip, q))
        impact = 0.0
        if self.costs.impact_k is not None:
            vol = float(self.m.volume[d, i])
            impact = (
                scalar(self.costs.impact_k, d, i)
                * (abs(q) / vol) ** self.costs.impact_exponent
                if math.isfinite(vol) and vol > 0
                else math.inf
            )
        price = slipped * (1 + math.copysign(impact, q))
        amount = abs(q) * float(self.f.multiplier[i]) * price
        fee = amount * scalar(self.f.fee_rate, d, i) + abs(q) * scalar(
            self.f.fee_per_contract, d, i
        )
        return (
            price,
            fee,
            amount,
            abs(q) * self.f.multiplier[i] * raw * slip,
            abs(q) * self.f.multiplier[i] * abs(price - slipped),
        )

    def ledger_fill(self, cash, quantities, bases, i, q, price, fee):
        old = quantities[i]
        realized = (
            math.copysign(min(abs(q), abs(old)), old)
            * self.f.multiplier[i]
            * (price - bases[i])
            if old * q < 0
            else 0.0
        )
        after = old + q
        if after == 0:
            bases[i] = 0.0
        elif old == 0:
            bases[i] = price
        elif old * q > 0:
            bases[i] = (abs(old) * bases[i] + abs(q) * price) / abs(after)
        quantities[i] = after
        return cash + realized - fee, realized

    def fill(self, i, q, *, raw=None, forced=False, carry=False):
        price, fee, amount, slip, impact = self.quote(i, q, raw)
        old = self.q[i]
        self.cash, pnl = self.ledger_fill(self.cash, self.q, self.b, i, q, price, fee)
        self.contribution[i] += pnl - fee
        for key, value in [
            ("futures_pnl", pnl),
            ("commission", fee),
            ("turnover", amount),
            ("slippage_cost", slip),
            ("impact_cost", impact),
        ]:
            self.add(key, value)
        self.count("trades")
        self.count("carry_trade", int(carry))
        self.count("impact", int(impact > 0))
        self.count("futures_open" if abs(self.q[i]) > abs(old) else "futures_close")
        if forced:
            self.count("liquidation")
            self.count("futures_liquidation")
            self.liquidations.append(LiquidationEvent(self.day, i, q))
        if self.trades is not None:
            self.trades.append(Trade(self.day, i, q, price, fee, 0, 0, slip, impact))
        return amount

    def balance(self, cash=None, quantities=None, bases=None, close=False):
        cash = self.cash if cash is None else cash
        quantities = self.q if quantities is None else quantities
        bases = self.b if bases is None else bases
        equity, im, mm, nominal = cash, 0.0, 0.0, 0.0
        for i, q in enumerate(quantities):
            if q == 0:
                continue
            price = float(
                (self.f.settlement if close else self.l.mark_open)[self.day, i]
            )
            equity += q * self.f.multiplier[i] * (price - bases[i])
            value = abs(q) * self.f.multiplier[i] * price
            nominal += value
            im += value * scalar(self.f.margin_rate, self.day, i)
            mm += value * scalar(
                self.f.margin_rate if self.f.maint_rate is None else self.f.maint_rate,
                self.day,
                i,
            )
        return equity, im, mm, nominal

    def failed(self, close=False):
        equity, _, mm, nominal = self.balance(close=close)
        return bool(self.held()) and equity - mm <= 1e-12 * nominal

    def check_marks(self, cols):
        for i in cols:
            if any(
                not math.isfinite(a[self.day, i]) or a[self.day, i] <= 0
                for a in (self.l.mark_open, self.l.mark_high, self.l.mark_low)
            ):
                raise ValueError(
                    f"mark: missing held value at row {self.day}, column {i}"
                )

    def tier(self, prices):
        return any(
            abs(self.q[i]) * self.f.multiplier[i] * prices[self.day, i]
            > scalar(self.l.tier_cap, self.day, i)
            for i in self.held()
        )

    def liquidate(self, phase, fraction=0):
        self.count("liquidation_" + phase)
        self.seen_liquidation = True
        dues = []
        for i in self.held():
            start = float(self.l.mark_open[self.day, i])
            end = float(
                (self.l.mark_low if self.q[i] > 0 else self.l.mark_high)[self.day, i]
            )
            raw = (
                float(self.f.settlement[self.day, i])
                if phase == "close"
                else start + fraction * (end - start)
            )
            quoted = self.quote(i, -self.q[i], raw)
            if not math.isfinite(quoted[0]) or quoted[0] <= 0:
                raise ValueError("liquidation quote: invalid")
            amount = self.fill(i, -self.q[i], raw=raw, forced=True)
            dues.append((i, amount * scalar(self.l.clearance_fee_rate, self.day, i)))
        total = sum(v for _, v in dues)
        fee = min(total, max(self.cash, 0.0))
        self.count("clearance_capped", int(fee < total))
        if total:
            for i, due in dues:
                self.contribution[i] -= fee * due / total
        self.cash -= fee
        self.add("clearance_fee", fee)
        self.add("other_fee", fee)
        if self.cash < 0:
            self.add("bankruptcy_deficit", -self.cash)
            self.count("bankruptcy_deficit")
            self.count("bankruptcy")
            self.count("genuine_bankruptcy")
            self.bankruptcies.append(self.day)
            self.cash = 0.0
            self.bankrupt = True
        self.pending.clear()

    def terms(self, i):
        return (
            1 if self.f.min_qty is None else float(self.f.min_qty[self.day, i]),
            1 if self.f.qty_step is None else float(self.f.qty_step[self.day, i]),
        )

    def lot(self, i, size, closing=False):
        if closing and size >= abs(self.q[i]):
            return abs(self.q[i])
        minimum, step = self.terms(i)
        result = math.floor(size / step) * step
        return result if result >= minimum else 0.0

    def opening(self, i, desired):
        qty = self.lot(i, abs(desired))
        if (
            self.f.min_notional is not None
            and qty * self.f.multiplier[i] * self.l.mark_open[self.day, i]
            < self.f.min_notional[self.day, i]
        ):
            return 0.0
        return math.copysign(qty, desired)

    def affordable(self, orders):
        # SPEC Ordinary-order decisions (3): collateral must exist BEFORE
        # simulating any candidate profit. Re-evaluate after reductions/fills.
        if any(qty for _, qty in orders) and self.balance()[0] <= 0:
            return False
        cash, q, b = self.cash, self.q.copy(), self.b.copy()
        for i, qty in orders:
            if qty:
                price, fee, *_ = self.quote(i, qty)
                cash, _ = self.ledger_fill(cash, q, b, i, qty, price, fee)
        equity, im, *_ = self.balance(cash, q, b)
        return equity >= im

    def market_obstacles(self, i, intent, capacity=math.inf):
        """SPEC (1): classify causes before rounding; size is not a cause."""
        reasons = set()
        if not math.isfinite(self.m.exec_price[self.day, i]):
            reasons.add("price")
        mask = self.m.can_buy if intent > 0 else self.m.can_sell
        if not mask[self.day, i]:
            reasons.add("mask")
        if abs(intent) > capacity:
            reasons.add("volume")
        return reasons

    def valid_ordinary_quote(self, i, qty):
        """SPEC (2): an unusable ordinary quote is another market obstacle."""
        price = self.quote(i, qty)[0]
        if math.isfinite(price) and price > 0:
            return True
        self.count("invalid_quote")
        self.add("n_invalid_quote", 1)
        return False

    def validate_intent_marks(self, i, intent):
        """SPEC (4): validate data before converting a nonzero weight intent.

        Held marks are checked by the row's risk stage. Expired commands have
        already been removed. An unheld masked/unquoted column needs no mark.
        """
        if intent and not self.q[i] and not self.market_obstacles(i, intent):
            self.check_marks([i])

    def execute(self, commands, scheduled, equity):
        openings = []
        for i, weight in sorted(commands.items()):
            d = self.day
            if d > self.f.expiry[i]:
                if weight:
                    self.add("n_expired_target", 1)
                    self.count("expired_target")
                continue
            current = self.q[i]
            self.validate_intent_marks(i, weight)
            mark = float(self.l.mark_open[d, i])
            if not math.isfinite(mark):
                # SPEC lifecycle placeholder: only for blocked, unheld sizing;
                # validate_intent_marks has already rejected executable gaps.
                mark = float(self.f.settlement[max(d - 1, 0), i])
            if self.plan.kind == "shares":
                self.count("shares_target")
                target = weight
            else:
                raw = (
                    weight
                    * max(equity, 0)
                    * (1 - self.cfg.cash_buffer)
                    / (mark * self.f.multiplier[i])
                    if mark > 0
                    else 0.0
                )
                target = math.copysign(math.floor(abs(raw)), raw)
                self.count("futures_rounded_to_zero", int(raw != 0 and target == 0))
                if (
                    weight
                    and current
                    and equity > 0
                    and abs(weight - current * mark * self.f.multiplier[i] / equity)
                    < self.cfg.trade_band
                ):
                    self.count("trade_band")
                    continue
            limit = math.inf
            if self.cfg.max_participation > 0 and self.m.volume is not None:
                volume = float(self.m.volume[d, i])
                limit = (
                    volume * self.cfg.max_participation
                    if math.isfinite(volume) and volume > 0
                    else 0
                )
            reduction = (
                -max(current - max(target, 0), 0)
                if current > 0
                else max(min(target, 0) - current, 0)
                if current < 0
                else 0
            )
            blocked = set()
            used = 0.0

            if reduction:
                obstacles = self.market_obstacles(i, reduction, limit)
                permitted = not obstacles.intersection({"price", "mask"})
                qty = (
                    math.copysign(
                        self.lot(i, min(abs(reduction), limit), True), reduction
                    )
                    if permitted
                    else 0.0
                )
                if obstacles:
                    blocked.add("buy" if reduction > 0 else "sell")
                if qty:
                    if self.valid_ordinary_quote(i, qty):
                        self.fill(i, qty, carry=not scheduled)
                        used = abs(qty)
                        self.count("futures_full_close", int(abs(qty) == abs(current)))
                        if (
                            self.f.min_notional is not None
                            and abs(qty) * self.f.multiplier[i] * mark
                            < self.f.min_notional[d, i]
                        ):
                            self.count("futures_reduce_exempt")
                    else:
                        blocked.add("buy" if reduction > 0 else "sell")
                elif permitted:
                    self.count("futures_rounded_to_zero")
            after = self.q[i]
            desired = (
                target - after
                if (target > after and target > current and after >= 0)
                or (target < after and target < current and after <= 0)
                else 0.0
            )
            if desired:
                obstacles = self.market_obstacles(i, desired, max(limit - used, 0))
                if obstacles:
                    blocked.add("buy" if desired > 0 else "sell")
                if not obstacles.intersection({"price", "mask"}):
                    self.check_marks([i])
                    limited = math.copysign(
                        min(abs(desired), max(limit - used, 0)), desired
                    )
                    qty = self.opening(i, limited)
                    self.count("futures_rule_scaled", int(abs(qty) < abs(desired)))
                    if qty == 0:
                        self.count("futures_rounded_to_zero")
                        if (
                            self.f.min_notional is not None
                            and self.lot(i, abs(limited)) * self.f.multiplier[i] * mark
                            < self.f.min_notional[d, i]
                        ):
                            self.count("futures_min_notional")
                    elif self.valid_ordinary_quote(i, qty):
                        self.count("reversal", int(current * qty < 0 and used > 0))
                        openings.append((i, qty))
                    else:
                        blocked.add("buy" if desired > 0 else "sell")
            for side in blocked:
                self.count("blocked_" + side)
                self.add("n_blocked_" + side, 1)
            if blocked and self.cfg.carry_blocked:
                self.pending[i] = weight

        def scaled(scale):
            return [(i, self.opening(i, q * scale)) for i, q in openings]

        if self.affordable(scaled(1)):
            scale = 1.0
        else:
            self.count("futures_margin_scaled")
            lo, hi = 0.0, 1.0
            for _ in range(64):
                mid = (lo + hi) / 2
                if self.affordable(scaled(mid)):
                    lo = mid
                else:
                    hi = mid
            scale = lo
        for i, desired in openings:
            q = self.opening(i, desired * scale)
            if q and not self.affordable([(i, q)]):
                lo, hi = 0.0, abs(q)
                for _ in range(64):
                    mid = (lo + hi) / 2
                    trial = self.opening(i, math.copysign(mid, q))
                    if trial == 0 or self.affordable([(i, trial)]):
                        lo = mid
                    else:
                        hi = mid
                q = self.opening(i, math.copysign(lo, q))
            self.count("futures_rule_scaled", int(abs(q) < abs(desired)))
            if not q:
                self.count("futures_rounded_to_zero")
                continue
            self.count("futures_reopen", int(self.seen_liquidation))
            self.fill(i, q, carry=not scheduled)

    def run(self):
        dense = self.plan.to_dense(self.t, self.n)
        for d in range(self.t):
            self.day = d
            # Pre-open cash flow uses the previous basis and this row's IM.
            reserved = sum(
                abs(self.q[i])
                * self.f.multiplier[i]
                * self.b[i]
                * scalar(self.f.margin_rate, d, i)
                for i in self.held()
            )
            requested = scalar(self.cfg.cash_flow, d)
            flow = max(requested, -max(self.cash - reserved, 0))
            flow = 0.0 if flow == 0 else flow
            self.cash += flow
            self.add("cash_flow", flow)
            self.add("withdrawal_shortfall", flow - requested)
            self.count("cash_flow", int(flow != 0))
            self.count("withdrawal_shortfall", int(flow > requested))
            self.check_marks(self.held())
            tier = self.tier(self.l.mark_open)
            equity = self.balance()[0]
            failed = self.failed()
            if failed:
                self.liquidate("open")
            scheduled = not np.isnan(dense[d]).all()
            commands = {}
            if not failed and not self.bankrupt:
                if scheduled:
                    commands = {
                        i: float(np.nan_to_num(w)) for i, w in enumerate(dense[d])
                    }
                    commands = {
                        i: w for i, w in commands.items() if w != 0 or self.q[i] != 0
                    }
                    self.pending.clear()
                elif self.cfg.carry_blocked:
                    commands, self.pending = self.pending, {}
            self.execute(commands, scheduled, equity)
            tier |= self.tier(self.l.mark_open)
            if self.failed():
                self.liquidate("post_trade")
            tier |= self.tier(self.l.mark_high)
            if self.held():
                before = self.balance()
                # Evaluate the full endpoint ledger instead of using the
                # kernel's closed-form derivative.
                end_equity, end_maintenance = self.cash, 0.0
                for i in self.held():
                    p = float(
                        (self.l.mark_low if self.q[i] > 0 else self.l.mark_high)[d, i]
                    )
                    end_equity += self.q[i] * self.f.multiplier[i] * (p - self.b[i])
                    end_maintenance += (
                        abs(self.q[i])
                        * self.f.multiplier[i]
                        * p
                        * scalar(
                            self.f.margin_rate
                            if self.f.maint_rate is None
                            else self.f.maint_rate,
                            d,
                            i,
                        )
                    )
                start = before[0] - before[2]
                end = end_equity - end_maintenance
                if end >= start:
                    self.count("liquidation_nondecreasing")
                elif end <= before[3] * 1e-12:
                    fraction = min(start / (start - end), 1.0)
                    self.count(
                        "liquidation_single"
                        if len(self.held()) == 1
                        else "liquidation_multi"
                    )
                    self.count(
                        "liquidation_mixed",
                        int(any(q > 0 for q in self.q) and any(q < 0 for q in self.q)),
                    )
                    self.count(
                        "liquidation_lambda_one", int(abs(fraction - 1) <= 1e-12)
                    )
                    tier = True
                    self.liquidate("intraday", fraction)
            tier |= self.tier(self.f.settlement)
            for i in self.held():
                price = float(self.f.settlement[d, i])
                pnl = self.q[i] * self.f.multiplier[i] * (price - self.b[i])
                self.cash += pnl
                self.contribution[i] += pnl
                self.add("futures_pnl", pnl)
                self.b[i] = price
                if d == self.f.expiry[i]:
                    fee = abs(self.q[i]) * (
                        self.f.multiplier[i]
                        * price
                        * scalar(self.f.delivery_fee_rate, d, i)
                        + scalar(self.f.delivery_fee_per_contract, d, i)
                    )
                    self.cash -= fee
                    self.contribution[i] -= fee
                    self.add("delivery_fee", fee)
                    self.add("other_fee", fee)
                    self.expiries.append(ExpiryEvent(d, i, self.q[i], price, fee))
                    self.count("futures_expiry")
                    self.q[i] = self.b[i] = 0.0
            if self.f.funding is not None:
                for i in self.held():
                    per = float(self.f.funding[d, i])
                    if not math.isfinite(per):
                        raise ValueError("funding: missing held value")
                    paid = self.q[i] * per
                    self.cash -= paid
                    self.contribution[i] -= paid
                    self.add("funding", paid)
                    self.count("funding_paid", int(paid > 0))
                    self.count("funding_received", int(paid < 0))
            if self.failed(close=True):
                self.liquidate("close")
            self.count("tier_cap_exceeded", int(tier))
            im = self.balance(close=True)[1]
            own = self.cash - im
            interest = max(own, 0) * scalar(self.cfg.cash_rate, d)
            financing = max(-own, 0) * scalar(self.cfg.margin.financing_rate, d)
            self.cash += interest - financing
            self.count("cash_interest", int(interest > 1e-10))
            self.count("financing_interest", int(financing > 1e-10))
            if self.cash < 0 and not self.bankrupt:
                self.bankrupt = True
                self.bankruptcies.append(d)
                self.count("bankruptcy")
                self.count("genuine_bankruptcy")
            denom = self.previous + flow
            ret = (
                self.cash / denom - 1
                if denom
                > 8
                * np.finfo(float).eps
                * max(self.cfg.initial_cash, abs(self.previous), abs(flow))
                else 0.0
            )
            self.unit *= 1 + ret
            for name, value in [
                ("cash", self.cash),
                ("equity", self.cash),
                ("nav", self.cash / self.cfg.initial_cash),
                ("cash_interest", interest),
                ("financing_interest", financing),
                ("returns", ret),
                ("unit_nav", self.unit),
                ("futures_margin", im),
                (
                    "margin_usage",
                    im / self.cash if self.cash > 0 else math.inf if im > 0 else 0,
                ),
            ]:
                self.data[name][d] = value
            self.data["turnover"][d] *= 0.5 / equity if equity > 0 else 0
            self.previous = self.cash
            if self.positions is not None:
                self.positions[d] = self.q
        return Result(
            **self.data,
            positions=self.positions,
            trades=self.trades,
            delist_events=[],
            diagnostics={k: self.counters[k] for k in COUNTERS},
            coverage={k: self.counters[k] for k in (*EXTRA_COUNTERS, *CRYPTO_COUNTERS)},
            contribution=self.contribution,
            liquidation_events=self.liquidations,
            bankruptcy_events=self.bankruptcies,
            expiry_events=self.expiries,
        )


def run(market, targets, costs=None, config=None):
    return CryptoEngine(market, targets, costs, config).run()
