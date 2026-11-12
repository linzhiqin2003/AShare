"""Independent scalar specification for one daily spot and futures account."""

from __future__ import annotations

import math

import numpy as np

from .models import (
    COUNTERS,
    EXTRA_COUNTERS,
    METRICS,
    Config,
    Costs,
    DelistEvent,
    ExpiryEvent,
    LiquidationEvent,
    Result,
    Trade,
)


def run(market, targets, costs=None, config=None):
    """Execute mixed columns without using Rust or its prepared argument arrays."""
    costs, config = costs or Costs(), config or Config()
    spec = market.futures
    margin = config.margin
    t, n = market.shape
    targets.validate_shape(t, n)
    for col, target in zip(targets.cols, targets.weights, strict=True):
        if target < 0 and not spec.is_future[col] and not margin.allow_short:
            raise ValueError("negative spot targets require allow_short")
        if (
            targets.kind == "shares"
            and spec.is_future[col]
            and target != math.trunc(target)
        ):
            raise ValueError("futures shares: target contracts must be integers")

    def daily(value, size):
        x = np.asarray(value, dtype=float)
        return np.full(size, float(x)) if x.ndim == 0 else x.copy()

    rates = {
        name: daily(getattr(costs, name), t)
        for name in (
            "tax_sell",
            "tax_buy",
            "fee_rate_sell",
            "fee_rate_buy",
            "fee_per_share_sell",
            "fee_per_share_buy",
        )
    }
    rates.update(
        {
            name: daily(getattr(margin, name), t)
            for name in (
                "long_initial",
                "short_initial",
                "long_maintenance",
                "short_maintenance",
                "financing_rate",
            )
        }
    )
    rates["cash_rate"] = daily(config.cash_rate, t)
    rates["cash_flow"] = daily(config.cash_flow, t)
    rates["dividend_tax_rate"] = daily(config.dividend_tax_rate, t)
    minimum = daily(config.min_qty, n)
    step_size = daily(config.qty_step, n)
    pos = [0.0] * n
    mark = [0.0] * n
    basis = [0.0] * n
    cash = float(config.initial_cash)
    data = {
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
    counts = dict.fromkeys(COUNTERS, 0)
    coverage = dict.fromkeys(EXTRA_COUNTERS, 0)
    contribution = np.zeros(n)
    positions = np.zeros((t, n)) if config.record_trades else None
    trades = [] if config.record_trades else None
    events, expiries, liquidations, bankruptcies = [], [], [], []
    pending = {}
    cursor = 0
    previous = cash
    unit = 1.0
    call = bankrupt = False

    def at(value, day, i):
        a = np.asarray(value)
        if a.ndim == 0:
            return float(a)
        if a.ndim == 1:
            return float(a[day])
        return float(a[day, i])

    def fee_at(value, day, i):
        a = np.asarray(value)
        if a.size == 1:
            return float(a.flat[0])
        if a.shape == (1, n):
            return float(a[0, i])
        return float(a[day, i])

    def future_margin(i, q, p, day):
        return abs(q) * spec.multiplier[i] * p * fee_at(spec.margin_rate, day, i)

    def total_future_margin(day):
        return sum(
            future_margin(i, pos[i], basis[i], day)
            for i in range(n)
            if spec.is_future[i] and pos[i] != 0
        )

    def lot_open(quantity, i):
        if spec.is_future[i]:
            return math.floor(max(quantity, 0))
        step = 0 if targets.kind == "shares" else step_size[i]
        if step == 0:
            return max(quantity, 0)
        return (
            0.0
            if quantity < minimum[i]
            else minimum[i] + math.floor((quantity - minimum[i]) / step) * step
        )

    def lot_close(quantity, current, i):
        if quantity >= current:
            return current
        if spec.is_future[i]:
            return math.floor(max(quantity, 0))
        step = 0 if targets.kind == "shares" else step_size[i]
        if step == 0:
            return max(quantity, 0)
        return 0 if quantity < minimum[i] else math.floor(quantity / step) * step

    def tolerance(a, b):
        return 1e-9 * max(abs(a), abs(b), 1)

    def quote(i, q, day):
        raw = float(market.exec_price[day, i])
        slip = at(costs.slippage, day, i)
        slipped = raw * (1 + math.copysign(slip, q))
        impact = 0.0
        if costs.impact_k is not None:
            volume = float(market.volume[day, i])
            impact = (
                float(costs.impact_k[day, i])
                * (abs(q) / volume) ** costs.impact_exponent
                if math.isfinite(volume) and volume > 0
                else math.inf
            )
        price = slipped * (1 + math.copysign(impact, q))
        multiplier = spec.multiplier[i] if spec.is_future[i] else 1.0
        amount = abs(q) * price * multiplier
        slip_cost = abs(q) * raw * slip * multiplier
        impact_cost = abs(q) * abs(price - slipped) * multiplier
        if spec.is_future[i]:
            commission = abs(q) * (
                multiplier * price * fee_at(spec.fee_rate, day, i)
                + fee_at(spec.fee_per_contract, day, i)
            )
            return price, amount, commission, 0.0, 0.0, slip_cost, impact_cost
        commission = max(
            amount * costs.commission_rate + abs(q) * costs.commission_per_share,
            costs.commission_min or 0.0,
        )
        if costs.commission_max_rate is not None:
            commission = min(commission, costs.commission_max_rate * amount)
        direction = "buy" if q > 0 else "sell"
        tax = amount * rates[f"tax_{direction}"][day]
        other = (
            amount * rates[f"fee_rate_{direction}"][day]
            + abs(q) * rates[f"fee_per_share_{direction}"][day]
        )
        return price, amount, commission, tax, other, slip_cost, impact_cost

    def spot_movement(i, q, day):
        _, amount, commission, tax, other, _, _ = quote(i, q, day)
        return (
            -amount - commission - tax - other
            if q > 0
            else amount - commission - tax - other
        )

    def realized(i, q, price):
        before = pos[i]
        if before * q >= 0:
            return 0.0
        closed = math.copysign(min(abs(before), abs(q)), before)
        return closed * spec.multiplier[i] * (price - basis[i])

    def balance(day):
        long = short = 0.0
        for i, q in enumerate(pos):
            if spec.is_future[i] or q == 0:
                continue
            p = float(market.exec_price[day, i])
            value = q * (p if math.isfinite(p) else mark[i])
            if value > 0:
                long += value
            else:
                short -= value
        return cash, long, short

    def project(account, fm, i, q, day):
        c, long, short = account
        if q == 0:
            return account, fm
        price = quote(i, q, day)[0]
        if spec.is_future[i]:
            c += realized(i, q, price) - quote(i, q, day)[2]
            fm += future_margin(i, pos[i] + q, price, day)
            fm -= future_margin(i, pos[i], basis[i], day)
        else:
            c += spot_movement(i, q, day)
            raw = float(market.exec_price[day, i])
            before, after = pos[i], pos[i] + q
            long += (max(after, 0) - max(before, 0)) * raw
            short -= (min(after, 0) - min(before, 0)) * raw
        return (c, long, short), fm

    def feasible(account, fm, day):
        c, long, short = account
        equity = c + long - short
        needed = (
            rates["long_initial"][day] * long + rates["short_initial"][day] * short + fm
        )
        own = c - (0 if margin.short_proceeds_usable else short) - fm
        return needed <= equity and (margin.allow_financing or own >= 0)

    def target_quantity(i, w, value, day):
        if targets.kind == "shares":
            return w
        price = float(market.exec_price[day, i])
        mark_price = (
            price
            if math.isfinite(price)
            else basis[i]
            if spec.is_future[i]
            else mark[i]
        )
        if mark_price <= 0:
            return 0.0
        multiplier = spec.multiplier[i] if spec.is_future[i] else 1
        raw = w * max(value, 0) * (1 - config.cash_buffer) / (mark_price * multiplier)
        if spec.is_future[i]:
            return math.copysign(math.floor(abs(raw)), raw)
        return (
            raw
            if abs(raw) < minimum[i] and pos[i] * raw > 0
            else math.copysign(lot_open(abs(raw), i), raw)
        )

    def previous_close(day, i):
        for earlier in range(day - 1, -1, -1):
            px = float(market.close[earlier, i])
            if not math.isfinite(px):
                continue
            for later in range(earlier + 1, day + 1):
                if market.cash_div is not None:
                    px = max(px - float(market.cash_div[later, i]), 0)
                if market.split is not None:
                    px /= float(market.split[later, i])
            return px
        return 0.0

    def execute(i, q, day, rebalance, forced):
        nonlocal cash
        price, amount, commission, tax, other, slip, impact = quote(i, q, day)
        before = pos[i]
        if spec.is_future[i]:
            pnl = realized(i, q, price)
            cash += pnl - commission
            contribution[i] += pnl - commission
            data["futures_pnl"][day] += pnl
            after = before + q
            basis[i] = (
                0.0
                if after == 0
                else price
                if before == 0 or math.copysign(1, before) != math.copysign(1, after)
                else (abs(before) * basis[i] + abs(q) * price) / abs(after)
                if math.copysign(1, before) == math.copysign(1, q)
                else basis[i]
            )
            coverage[
                "futures_open" if abs(after) > abs(before) else "futures_close"
            ] += 1
        else:
            change = -q * price - commission - tax - other
            cash += change
            contribution[i] += change
            counts["min_commission"] += int(
                commission
                > amount * costs.commission_rate + abs(q) * costs.commission_per_share
            )
            coverage["per_share_fee"] += int(
                costs.commission_per_share > 0
                or rates[f"fee_per_share_{'buy' if q > 0 else 'sell'}"][day] > 0
            )
            coverage["commission_cap"] += int(
                costs.commission_max_rate is not None
                and commission
                < max(
                    amount * costs.commission_rate
                    + abs(q) * costs.commission_per_share,
                    costs.commission_min or 0,
                )
            )
            coverage["open_short"] += int(before + q < min(before, 0))
            coverage["cover_short"] += int(before < 0 and q > 0)
        pos[i] += q
        for key, x in (
            ("turnover", amount),
            ("commission", commission),
            ("tax", tax),
            ("other_fee", other),
            ("slippage_cost", slip),
            ("impact_cost", impact),
        ):
            data[key][day] += x
        counts["trades"] += 1
        counts["carry_trade"] += int(not rebalance)
        coverage["impact"] += int(impact > 0)
        if forced:
            coverage["liquidation"] += 1
            coverage["futures_liquidation"] += int(spec.is_future[i])
            liquidations.append(LiquidationEvent(day, i, q))
        if trades is not None:
            trades.append(Trade(day, i, q, price, commission, tax, other, slip, impact))

    for day in range(t):
        fm = total_future_margin(day)
        restricted = (
            sum(
                -pos[i] * mark[i]
                for i in range(n)
                if not spec.is_future[i] and pos[i] < 0
            )
            if not margin.short_proceeds_usable
            else 0.0
        )
        available = max(cash - restricted - fm, 0)
        requested = rates["cash_flow"][day]
        flow = max(requested, -available)
        cash += flow
        data["cash_flow"][day] = flow
        data["withdrawal_shortfall"][day] = flow - requested
        coverage["cash_flow"] += int(flow != 0)
        coverage["withdrawal_shortfall"] += int(flow > requested)
        for i in range(n):
            if spec.is_future[i]:
                continue
            div = float(market.cash_div[day, i]) if market.cash_div is not None else 0.0
            split = float(market.split[day, i]) if market.split is not None else 1.0
            if pos[i] != 0:
                gross = pos[i] * div
                tax = max(gross, 0) * rates["dividend_tax_rate"][day]
                cash += gross - tax
                contribution[i] += gross - tax
                data["dividend_gross"][day] += gross
                data["dividend_tax"][day] += tax
                counts["dividend"] += int(div > 0)
                coverage["dividend_tax"] += int(tax > 0)
                coverage["short_dividend"] += int(gross < 0)
                pos[i] *= split
                counts["split"] += int(split != 1)
            mark[i] = max(mark[i] - div, 0) / split if mark[i] > 0 else 0
            if market.delist is not None and market.delist[day, i] and pos[i] != 0:
                exit_price = mark[i] * config.delist_recovery
                proceeds = pos[i] * exit_price
                cash += proceeds
                contribution[i] += proceeds
                events.append(DelistEvent(day, i, pos[i], exit_price, proceeds))
                pos[i] = 0
                counts["delist"] += 1

        rebalance = cursor < len(targets.days) and targets.days[cursor] == day
        forced = (call and margin.margin_call == "liquidate") or bankrupt
        if rebalance:
            pending = {}
            commands = {i: 0.0 for i, q in enumerate(pos) if q != 0}
            for j in range(targets.indptr[cursor], targets.indptr[cursor + 1]):
                commands[int(targets.cols[j])] = float(targets.weights[j])
            cursor += 1
        else:
            commands = dict(pending) if config.carry_blocked else {}
            pending = {}
        account = balance(day)
        value = account[0] + account[1] - account[2]
        planned = {}
        if forced:
            requested_commands = commands.copy()
            commands = {}

            def liquidated(keep, account=account, fm=fm, day=day):
                a, collateral = account, fm
                held = sorted(
                    (i for i, q in enumerate(pos) if q != 0),
                    key=lambda i: (not spec.is_future[i], pos[i] < 0, i),
                )
                for i in held:
                    q = -math.copysign(
                        lot_close(abs(pos[i]) * (1 - keep), abs(pos[i]), i), pos[i]
                    )
                    if q != 0 and math.isfinite(market.exec_price[day, i]):
                        p = quote(i, q, day)[0]
                        if math.isfinite(p) and p > 0:
                            a, collateral = project(a, collateral, i, q, day)
                return a, collateral

            full_a, full_fm = liquidated(0)
            if bankrupt or value <= 0 or not feasible(full_a, full_fm, day):
                keep = 0.0
            elif feasible(account, fm, day):
                keep = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(64):
                    mid = (lo + hi) / 2
                    a, collateral = liquidated(mid)
                    if feasible(a, collateral, day):
                        lo = mid
                    else:
                        hi = mid
                keep = lo
            for i, current in enumerate(pos):
                if current == 0:
                    continue
                amount = lot_close(abs(current) * (1 - keep), abs(current), i)
                if rebalance and i in requested_commands:
                    desired = target_quantity(i, requested_commands[i], value, day)
                    reduction = (
                        current - max(desired, 0)
                        if current > 0
                        else min(desired, 0) - current
                    )
                    amount = max(amount, lot_close(reduction, abs(current), i))
                planned[i] = -math.copysign(amount, current)
                commands[i] = current + planned[i]

        orders = []
        unfilled_liquidation = False
        command_order = sorted(
            commands, key=lambda i: (not spec.is_future[i], pos[i] < 0, i)
        )
        for i in command_order:
            w = commands[i]
            if not spec.is_future[i] and market.first_delist[i] <= day:
                continue
            if spec.is_future[i] and day > spec.expiry[i]:
                if w != 0:
                    data["n_expired_target"][day] += 1
                    coverage["expired_target"] += 1
                continue
            if not spec.is_future[i] and pos[i] == 0:
                mark[i] = previous_close(day, i)
            current = pos[i]
            p = float(market.exec_price[day, i])
            mark_price = (
                p if math.isfinite(p) else basis[i] if spec.is_future[i] else mark[i]
            )
            target = w if forced else target_quantity(i, w, value, day)
            coverage["shares_target"] += int(targets.kind == "shares" and not forced)
            if abs(target - current) <= tolerance(target, current):
                target = current
            current_notional = (
                current * mark_price * (spec.multiplier[i] if spec.is_future[i] else 1)
            )
            if (
                not forced
                and targets.kind == "weights"
                and w != 0
                and current != 0
                and value > 0
                and abs(w - current_notional / value) < config.trade_band
            ):
                coverage["trade_band"] += 1
                continue
            cap = math.inf
            if market.volume is not None and config.max_participation > 0:
                vol = float(market.volume[day, i])
                cap = (
                    vol * config.max_participation
                    if math.isfinite(vol) and vol > 0
                    else 0.0
                )
                step = (
                    1
                    if spec.is_future[i]
                    else 0
                    if targets.kind == "shares"
                    else step_size[i]
                )
                if step > 0:
                    cap = math.floor(cap / step) * step
            reduction = (
                planned.get(i, 0.0)
                if forced
                else -lot_close(current - max(target, 0), current, i)
                if current > 0 and target < current
                else lot_close(min(target, 0) - current, -current, i)
                if current < 0 and target > current
                else 0.0
            )
            blocked_buy = blocked_sell = False
            used = 0.0
            if reduction != 0:
                allowed = math.isfinite(p) and (
                    market.can_buy[day, i] if reduction > 0 else market.can_sell[day, i]
                )
                q = (
                    math.copysign(
                        lot_close(min(abs(reduction), cap), abs(current), i), reduction
                    )
                    if allowed
                    else 0.0
                )
                if q != reduction:
                    blocked_buy |= reduction > 0
                    blocked_sell |= reduction < 0
                if q != 0:
                    px = quote(i, q, day)[0]
                    if math.isfinite(px) and px > 0:
                        candidate, candidate_fm = project(account, fm, i, q, day)
                        own_before = (
                            account[0]
                            - (0 if margin.short_proceeds_usable else account[2])
                            - fm
                        )
                        own_after = (
                            candidate[0]
                            - (0 if margin.short_proceeds_usable else candidate[2])
                            - candidate_fm
                        )
                        if (
                            not margin.allow_financing
                            and own_before >= -1e-10 * config.initial_cash
                            and own_after < 0
                        ):
                            requested_q = q
                            lo, hi = 0.0, abs(q)
                            for _ in range(64):
                                mid = (lo + hi) / 2
                                trial_q = math.copysign(
                                    lot_close(mid, abs(current), i), q
                                )
                                a, collateral = project(account, fm, i, trial_q, day)
                                own = (
                                    a[0]
                                    - (0 if margin.short_proceeds_usable else a[2])
                                    - collateral
                                )
                                if own >= 0:
                                    lo = mid
                                else:
                                    hi = mid
                            q = math.copysign(lot_close(lo, abs(current), i), q)
                            if abs(q - requested_q) > tolerance(q, requested_q):
                                data["n_reduction_clipped"][day] += 1
                                coverage["reduction_clipped"] += 1
                        if q != 0:
                            candidate, candidate_fm = project(account, fm, i, q, day)
                            x = quote(i, q, day)
                            if spec.is_future[i] or cash + x[1] >= x[2] + x[3] + x[4]:
                                execute(i, q, day, rebalance, forced)
                                account, fm = candidate, candidate_fm
                                used = abs(q)
                    else:
                        data["n_invalid_quote"][day] += 1
                        coverage["invalid_quote"] += 1
                if forced and abs(reduction) - used > tolerance(reduction, current):
                    unfilled_liquidation = True
            after = pos[i]
            opening = (
                lot_open(target - after, i)
                if target > 0 and after >= 0 and target > after and target > current
                else -lot_open(after - target, i)
                if target < 0 and after <= 0 and target < after and target < current
                else 0.0
            )
            if abs(opening) > tolerance(target, after) and not forced and not bankrupt:
                allowed = math.isfinite(p) and (
                    market.can_buy[day, i]
                    if opening > 0
                    else market.can_sell[day, i]
                    and (
                        spec.is_future[i]
                        or margin.allow_short
                        and (market.can_short is None or market.can_short[day, i])
                    )
                )
                q = (
                    math.copysign(
                        lot_open(min(abs(opening), max(cap - used, 0)), i), opening
                    )
                    if allowed
                    else 0.0
                )
                if q != 0:
                    px = quote(i, q, day)[0]
                    if not math.isfinite(px) or px <= 0:
                        q = 0.0
                        data["n_invalid_quote"][day] += 1
                        coverage["invalid_quote"] += 1
                if q != opening:
                    blocked_buy |= opening > 0
                    blocked_sell |= opening < 0
                if q != 0:
                    coverage["reversal"] += int(current * q < 0 and used > 0)
                    orders.append((i, q))
            if mark_price == 0 and w != 0 and target >= current:
                blocked_buy |= w > 0
                blocked_sell |= w < 0
            if blocked_buy:
                data["n_blocked_buy"][day] += 1
                counts["blocked_buy"] += 1
            if blocked_sell:
                data["n_blocked_sell"][day] += 1
                counts["blocked_sell"] += 1
            if (blocked_buy or blocked_sell) and config.carry_blocked and not forced:
                pending[i] = w
        orders.sort(key=lambda x: (not spec.is_future[x[0]], x[1] > 0, x[0]))
        if orders:

            def trial(scale, account=account, fm=fm, orders=orders, day=day):
                a, collateral = account, fm
                for i, desired in orders:
                    q = math.copysign(lot_open(abs(desired) * scale, i), desired)
                    if q != 0:
                        a, collateral = project(a, collateral, i, q, day)
                return a, collateral

            a, collateral = trial(1)
            scale = 1.0
            if not feasible(a, collateral, day):
                coverage["margin_scaled"] += 1
                coverage["futures_margin_scaled"] += int(
                    any(spec.is_future[i] for i, _ in orders)
                )
                lo, hi = 0.0, 1.0
                for _ in range(64):
                    mid = (lo + hi) / 2
                    a, collateral = trial(mid)
                    if feasible(a, collateral, day):
                        lo = mid
                    else:
                        hi = mid
                scale = lo
            for i, desired in orders:
                q = math.copysign(lot_open(abs(desired) * scale, i), desired)
                if q != 0:
                    a, collateral = project(account, fm, i, q, day)
                    if not feasible(a, collateral, day):
                        lo, hi = 0.0, abs(q)
                        for _ in range(64):
                            mid = (lo + hi) / 2
                            trial_q = math.copysign(lot_open(mid, i), q)
                            a, collateral = project(account, fm, i, trial_q, day)
                            if feasible(a, collateral, day):
                                lo = mid
                            else:
                                hi = mid
                        q = math.copysign(lot_open(lo, i), q)
                if abs(q) > tolerance(pos[i] + desired, pos[i]):
                    account, fm = project(account, fm, i, q, day)
                    execute(i, q, day, rebalance, False)

        long = short = 0.0
        for i, q in enumerate(pos):
            if spec.is_future[i]:
                continue
            px = float(market.close[day, i])
            if math.isfinite(px):
                mark[i] = px
            if q == 0:
                continue
            elif mark[i] == 0 and math.isfinite(market.exec_price[day, i]):
                mark[i] = float(market.exec_price[day, i])
            value_i = q * mark[i]
            if value_i > 0:
                long += value_i
            else:
                short -= value_i
                borrow = -value_i * at(margin.borrow_rate, day, i)
                cash -= borrow
                contribution[i] -= borrow
                data["borrow_cost"][day] += borrow
                coverage["borrow_cost"] += int(borrow > 0)
        closing_fm = 0.0
        for i, q in enumerate(pos):
            if not spec.is_future[i] or q == 0:
                continue
            settle = float(spec.settlement[day, i])
            pnl = q * spec.multiplier[i] * (settle - basis[i])
            cash += pnl
            contribution[i] += pnl
            data["futures_pnl"][day] += pnl
            basis[i] = settle
            if day == spec.expiry[i]:
                fee = abs(q) * (
                    spec.multiplier[i] * settle * fee_at(spec.delivery_fee_rate, day, i)
                    + fee_at(spec.delivery_fee_per_contract, day, i)
                )
                cash -= fee
                contribution[i] -= fee
                data["other_fee"][day] += fee
                data["delivery_fee"][day] += fee
                expiries.append(ExpiryEvent(day, i, q, settle, fee))
                coverage["futures_expiry"] += 1
                pos[i] = 0.0
                basis[i] = 0.0
            else:
                closing_fm += future_margin(i, q, settle, day)
        own = cash - (0 if margin.short_proceeds_usable else short) - closing_fm
        interest = max(own, 0) * rates["cash_rate"][day]
        financing = max(-own, 0) * rates["financing_rate"][day]
        cash += interest - financing
        coverage["cash_interest"] += int(interest > 1e-10)
        coverage["financing_interest"] += int(financing > 1e-10)
        equity = cash + long - short
        denominator = previous + flow
        ret = (
            equity / denominator - 1
            if denominator
            > 8
            * np.finfo(float).eps
            * max(config.initial_cash, abs(previous), abs(flow))
            else 0.0
        )
        unit *= 1 + ret
        required = (
            rates["long_initial"][day] * long
            + rates["short_initial"][day] * short
            + closing_fm
        )
        tol = 1e-10 * max(abs(equity), long, short, closing_fm, 1)
        call = (
            rates["long_maintenance"][day] * long
            + rates["short_maintenance"][day] * short
            + closing_fm
            > equity + tol
            or unfilled_liquidation
        )
        coverage["margin_call"] += int(call)
        coverage["futures_margin_call"] += int(call and closing_fm > 0)
        valued = all(
            mark[i] > 0 for i, q in enumerate(pos) if q != 0 and not spec.is_future[i]
        )
        if equity <= 0 and not bankrupt and valued and (any(pos) or cash < 0):
            bankrupt = True
            bankruptcies.append(day)
            coverage["bankruptcy"] += 1
            coverage["genuine_bankruptcy"] += int(short > 0 or own < -tol)
        for name, val in (
            ("nav", equity / config.initial_cash),
            ("cash", cash),
            ("cash_interest", interest),
            ("returns", ret),
            ("unit_nav", unit),
            ("long_value", long),
            ("short_value", short),
            ("financing_interest", financing),
            (
                "margin_usage",
                required / equity if equity > 0 else math.inf if required > 0 else 0,
            ),
            ("margin_call", call),
            ("equity", equity),
            ("futures_margin", closing_fm),
        ):
            data[name][day] = val
        data["turnover"][day] = data["turnover"][day] * 0.5 / value if value > 0 else 0
        previous = equity
        if positions is not None:
            positions[day] = pos
    for i, q in enumerate(pos):
        if q != 0 and not spec.is_future[i]:
            contribution[i] += q * mark[i]
    return Result(
        **data,
        positions=positions,
        trades=trades,
        delist_events=events,
        diagnostics=counts,
        coverage=coverage,
        contribution=contribution,
        liquidation_events=liquidations,
        bankruptcy_events=bankruptcies,
        expiry_events=expiries,
    )
