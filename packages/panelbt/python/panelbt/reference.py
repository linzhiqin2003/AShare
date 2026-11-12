"""Deliberately scalar executable specification, independent of Rust execution."""

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
    LiquidationEvent,
    Result,
    Trade,
)


def run(market, targets, costs=None, config=None):
    costs, config = costs or Costs(), config or Config()
    margin = config.margin
    t, n = market.shape
    targets.validate_shape(t, n)
    if not margin.allow_short and np.any(targets.weights < 0):
        raise ValueError("negative targets require allow_short")

    def expand(value, size):
        a = np.asarray(value, dtype=float)
        return np.full(size, float(a)) if a.ndim == 0 else a.copy()

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
    quantities, marks = [0.0] * n, [0.0] * n
    dead = [False] * n
    cash = config.initial_cash
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
    events, liquidations, bankruptcies = [], [], []
    pending = {}
    cursor = 0
    previous, unit = cash, 1.0
    call = bankrupt = False
    classic = (
        not margin.allow_short
        and not margin.allow_financing
        and np.all(np.asarray(margin.long_initial) == 1)
        and np.all(np.asarray(margin.long_maintenance) == 0)
        and np.all(np.asarray(margin.short_maintenance) == 0)
    )

    def rate(a, day, col):
        a = np.asarray(a)
        return (
            float(a)
            if a.ndim == 0
            else float(a[day])
            if a.ndim == 1
            else float(a[day, col])
        )

    def step(i):
        return 0.0 if targets.kind == "shares" else float(v[7][i])

    def opening(q, i):
        d = step(i)
        if d == 0:
            return max(q, 0.0)
        return (
            0.0 if q < v[6][i] else float(v[6][i] + math.floor((q - v[6][i]) / d) * d)
        )

    def closing(q, pos, i):
        return (
            min(q, pos)
            if q >= pos or step(i) == 0
            else 0.0
            if q < v[6][i]
            else math.floor(q / step(i)) * step(i)
        )

    def quote(q, i, day):
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

    def movement(q, i, day):
        _, a, c, tax, fee, _, _ = quote(q, i, day)
        return -(a + c + tax + fee) if q > 0 else a - c - tax - fee

    def balance(day):
        long = short = 0.0
        for i in range(n):
            p = float(market.exec_price[day, i])
            x = quantities[i] * (p if math.isfinite(p) else marks[i])
            if x > 0:
                long += x
            else:
                short -= x
        return cash, long, short

    def project(a, i, q, day):
        c, long, s = a
        p = float(market.exec_price[day, i])
        old = quantities[i]
        return (
            c + movement(q, i, day),
            long + (max(old + q, 0) - max(old, 0)) * p,
            s - (min(old + q, 0) - min(old, 0)) * p,
        )

    def feasible(a, day):
        c, long, s = a
        return v[11][day] * long + v[12][day] * s <= c + long - s and (
            margin.allow_financing
            or c - (0 if margin.short_proceeds_usable else s) >= 0
        )

    def execute(i, q, day, rebalance, forced):
        nonlocal cash
        price, amount, comm, tax, fee, slip, impact = quote(q, i, day)
        before = quantities[i]
        change = movement(q, i, day)
        cash += change
        contribution[i] += change
        quantities[i] += q
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
        coverage["open_short"] += int(quantities[i] < min(before, 0))
        coverage["cover_short"] += int(before < 0 and q > 0)
        if forced:
            coverage["liquidation"] += 1
            liquidations.append(LiquidationEvent(day, i, q))
        if trades is not None:
            trades.append(Trade(day, i, q, price, comm, tax, fee, slip, impact))

    for day in range(t):
        short = sum(-quantities[i] * marks[i] for i in range(n) if quantities[i] < 0)
        available = max(cash - (0 if margin.short_proceeds_usable else short), 0)
        requested = float(v[10][day])
        flow = max(requested, -available)
        cash += flow
        data["cash_flow"][day] = flow
        data["withdrawal_shortfall"][day] = flow - requested
        coverage["cash_flow"] += int(flow != 0)
        coverage["withdrawal_shortfall"] += int(flow > requested)
        for i in range(n):
            if quantities[i] != 0:
                div = (
                    float(market.cash_div[day, i])
                    if market.cash_div is not None
                    else 0.0
                )
                gross = quantities[i] * div
                tax = max(gross, 0) * v[8][day]
                cash += gross - tax
                contribution[i] += gross - tax
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
            if market.delist is not None and market.delist[day, i]:
                dead[i] = True
                if quantities[i] != 0:
                    price = marks[i] * config.delist_recovery
                    proceeds = quantities[i] * price
                    cash += proceeds
                    contribution[i] += proceeds
                    events.append(DelistEvent(day, i, quantities[i], price, proceeds))
                    quantities[i] = 0.0
                    counts["delist"] += 1
        rebalance = cursor < len(targets.days) and targets.days[cursor] == day
        forced = not classic and (
            (call and margin.margin_call == "liquidate") or bankrupt
        )
        unfilled_liquidation = False
        planned = {}
        if rebalance:
            pending = {}
            commands = {i: 0.0 for i in range(n) if quantities[i] != 0}
            for j in range(targets.indptr[cursor], targets.indptr[cursor + 1]):
                commands[int(targets.cols[j])] = float(targets.weights[j])
            cursor += 1
        else:
            commands = dict(pending) if config.carry_blocked else {}
            pending = {}
        account = balance(day)
        value = account[0] + account[1] - account[2]
        account_start = value
        if forced:
            pending = {}
            requested_commands = commands.copy()

            def liquidated(keep, account=account, day=day):
                a = account
                for i in range(n):
                    current = quantities[i]
                    if current == 0:
                        continue
                    q = -math.copysign(
                        closing(abs(current) * (1 - keep), abs(current), i), current
                    )
                    if q != 0 and math.isfinite(market.exec_price[day, i]):
                        price = quote(q, i, day)[0]
                        if price > 0 and math.isfinite(price):
                            a = project(a, i, q, day)
                return a

            if bankrupt or value <= 0 or not feasible(liquidated(0), day):
                keep = 0.0
            elif feasible(liquidated(1), day):
                keep = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(64):
                    mid = (lo + hi) / 2
                    if feasible(liquidated(mid), day):
                        lo = mid
                    else:
                        hi = mid
                keep = lo
            commands = {}
            for i, current in enumerate(quantities):
                if current == 0:
                    continue
                amount = closing(abs(current) * (1 - keep), abs(current), i)
                if rebalance and i in requested_commands:
                    w = requested_commands[i]
                    p = float(market.exec_price[day, i])
                    mark = p if math.isfinite(p) else marks[i]
                    desired = (
                        w
                        if targets.kind == "shares"
                        else w * max(value, 0) * (1 - config.cash_buffer) / mark
                        if mark > 0
                        else current
                    )
                    reduction = (
                        current - max(desired, 0)
                        if current > 0
                        else min(desired, 0) - current
                    )
                    amount = max(amount, closing(reduction, abs(current), i))
                planned[i] = -math.copysign(amount, current)
                commands[i] = current + planned[i]
        orders = []
        command_order = (
            sorted(commands)
            if classic
            else sorted(commands, key=lambda i: (quantities[i] < 0, i))
        )
        for i in command_order:
            if dead[i]:
                continue
            w = commands[i]
            p = float(market.exec_price[day, i])
            mark = p if math.isfinite(p) else marks[i]
            if forced or targets.kind == "shares":
                target = w
                coverage["shares_target"] += int(not forced)
            else:
                x = (
                    w * max(value, 0.0) * (1 - config.cash_buffer) / mark
                    if mark > 0
                    else 0
                )
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
                and targets.kind == "weights"
                and w != 0
                and quantities[i] != 0
                and value > 0
                and abs(w - quantities[i] * mark / value) < config.trade_band
            ):
                coverage["trade_band"] += 1
                continue
            cap = math.inf
            if market.volume is not None and config.max_participation > 0:
                volume = float(market.volume[day, i])
                cap = (
                    volume * config.max_participation
                    if math.isfinite(volume) and volume > 0
                    else 0.0
                )
                if step(i) > 0:
                    cap = math.floor(cap / step(i)) * step(i)
            current = quantities[i]
            reduction = (
                planned[i]
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

                def own_balance(a):
                    return a[0] - (0 if margin.short_proceeds_usable else a[2])

                if (
                    q != 0
                    and quote(q, i, day)[0] > 0
                    and math.isfinite(quote(q, i, day)[0])
                    and not classic
                    and not margin.allow_financing
                    and own_balance(account) >= -1e-10 * config.initial_cash
                    and own_balance(project(account, i, q, day)) < 0
                ):
                    requested_quantity = q
                    lo, hi = 0.0, abs(q)
                    for _ in range(64):
                        mid = (lo + hi) / 2
                        x = math.copysign(closing(mid, abs(current), i), q)
                        if own_balance(project(account, i, x, day)) >= 0:
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
                    candidate = project(account, i, q, day)
                    if (
                        price > 0
                        and math.isfinite(price)
                        and (not classic or cash + a >= c + tax + fee)
                    ):
                        execute(i, q, day, rebalance, forced)
                        account = candidate
                        used = abs(q)
                    elif not math.isfinite(price) or price <= 0:
                        data["n_invalid_quote"][day] += 1
                        coverage["invalid_quote"] += 1
                if forced and abs(reduction) - used > 1e-9 * max(
                    abs(reduction), abs(current), 1
                ):
                    unfilled_liquidation = True
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
                and not bankrupt
            ):
                allowed = math.isfinite(p) and (
                    market.can_buy[day, i]
                    if wanted > 0
                    else market.can_sell[day, i]
                    and margin.allow_short
                    and (market.can_short is None or market.can_short[day, i])
                )
                q = (
                    math.copysign(
                        opening(min(abs(wanted), max(cap - used, 0)), i), wanted
                    )
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
                    orders.append((i, q))
            if mark == 0 and w != 0 and target >= current:
                blocked_buy |= w > 0
                blocked_sell |= w < 0
            for flag, name in ((blocked_buy, "buy"), (blocked_sell, "sell")):
                if flag:
                    data[f"n_blocked_{name}"][day] += 1
                    counts[f"blocked_{name}"] += 1
            if (blocked_buy or blocked_sell) and config.carry_blocked and not forced:
                pending[i] = w
        if orders:
            if classic:
                total = sum(-movement(q, i, day) for i, q in orders)
                scale = cash / total if total > cash else 1.0
                counts["cash_scaled"] += int(total > cash)
                for i, desired in orders:
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
                                if (
                                    -movement(float(v[6][i] + mid * step(i)), i, day)
                                    <= cash
                                ):
                                    lo = mid
                                else:
                                    hi = mid - 1
                            q = float(v[6][i] + lo * step(i))
                    if q > 0:
                        execute(i, q, day, rebalance, False)
            else:
                orders.sort(key=lambda x: (x[1] > 0, x[0]))

                def trial(scale, account=account, orders=orders, day=day):
                    a = account
                    for i, q in orders:
                        x = math.copysign(opening(abs(q) * scale, i), q)
                        if x != 0:
                            a = project(a, i, x, day)
                    return a

                scale = 1.0
                if not feasible(trial(1.0), day):
                    coverage["margin_scaled"] += 1
                    lo, hi = 0.0, 1.0
                    for _ in range(64):
                        mid = (lo + hi) / 2
                        if feasible(trial(mid), day):
                            lo = mid
                        else:
                            hi = mid
                    scale = lo
                for i, desired in orders:
                    q = math.copysign(opening(abs(desired) * scale, i), desired)
                    if q != 0 and not feasible(project(account, i, q, day), day):
                        lo, hi = 0.0, abs(q)
                        for _ in range(64):
                            mid = (lo + hi) / 2
                            x = math.copysign(opening(mid, i), q)
                            if feasible(project(account, i, x, day), day):
                                lo = mid
                            else:
                                hi = mid
                        q = math.copysign(opening(lo, i), q)
                    if q != 0:
                        account = project(account, i, q, day)
                        execute(i, q, day, rebalance, False)
        long = short = 0.0
        for i in range(n):
            p = float(market.close[day, i])
            if math.isfinite(p):
                marks[i] = p
            elif (
                not classic
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
                cash -= fee
                contribution[i] -= fee
                data["borrow_cost"][day] += fee
                coverage["borrow_cost"] += int(fee > 0)
        own = cash - (0 if margin.short_proceeds_usable else short)
        interest = max(own, 0) * v[9][day]
        financing = max(-own, 0) * v[15][day]
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
            else 0
        )
        unit *= 1 + ret
        required = v[11][day] * long + v[12][day] * short
        tolerance = 1e-10 * max(abs(equity), long, short, 1)
        call = not classic and (
            v[13][day] * long + v[14][day] * short > equity + tolerance
            or unfilled_liquidation
        )
        coverage["margin_call"] += int(call)
        valued = all(marks[i] > 0 for i in range(n) if quantities[i] != 0)
        if (
            not classic
            and equity <= 0
            and not bankrupt
            and valued
            and (any(quantities) or cash < 0)
        ):
            bankrupt = True
            bankruptcies.append(day)
            coverage["bankruptcy"] += 1
            coverage["genuine_bankruptcy"] += int(short > 0 or own < -tolerance)
        for key, x in (
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
        ):
            data[key][day] = x
        data["turnover"][day] = (
            data["turnover"][day] * 0.5 / account_start if account_start > 0 else 0.0
        )
        previous = equity
        if positions is not None:
            positions[day] = quantities
    contribution += np.asarray(quantities) * np.asarray(marks)
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
    )
