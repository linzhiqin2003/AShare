"""Independent given-fill cash ledger; no execution or preparation helpers."""

import numpy as np


def audit(inputs, result):
    m, t, c, g = inputs
    if m.futures is not None:
        return audit_futures(inputs, result)
    periods, n = m.shape
    q = np.zeros(n)
    mark = np.zeros(n)
    cash = g.initial_cash
    previous_equity, unit = g.initial_cash, 1.0
    contribution = np.zeros(n)
    fills = [[] for _ in range(periods)]
    events = [[] for _ in range(periods)]
    for x in result.trades:
        fills[x.day].append(x)
    for x in result.delist_events:
        events[x.day].append(x)

    def rate(value, day, col=0):
        a = np.asarray(value)
        return float(a if a.ndim == 0 else a[day] if a.ndim == 1 else a[day, col])

    def equal(a, b):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-8)

    classic = (
        not g.margin.allow_short
        and not g.margin.allow_financing
        and np.all(np.asarray(g.margin.long_initial) == 1)
        and np.all(np.asarray(g.margin.long_maintenance) == 0)
        and np.all(np.asarray(g.margin.short_maintenance) == 0)
    )
    opening_checks = 0
    for d in range(periods):
        short = float(np.dot(np.maximum(-q, 0), mark))
        own = cash - (0 if g.margin.short_proceeds_usable else short)
        flow = max(rate(g.cash_flow, d), -max(own, 0))
        cash += flow
        equal(flow, result.cash_flow[d])
        equal(flow - rate(g.cash_flow, d), result.withdrawal_shortfall[d])
        div = m.cash_div[d] if m.cash_div is not None else np.zeros(n)
        split = m.split[d] if m.split is not None else np.ones(n)
        gross = q * div
        net = gross - np.maximum(gross, 0) * rate(g.dividend_tax_rate, d)
        cash += float(net.sum())
        contribution += net
        equal(gross.sum(), result.dividend_gross[d])
        equal(
            np.maximum(gross, 0).sum() * rate(g.dividend_tax_rate, d),
            result.dividend_tax[d],
        )
        mark = np.maximum(mark - div, 0) / split
        q *= split
        for e in events[d]:
            equal(e.qty, q[e.col])
            equal(e.price, mark[e.col] * g.delist_recovery)
            proceeds = q[e.col] * e.price
            cash += proceeds
            contribution[e.col] += proceeds
            q[e.col] = 0
        used = np.zeros(n)
        daily_fees = np.zeros(5)
        previous_rank = -1
        for x in fills[d]:
            i = x.col
            quantity = abs(x.qty)
            buy = x.qty > 0
            assert (m.can_buy if buy else m.can_sell)[d, i]
            slipped = m.exec_price[d, i] * (
                1 + (1 if buy else -1) * rate(c.slippage, d, i)
            )
            impact = (
                0
                if c.impact_k is None
                else c.impact_k[d, i] * (quantity / m.volume[d, i]) ** c.impact_exponent
            )
            price = slipped * (1 + (1 if buy else -1) * impact)
            equal(price, x.price)
            amount = quantity * price
            commission = max(
                c.commission_rate * amount + c.commission_per_share * quantity,
                c.commission_min or 0,
            )
            if c.commission_max_rate is not None:
                commission = min(commission, c.commission_max_rate * amount)
            tax = amount * rate(c.tax_buy if buy else c.tax_sell, d)
            other = amount * rate(
                c.fee_rate_buy if buy else c.fee_rate_sell, d
            ) + quantity * rate(c.fee_per_share_buy if buy else c.fee_per_share_sell, d)
            equal([commission, tax, other], [x.commission, x.tax, x.other_fee])
            daily_fees += [
                commission,
                tax,
                other,
                quantity * abs(slipped - m.exec_price[d, i]),
                quantity * abs(price - slipped),
            ]
            before = q[i]
            reducing = before * x.qty < 0
            rank = (0 if before > 0 else 1) if reducing else (3 if buy else 2)
            assert rank >= previous_rank
            previous_rank = rank
            if not reducing and not buy:
                assert g.margin.allow_short
                assert m.can_short is None or m.can_short[d, i]
            if t.kind != "shares":
                step = float(np.broadcast_to(g.qty_step, (n,))[i])
                minimum = float(np.broadcast_to(g.min_qty, (n,))[i])
                if step > 0:
                    full_close = reducing and abs(quantity - abs(before)) <= 1e-9 * max(
                        abs(before), 1
                    )
                    if not full_close:
                        assert quantity >= minimum - 1e-9
                        units = (
                            quantity / step if reducing else (quantity - minimum) / step
                        )
                        equal(units, round(units))
            change = -x.qty * price - commission - tax - other
            cash += change
            contribution[i] += change
            q[i] += x.qty
            used[i] += quantity
            if not reducing:
                opening_checks += 1
                exec_marks = np.where(
                    np.isfinite(m.exec_price[d]), m.exec_price[d], mark
                )
                lv = float(np.dot(np.maximum(q, 0), exec_marks))
                sv = float(np.dot(np.maximum(-q, 0), exec_marks))
                equity = cash + lv - sv
                tolerance = 1e-8 + 1e-12 * max(abs(cash), lv, sv)
                required = (
                    rate(g.margin.long_initial, d) * lv
                    + rate(g.margin.short_initial, d) * sv
                )
                assert required <= equity + tolerance
                if not g.margin.allow_financing:
                    assert (
                        cash - (0 if g.margin.short_proceeds_usable else sv)
                        >= -tolerance
                    )
        if m.volume is not None and g.max_participation > 0:
            assert np.all(
                used <= np.nan_to_num(m.volume[d]) * g.max_participation + 1e-7
            )
        equal(
            daily_fees,
            [
                result.commission[d],
                result.tax[d],
                result.other_fee[d],
                result.slippage_cost[d],
                result.impact_cost[d],
            ],
        )
        mark = np.where(np.isfinite(m.close[d]), m.close[d], mark)
        if not classic:
            mark = np.where(
                (mark == 0) & (q != 0) & np.isfinite(m.exec_price[d]),
                m.exec_price[d],
                mark,
            )
        long = float(np.dot(np.maximum(q, 0), mark))
        short = float(np.dot(np.maximum(-q, 0), mark))
        fees = np.array(
            [
                max(-q[i], 0) * mark[i] * rate(g.margin.borrow_rate, d, i)
                for i in range(n)
            ]
        )
        cash -= float(fees.sum())
        contribution -= fees
        own = cash - (0 if g.margin.short_proceeds_usable else short)
        interest = max(own, 0) * rate(g.cash_rate, d)
        financing = max(-own, 0) * rate(g.margin.financing_rate, d)
        cash += interest - financing
        equal(
            [cash, long, short, cash + long - short, interest, financing, fees.sum()],
            [
                result.cash[d],
                result.long_value[d],
                result.short_value[d],
                result.equity[d],
                result.cash_interest[d],
                result.financing_interest[d],
                result.borrow_cost[d],
            ],
        )
        equal(q, result.positions[d])
        equity = cash + long - short
        equal(equity / g.initial_cash, result.nav[d])
        denominator = previous_equity + flow
        zero_capital = (
            8
            * np.finfo(float).eps
            * max(g.initial_cash, abs(previous_equity), abs(flow))
        )
        period_return = equity / denominator - 1 if denominator > zero_capital else 0
        unit *= 1 + period_return
        equal([period_return, unit], [result.returns[d], result.unit_nav[d]])
        previous_equity = equity
    equal(contribution + q * mark, result.contribution)
    return opening_checks


def audit_futures(inputs, result):
    """Rebuild mixed-account balances using only recorded fills and market facts."""
    m, targets, costs, config = inputs
    spec = m.futures
    t, n = m.shape
    q = np.zeros(n)
    mark = np.zeros(n)
    basis = np.zeros(n)
    margin_mark = np.zeros(n)
    cash = float(config.initial_cash)
    previous, unit = cash, 1.0
    contribution = np.zeros(n)
    fills = [[] for _ in range(t)]
    delists = [[] for _ in range(t)]
    expiries = [[] for _ in range(t)]
    for fill in result.trades:
        fills[fill.day].append(fill)
    for event in result.delist_events:
        delists[event.day].append(event)
    for event in result.expiry_events:
        expiries[event.day].append(event)

    def spot_rate(value, day, i=0):
        a = np.asarray(value)
        return float(a if a.ndim == 0 else a[day] if a.ndim == 1 else a[day, i])

    def future_rate(value, day, i):
        a = np.asarray(value)
        return float(
            a.flat[0] if a.size == 1 else a[0, i] if a.shape == (1, n) else a[day, i]
        )

    def equal(left, right):
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-8)

    def collateral(day):
        return sum(
            abs(q[i])
            * spec.multiplier[i]
            * margin_mark[i]
            * future_rate(spec.margin_rate, day, i)
            for i in range(n)
            if spec.is_future[i]
        )

    def previous_close(day, i):
        for old in range(day - 1, -1, -1):
            price = float(m.close[old, i])
            if not np.isfinite(price):
                continue
            for later in range(old + 1, day + 1):
                if m.cash_div is not None:
                    price = max(price - m.cash_div[later, i], 0)
                if m.split is not None:
                    price /= m.split[later, i]
            return price
        return 0.0

    checks = 0
    for day in range(t):
        reserved_short = (
            sum(-q[i] * mark[i] for i in range(n) if not spec.is_future[i] and q[i] < 0)
            if not config.margin.short_proceeds_usable
            else 0.0
        )
        requested = spot_rate(config.cash_flow, day)
        flow = max(requested, -max(cash - reserved_short - collateral(day), 0))
        cash += flow
        equal(
            [flow, flow - requested],
            [result.cash_flow[day], result.withdrawal_shortfall[day]],
        )
        gross_div = tax_div = 0.0
        for i in range(n):
            if spec.is_future[i]:
                continue
            div = float(m.cash_div[day, i]) if m.cash_div is not None else 0.0
            factor = float(m.split[day, i]) if m.split is not None else 1.0
            gross = q[i] * div
            tax = max(gross, 0) * spot_rate(config.dividend_tax_rate, day)
            cash += gross - tax
            contribution[i] += gross - tax
            gross_div += gross
            tax_div += tax
            mark[i] = max(mark[i] - div, 0) / factor
            q[i] *= factor
        equal(
            [gross_div, tax_div], [result.dividend_gross[day], result.dividend_tax[day]]
        )
        for e in delists[day]:
            assert not spec.is_future[e.col]
            equal([e.qty, e.price], [q[e.col], mark[e.col] * config.delist_recovery])
            cash += e.proceeds
            contribution[e.col] += e.proceeds
            q[e.col] = 0

        start_pos = q.copy()
        fees = np.zeros(5)
        pnl = 0.0
        used = np.zeros(n)
        rank = -1
        for fill in fills[day]:
            i, amount = fill.col, abs(fill.qty)
            before = q[i]
            future = bool(spec.is_future[i])
            if not future and before == 0.0 and start_pos[i] == 0.0:
                mark[i] = previous_close(day, i)
            assert day <= spec.expiry[i] if future else day < m.first_delist[i]
            assert (m.can_buy if fill.qty > 0 else m.can_sell)[day, i]
            reducing = before * fill.qty < 0
            this_rank = (
                (0 if future else 1 if before > 0 else 2)
                if reducing
                else (3 if future else 4 if fill.qty < 0 else 5)
            )
            assert this_rank >= rank
            rank = this_rank
            raw = float(m.exec_price[day, i])
            slip = spot_rate(costs.slippage, day, i)
            slipped = raw * (1 + (1 if fill.qty > 0 else -1) * slip)
            impact = (
                0.0
                if costs.impact_k is None
                else costs.impact_k[day, i]
                * (amount / m.volume[day, i]) ** costs.impact_exponent
            )
            price = slipped * (1 + (1 if fill.qty > 0 else -1) * impact)
            equal(price, fill.price)
            multiplier = spec.multiplier[i] if future else 1.0
            notional = amount * price * multiplier
            if future:
                assert amount == round(amount)
                commission = amount * (
                    multiplier * price * future_rate(spec.fee_rate, day, i)
                    + future_rate(spec.fee_per_contract, day, i)
                )
                tax = other = 0.0
            else:
                commission = max(
                    costs.commission_rate * notional
                    + costs.commission_per_share * amount,
                    costs.commission_min or 0.0,
                )
                if costs.commission_max_rate is not None:
                    commission = min(commission, costs.commission_max_rate * notional)
                tax = notional * spot_rate(
                    costs.tax_buy if fill.qty > 0 else costs.tax_sell, day
                )
                other = notional * spot_rate(
                    costs.fee_rate_buy if fill.qty > 0 else costs.fee_rate_sell, day
                ) + amount * spot_rate(
                    costs.fee_per_share_buy
                    if fill.qty > 0
                    else costs.fee_per_share_sell,
                    day,
                )
            equal([commission, tax, other], [fill.commission, fill.tax, fill.other_fee])
            fees += [
                commission,
                tax,
                other,
                amount * raw * slip * multiplier,
                amount * abs(price - slipped) * multiplier,
            ]
            if future:
                closed = min(abs(before), amount) * np.sign(before) if reducing else 0.0
                realized = closed * multiplier * (price - basis[i])
                cash += realized - commission
                contribution[i] += realized - commission
                pnl += realized
                after = before + fill.qty
                basis[i] = (
                    0.0
                    if after == 0
                    else price
                    if before == 0 or np.sign(before) != np.sign(after)
                    else (abs(before) * basis[i] + amount * price) / abs(after)
                    if np.sign(before) == np.sign(fill.qty)
                    else basis[i]
                )
                margin_mark[i] = price
            else:
                change = -fill.qty * price - commission - tax - other
                cash += change
                contribution[i] += change
            q[i] += fill.qty
            used[i] += amount
            if not reducing:
                checks += 1
                exec_mark = np.where(
                    np.isfinite(m.exec_price[day]), m.exec_price[day], mark
                )
                lv = sum(
                    max(q[j], 0) * exec_mark[j]
                    for j in range(n)
                    if not spec.is_future[j]
                )
                sv = sum(
                    max(-q[j], 0) * exec_mark[j]
                    for j in range(n)
                    if not spec.is_future[j]
                )
                equity = cash + lv - sv
                needed = (
                    spot_rate(config.margin.long_initial, day) * lv
                    + spot_rate(config.margin.short_initial, day) * sv
                    + collateral(day)
                )
                tol = 1e-8 + 1e-12 * max(abs(cash), lv, sv, needed)
                assert needed <= equity + tol
                if not config.margin.allow_financing:
                    assert (
                        cash
                        - (0 if config.margin.short_proceeds_usable else sv)
                        - collateral(day)
                        >= -tol
                    )
        if m.volume is not None and config.max_participation > 0:
            assert np.all(used <= m.volume[day] * config.max_participation + 1e-7)
        for i in range(n):
            if not spec.is_future[i]:
                close = float(m.close[day, i])
                if np.isfinite(close):
                    mark[i] = close
                elif q[i] != 0 and mark[i] == 0 and np.isfinite(m.exec_price[day, i]):
                    mark[i] = float(m.exec_price[day, i])
        long = sum(max(q[i], 0) * mark[i] for i in range(n) if not spec.is_future[i])
        short = sum(max(-q[i], 0) * mark[i] for i in range(n) if not spec.is_future[i])
        borrow = 0.0
        for i in range(n):
            if not spec.is_future[i] and q[i] < 0:
                fee = -q[i] * mark[i] * spot_rate(config.margin.borrow_rate, day, i)
                cash -= fee
                contribution[i] -= fee
                borrow += fee
        closing_margin = 0.0
        for i in range(n):
            if not spec.is_future[i] or q[i] == 0:
                continue
            settle = spec.settlement[day, i]
            variation = q[i] * spec.multiplier[i] * (settle - basis[i])
            cash += variation
            contribution[i] += variation
            pnl += variation
            basis[i] = settle
            margin_mark[i] = settle
            if day == spec.expiry[i]:
                fee = abs(q[i]) * (
                    spec.multiplier[i]
                    * settle
                    * future_rate(spec.delivery_fee_rate, day, i)
                    + future_rate(spec.delivery_fee_per_contract, day, i)
                )
                assert any(
                    e.col == i and e.qty == q[i] and e.fee == fee for e in expiries[day]
                )
                cash -= fee
                contribution[i] -= fee
                fees[2] += fee
                q[i] = 0.0
                basis[i] = 0.0
                margin_mark[i] = 0.0
            else:
                closing_margin += (
                    abs(q[i])
                    * spec.multiplier[i]
                    * settle
                    * future_rate(spec.margin_rate, day, i)
                )
        own = (
            cash
            - (0 if config.margin.short_proceeds_usable else short)
            - closing_margin
        )
        interest = max(own, 0) * spot_rate(config.cash_rate, day)
        financing = max(-own, 0) * spot_rate(config.margin.financing_rate, day)
        cash += interest - financing
        equal(
            [
                cash,
                long,
                short,
                cash + long - short,
                pnl,
                closing_margin,
                borrow,
                interest,
                financing,
            ],
            [
                result.cash[day],
                result.long_value[day],
                result.short_value[day],
                result.equity[day],
                result.futures_pnl[day],
                result.futures_margin[day],
                result.borrow_cost[day],
                result.cash_interest[day],
                result.financing_interest[day],
            ],
        )
        equal(
            fees,
            [
                result.commission[day],
                result.tax[day],
                result.other_fee[day],
                result.slippage_cost[day],
                result.impact_cost[day],
            ],
        )
        equal(q, result.positions[day])
        equal((cash + long - short) / config.initial_cash, result.nav[day])
        denom = previous + flow
        period_return = (
            (cash + long - short) / denom - 1
            if denom
            > 8
            * np.finfo(float).eps
            * max(config.initial_cash, abs(previous), abs(flow))
            else 0.0
        )
        unit *= 1 + period_return
        equal([period_return, unit], [result.returns[day], result.unit_nav[day]])
        previous = cash + long - short
    contribution += q * mark * ~spec.is_future
    equal(contribution, result.contribution)
    return checks
