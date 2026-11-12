"""Independent fill-ledger reconciliation, without api._prepare or reference.

Reconstructs monetary state from public inputs and observed fills. Hand-calculated
order-decision tests are separate; agreeing with this ledger does not validate sizing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import panelbt as bt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test.generators import case  # noqa: E402
from test.test_extended import extended_case  # noqa: E402


def audit(inputs, result):
    m, t, c, g = inputs
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


def main():
    totals = {}
    for name, gen, count in [("legacy", case, 20000), ("healthy", extended_case, 3000)]:
        stats = {"cases": count, "trades": 0, "failures": 0}
        for seed in range(count):
            inputs = gen(seed)
            r = bt.run(*inputs)
            try:
                checks = audit(inputs, r)
                stats["opening_constraint_checks"] = (
                    stats.get("opening_constraint_checks", 0) + checks
                )
            except AssertionError:
                print(name, seed, flush=True)
                raise
            stats["trades"] += len(r.trades)
            for key, value in r.coverage.items():
                stats[key] = stats.get(key, 0) + value
        totals[name] = stats
        print(name, json.dumps(stats), flush=True)
    (ROOT / "bench/results/review-audit.json").write_text(
        json.dumps(totals, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
