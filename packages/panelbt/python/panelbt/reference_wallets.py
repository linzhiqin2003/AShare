"""Independent C4 oracle, specified directly in terms of wallet money.

No native calls, no imports from reference_crypto, and no settlement-cash
ledger. Futures P&L stays unrealized until a position is actually closed.
This intentionally slow oracle is for small synthetic panels, not production.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np


def contract_count(value):
    # SPEC: choose the adjacent integer only within 1e-9 contract; otherwise
    # discard the fractional contract. Use scalar distances, not vector rounding.
    whole = math.trunc(value)
    adjacent = whole + (1 if value > 0 else -1)
    return adjacent if abs(value - adjacent) <= 1e-9 else whole


def term(value, row, col=0):
    a = np.asarray(value)
    if a.ndim == 0:
        return float(a)
    if a.ndim == 1:
        return float(a[row])
    return float(a[min(row, a.shape[0] - 1), min(col, a.shape[1] - 1)])


class WalletOracle:
    def __init__(self, market, costs, config):
        self.market, self.costs, self.config = market, costs, config
        self.future = market.futures
        self.rows, self.cols = market.shape
        self.row = 0
        self.money = float(config.initial_cash)
        self.position = np.zeros(self.cols)
        self.entry = np.zeros(self.cols)
        self.bankrupt = False
        self.pending = {}
        self.records = []
        self.previous = self.money
        if np.any(np.asarray(config.cash_flow) != 0):
            raise ValueError("oracle: use dynamic flow, not static flow")
        if (
            config.rules is not None
            or config.margin.allow_short
            or config.margin.allow_financing
        ):
            raise ValueError("oracle: C4 cash spot / USD-M scope only")

    def _marks(self, stage):
        if self.future is None:
            return self.market.close[min(self.row, self.rows - 1)]
        return (
            self.future.settlement
            if stage == "close"
            else self.future.liquidation.mark_open
        )[self.row]

    def value(self, marks):
        if self.future is None:
            return self.money + sum(
                q * p for q, p in zip(self.position, marks, strict=True) if q
            )
        return self.money + sum(
            q * mult * (p - e)
            for q, mult, p, e in zip(
                self.position, self.future.multiplier, marks, self.entry, strict=True
            )
            if q
        )

    def state(self):
        d = max(self.row - 1, 0)
        rule = min(self.row, self.rows - 1)
        if self.future is None:
            marks = self.market.close[d]
            return SimpleNamespace(
                shares=self.position.copy(),
                cash=self.money,
                equity=self.value(marks),
                transferable=max(self.money, 0),
                bankrupt=self.bankrupt,
                wallet_balance=None,
                unrealized=None,
                entry_price=None,
                notional=0,
                initial_margin=0,
            )
        f = self.future
        marks = f.settlement[d]
        unrealized = self.value(marks) - self.money
        nominal = sum(
            abs(q) * mult * p
            for q, mult, p in zip(self.position, f.multiplier, marks, strict=True)
            if q
        )
        reserve = sum(
            abs(q) * f.multiplier[j] * marks[j] * term(f.margin_rate, rule, j)
            for j, q in enumerate(self.position)
            if q
        )
        return SimpleNamespace(
            shares=self.position.copy(),
            cash=self.value(marks),
            equity=self.value(marks),
            wallet_balance=self.money,
            unrealized=unrealized,
            entry_price=self.entry.copy(),
            notional=nominal,
            initial_margin=reserve,
            transferable=0
            if self.bankrupt
            else max(0, self.money - max(-unrealized, 0) - reserve),
            bankrupt=self.bankrupt,
        )

    def quote(self, j, change, raw=None):
        d, c = self.row, self.costs
        raw = float(self.market.exec_price[d, j]) if raw is None else raw
        price = raw * (1 + np.sign(change) * term(c.slippage, d, j))
        if c.impact_k is not None:
            v = self.market.volume[d, j]
            impact = (
                term(c.impact_k, d, j) * (abs(change) / v) ** c.impact_exponent
                if v > 0
                else math.inf
            )
            price *= 1 + np.sign(change) * impact
        if self.future is not None:
            f = self.future
            fee = abs(change) * (
                f.multiplier[j] * price * term(f.fee_rate, d, j)
                + term(f.fee_per_contract, d, j)
            )
        else:
            amount = abs(change) * price
            commission = max(
                c.commission_min or 0,
                c.commission_rate * amount + c.commission_per_share * abs(change),
            )
            if c.commission_max_rate is not None:
                commission = min(commission, c.commission_max_rate * amount)
            suffix = "buy" if change > 0 else "sell"
            fee = (
                commission
                + amount
                * (
                    term(getattr(c, "tax_" + suffix), d)
                    + term(getattr(c, "fee_rate_" + suffix), d)
                )
                + abs(change) * term(getattr(c, "fee_per_share_" + suffix), d)
            )
        return price, fee

    def _fill(self, j, change, price, fee):
        old = self.position[j]
        if self.future is None:
            self.money -= change * price + fee
        else:
            closed = min(abs(old), abs(change)) if old * change < 0 else 0
            self.money += (
                np.sign(old)
                * closed
                * self.future.multiplier[j]
                * (price - self.entry[j])
                - fee
            )
            # Compute the remaining old units and the new units separately.
            remaining = abs(old) - closed
            new = abs(change) - closed
            final = remaining + new
            self.entry[j] = (
                (remaining * self.entry[j] + new * price) / final if final else 0
            )
        self.position[j] += change
        self.current["fees"] += fee

    def _filter(self, j, size, *, closing=False):
        if self.future is None:
            minimum = np.broadcast_to(self.config.min_qty, (self.cols,))[j]
            step = np.broadcast_to(self.config.qty_step, (self.cols,))[j]
        else:
            f = self.future
            minimum = 1 if f.min_qty is None else f.min_qty[self.row, j]
            step = 1 if f.qty_step is None else f.qty_step[self.row, j]
        # SPEC quantity tolerance is a fraction of a step, not a cash allowance.
        tolerance = step * 1e-9 if self.future is None else 0.0
        if closing and abs(self.position[j]) - size <= tolerance:
            return abs(self.position[j])
        if step and self.future is None:
            lower = math.floor(size / step)
            upper = (lower + 1) * step
            q = upper if abs(upper - size) <= tolerance else lower * step
        else:
            q = math.floor(size / step) * step if step else size
        if q < minimum:
            return 0.0
        if (
            self.future is not None
            and not closing
            and self.future.min_notional is not None
        ):
            if (
                q * self.future.multiplier[j] * self._marks("open")[j]
                < self.future.min_notional[self.row, j]
            ):
                return 0.0
        return q

    def _risk(self, marks):
        f, d = self.future, self.row
        mm = sum(
            abs(q)
            * f.multiplier[j]
            * marks[j]
            * term(f.margin_rate if f.maint_rate is None else f.maint_rate, d, j)
            for j, q in enumerate(self.position)
            if q
        )
        nominal = sum(
            abs(q) * f.multiplier[j] * marks[j]
            for j, q in enumerate(self.position)
            if q
        )
        return self.value(marks) - mm, nominal

    def _liquidate(self, marks, stage):
        due = 0.0
        for j, q in enumerate(self.position.copy()):
            if q:
                price, fee = self.quote(j, -q, float(marks[j]))
                if not math.isfinite(price) or price <= 0:
                    raise ValueError("oracle invalid liquidation quote")
                due += (
                    abs(q)
                    * self.future.multiplier[j]
                    * price
                    * term(self.future.liquidation.clearance_fee_rate, self.row, j)
                )
                self._fill(j, -q, price, fee)
        clearance = min(due, max(self.money, 0))
        self.money -= clearance
        self.current["clearance_fee"] += clearance
        if self.money < 0:
            self.current["deficit"] += -self.money
            self.money = 0.0
            self.bankrupt = True
        self.current["liquidations"].append(stage)
        self.pending.clear()

    def _trade(self, target):
        if self.bankrupt:
            return
        commands = dict(enumerate(target)) if target is not None else dict(self.pending)
        self.pending.clear()
        openings = []
        for j, targetq in commands.items():
            if self.future is not None and self.row > self.future.expiry[j]:
                continue
            current = self.position[j]
            limit = math.inf
            if self.config.max_participation and self.market.volume is not None:
                v = self.market.volume[self.row, j]
                limit = (
                    v * self.config.max_participation if np.isfinite(v) and v > 0 else 0
                )
            used, blocked = 0.0, False
            close = (
                min(abs(current), abs(targetq - current))
                if current * (targetq - current) < 0
                else 0
            )
            close = -np.sign(current) * close
            for reducing in (True, False):
                difference = close if reducing else targetq - self.position[j]
                if not difference or (
                    not reducing and self.position[j] * difference < 0
                ):
                    continue
                mask = self.market.can_buy if difference > 0 else self.market.can_sell
                valid = mask[self.row, j] and np.isfinite(
                    self.market.exec_price[self.row, j]
                )
                capacity = max(limit - used, 0)
                blocked |= not valid or abs(difference) > capacity
                q = (
                    np.sign(difference)
                    * self._filter(j, min(abs(difference), capacity), closing=reducing)
                    if valid
                    else 0
                )
                if not q:
                    continue
                price, fee = self.quote(j, q)
                if not np.isfinite(price) or price <= 0:
                    blocked = True
                elif reducing:
                    self._fill(j, q, price, fee)
                    used += abs(q)
                else:
                    openings.append((j, q))
            if blocked and self.config.carry_blocked:
                self.pending[j] = targetq

        if self.future is None:
            total = sum(
                q * self.quote(j, q)[0] + self.quote(j, q)[1] for j, q in openings
            )
            scale = min(1.0, self.money / total) if total else 1.0
            for j, q in openings:
                q = self._filter(j, q * scale)
                price, fee = self.quote(j, q)
                if q and q * price + fee <= self.money:
                    self._fill(j, q, price, fee)
            return

        marks = self._marks("open")

        def feasible(orders):
            if orders and self.value(marks) <= 0:
                return False
            equity = self.value(marks)
            quantities = self.position.copy()
            for j, q in orders:
                price, fee = self.quote(j, q)
                equity += q * self.future.multiplier[j] * (marks[j] - price) - fee
                quantities[j] += q
            collateral = sum(
                abs(q)
                * self.future.multiplier[j]
                * marks[j]
                * term(self.future.margin_rate, self.row, j)
                for j, q in enumerate(quantities)
                if q
            )
            return equity >= collateral

        # Enumerate all lattice changes; no native basket search is reused.
        candidates = {1.0, 0.0}
        for j, q in openings:
            step = (
                1 if self.future.qty_step is None else self.future.qty_step[self.row, j]
            )
            candidates.update(
                k * step / abs(q) for k in range(1, int(abs(q) / step) + 1)
            )
        orders = []
        for scale in sorted(candidates, reverse=True):
            trial = [
                (j, np.sign(q) * self._filter(j, abs(q) * scale)) for j, q in openings
            ]
            trial = [(j, q) for j, q in trial if q]
            if feasible(trial):
                orders = trial
                break
        for j, q in orders:
            if feasible([(j, q)]):
                self._fill(j, q, *self.quote(j, q))

    def step(self, target=None, *, flow=0.0):
        before = self.state()
        actual = (
            0.0
            if self.future is not None and self.bankrupt
            else max(flow, -before.transferable)
        )
        self.money += actual
        self.current = dict(
            flow=actual,
            shortfall=actual - flow,
            fees=0.0,
            funding=0.0,
            clearance_fee=0.0,
            deficit=0.0,
            liquidations=[],
            expiries=[],
        )
        f, d = self.future, self.row
        skip = False
        if f is not None and np.any(self.position):
            distance, nominal = self._risk(self._marks("open"))
            if distance <= nominal * 1e-12:
                self._liquidate(self._marks("open"), "open")
                skip = True
        if not skip:
            self._trade(target)
        if f is not None:
            if np.any(self.position):
                op = self._marks("open")
                distance, nominal = self._risk(op)
                if distance <= nominal * 1e-12:
                    self._liquidate(op, "post_trade")
                else:
                    endpoint = np.where(
                        self.position > 0,
                        f.liquidation.mark_low[d],
                        f.liquidation.mark_high[d],
                    )
                    end, _ = self._risk(endpoint)
                    if end < distance and end <= nominal * 1e-12:
                        fraction = min(distance / (distance - end), 1.0)
                        self._liquidate(op + fraction * (endpoint - op), "intraday")
            for j, q in enumerate(self.position.copy()):
                if q and d == f.expiry[j]:
                    price = f.settlement[d, j]
                    fee = abs(q) * (
                        f.multiplier[j] * price * term(f.delivery_fee_rate, d, j)
                        + term(f.delivery_fee_per_contract, d, j)
                    )
                    self._fill(j, -q, price, fee)
                    self.current["expiries"].append(j)
            paid = (
                0.0
                if f.funding is None
                else sum(q * f.funding[d, j] for j, q in enumerate(self.position) if q)
            )
            self.money -= paid
            self.current["funding"] = paid
            if np.any(self.position):
                distance, nominal = self._risk(self._marks("close"))
                if distance <= nominal * 1e-12:
                    self._liquidate(self._marks("close"), "close")
            marks = self._marks("close")
            equity = self.value(marks)
            im = sum(
                abs(q) * f.multiplier[j] * marks[j] * term(f.margin_rate, d, j)
                for j, q in enumerate(self.position)
                if q
            )
            interest = max(equity - im, 0) * term(self.config.cash_rate, d) - max(
                im - equity, 0
            ) * term(self.config.margin.financing_rate, d)
            self.money += interest
            if self.value(marks) < 0:
                self.bankrupt = True
        else:
            self.money += max(self.money, 0) * term(self.config.cash_rate, d)
            marks = self._marks("close")
        equity = self.value(marks)
        self.current.update(
            equity=equity,
            cash=equity if f is not None else self.money,
            wallet_balance=self.money if f is not None else None,
            unrealized=equity - self.money if f is not None else None,
            entry=self.entry.copy(),
            positions=self.position.copy(),
            bankrupt=self.bankrupt,
        )
        self.records.append(self.current)
        self.previous = equity
        self.row += 1
        return self.current


class PairOracle:
    """Independent paired coordinator for synthetic one-column-per-coin cases."""

    def __init__(
        self,
        spot,
        usdm,
        *,
        transfer=None,
        hedge="independent",
        kind="coins",
        trade_hour=None,
    ):
        self.markets = spot, usdm
        self.spot = WalletOracle(spot.market, spot.costs, spot.config)
        self.usdm = WalletOracle(usdm.market, usdm.costs, usdm.config)
        self.transfer, self.hedge, self.kind = transfer, hedge, kind
        self.coins = list(
            dict.fromkeys(p["coin"] for m in self.markets for p in m.profiles)
        )
        self.trade_hour = trade_hour
        self.ratio = dict.fromkeys(self.coins, 1.0)
        self.records = []

    def step(self, target=None):
        d = self.spot.row
        ss, us = self.spot.state(), self.usdm.state()
        if (
            self.trade_hour is not None
            and (int(self.markets[0].times[d]) // 3600000) % 24 != self.trade_hour
        ):
            target = None
        refs = {}
        if d:
            for m in reversed(self.markets):
                for j, p in enumerate(m.profiles):
                    prices = (
                        m.market.close
                        if m.kind == "spot"
                        else m.market.futures.settlement
                    )
                    refs[p["coin"]] = (
                        prices[d - 1, j] / m.arrays["coin_units"][d - 1, j]
                    )
        targets = [None, None]
        if target is not None:
            a = np.asarray(target, float)
            if a.any() and not d:
                raise ValueError("previous price required")
            for side, m in enumerate(self.markets):
                targets[side] = []
                for j, p in enumerate(m.profiles):
                    k = self.coins.index(p["coin"])
                    q = a[k, side]
                    if self.kind == "weights" and q:
                        q *= (ss.equity + us.equity) / refs[p["coin"]]
                    unit = m.arrays["coin_units"][d, j]
                    if side:
                        q = contract_count(q / (unit * m.futures.multiplier[j]))
                    else:
                        q /= unit
                    targets[side].append(q)
                    if self.hedge == "follow_spot" and a[k, 0] > 0:
                        self.ratio[p["coin"]] = -a[k, 1] / a[k, 0]
        exiting = set()
        spot_market = self.markets[0]
        for event in spot_market.events:
            if (
                event["kind"] == "delist"
                and event["status"] == "active"
                and event["known_at"] <= spot_market.times[d]
            ):
                for j, p in enumerate(spot_market.profiles):
                    if (
                        event["instrument"]
                        == spot_market.arrays["execution_symbol"][d, j]
                    ):
                        exiting.add(p["coin"])
                        if targets[0] is None:
                            targets[0] = list(ss.shares)
                        targets[0][j] = 0
        request = self.transfer(d, ss, us) if self.transfer else 0
        if target is not None and getattr(self.transfer, "pre_fund", False):
            desired = list(targets[1])
            f = self.markets[1].futures
            if self.hedge == "follow_spot":
                spot_coins = {
                    p["coin"]: targets[0][j] * spot_market.arrays["coin_units"][d, j]
                    for j, p in enumerate(spot_market.profiles)
                }
                desired = [
                    contract_count(
                        -self.ratio[p["coin"]]
                        * spot_coins.get(p["coin"], 0)
                        / (self.markets[1].arrays["coin_units"][d, j] * f.multiplier[j])
                    )
                    for j, p in enumerate(self.markets[1].profiles)
                ]
            margin = fees = cushion = 0.0
            for j, quantity in enumerate(desired):
                if quantity == 0 and us.shares[j] == 0:
                    continue
                price = f.settlement[max(0, d - 1), j]
                if not math.isfinite(price) or price <= 0:
                    raise ValueError(
                        "oracle: previous settlement required for prefunding"
                    )
                # Convert contracts to underlying units before valuing IM,
                # as in the independently enumerated feasibility check.
                amount = abs(quantity) * f.multiplier[j] * price
                margin += amount * term(f.margin_rate, d, j)
                fees += abs(quantity - us.shares[j]) * (
                    price * f.multiplier[j] * term(f.fee_rate, d, j)
                    + term(f.fee_per_contract, d, j)
                )
                cushion += amount * self.transfer.buffer
            required = max(margin, us.initial_margin) + fees + cushion
            request = max(request, required - us.equity)

            # This oracle credits the wallet balance, then values its holdings.
            # Check that operation, not merely the mathematical difference.
            def sufficient(credit):
                # Independently simulate reductions and both opening checks at
                # the supplied previous marks. Never call the native preview.
                balance = self.usdm.money + credit
                positions = list(us.shares)
                openings = []
                for j, goal in enumerate(desired):
                    if goal == 0 and positions[j] == 0:
                        continue
                    p = f.settlement[max(0, d - 1), j]
                    mult = f.multiplier[j]
                    delta = goal - positions[j]
                    closed = (
                        min(abs(delta), abs(positions[j]))
                        if delta * positions[j] < 0
                        else 0
                    )
                    unit_fee = mult * p * term(f.fee_rate, d, j) + term(
                        f.fee_per_contract, d, j
                    )
                    if closed:
                        direction = math.copysign(1, positions[j])
                        balance += (
                            closed * mult * direction * (p - self.usdm.entry[j])
                            - closed * unit_fee
                        )
                        positions[j] -= direction * closed
                    q = goal - positions[j]
                    if q:
                        openings.append(
                            (
                                abs(q) * unit_fee,
                                abs(q) * mult * p * term(f.margin_rate, d, j),
                            )
                        )
                equity = balance + sum(
                    q
                    * f.multiplier[j]
                    * (f.settlement[max(0, d - 1), j] - self.usdm.entry[j])
                    for j, q in enumerate(positions)
                    if q
                )
                collateral = sum(
                    abs(q)
                    * f.multiplier[j]
                    * f.settlement[max(0, d - 1), j]
                    * term(f.margin_rate, d, j)
                    for j, q in enumerate(positions)
                    if q
                )
                for charge, extra in openings:
                    if equity <= 0 or equity - charge < collateral + extra:
                        return False
                    equity -= charge
                    collateral += extra
                return equity >= collateral

            if not us.bankrupt:
                while not sufficient(min(request, ss.transferable)):
                    if request >= ss.transferable:
                        request = max(
                            request, math.nextafter(ss.transferable, math.inf)
                        )
                        break
                    # Step the credited balance too: a subnormal request at
                    # zero would take unbounded iterations to affect money.
                    request = min(
                        ss.transferable,
                        max(
                            math.nextafter(request, math.inf),
                            math.nextafter(self.usdm.money + request, math.inf)
                            - self.usdm.money,
                        ),
                    )
        limit = ss.transferable if request >= 0 else us.transferable
        accepted = math.copysign(min(abs(request), limit), request)
        if us.bankrupt:
            accepted = 0.0
        sr = self.spot.step(targets[0], flow=-accepted)
        if self.hedge == "follow_spot" and (target is not None or exiting):
            coins = {
                p["coin"]: self.spot.position[j]
                * self.markets[0].arrays["coin_units"][d, j]
                for j, p in enumerate(self.markets[0].profiles)
            }
            targets[1] = [
                contract_count(
                    -self.ratio[p["coin"]]
                    * coins.get(p["coin"], 0)
                    / (
                        self.markets[1].arrays["coin_units"][d, j]
                        * self.markets[1].futures.multiplier[j]
                    )
                )
                if target is not None or p["coin"] in exiting
                else us.shares[j]
                for j, p in enumerate(self.markets[1].profiles)
            ]
        ur = self.usdm.step(targets[1], flow=accepted)
        residual = dict.fromkeys(self.coins, 0.0)
        for side, (m, wallet) in enumerate(
            zip(self.markets, (self.spot, self.usdm), strict=True)
        ):
            for j, p in enumerate(m.profiles):
                residual[p["coin"]] += (
                    wallet.position[j]
                    * m.arrays["coin_units"][d, j]
                    * (m.futures.multiplier[j] if side else 1)
                )
        record = dict(
            spot=sr,
            usdm=ur,
            equity=sr["equity"] + ur["equity"],
            requested=request,
            actual=accepted,
            residual=list(residual.values()),
        )
        self.records.append(record)
        return record
