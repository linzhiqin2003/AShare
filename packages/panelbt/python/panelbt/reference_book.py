"""Naive scalar reference for `FuturesBook` (pure futures, cash account).

Written from SPEC.md ("Stand-alone futures portfolios" and "Futures"), not
from `futures_book.py` or the kernel: it reads the raw book inputs cell by
cell, applies the account terms, forced closes, limit masks and target kinds
itself, and runs one plain day-by-day, contract-by-contract loop:

    pre-open collateral -> commands (rebalance row, carried orders or the
    liquidation plan) -> reductions (by column) -> openings scaled by one
    common factor so that cash after fees covers all collateral -> closing
    variation at settlement -> end-row close (delivery or forced) ->
    maintenance call (collateral > equity) and insolvency.

Supported: kinds lots / notional / margin, carry_blocked, max_participation,
scalar or T×N slippage, cash_buffer. Not supported (the tests do not use
them here): spot columns, cash flows, interest, trade band, impact.
"""

from __future__ import annotations

import math

import numpy as np

TOL = 1e-9


def _get(value, day, col):
    """Scalar, (1, N), (T, 1) or T×N cell."""
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        return float(a)
    r = day if a.shape[0] > 1 else 0
    c = col if a.shape[1] > 1 else 0
    return float(a[r, c])


def _lot(x):
    return float(math.floor(x)) if x >= 1 else 0.0


def _close_qty(x, position):
    """Quantity for closing x of a position of size `position` (both >= 0)."""
    if x >= position:
        return position
    return _lot(x)


def _tol(a, b):
    return 1e-9 * max(abs(a), abs(b), 1.0)


class _Spec:
    """Per-cell effective inputs, derived from the raw book arguments."""

    def __init__(self, kw):
        price = np.asarray(
            kw["exec_price"] if kw.get("exec_price") is not None else kw["open"], float
        )
        self.t, self.n = t, n = price.shape
        self.price = price
        self.mult = [float(x) for x in np.broadcast_to(kw["multiplier"], (n,))]
        expiry = [min(int(x), t) for x in np.broadcast_to(kw["expiry"], (n,))]
        hold = kw.get("last_hold")
        hold = (
            [t] * n
            if hold is None
            else [min(int(x), t) for x in np.broadcast_to(hold, (n,))]
        )
        cash_settled = kw.get("cash_settled")
        cash_settled = (
            [False] * n
            if cash_settled is None
            else [bool(x) for x in np.broadcast_to(cash_settled, (n,))]
        )
        self.end = []
        self.personal = []
        for j in range(n):
            e = expiry[j]
            if hold[j] + 1 < e:
                self.end.append(hold[j] + 1)
                self.personal.append(True)
            else:
                self.end.append(e)
                self.personal.append(False)
        self.cash_settled = cash_settled
        tradable = kw.get("tradable")
        self.tradable = [
            [
                math.isfinite(price[d, j])
                and (True if tradable is None else bool(_get(tradable, d, j)))
                for j in range(n)
            ]
            for d in range(t)
        ]
        settle = np.asarray(kw["settlement"], float)
        self.settle = [[0.0] * n for _ in range(t)]
        for j in range(n):
            first = None
            last = None
            for d in range(t):
                v = _get(settle, d, j)
                if math.isfinite(v):
                    last = v
                    if first is None:
                        first = v
                self.settle[d][j] = last
            for d in range(t):
                if self.settle[d][j] is None:
                    self.settle[d][j] = first
        # A cell that is not tradable has no execution quote at all.
        self.price = price.copy()
        for d in range(t):
            for j in range(n):
                if not self.tradable[d][j]:
                    self.price[d, j] = math.nan
        self.start = []
        for j in range(n):
            rows = [d for d in range(t) if self.tradable[d][j]]
            self.start.append(rows[0] if rows else t)
        acct = kw.get("account")
        m_add = 0.0 if acct is None else acct.margin_add
        m_scale = 1.0 if acct is None else acct.margin_scale
        f_scale = 1.0 if acct is None else acct.fee_scale
        missing = kw.get("missing", "error")

        def window(d, j):
            return self.start[j] <= d <= self.end[j]

        def rate(name, default):
            raw = kw.get(name, default)
            out = [[0.0] * n for _ in range(t)]
            for j in range(n):
                prev = math.nan
                for d in range(t):
                    if not window(d, j):
                        continue
                    v = _get(raw, d, j)
                    if not math.isfinite(v):
                        if missing == "previous":
                            v = prev
                        elif not isinstance(missing, str):
                            v = float(missing)
                    if not math.isfinite(v):
                        raise ValueError(f"{name}: missing value at ({d}, {j})")
                    prev = v
                    out[d][j] = v
            return out

        margin = rate("margin_rate", None)
        self.margin = [
            [margin[d][j] * m_scale + m_add if window(d, j) else 0.0 for j in range(n)]
            for d in range(t)
        ]
        fee = rate("fee_rate", 0.0)
        fixed = rate("fee_per_contract", 0.0)
        self.fee = [[fee[d][j] * f_scale for j in range(n)] for d in range(t)]
        self.fixed = [[fixed[d][j] * f_scale for j in range(n)] for d in range(t)]
        self.costs = kw.get("costs")
        self.slip = (
            (lambda d, j: 0.0)
            if self.costs is None
            else (lambda d, j: _get(self.costs.slippage, d, j))
        )
        self.d_rate = kw.get("delivery_fee_rate", 0.0)
        self.d_fixed = kw.get("delivery_fee_per_contract", 0.0)
        # Trading permissions.
        self.buy = [row[:] for row in self.tradable]
        self.sell = [row[:] for row in self.tradable]
        for d in range(t):
            for j in range(n):
                p = price[d, j]
                if kw.get("no_buy") is not None and _get(kw["no_buy"], d, j):
                    self.buy[d][j] = False
                if kw.get("no_sell") is not None and _get(kw["no_sell"], d, j):
                    self.sell[d][j] = False
                if kw.get("limit_up") is not None:
                    up = _get(kw["limit_up"], d, j)
                    if math.isfinite(up) and p >= up:
                        self.buy[d][j] = False
                if kw.get("limit_down") is not None:
                    dn = _get(kw["limit_down"], d, j)
                    if math.isfinite(dn) and p <= dn:
                        self.sell[d][j] = False
                if kw.get("infer_locked") and d > self.start[j]:
                    o = _get(kw["open"], d, j)
                    h = _get(kw["high"], d, j)
                    lo = _get(kw["low"], d, j)
                    v = _get(kw["volume"], d, j)
                    if o == h == lo and v > 0:
                        if o > self.settle[d - 1][j]:
                            self.buy[d][j] = False
                        elif o < self.settle[d - 1][j]:
                            self.sell[d][j] = False
        vol = kw.get("volume")
        self.volume = None if vol is None else np.nan_to_num(np.asarray(vol, float))

    def end_fee(self, q, d, j):
        """Fee on the end row: delivery (cash settled expiry) or forced close."""
        s = self.settle[d][j]
        if self.cash_settled[j] and not self.personal[j]:
            r, f = _get(self.d_rate, d, j), _get(self.d_fixed, d, j)
        else:
            r, f = self.fee[d][j] + self.slip(d, j), self.fixed[d][j]
        return abs(q) * (self.mult[j] * s * r + f)


def run(kw, plan, kind="notional", config=None):
    """Run the naive loop; returns a dict of daily arrays and event lists.

    `plan` is a T×N dense array (all-NaN row: no rebalance; NaN cell: 0).
    """
    from .models import Config

    config = (
        config or kw.get("config") or Config(initial_cash=kw.get("initial_cash", 1e7))
    )
    sp = _Spec(kw)
    t, n = sp.t, sp.n
    plan = np.asarray(plan, float)
    if kind == "margin":
        plan = plan.copy()
        for d in range(t):
            for j in range(n):
                if not math.isnan(plan[d, j]) and plan[d, j] != 0:
                    m = sp.margin[d][j]
                    if m <= 0:
                        raise ValueError("margin target outside the trading window")
                    plan[d, j] = plan[d, j] / m
    lots = kind == "lots"
    part = config.max_participation
    carry = config.carry_blocked
    buffer = config.cash_buffer

    cash = float(config.initial_cash)
    pos = [0.0] * n
    basis = [0.0] * n
    seen = False
    call = False
    bankrupt = False
    pending = []
    out = dict(
        equity=np.zeros(t),
        cash=np.zeros(t),
        commission=np.zeros(t),
        other_fee=np.zeros(t),
        margin_usage=np.zeros(t),
        margin_call=np.zeros(t, bool),
        positions=np.zeros((t, n)),
        n_expired_target=np.zeros(t, np.int64),
        n_blocked_buy=np.zeros(t, np.int64),
        n_blocked_sell=np.zeros(t, np.int64),
        expiries=[],
        liquidations=[],
        trades=[],
        bankruptcies=[],
        rounded_to_zero=0,
    )

    def price_of(q, d, j):
        raw = sp.price[d, j]
        s = sp.slip(d, j)
        return raw * (1 + s) if q > 0 else raw * (1 - s)

    def commission(q, p, d, j):
        return abs(q) * (sp.mult[j] * p * sp.fee[d][j] + sp.fixed[d][j])

    def margin_of(q, p, d, j):
        return abs(q) * sp.mult[j] * p * sp.margin[d][j]

    def realized(before, q, b, p, j):
        if before * q < 0:
            closed = math.copysign(min(abs(before), abs(q)), before)
            return closed * sp.mult[j] * (p - b)
        return 0.0

    def project(c, fm, j, q, d):
        if q == 0:
            return c, fm
        p = price_of(q, d, j)
        c += realized(pos[j], q, basis[j], p, j) - commission(q, p, d, j)
        fm -= margin_of(pos[j], basis[j], d, j)
        fm += margin_of(pos[j] + q, p, d, j)
        return c, fm

    def feasible(c, fm):
        return fm <= c and c - fm >= 0.0

    def execute(j, q, d, forced):
        nonlocal cash, seen
        p = price_of(q, d, j)
        before = pos[j]
        after = before + q
        pnl = realized(before, q, basis[j], p, j)
        fee = commission(q, p, d, j)
        cash += pnl - fee
        out["commission"][d] += fee
        pos[j] = after
        if after == 0:
            basis[j] = 0.0
        elif before == 0:
            basis[j] = p
            seen = True
        elif (before > 0) == (q > 0):
            basis[j] = (abs(before) * basis[j] + abs(q) * p) / abs(after)
        out["trades"].append((d, j, q, p, fee))
        if forced:
            out["liquidations"].append((d, j, q))

    def cap(d, j):
        if sp.volume is None or part <= 0:
            return math.inf
        v = sp.volume[d, j]
        return float(math.floor(v * part)) if v > 0 else 0.0

    for d in range(t):
        held = sorted(j for j in range(n) if pos[j] != 0)
        fm = sum(margin_of(pos[j], basis[j], d, j) for j in held)
        row = plan[d]
        rebalance = not np.all(np.isnan(row))
        forced = (call and config.margin.margin_call == "liquidate") or bankrupt
        unfilled = False
        if rebalance:
            pending = []
            weights = {j: 0.0 for j in held}
            for j in range(n):
                if not math.isnan(row[j]) and row[j] != 0:
                    weights[j] = float(row[j])
            commands = sorted(weights.items())
        elif carry:
            commands, pending = pending, []
        else:
            commands = []
        value = cash
        orders = []
        quantities = {}
        if forced:
            pending = []
            requested = dict(commands)
            for j, w in commands:
                if d > sp.end[j] and w != 0:
                    out["n_expired_target"][d] += 1
            commands = []

            def trial(keep, cash=cash, fm=fm, held=held, d=d):
                c, f = cash, fm
                for j in held:
                    mag = _close_qty(abs(pos[j]) * (1 - keep), abs(pos[j]))
                    q = -math.copysign(mag, pos[j])
                    if q != 0 and math.isfinite(sp.price[d, j]):
                        p = price_of(q, d, j)
                        if p > 0 and math.isfinite(p):
                            c, f = project(c, f, j, q, d)
                return c, f

            if bankrupt or value <= 0 or not feasible(*trial(0.0)):
                keep = 0.0
            elif feasible(*trial(1.0)):
                keep = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(64):
                    mid = (lo + hi) * 0.5
                    if feasible(*trial(mid)):
                        lo = mid
                    else:
                        hi = mid
                keep = lo
            for j in held:
                cur = pos[j]
                amount = _close_qty(abs(cur) * (1 - keep), abs(cur))
                if rebalance and j in requested:
                    w = requested[j]
                    p = sp.price[d, j]
                    mark = p if math.isfinite(p) else basis[j]
                    if lots:
                        desired = w
                    elif mark > 0:
                        desired = w * max(value, 0) * (1 - buffer) / (mark * sp.mult[j])
                    else:
                        desired = cur
                    risk = cur - max(desired, 0) if cur > 0 else min(desired, 0) - cur
                    amount = max(amount, _close_qty(risk, abs(cur)))
                quantities[j] = -math.copysign(amount, cur)
                commands.append((j, cur + quantities[j]))
        for j, w in commands:
            if d > sp.end[j]:
                if w != 0:
                    out["n_expired_target"][d] += 1
                continue
            p = sp.price[d, j]
            if math.isfinite(p):
                mark = p
            elif pos[j] != 0:
                mark = basis[j]
            elif d > 0:
                mark = sp.settle[d - 1][j]
            else:
                mark = 0.0
            cur = pos[j]
            if forced or lots:
                target = w
            elif mark > 0:
                raw = w * max(value, 0) * (1 - buffer) / (mark * sp.mult[j])
                target = math.copysign(math.floor(abs(raw)), raw)
                if raw != 0 and target == 0:
                    out["rounded_to_zero"] += 1
            else:
                target = 0.0
            if abs(target - cur) <= _tol(target, cur):
                target = cur
            limit = cap(d, j)
            used = 0.0
            blocked_buy = blocked_sell = False
            if forced:
                reduction = quantities[j]
            elif cur > 0 and target < cur:
                reduction = -_close_qty(cur - max(target, 0), cur)
            elif cur < 0 and target > cur:
                reduction = _close_qty(min(target, 0) - cur, -cur)
            else:
                reduction = 0.0
            if reduction != 0:
                ok = math.isfinite(p) and (
                    sp.buy[d][j] if reduction > 0 else sp.sell[d][j]
                )
                q = (
                    math.copysign(
                        _close_qty(min(abs(reduction), limit), abs(cur)), reduction
                    )
                    if ok
                    else 0.0
                )
                if q != reduction:
                    if reduction > 0:
                        blocked_buy = True
                    else:
                        blocked_sell = True
                if q != 0:
                    px = price_of(q, d, j)
                    if px > 0 and math.isfinite(px):
                        cash_f, fm = project(cash, fm, j, q, d)
                        execute(j, q, d, forced)
                        used = abs(q)
                if forced and abs(reduction) - used > _tol(reduction, cur):
                    unfilled = True
            after = pos[j]
            if target > 0 and after >= 0 and target > after and target > cur:
                opening = _lot(target - after)
            elif target < 0 and after <= 0 and target < after and target < cur:
                opening = -_lot(after - target)
            else:
                opening = 0.0
            if abs(opening) > _tol(target, after) and not forced and not bankrupt:
                ok = math.isfinite(p) and (
                    sp.buy[d][j] if opening > 0 else sp.sell[d][j]
                )
                q = (
                    math.copysign(
                        _lot(min(abs(opening), max(limit - used, 0.0))), opening
                    )
                    if ok
                    else 0.0
                )
                if opening != q:
                    if opening > 0:
                        blocked_buy = True
                    else:
                        blocked_sell = True
                if q != 0:
                    orders.append((j, q))
            if mark == 0 and w != 0 and target >= cur:
                if w > 0:
                    blocked_buy = True
                else:
                    blocked_sell = True
            if blocked_buy:
                out["n_blocked_buy"][d] += 1
            if blocked_sell:
                out["n_blocked_sell"][d] += 1
            if (blocked_buy or blocked_sell) and carry and not forced:
                pending.append((j, w))
        if orders:
            orders.sort(key=lambda x: (x[1] > 0, x[0]))

            def basket(scale, cash=cash, fm=fm, orders=orders, d=d):
                c, f = cash, fm
                for j, q in orders:
                    x = math.copysign(_lot(abs(q) * scale), q)
                    if x != 0:
                        c, f = project(c, f, j, x, d)
                return c, f

            if feasible(*basket(1.0)):
                scale = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(64):
                    mid = (lo + hi) * 0.5
                    if feasible(*basket(mid)):
                        lo = mid
                    else:
                        hi = mid
                scale = lo
            for j, desired in orders:
                q = math.copysign(_lot(abs(desired) * scale), desired)
                if q != 0 and not feasible(*project(cash, fm, j, q, d)):
                    lo, hi = 0.0, abs(q)
                    for _ in range(64):
                        mid = (lo + hi) * 0.5
                        if feasible(
                            *project(cash, fm, j, math.copysign(_lot(mid), q), d)
                        ):
                            lo = mid
                        else:
                            hi = mid
                    q = math.copysign(_lot(lo), q)
                if abs(q) > _tol(pos[j] + desired, pos[j]):
                    _, fm = project(cash, fm, j, q, d)
                    execute(j, q, d, False)
        # Close: variation at settlement, then the end-row close.
        closing = 0.0
        for j in sorted(j for j in range(n) if pos[j] != 0):
            s = sp.settle[d][j]
            cash += pos[j] * sp.mult[j] * (s - basis[j])
            basis[j] = s
            if d == sp.end[j]:
                fee = sp.end_fee(pos[j], d, j)
                cash -= fee
                out["other_fee"][d] += fee
                out["expiries"].append((d, j, pos[j], s, fee))
                pos[j] = 0.0
                basis[j] = 0.0
            else:
                closing += margin_of(pos[j], s, d, j)
        equity = cash
        tol = max(1e-10 * max(abs(equity), 1.0), 1e-10 * closing)
        call = seen and (closing > equity + tol or unfilled)
        if seen and equity <= 0 and not bankrupt and (any(pos) or cash < 0):
            bankrupt = True
            out["bankruptcies"].append(d)
        out["equity"][d] = equity
        out["cash"][d] = cash
        out["margin_usage"][d] = (
            closing / equity if equity > 0 else (math.inf if closing > 0 else 0.0)
        )
        out["margin_call"][d] = call
        out["positions"][d] = pos
    return out


__all__ = ["run"]
