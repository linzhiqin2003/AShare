"""品种层面的目标逐行推进到具体合约（``FuturesPortfolio.run`` 的实现）。

规则见 ``FuturesPortfolio.simulate`` 和 ``docs/FUTURES_BACKTEST.md`` 第 6 条。这里只管
「哪一行给引擎下什么单」；账户、成交、追保、到期都由 ``panelbt.Simulator`` 按内核规则做。
"""

from __future__ import annotations

import numpy as np

# Invariants of the driver (checked by test/backtest/test_futures_step.py):
#
# The only state the driver keeps is wait[k]: something to open once product k
# is flat on a tradable contract, either the caller's target set on a row where
# k had none ("goal") or the lots x multiplier of a position the engine closed
# on its end row ("units"). It keeps no record of orders it has sent.
#
# I1  Orders in flight are read from the engine every time they matter
#     (State.pending_cols / pending_targets at the start of the row). The
#     driver never re-sends an order from its own memory, so a carried order
#     that has filled, or was dropped, cannot come back. (The previous version
#     kept its own copy and cleaned it only on rows it wrote; a carried fill on
#     a quiet row left the copy behind and the next roll of another product
#     re-sent it, adding lots back after a margin-call liquidation.)
# I2  The driver writes a row only on rebalance rows (the caller's targets),
#     roll rows (k holds a contract other than held[t, k] and no engine order
#     for held[t, k] is pending) and landing rows (wait[k] and k flat, with no
#     pending order). On a written row every other product keeps exactly its
#     current lots, every pending engine order is re-sent with the engine's own
#     carried value, and only a leftover old contract that is neither current
#     nor pending is closed (recorded as touched for that product).
# I3  Hence a fill of product k on a row that is neither a rebalance row nor a
#     row on which the driver touched k comes from the engine alone: an order
#     pending at the start of the row, a margin-call liquidation or an end-row
#     close.
# I4  wait[k] is cleared by every rebalance row and on landing. On a
#     margin-call liquidation row every wait that existed before the row is
#     dropped (the engine suspends the plan until the next rebalance, so
#     nothing liquidated may be reopened); what the engine closes on that same
#     row because a contract ends is the position left after the liquidation,
#     and that is carried like on any other end row (the same result as a call
#     one row earlier). A leftover old contract closed on its end row after a
#     roll sets wait[k] to the lots the roll already opened, so its landing
#     changes nothing unless the roll was only partly filled, in which case it
#     completes the roll.


def simulate(pf, w, kind, trace=None):
    """逐行推进。``trace``（列表）给测试用：每行记 (行, 下单的品种, 行首顺延中的列)。"""
    book = pf.book
    sim = book.simulator("lots" if kind == "lots" else "notional")
    held = pf.held
    t, p = held.shape
    n = len(pf.symbols)
    mult = np.asarray(book.market.futures.multiplier, dtype=float)
    price = np.asarray(book.market.exec_price, dtype=float)
    end_row = np.asarray(book.end_row)
    finite = np.isfinite(price)
    first = np.where(finite.any(axis=0), np.argmax(finite, axis=0), t)
    buffer = 1.0 - book.config.cash_buffer
    owner = np.full(n, -1, dtype=np.int64)
    for k in range(p):
        cols = held[:, k]
        owner[cols[cols >= 0]] = k
    changed = np.ones(t, dtype=bool)
    changed[1:] = np.any(held[1:] != held[:-1], axis=1)
    rebalance_rows = ~np.all(np.isnan(w), axis=1)
    ending_on = {}
    for c in np.flatnonzero(end_row < t):
        ending_on.setdefault(int(end_row[c]), []).append(int(c))
    ending_on = {d: np.asarray(c) for d, c in ending_on.items()}
    wait = [None] * p
    goal_now = np.zeros(p)
    stats = dict(
        roll_rounding_events=0,
        roll_rounding_notional=0.0,
        roll_overlap_events=0,
        landings=0,
        liquidation_rows=0,
        liquidation_dropped_waits=0,
        liquidation_dropped_units=0.0,
    )

    def tradable(day, h):
        return h >= 0 and first[h] <= day <= end_row[h]

    def mark(day, col, basis):
        m = price[day, col]
        return m if np.isfinite(m) else basis[col]

    def keep_value(day, col, lots, equity, basis):
        """「这一列持有 lots 手」写成模拟器的目标值。"""
        if kind == "lots":
            return float(lots)
        m = mark(day, col, basis)
        if lots == 0 or not (equity > 0 and np.isfinite(m) and m > 0):
            return 0.0
        # 引擎按 trunc(|w| × E × (1 − buffer) / (价格 × 乘数)) 取手数：多给半手，取整后正好是 lots。
        return (lots + 0.5 * np.sign(lots)) * m * mult[col] / (equity * buffer)

    def goal_value(day, h, g):
        if kind == "margin":
            return float(book.margin_to_notional(day, [h], [g])[1][0])
        return float(g)

    def landing_value(day, h, e, equity, basis):
        tag, amount, ref = e
        if tag == "units":
            return keep_value(day, h, np.trunc(amount / mult[h]), equity, basis)
        if kind == "lots":
            lots = amount if ref is None else np.trunc(amount * ref / mult[h])
            return float(lots)
        # 名义 / 保证金：按落地这一行的权益和价格定手数（这一行开盘前已知，不是前视）。
        return goal_value(day, h, amount)

    def remaining_at_close(day, target):
        """追保强平行收盘前（强平之后、结束行了结之前）各列的手数。

        模拟器不在行内给出这个数，所以从行首存档复制一份，同一行同样的目标推进到底，读这一行的到期
        事件（只在「既是强平行又是结束行」时用到，很少见）。
        """
        from panelbt.step import Simulator

        probe = Simulator.restore(
            sim.snapshot(), book.market, book.costs, book.config, kind=sim.kind
        )
        probe.step(target)
        while probe.day < t:
            probe.step(None)
        out = np.zeros(n)
        for e in probe.result().expiry_events:
            if e.day == day:
                out[e.col] = e.qty
        return out

    def submit(day, cols, vals, rebalance):
        call = sim.state.margin_call
        if call:
            # 追保强平行：引擎丢弃这一行的加仓、暂停原计划直到下一次调仓；强平之前就在等的开仓和
            # 接续同样丢掉（计入诊断），否则强平掉的仓位会在之后被加回去。
            for k in range(p):
                if wait[k] is not None:
                    stats["liquidation_dropped_waits"] += 1
                    if wait[k][0] == "units":
                        stats["liquidation_dropped_units"] += abs(wait[k][1])
                    wait[k] = None
            stats["liquidation_rows"] += 1
        ends = ending_on.get(day)
        target = None
        if cols is not None:
            idx = np.asarray(cols, dtype=np.int64)
            o = np.argsort(idx, kind="stable")
            target = (idx[o], np.asarray(vals, dtype=float)[o])
        if ends is not None:
            q0 = np.array(sim.state.shares, dtype=float)
            kept = q0 != 0
            if cols is not None:
                kept = np.zeros(n, dtype=bool)
                kept[target[0][target[1] != 0]] = True
            if call and (q0[ends] != 0).any():
                q0 = remaining_at_close(day, target)
                kept &= q0 != 0
        sim.step(target)
        if ends is None:
            return
        shares = sim.state.shares
        for c in ends[kept[ends]]:
            k = owner[c]
            if k < 0 or shares[c] != 0:
                continue
            # 本行因合约结束而了结的仓位（追保行是强平后剩下的部分）接到品种的下一个合约。
            if rebalance and not call:
                wait[k] = ("goal", goal_now[k], mult[c])
            elif wait[k] is None or wait[k][0] != "units":
                wait[k] = ("units", q0[c] * mult[c], None)
            else:
                wait[k] = ("units", wait[k][1] + q0[c] * mult[c], None)

    def pending_now():
        st = sim.state
        return dict(
            zip(
                np.asarray(st.pending_cols).tolist(),
                np.asarray(st.pending_targets).tolist(),
                strict=True,
            )
        )

    watch = False  # 下一行要不要看持仓（有顺延单、或刚有过）
    carried_last = False
    for day in range(t):
        if rebalance_rows[day]:
            row = np.nan_to_num(w[day], nan=0.0)
            cols, vals = [], []
            for k in range(p):
                h = held[day, k]
                goal_now[k] = row[k]
                wait[k] = None
                if row[k] == 0:
                    continue
                if tradable(day, h):
                    cols.append(h)
                    vals.append(goal_value(day, h, row[k]))
                else:
                    wait[k] = ("goal", row[k], None)
            if trace is not None:
                trace.append((day, set(range(p)), set(pending_now())))
            submit(day, cols, vals, True)
        elif not (changed[day] or watch or any(e is not None for e in wait)):
            if trace is not None:
                trace.append((day, set(), set()))
            submit(day, None, None, False)
        else:
            st = sim.state
            pend = pending_now()
            shares = st.shares
            held_now = np.flatnonzero(shares)
            mine = {}
            for c in held_now.tolist():
                mine.setdefault(int(owner[c]), []).append(c)
            pend_by = {}
            for c in pend:
                pend_by.setdefault(int(owner[c]), []).append(c)
            equity, basis = st.equity, np.asarray(st.futures_basis)
            fresh = {}
            if not st.bankrupt:
                for k in sorted(
                    set(mine) | set(pend_by) | {k for k in range(p) if wait[k]}
                ):
                    h = held[day, k] if k >= 0 else -1
                    if k < 0 or not tradable(day, h):
                        continue
                    own = mine.get(k, [])
                    if wait[k] is not None and not pend_by.get(k):
                        # 落地：调用方的目标（或接续的手数）优先；还留着的旧合约这一行平掉。
                        fresh[k] = landing_value(day, h, wait[k], equity, basis)
                        wait[k] = None
                        stats["landings"] += 1
                    elif wait[k] is None and own and h not in own and h not in pend:
                        # 换月：平旧开新，保持「手数 × 乘数」，向零取整。
                        units = float(sum(shares[c] * mult[c] for c in own))
                        lots = np.trunc(units / mult[h])
                        rest = units - lots * mult[h]
                        if rest != 0:
                            stats["roll_rounding_events"] += 1
                            stats["roll_rounding_notional"] += abs(rest) * mark(
                                day, h, basis
                            )
                        fresh[k] = keep_value(day, h, lots, equity, basis)
            if not fresh:
                if trace is not None:
                    trace.append((day, set(), set(pend)))
                submit(day, None, None, False)
            else:
                cols, vals, closing = [], [], set()
                for k in sorted(set(mine) | set(pend_by) | set(fresh)):
                    h = held[day, k] if k >= 0 else -1
                    if k in fresh:
                        if fresh[k] != 0:
                            cols.append(h)
                            vals.append(fresh[k])
                        continue  # 旧合约不写 = 平掉
                    for c in pend_by.get(k, []):
                        cols.append(c)
                        vals.append(pend[c])  # 引擎自己的顺延目标原样带上
                    for c in mine.get(k, []):
                        if c in pend:
                            continue
                        if k < 0 or h < 0 or c == h:
                            cols.append(c)
                            vals.append(keep_value(day, c, shares[c], equity, basis))
                        else:
                            # 当前合约以外的残余（上次换月没平掉、也没在顺延）不写 = 这一行平掉
                            closing.add(k)
                if trace is not None:
                    trace.append((day, set(fresh) | closing, set(pend)))
                submit(day, cols, vals, False)
                after = sim.state.shares
                for k in fresh:
                    if any(after[c] != 0 for c in mine.get(k, []) if c != held[day, k]):
                        stats["roll_overlap_events"] += 1
        pend_after = sim.state.pending_cols
        # 顺延单可能在下一行成交、成交的列可能已不是品种当前的合约：下两行都要看持仓。
        has_pending = len(pend_after) > 0
        watch = has_pending or carried_last
        carried_last = has_pending
    result = sim.result()
    result.diagnostics["futures_portfolio"] = stats
    return result
