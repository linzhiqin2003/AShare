"""Native target-percent orders, shared cash, automatic sell-before-buy sorting."""

import numpy as np
import vectorbt as vbt
from common import FEE, INITIAL

ENGINE = "numba"


def exit_first_sequence(sizes, days, n):
    """Native call order: zero-target exits precede target holdings.

    Task one has 5%-15% reserve cash, so remaining within-target trims can run
    in column order without affecting affordability. Every resulting NAV is
    checked against panelbt; no account path is computed in this adapter.
    This avoids automatic sorting of every inactive row (especially in Rust).
    """
    groups = sizes.shape[1] // n
    sequence = np.tile(np.arange(n, dtype=np.int64), (sizes.shape[0], groups))
    active = sizes[days].reshape(len(days), groups, n) > 0
    sequence[days] = np.argsort(active, axis=2, kind="stable").reshape(len(days), -1)
    return sequence


def run_orders(market, target):
    p = market.exec_price
    sizes = target.to_dense(*p.shape)
    return execute_orders(
        (
            p,
            sizes,
            exit_first_sequence(sizes, target.days, p.shape[1]),
            len(target.weights),
        )
    )


def execute_orders(prepared):
    p, sizes, sequence, nnz = prepared
    portfolio = vbt.Portfolio.from_orders(
        p,
        engine=ENGINE,
        size=sizes,
        size_type="targetpercent",
        price=p,
        val_price=p,
        init_cash=INITIAL,
        fees=FEE,
        cash_sharing=True,
        group_by=True,
        call_seq=sequence,
        update_value=False,
        max_orders=nnz * 2 + 100,
    )
    return (
        np.asarray(portfolio.value(engine="numba" if ENGINE == "numba" else "rust"))
        / INITIAL
    )


def prepare_batch(market, targets):
    p = market.exec_price
    n = p.shape[1]
    chunks = []
    for start in range(0, len(targets), 20):
        group = targets[start : start + 20]
        prices = np.tile(p, (1, len(group)))
        sizes = np.concatenate([x.to_dense(*p.shape) for x in group], axis=1)
        chunks.append(
            (
                prices,
                sizes,
                exit_first_sequence(sizes, group[0].days, n),
                np.repeat(np.arange(len(group)), n),
                sum(len(x.weights) * 2 for x in group) + 100,
            )
        )
    return chunks


def execute_batch(chunks):
    outputs = []
    for prices, sizes, sequence, groups, limit in chunks:
        portfolio = vbt.Portfolio.from_orders(
            prices,
            engine=ENGINE,
            size=sizes,
            size_type="targetpercent",
            price=prices,
            val_price=prices,
            init_cash=INITIAL,
            fees=FEE,
            cash_sharing=True,
            group_by=groups,
            call_seq=sequence,
            update_value=False,
            max_orders=limit,
        )
        outputs.extend(np.asarray(portfolio.value(engine=ENGINE)).T / INITIAL)
    return np.asarray(outputs)


def batch(market, targets):
    return execute_batch(prepare_batch(market, targets))


# Task two uses the documented pre-segment sizing hook. Portfolio accounting
# and order execution still run inside vectorbt's Numba simulation engine.
from numba import njit  # noqa: E402
from vectorbt.portfolio import nb  # noqa: E402


@njit
def lot_segment(c, weights, fee, step):
    size = np.zeros(c.group_len)
    value = c.last_cash[c.group]
    for j in range(c.group_len):
        col = c.from_col + j
        c.last_val_price[col] = c.close[c.i, col]
        value += c.last_position[col] * c.close[c.i, col]
    available = c.last_cash[c.group]
    needed = 0.0
    for j in range(c.group_len):
        col = c.from_col + j
        desired = weights[c.i, col] * value / c.close[c.i, col]
        if step > 0:
            desired = np.floor(desired / step) * step
        size[j] = desired - c.last_position[col]
        if size[j] < 0:
            available -= size[j] * c.close[c.i, col] * (1 - fee)
        else:
            needed += size[j] * c.close[c.i, col] * (1 + fee)
    if needed > available:
        for j in range(c.group_len):
            if size[j] > 0:
                size[j] = size[j] * (available / needed)
                if step > 0:
                    size[j] = np.floor(size[j] / step) * step
    cursor = 0
    for j in range(c.group_len):
        if size[j] < 0:
            c.call_seq_now[cursor] = j
            cursor += 1
    for j in range(c.group_len):
        if size[j] >= 0:
            c.call_seq_now[cursor] = j
            cursor += 1
    return (size,)


@njit
def lot_order(c, size):
    return nb.order_nb(
        size=size[c.col - c.from_col], price=c.close[c.i, c.col], fees=FEE
    )


def prepare(market, target, task=1):
    p = market.exec_price
    return (
        p,
        target.to_dense(*p.shape),
        np.isin(np.arange(len(p)), target.days)[:, None],
        len(target.weights),
        task,
    )


def execute(prepared):
    p, weights, mask, nnz, task = prepared
    step = 100.0 if task == 2 else 0.0
    initial = INITIAL * (100 if task == 2 else 1)
    portfolio = vbt.Portfolio.from_order_func(
        p,
        lot_order,
        pre_segment_func_nb=lot_segment,
        pre_segment_args=(weights, FEE, step),
        segment_mask=mask,
        init_cash=initial,
        cash_sharing=True,
        group_by=True,
        max_orders=nnz * 2 + 100,
    )
    return np.asarray(portfolio.value(engine="numba")) / initial


def _segmented(market, target, step):
    return execute(prepare(market, target, 2 if step else 1))


def run(market, target):
    """Skip inactive segments instead of sorting every all-NaN order row."""
    return _segmented(market, target, 0.0)


def run_lots(market, target):
    return _segmented(market, target, 100.0)
