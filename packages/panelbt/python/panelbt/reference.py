"""独立逐日逐股参考实现：只共享数据结构和输入校验，不调用内核交易逻辑。"""

from __future__ import annotations

import math

import numpy as np

from .api import _prepare
from .models import (
    COUNTERS,
    METRICS,
    Config,
    Costs,
    DelistEvent,
    Market,
    Result,
    Targets,
    Trade,
)


def run(
    market: Market,
    targets: Targets,
    costs: Costs | None = None,
    config: Config | None = None,
) -> Result:
    costs = costs or Costs()
    config = config or Config()
    prepared = _prepare(market, [targets], costs, config)
    sell_tax, buy_tax, minimum, steps = prepared[3]
    t, n = market.shape
    shares = [0.0] * n
    last = [0.0] * n
    dead = [False] * n
    cash = config.initial_cash
    data = {
        name: np.zeros(
            t, dtype=np.int64 if name.startswith("n_blocked") else np.float64
        )
        for name in METRICS
    }
    counts = dict.fromkeys(COUNTERS, 0)
    trades = [] if config.record_trades else None
    positions = np.zeros((t, n)) if config.record_trades else None
    events = []
    pending: dict[int, float] = {}
    cursor = 0

    def round_buy(x: float, col: int) -> float:
        if steps[col] == 0:
            return max(x, 0.0)
        if x < minimum[col]:
            return 0.0
        return float(
            minimum[col] + math.floor((x - minimum[col]) / steps[col]) * steps[col]
        )

    def charge(
        qty: float, price: float, tax_rate: float
    ) -> tuple[float, float, float, float]:
        amount = abs(qty) * price
        commission = (
            max(amount * costs.commission, costs.min_commission) if amount > 0 else 0.0
        )
        return amount, commission, amount * tax_rate, amount * costs.transfer_fee

    def required(qty: float, price: float, tax_rate: float) -> float:
        amount, comm, tax, transfer = charge(qty, price, tax_rate)
        return amount + comm + tax + transfer

    for day in range(t):
        for i in range(n):
            if shares[i] > 0:
                div = (
                    float(market.cash_div[day, i])
                    if market.cash_div is not None
                    else 0.0
                )
                cash += shares[i] * div
                if div > 0:
                    counts["dividend"] += 1
                split = float(market.split[day, i]) if market.split is not None else 1.0
                shares[i] *= split
                if split != 1:
                    counts["split"] += 1
            if market.delist is not None and market.delist[day, i]:
                dead[i] = True
                if shares[i] > 0:
                    price = last[i] * config.delist_recovery
                    proceeds = shares[i] * price
                    cash += proceeds
                    events.append(DelistEvent(day, i, shares[i], price, proceeds))
                    counts["delist"] += 1
                    shares[i] = 0.0
        rebalance = cursor < len(targets.days) and targets.days[cursor] == day
        if rebalance:
            pending = {}
            weights = {i: 0.0 for i in range(n) if shares[i] > 0}
            for j in range(
                int(targets.indptr[cursor]), int(targets.indptr[cursor + 1])
            ):
                weights[int(targets.cols[j])] = float(targets.weights[j])
            cursor += 1
        else:
            weights = dict(pending) if config.carry_blocked else {}
            pending = {}
        # 保持独立直白实现；每个证券的历史收盘价已在前一日末更新。
        holding_value = 0.0
        for i in range(n):
            p = float(market.exec_price[day, i])
            holding_value += shares[i] * (p if math.isfinite(p) else last[i])
        value = cash + holding_value
        sell_orders = []
        buy_orders = []
        for i in sorted(weights):
            if dead[i]:
                continue
            weight = weights[i]
            p = float(market.exec_price[day, i])
            mark = p if math.isfinite(p) else last[i]
            target = (
                round_buy(weight * value * (1 - config.cash_buffer) / mark, i)
                if mark > 0
                else 0.0
            )
            capacity = math.inf
            if market.volume is not None and config.max_participation > 0:
                vol = float(market.volume[day, i])
                capacity = (
                    vol * config.max_participation
                    if math.isfinite(vol) and vol > 0
                    else 0.0
                )
                if steps[i] > 0:
                    capacity = math.floor(capacity / steps[i]) * steps[i]
            blocked = False
            if shares[i] > target:
                wanted = shares[i] - target
                if target != 0 and steps[i] > 0:
                    wanted = math.floor(wanted / steps[i]) * steps[i]
                qty = min(wanted, capacity)
                if not math.isfinite(p) or not market.can_sell[day, i]:
                    qty = 0.0
                if wanted > qty:
                    data["n_blocked_sell"][day] += 1
                    counts["blocked_sell"] += 1
                    blocked = True
                if qty > 0:
                    sell_orders.append((i, qty))
            else:
                wanted = round_buy(target - shares[i], i)
                qty = round_buy(min(wanted, capacity), i)
                if not math.isfinite(p) or not market.can_buy[day, i]:
                    qty = 0.0
                if wanted > qty or (mark == 0 and weight > 0):
                    data["n_blocked_buy"][day] += 1
                    counts["blocked_buy"] += 1
                    blocked = True
                if qty > 0:
                    buy_orders.append((i, qty))
            if blocked and config.carry_blocked:
                pending[i] = weight

        def fill(
            i: int, qty: float, day: int = day, rebalance: bool = rebalance
        ) -> None:
            nonlocal cash
            p = float(market.exec_price[day, i])
            price = p * (1 + costs.slippage) if qty > 0 else p * (1 - costs.slippage)
            rate = buy_tax[day] if qty > 0 else sell_tax[day]
            amount, comm, tax, transfer = charge(qty, price, float(rate))
            if qty > 0:
                cash -= amount + comm + tax + transfer
            else:
                cash += amount - comm - tax - transfer
            shares[i] += qty
            slip = abs(qty) * p * costs.slippage
            for key, x in [
                ("turnover", amount),
                ("commission", comm),
                ("tax", tax),
                ("transfer_fee", transfer),
                ("slippage_cost", slip),
            ]:
                data[key][day] += x
            counts["trades"] += 1
            if comm > amount * costs.commission:
                counts["min_commission"] += 1
            if not rebalance:
                counts["carry_trade"] += 1
            if trades is not None:
                trades.append(Trade(day, i, qty, price, comm, tax, transfer, slip))

        for i, qty in sell_orders:
            amount, comm, tax, transfer = charge(
                qty,
                float(market.exec_price[day, i]) * (1 - costs.slippage),
                float(sell_tax[day]),
            )
            if cash + amount >= comm + tax + transfer:
                fill(i, -qty)
        total = 0.0
        for i, qty in buy_orders:
            price = float(market.exec_price[day, i]) * (1 + costs.slippage)
            total += required(qty, price, float(buy_tax[day]))
        scale = cash / total if total > cash else 1.0
        if total > cash:
            counts["cash_scaled"] += 1
        for i, qty in buy_orders:
            price = float(market.exec_price[day, i]) * (1 + costs.slippage)
            rate = float(buy_tax[day])
            qty = round_buy(qty * scale, i) if scale < 1 else qty
            if required(qty, price, rate) > cash:
                # 独立用二分搜索求现金可负担的股数，避免复制内核解析式。
                if steps[i] == 0:
                    lo, hi = 0.0, qty
                    for _ in range(80):
                        mid = (lo + hi) / 2
                        if required(mid, price, rate) <= cash:
                            lo = mid
                        else:
                            hi = mid
                    qty = lo
                elif required(float(minimum[i]), price, rate) > cash:
                    qty = 0.0
                else:
                    lo, hi = 0, math.floor((qty - minimum[i]) / steps[i])
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        trial = float(minimum[i] + mid * steps[i])
                        if required(trial, price, rate) <= cash:
                            lo = mid
                        else:
                            hi = mid - 1
                    qty = float(minimum[i] + lo * steps[i])
            if qty > 0:
                fill(i, qty)
        end_value = cash
        for i in range(n):
            closing = float(market.close[day, i])
            if math.isfinite(closing):
                last[i] = closing
            end_value += shares[i] * last[i]
        data["nav"][day] = end_value / config.initial_cash
        data["cash"][day] = cash
        data["turnover"][day] = (
            data["turnover"][day] * 0.5 / value if value > 0 else 0.0
        )
        if positions is not None:
            positions[day] = shares
    return Result(
        **data,
        trades=trades,
        positions=positions,
        delist_events=events,
        diagnostics=counts,
    )
