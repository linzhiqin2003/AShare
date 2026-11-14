"""逐小时调度：普通目标交给内核，下架/季度换月仅覆盖涉及的列。"""

import copy
from dataclasses import replace

import numpy as np
import panelbt as bt

from ashare.data.crypto.events import asof
from ashare.data.crypto.timeutil import DAY, HOUR

from .crypto_schedule import DailyTargets, quarter_chain


class InvalidCryptoResult(ValueError):
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def simulate(m, targets, *, kind, allow_tier_exceeded, record_trades, roll_before):
    if kind not in ("weights", "shares"):
        raise ValueError("kind 只能是 weights/shares（期货 shares 为整数张）")
    daily = isinstance(targets, DailyTargets)
    trade_hour = targets.trade_hour if daily else 0
    w = (
        targets.align(m.times, m.symbols)
        if daily
        else targets.to_dense(*m.market.shape)
        if isinstance(targets, bt.Targets)
        else np.asarray(targets, dtype=float)
    )
    if isinstance(targets, bt.Targets):
        kind = targets.kind
    if m.kind == "spot" and kind == "shares":
        raise ValueError("现货仅支持权重目标，避免内核 shares 模式绕过数量快照")
    if w.shape != m.market.shape or np.isinf(w).any():
        raise ValueError("目标必须为 T×N，不能含 infinity")
    if m.kind == "spot" and (np.nan_to_num(w) < 0).any():
        raise ValueError("现货现金账户只做多")
    config = replace(m.config, carry_blocked=False) if daily else m.config
    if record_trades is not None:
        config = replace(config, record_trades=record_trades)
    diag = copy.deepcopy(m.diagnostics)
    diag.update(
        valid=True,
        unrealized=[],
        rolls=[],
        roll_overlap_events=0,
        periods_per_year=8760,
        daily_periods_per_year=365,
        carry_blocked=config.carry_blocked,
    )
    groups = {}
    for j, p in enumerate(m.profiles):
        if p["contract_type"] in ("CURRENT_QUARTER", "NEXT_QUARTER"):
            groups.setdefault(p["coin"], []).append(j)
    chain = (
        quarter_chain(
            m.profiles, m.times, trade_hour=trade_hour, roll_before=roll_before
        )
        if groups and roll_before is not None
        else None
    )
    execution_names = set(m.arrays["execution_symbol"].ravel())
    has_events = m.kind == "spot" and any(
        e["instrument"] in execution_names
        and e["kind"] == "delist"
        and e["known_at"] <= m.times[-1]
        for e in m.events
    )
    if not has_events and chain is None:
        result = bt.run(m.market, bt.Targets.from_dense(w, kind=kind), m.costs, config)
    else:
        sim = bt.Simulator(m.market, m.costs, config, kind=kind)
        retired = np.zeros(len(m.symbols), bool)
        unrealized = {}
        mult = np.array(
            [p["multiplier"] if m.kind == "usdm" else 1.0 for p in m.profiles]
        )
        for i, t in enumerate(m.times):
            state = sim.state
            q = state.shares.copy()
            raw = m.market.exec_price[i]
            price = np.where(np.isfinite(raw), raw, state.last)
            if m.kind == "usdm":
                price = m.arrays["mark_open"][i]
                basis = state.futures_basis
                equity = state.cash + np.sum(
                    np.where(q != 0, q * mult * (price - basis), 0)
                )
            else:
                equity = state.cash + np.sum(np.where(q != 0, q * price, 0))

            def quantity_target(quantities, equity=equity, price=price, q=q):
                if kind == "shares":
                    return quantities.copy()
                if equity <= 0:
                    return np.zeros(len(q))
                # 数量格点边缘的舍入防护；原仓位的目标不会跨过下一个格点。
                return np.nan_to_num(
                    quantities * mult * price / (equity * (1 - config.cash_buffer))
                ) * (1 + 4e-14)

            ordinary = not np.isnan(w[i]).all()
            target = np.nan_to_num(w[i], nan=0.0) if ordinary else None
            exits = np.zeros(len(q), bool)
            if has_events:
                known = asof(m.events, int(t))
                for e in known:
                    if e["kind"] == "delist":
                        exits |= m.arrays["execution_symbol"][i] == e["instrument"]
                if exits.any() and (
                    ordinary
                    or np.any(q[exits] != 0)
                    or np.any(exits[state.pending_cols])
                ):
                    if target is None:
                        target = quantity_target(q)
                        if config.carry_blocked:
                            # 先保留其他列的顺延买入/减仓，再以零目标取消下架列买入并继续退出。
                            target[state.pending_cols] = state.pending_targets
                    target[exits] = 0.0
            if chain is not None:
                for k, cols in enumerate(groups.values()):
                    dest = chain[i, k]
                    if dest < 0:
                        continue
                    # 完成上一小时才成为可选择的合约；目标仍由当前交易掩码决定能否成交。
                    if i == 0 or not np.isfinite(m.signal_prices[i, dest]):
                        continue
                    earlier = [
                        c
                        for c in cols
                        if m.profiles[c]["delivery_at"]
                        < m.profiles[dest]["delivery_at"]
                    ]
                    if ordinary:
                        transfer = sum(
                            target[c]
                            * (mult[c] / mult[dest] if kind == "shares" else 1.0)
                            for c in earlier
                        )
                        target[dest] += (
                            np.trunc(transfer) if kind == "shares" else transfer
                        )
                        target[earlier] = 0.0
                    if t % DAY == trade_hour * HOUR and any(q[c] != 0 for c in earlier):
                        if target is None:
                            target = quantity_target(q)
                        if not ordinary:
                            transfer = sum(q[c] * mult[c] for c in earlier)
                            newq = q.copy()
                            newq[earlier] = 0
                            newq[dest] += np.trunc(transfer / mult[dest])
                            values = quantity_target(newq)
                            target[earlier] = 0
                            target[dest] = values[dest]
                        diag["rolls"].append(
                            dict(
                                time=int(t),
                                from_symbols=[m.symbols[c] for c in earlier],
                                to_symbol=m.symbols[dest],
                            )
                        )
            sim.step(target)
            if chain is not None and diag["rolls"] and diag["rolls"][-1]["time"] == t:
                for k, cols in enumerate(groups.values()):
                    dest = chain[i, k]
                    if (
                        dest >= 0
                        and sim.state.shares[dest] != 0
                        and any(sim.state.shares[c] != 0 for c in cols if c != dest)
                    ):
                        diag["roll_overlap_events"] += 1
            if m.kind == "spot":
                ended = m.arrays["lifecycle"][i] == "ended"
                for j in np.flatnonzero(~ended & retired):
                    retired[j] = False
                    unrealized.pop(m.symbols[j], None)
                for j in np.flatnonzero(ended & ~retired):
                    amount = float(sim.state.shares[j])
                    retired[j] = True
                    if amount:
                        unrealized[m.symbols[j]] = dict(
                            symbol=m.symbols[j],
                            time=int(t),
                            quantity=amount,
                            value=amount * float(m.market.close[i, j]),
                        )
        result = sim.result()
        diag["unrealized"] = list(unrealized.values())
    coverage = result.coverage
    diag["liquidation_counts"] = {
        stage: int(coverage.get("liquidation_" + stage, 0))
        for stage in ("open", "post_trade", "intraday", "close")
    }
    diag["liquidation_events"] = [
        dict(time=int(m.times[e.day]), symbol=m.symbols[e.col], quantity=e.qty)
        for e in result.liquidation_events
    ]
    diag["clearance_fee"] = (
        float(np.sum(result.clearance_fee)) if result.clearance_fee is not None else 0.0
    )
    diag["bankruptcy_deficit"] = (
        float(np.sum(result.bankruptcy_deficit))
        if result.bankruptcy_deficit is not None
        else 0.0
    )
    diag["tier_cap_exceeded"] = int(coverage.get("tier_cap_exceeded", 0))
    diag["filtered_orders"] = {
        k: int(v)
        for k, v in coverage.items()
        if "rounded" in k or "notional" in k or "qty" in k
    }
    diag["blocked_buy"] = int(result.n_blocked_buy.sum())
    diag["blocked_sell"] = int(result.n_blocked_sell.sum())
    if diag["tier_cap_exceeded"]:
        diag["valid"] = False
    result.diagnostics["crypto"] = diag
    if diag["tier_cap_exceeded"] and not allow_tier_exceeded:
        raise InvalidCryptoResult(
            f"tier_cap_exceeded={diag['tier_cap_exceeded']}；第一档模型结果无效，研究可显式 allow_tier_exceeded=True",
            result,
        )
    return result
