"""容量与成本：在一组资金规模下估计成交量占比、冲击成本、扣成本后的收益和策略容量。

冲击模型与 `panelbt` 同一口径（`Costs(impact_k=k, impact_exponent=e)`）：
一笔成交 q 股、成交价 p、当日成交量 V 股，

    滑点后价格 p_s = p (1 ± slippage)
    成交价     p_e = p_s (1 ± k (q / V)^e)
    冲击成本   = q · |p_e − p_s| = 成交额 · (1 ± slippage) · k · (q / V)^e

默认 e = 0.5（平方根律），k = 系数 × 该股此前 `vol_window` 日的日收益波动率（只用 t-1
及以前的数据），即常见的「冲击 = Y · σ · sqrt(成交量占比)」，Y 默认 1。也可以直接给 k。

解析估计（`capacity`）在每个规模下按股数记一遍现金账户（`simulate`）：调仓日按开盘
估值的权益把股数调到目标，先卖后买，买单超过现金时同比例缩小，不能买卖的挂起逐日重试，
送转、派现、退市按面板事件处理，逐笔按上式与佣金、税费计成本。这些规则与 panelbt 相同；
不同的只有两点：不做整手取整、不设成交量占比上限。所以它给出的是「交易规则最宽松时」
的成本，算一个规模只要毫秒级，可以在几十个规模上二分求容量。
`rerun=True` 时再对每个规模真的调用 `panelbt.run_batch` 两次：一次用与解析相同的宽松
规则（用来对账，冲击成本应几乎相等），一次用真实 A 股规则（整手、成交量上限 10%），
三者并排。

容量的两种定义：
1. 成本吃掉毛超额收益的 `cost_fraction`（默认一半）时的规模。成本曲线先降后升（小规模被
   最低佣金等固定费用拖累，大规模被冲击拖累），容量取上升段的交点；下降段的交点单列为
   「最小可行规模」`min_viable_aum`；
2. 成交量占比超过 `participation_cap`（默认 10%）的成交额占全部成交额的比例超过
   `breach_share`（默认 5%）时的规模。按成交额加权而不是按笔数，因为笔数会被大量小额
   调整主导。
两种容量都带状态字段；求不到时给 NaN 或 inf 并写明原因，不拿搜索边界充数。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .adapters import ashare_costs, market_from_panel, panel_config
from .inputs import benchmark_held, reference_prices, total_returns
from .inputs import field as panel_field
from .table import Table, plain

DEFAULT_AUM = (1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10)
COST_ITEMS = ("commission", "tax", "fee", "slippage", "impact")


def _cell(value, t, n):
    """标量、长度 T、T×N 三种形状的费率取第 t 行（长度 N）。"""
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.ndim == 1:
        return np.full(n, a[t])
    return a[t]


def volatility(panel, window=20, returns=None):
    """每只证券此前 window 日（不含当日）的日收益标准差，T×N；样本不足为 NaN。"""
    r = total_returns(panel) if returns is None else returns
    t, n = r.shape
    x = np.where(np.isfinite(r), r, 0.0)
    m = np.isfinite(r).astype(float)
    c1 = np.vstack([np.zeros((1, n)), np.cumsum(x, axis=0)])
    c2 = np.vstack([np.zeros((1, n)), np.cumsum(x * x, axis=0)])
    cn = np.vstack([np.zeros((1, n)), np.cumsum(m, axis=0)])
    out = np.full((t, n), np.nan)
    if t > window:
        # 第 t 行用 [t-window, t) 的收益。
        s1 = c1[window:t] - c1[: t - window]
        s2 = c2[window:t] - c2[: t - window]
        cnt = cn[window:t] - cn[: t - window]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2 - s1 * s1 / cnt) / (cnt - 1)
        var[cnt < max(5, window // 2)] = np.nan
        out[window:] = np.sqrt(np.maximum(var, 0))
    return out


def impact_coefficients(panel, coefficient=1.0, window=20, returns=None):
    """k = coefficient × 此前 window 日的日波动率；缺失时用当日截面中位数（再缺用 2%）。"""
    vol = volatility(panel, window, returns)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(vol, axis=1)
    median = np.where(np.isfinite(median), median, 0.02)
    vol = np.where(np.isfinite(vol), vol, median[:, None])
    return coefficient * vol


def _dense_targets(targets, shape):
    """支持 panelbt.Targets、ashare.portfolio.SparseTargets 或 T×N 稠密（整行 NaN 不调仓）。"""
    t, n = shape
    if hasattr(targets, "indptr"):
        days = np.asarray(targets.days, dtype=np.int64)
        indptr = np.asarray(targets.indptr, dtype=np.int64)
        cols = np.asarray(targets.cols, dtype=np.int64)
        w = np.asarray(targets.weights, dtype=float)
        rows = np.zeros((len(days), n))
        for j in range(len(days)):
            rows[j, cols[indptr[j] : indptr[j + 1]]] = w[indptr[j] : indptr[j + 1]]
        return days, rows
    w = np.asarray(targets, dtype=float)
    if w.shape != shape:
        raise ValueError(f"targets: 稠密目标须为 {shape}")
    days = np.flatnonzero(~np.all(np.isnan(w), axis=1))
    return days, np.nan_to_num(w[days])


@dataclass
class MarketArrays:
    """解析模拟用的行情数组（T×N）及预先算好的累计送转、累计派现。"""

    exec_price: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    can_buy: np.ndarray
    can_sell: np.ndarray
    split: np.ndarray
    cash_div: np.ndarray
    delist: np.ndarray

    def __post_init__(self):
        t, n = self.exec_price.shape
        # 估值价与 panelbt 相同：缺价时用最近收盘价逐次除权（停牌中送转、派现）。
        self.close_cf = reference_prices(self.close, self.cash_div, self.split)
        split = np.where(np.isfinite(self.split), self.split, 1.0)
        # S_t = Π_{u≤t} split_u；第 s 日派现按送转前股数：q_last·S_{s-1}/S_last·div_s。
        self.split_cum = np.cumprod(split, axis=0)
        before = np.vstack([np.ones((1, n)), self.split_cum[:-1]])
        self.div_cum = np.cumsum(before * np.nan_to_num(self.cash_div), axis=0)
        d = np.asarray(self.delist, bool)
        first = np.where(d.any(axis=0), d.argmax(axis=0), t)
        self.first_delist = first

    def carried(self, day, cols=slice(None)):
        """第 day 日开盘时的估值价：前一日估值价按当日派现、送转除权（panelbt `past_close`）。"""
        if day == 0:
            return np.full(self.close.shape[1], np.nan)[cols]
        div = np.nan_to_num(self.cash_div[day, cols])
        split = self.split[day, cols]
        split = np.where(np.isfinite(split) & (split > 0), split, 1.0)
        return np.maximum(self.close_cf[day - 1, cols] - div, 0.0) / split

    @classmethod
    def from_panel(cls, panel, exec_price="open"):
        p_exec = (
            panel_field(panel, exec_price)
            if isinstance(exec_price, str)
            else np.asarray(exec_price, float)
        )
        shape = p_exec.shape

        def optional(name, default, dtype=float):
            try:
                return np.asarray(panel[name], dtype=dtype)
            except KeyError:
                return np.full(shape, default, dtype=dtype)

        can_buy = can_sell = None
        if exec_price == "open":
            try:
                can_buy = np.asarray(panel["can_buy_open"], bool)
                can_sell = np.asarray(panel["can_sell_open"], bool)
            except KeyError:
                pass
        if can_buy is None:
            good = np.isfinite(p_exec) & (p_exec > 0)
            can_buy = can_sell = good
        return cls(
            p_exec,
            panel_field(panel, "close"),
            panel_field(panel, "volume"),
            can_buy,
            can_sell,
            optional("split", 1.0),
            optional("cash_dividend", 0.0),
            optional("delist", False, bool),
        )

    @classmethod
    def from_market(cls, market):
        def get(name, default):
            v = getattr(market, name)
            return np.full(market.shape, default) if v is None else np.asarray(v)

        return cls(
            np.asarray(market.exec_price, float),
            np.asarray(market.close, float),
            get("volume", np.nan).astype(float),
            np.asarray(market.can_buy, bool),
            np.asarray(market.can_sell, bool),
            get("split", 1.0).astype(float),
            get("cash_div", 0.0).astype(float),
            get("delist", False).astype(bool),
        )


@dataclass
class Simulation:
    """单个规模的解析模拟结果（逐个成交日）。"""

    aum: float
    days: np.ndarray  # 有成交尝试的日子
    equity: np.ndarray  # 这些日子前一日收盘的权益（成本率的分母）
    open_equity: np.ndarray  # 开盘估值的权益（目标股数按它算）
    costs: np.ndarray  # 日数 × 5：佣金、税、过户费、滑点、冲击（元）
    turnover: np.ndarray  # 单边换手 = 成交额之和 / 2 / 交易前权益
    participation: np.ndarray  # 所有成交的成交量占比
    notional: np.ndarray  # 对应的成交额
    final_equity: float
    years: float
    blocked: int
    overdrawn: int = (
        0  # 买单同比例缩小后，最低佣金等固定费用仍让现金透支的天数（诊断用）
    )


def _fills(q, side, day, i, m, costs, k, e):
    """一组成交（股数 q>0、方向 side）的成交额与五项成本，公式与 panelbt 的 quote 相同。"""
    n = m.exec_price.shape[1]
    p = m.exec_price[day, i]
    slip = _cell(costs.slippage, day, n)[i]
    part = q / m.volume[day, i]
    slipped = p * (1.0 + side * slip)
    imp = _cell(k, day, n)[i] * part**e
    price = slipped * (1.0 + side * imp)
    amount = q * price
    commission = np.maximum(
        costs.commission_rate * amount + costs.commission_per_share * q,
        costs.commission_min or 0.0,
    )
    if costs.commission_max_rate is not None:
        commission = np.minimum(commission, costs.commission_max_rate * amount)
    commission = np.where(q > 0, commission, 0.0)
    buy = side > 0
    tax = np.where(
        buy, _cell(costs.tax_buy, day, n)[i], _cell(costs.tax_sell, day, n)[i]
    )
    fee = np.where(
        buy, _cell(costs.fee_rate_buy, day, n)[i], _cell(costs.fee_rate_sell, day, n)[i]
    )
    per_share = np.where(
        buy,
        _cell(costs.fee_per_share_buy, day, n)[i],
        _cell(costs.fee_per_share_sell, day, n)[i],
    )
    return {
        "amount": amount,
        "commission": commission,
        "tax": tax * amount,
        "fee": fee * amount + per_share * q,
        "slippage": q * p * slip,
        "impact": q * np.abs(price - slipped),
        "participation": part,
        "notional": q * p,
    }


def simulate(
    rebalance_days,
    target_rows,
    market,
    costs,
    impact_k,
    impact_exponent,
    aum,
    cash_buffer=0.0,
    carry=True,
):
    """单个规模的解析模拟（`simulate_many` 只给一个规模）。"""
    return simulate_many(
        rebalance_days,
        target_rows,
        market,
        costs,
        impact_k,
        impact_exponent,
        [aum],
        cash_buffer,
        carry,
    )[0]


def simulate_many(
    rebalance_days,
    target_rows,
    market,
    costs,
    impact_k,
    impact_exponent,
    sizes,
    cash_buffer=0.0,
    carry=True,
):
    """解析模拟的数值核心，一次算多个资金规模：按股数记账的现金账户，规则与 panelbt
    相同，只是不做整手取整、不设成交量占比上限。

    - 目标股数 = 目标权重 × 开盘估值的权益 × (1 − cash_buffer) / 成交价；先卖后买。
    - 买单总花费超过现金时所有买单同比例缩小（panelbt 的比例缩放）。
    - 不能买卖（掩码、无成交价、无成交量）的证券挂起，之后每天用原目标与当日权益重试，
      直到下一次调仓（panelbt 的 carry_blocked）。
    - 送转、派现、退市按面板事件处理；只在有成交的日子结算，日子之间用累计量一次补上。

    各规模的账户互不影响，只是逐日一起推进（数组多一个规模维），所以算几十个规模与算
    一个规模的耗时相近。某个规模在某天没有挂单而别的规模有时，它也会在这天结算一次
    公司行为，结果与单独模拟只差浮点舍入。返回与 sizes 同序的 `Simulation` 列表。
    """
    m = market if isinstance(market, MarketArrays) else MarketArrays.from_market(market)
    t_len, n = m.exec_price.shape
    sizes = np.asarray(sizes, dtype=float).ravel()
    k_n = len(sizes)
    q = np.zeros((k_n, n))
    cash = sizes.copy()
    held = np.zeros(0, dtype=np.int64)  # 至少有一个规模持仓的列（可以偏多）
    first_delist = m.first_delist
    state = {"last": 0}
    plan = None
    pending = np.zeros((k_n, n), bool)
    rebalance = {int(d): j for j, d in enumerate(rebalance_days)}
    days = sorted(set(int(d) for d in rebalance_days))
    # 逐个成交日记录所有规模的值（rec_rows 标出当天有成交的规模），最后按规模拆开。
    rec_day, rec_rows, rec_eq, rec_open, rec_cost, rec_turn = [], [], [], [], [], []
    fill_size, fill_part, fill_notional = [], [], []
    blocked = np.zeros(k_n, dtype=np.int64)
    overdrawn = np.zeros(k_n, dtype=np.int64)  # 买单缩小后固定费用仍超过现金的天数

    def settle(day):
        nonlocal held
        last = state["last"]
        if len(held) and day > last:
            base = q[:, held] / m.split_cum[last, held]
            fd = first_delist[held]
            gone = (fd > last) & (fd <= day)
            stop = np.where(gone, fd, day)
            div = m.div_cum[stop, held] - m.div_cum[last, held]
            # 退市日的派现在变现之前：取到退市日（含）为止。
            cash[:] += (base * div).sum(axis=1)
            if gone.any():
                g = held[gone]
                d = first_delist[g]
                shares = base[:, gone] * m.split_cum[d, g]
                # panelbt：按退市日的估值价（最近有效收盘价逐次除权）全额变现。
                price = np.array(
                    [m.carried(int(x), int(c)) for x, c in zip(d, g, strict=True)]
                )
                proceeds = np.where(shares != 0, shares * price, 0.0)
                cash[:] += proceeds.sum(axis=1)
                q[:, g] = 0.0
            keep = held[~gone]
            q[:, keep] = base[:, ~gone] * m.split_cum[day, keep]
            held = keep
        state["last"] = day

    def valued(day, prices):
        """持仓按 prices（第 day 日的价格行，只取 held 列）估值。"""
        if not len(held):
            return cash.copy()
        return cash + np.nansum(q[:, held] * prices, axis=1)

    queue = list(days)
    visited = set()
    upcoming_index = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        while queue:
            day = queue.pop(0)
            if day in visited or day >= t_len:
                continue
            visited.add(day)
            # 成本率的分母取前一日收盘的权益（与 panelbt 收益和报告模块的口径相同）。
            if day > 0:
                settle(day - 1)
                previous = valued(day, m.close_cf[day - 1, held])
            else:
                previous = cash.copy()
            settle(day)
            if day in rebalance:
                plan = target_rows[rebalance[day]]
                # 新调仓行清掉挂单，所有列都按新目标处理（不在计划里的目标为 0）。
                names = np.union1d(np.flatnonzero(plan != 0), held)
                active = np.ones((k_n, len(names)), bool)
            else:
                names = np.flatnonzero(pending.any(axis=0))
                active = pending[:, names]
            if plan is None or not len(names):
                continue
            p_row = m.exec_price[day]
            p_held = p_row[held]
            mark_held = np.where(np.isfinite(p_held), p_held, m.carried(day, held))
            value = valued(day, mark_held)
            alive = first_delist[names] > day
            names, active = names[alive], active[:, alive]
            p = p_row[names]
            mark = np.where(np.isfinite(p), p, m.carried(day, names))
            good_mark = np.isfinite(mark) & (mark > 0)
            scale = np.maximum(value, 0.0)[:, None] * (1 - cash_buffer)
            target = np.where(
                good_mark,
                plan[names] * scale / np.where(good_mark, mark, 1.0),
                0.0,
            )
            delta = np.where(active, target - q[:, names], 0.0)
            vol = m.volume[day, names]
            quote_ok = np.isfinite(p) & (p > 0) & np.isfinite(vol) & (vol > 0)
            sell = (delta < 0) & (m.can_sell[day, names] & quote_ok)
            buy = (delta > 0) & (m.can_buy[day, names] & quote_ok)
            stuck = (delta != 0) & ~sell & ~buy
            blocked += stuck.sum(axis=1)
            sold = bought = None
            if sell.any():
                f = _fills(
                    np.where(sell, -delta, 0.0),
                    -1.0,
                    day,
                    names,
                    m,
                    costs,
                    impact_k,
                    impact_exponent,
                )
                sold = {c: np.where(sell, v, 0.0) for c, v in f.items()}
                cash[:] += (
                    sold["amount"] - sold["commission"] - sold["tax"] - sold["fee"]
                ).sum(axis=1)
                q[:, names] += np.where(sell, delta, 0.0)
            if buy.any():
                want = np.where(buy, delta, 0.0)
                f = _fills(want, 1.0, day, names, m, costs, impact_k, impact_exponent)
                bought = {c: np.where(buy, v, 0.0) for c, v in f.items()}
                spend = (
                    bought["amount"]
                    + bought["commission"]
                    + bought["tax"]
                    + bought["fee"]
                ).sum(axis=1)
                over = spend > cash
                if over.any():
                    ratio = np.where(
                        over & (spend > 0), np.maximum(cash, 0.0) / spend, 1.0
                    )
                    want = want * ratio[:, None]
                    f = _fills(
                        want, 1.0, day, names, m, costs, impact_k, impact_exponent
                    )
                    bought = {c: np.where(buy, v, 0.0) for c, v in f.items()}
                    spend = (
                        bought["amount"]
                        + bought["commission"]
                        + bought["tax"]
                        + bought["fee"]
                    ).sum(axis=1)
                    # 同比例缩小后仍超过现金：最低佣金这类固定费用不随股数缩小，现金会出现
                    # 小额透支（panelbt 会少买一点）。只计数；透支到权益 ≤ 0 才算打穿。
                    overdrawn += spend > np.maximum(cash, 0.0) * (1 + 1e-12) + 1e-9
                cash[:] -= spend
                q[:, names] += want
            traded_names = names[(sell | buy).any(axis=0)]
            if len(traded_names):
                held = np.union1d(held, traded_names)
            if carry:
                pending = np.zeros((k_n, n), bool)
                pending[:, names] = stuck
            if pending.any():
                nxt = day + 1
                while upcoming_index < len(days) and days[upcoming_index] <= day:
                    upcoming_index += 1
                upcoming = days[upcoming_index] if upcoming_index < len(days) else t_len
                if nxt < upcoming:
                    queue.insert(0, nxt)
            traded_rows = sell.any(axis=1) | buy.any(axis=1)
            if traded_rows.any():
                parts = [
                    (x, mk) for x, mk in ((sold, sell), (bought, buy)) if x is not None
                ]
                item = np.zeros((k_n, len(COST_ITEMS)))
                amount = np.zeros(k_n)
                for x, _ in parts:
                    item += np.column_stack([x[c].sum(axis=1) for c in COST_ITEMS])
                    amount += x["amount"].sum(axis=1)
                rec_day.append(day)
                rec_rows.append(traded_rows)
                rec_eq.append(previous)
                rec_open.append(value.copy())
                rec_cost.append(item)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rec_turn.append(np.where(value > 0, amount / 2 / value, 0.0))
                for x, mk in parts:
                    rows_i, cols_i = np.nonzero(mk)
                    fill_size.append(rows_i)
                    fill_part.append(x["participation"][rows_i, cols_i])
                    fill_notional.append(x["notional"][rows_i, cols_i])
        settle(t_len - 1)
        final = valued(t_len - 1, m.close_cf[-1, held])
    first = days[0] if days else 0
    years = (t_len - first) / 252.0
    if rec_day:
        day_arr = np.asarray(rec_day, dtype=np.int64)
        mask = np.array(rec_rows)  # 成交日数 × 规模
        eq, opn = np.array(rec_eq), np.array(rec_open)
        cost, turn = np.array(rec_cost), np.array(rec_turn)
    if fill_size:
        who = np.concatenate(fill_size)
        order = np.argsort(who, kind="stable")  # 稳定排序保留日内与日间的先后
        part_all = np.concatenate(fill_part)[order]
        notional_all = np.concatenate(fill_notional)[order]
        bounds = np.searchsorted(who[order], np.arange(k_n + 1))
    result = []
    for s in range(k_n):
        if rec_day:
            rows_s = mask[:, s]
            sim_days, sim_eq, sim_open = day_arr[rows_s], eq[rows_s, s], opn[rows_s, s]
            sim_cost, sim_turn = cost[rows_s, s], turn[rows_s, s]
        else:
            sim_days = np.empty(0, dtype=np.int64)
            sim_eq = sim_open = sim_turn = np.empty(0)
            sim_cost = np.empty((0, len(COST_ITEMS)))
        if fill_size:
            lo, hi = bounds[s], bounds[s + 1]
            sim_part, sim_notional = part_all[lo:hi], notional_all[lo:hi]
        else:
            sim_part = sim_notional = np.empty(0)
        result.append(
            Simulation(
                float(sizes[s]),
                sim_days,
                sim_eq,
                sim_open,
                sim_cost.reshape(-1, len(COST_ITEMS)),
                sim_turn,
                sim_part,
                sim_notional,
                float(final[s]),
                years,
                int(blocked[s]),
                int(overdrawn[s]),
            )
        )
    return result


@dataclass
class Capacity:
    """容量分析结果。`table` 每行一个规模；`capacity` 是两种定义下的容量（元）。"""

    table: Table
    capacity: dict
    liquidation: Table
    rerun: Table | None = None
    simulations: list = field(default_factory=list, repr=False)
    assumptions: dict = field(default_factory=dict)

    def to_dict(self):
        out = {
            "table": self.table.to_dict(),
            "capacity": plain(self.capacity),
            "liquidation": self.liquidation.to_dict(),
            "assumptions": plain(self.assumptions),
        }
        if self.rerun is not None:
            out["rerun"] = self.rerun.to_dict()
        return out


def _summary(sim, cap, breach_share_unused=None):
    eq = sim.equity
    rates = sim.costs / eq[:, None] if len(eq) else np.zeros((0, len(COST_ITEMS)))
    annual = rates.sum(axis=0) / sim.years if sim.years > 0 else rates.sum(axis=0)
    part, notional = sim.participation, sim.notional
    if len(part):
        pct = np.percentile(part, [50, 90, 99])
        over = notional[part > cap].sum() / notional.sum() if notional.sum() > 0 else 0
        top = float(part.max())
    else:
        pct, over, top = np.zeros(3), 0.0, 0.0
    return annual, pct, top, float(over), float(sim.turnover.sum() / sim.years)


def capacity(
    targets,
    panel,
    aum=DEFAULT_AUM,
    *,
    benchmark=None,
    gross_excess=None,
    costs=None,
    impact_coefficient=1.0,
    impact_k=None,
    impact_exponent=0.5,
    vol_window=20,
    exec_price="open",
    cost_fraction=0.5,
    participation_cap=0.1,
    breach_share=0.05,
    adv_window=20,
    rerun=False,
    market=None,
    config=None,
    threads=0,
    search=(1e4, 1e13),
    rtol=1e-6,
    points=15,
):
    """一行估计容量与成本。

    ```python
    cap = an.capacity(targets, panel, aum=[1e7, 1e8, 1e9, 1e10], benchmark="000300.SH")
    cap.table.to_dict()          # 每个规模的占比分布、年化成本、扣成本后的超额
    cap.capacity                 # 两种定义下的容量（元）及其状态
    ```
    容量字段（`cap.capacity`）：
    - `by_cost`：年化成本在**上升段**第一次达到 `cost_fraction × 毛超额` 的规模。成本曲线
      先降后升：小规模时最低佣金 5 元这类固定费用占比高，规模变大后冲击成本占主导。先在
      `search` 区间的对数网格上找成本最低点（`min_cost_aum`、`min_cost`），再从它往大规模
      方向找交点。`min_viable_aum` 是最低点往小规模方向的交点：比它更小的资金被固定费用
      吃掉超过门槛。
    - 每个值都带 `*_status`：`"found"`（求到交点）、`"no_gross_excess"`（毛超额 ≤ 0，值为
      NaN）、`"never_viable"`（最低成本也超过门槛，NaN）、`"beyond_search_range"`（到
      搜索上限也没达到，inf）、`"below_search_floor"`（搜索下限处已经越过，NaN；对
      `min_viable_aum` 表示固定费用在下限处也不构成约束）。不会用搜索边界冒充结果。
    - 资金小到最低佣金把账户打穿（权益 ≤ 0）的规模，成本记为无穷大（不可行），不参与找最低点。
    - `by_participation`：成交量占比超过 `participation_cap` 的成交额占比**超过**
      `breach_share` 的最小规模，状态同上。
    targets：`panelbt.Targets`（行号是成交日）、`ashare.portfolio.SparseTargets` 或 T×N 稠密权重。
    panel：与回测同日期同证券的面板，需要 open（或 exec_price 指定的字段）、close、
    adj_factor、volume、amount。
    gross_excess：年化毛超额收益；默认用目标权重的不扣成本收益减基准收益（无基准时为绝对收益）。
    costs：`panelbt.Costs`；默认主干的 A 股费用表（`ASHARE.costs`：佣金万 2.5 最低 5 元、
    滑点万 5、历史印花税与过户费）。其中的 impact 字段会被 impact_k / impact_coefficient 覆盖。
    rerun：对每个规模真的跑 `panelbt.run_batch`（需要 market，或由面板按 A 股规则构建）。
    """
    dates = np.asarray(panel.dates).astype(str)
    shape = (len(dates), len(panel.symbols))
    arrays = (
        MarketArrays.from_market(market)
        if market is not None
        else MarketArrays.from_panel(panel, exec_price)
    )
    returns = total_returns(panel)
    costs = ashare_costs(dates) if costs is None else costs
    k = (
        impact_coefficients(panel, impact_coefficient, vol_window, returns)
        if impact_k is None
        else np.ascontiguousarray(np.broadcast_to(np.asarray(impact_k, float), shape))
    )
    days, rows = _dense_targets(targets, shape)
    if not len(days):
        raise ValueError("targets 没有调仓日")

    def run(size, c=costs, kk=k):
        return simulate(days, rows, arrays, c, kk, impact_exponent, size)

    # 毛收益：同一套规则、全部成本为零的路径（与规模无关）。
    free = replace(
        costs,
        commission_rate=0.0,
        commission_min=0.0,
        commission_max_rate=None,
        tax_buy=0.0,
        tax_sell=0.0,
        fee_rate_buy=0.0,
        fee_rate_sell=0.0,
        fee_per_share_buy=0.0,
        fee_per_share_sell=0.0,
        commission_per_share=0.0,
        slippage=0.0,
    )
    base = run(1e8, free, np.zeros(shape))
    years = base.years
    gross_final = base.final_equity / 1e8
    gross_annual = gross_final ** (1 / years) - 1 if years > 0 else np.nan
    bench_annual = 0.0
    if benchmark is not None:
        hb = benchmark_held(panel, benchmark, returns)
        rb = (hb * np.nan_to_num(returns)).sum(axis=1)[days[0] + 1 :]
        bench_annual = float(np.prod(1 + rb) ** (1 / years) - 1) if years > 0 else 0.0
    excess = (
        gross_annual - bench_annual if gross_excess is None else float(gross_excess)
    )

    cache = {}
    batches = []  # 每批实际模拟的规模个数

    def evaluate(sizes):
        """一组规模的模拟（一次 `simulate_many` 批量算完没算过的规模）。"""
        sizes = [float(x) for x in sizes]
        need = [x for x in dict.fromkeys(sizes) if x not in cache]
        if need:
            batches.append(len(need))
            got = simulate_many(days, rows, arrays, costs, k, impact_exponent, need)
            cache.update(zip(need, got, strict=True))
        return [cache[x] for x in sizes]

    def cost_of(sim):
        # 资金太小时最低佣金能把账户打穿（权益 ≤ 0），成本率失去意义：记为不可行。
        if _wiped(sim):
            return float("inf")
        return float(_summary(sim, participation_cap)[0].sum())

    def breach_of(sim):
        return _summary(sim, participation_cap)[3]

    aum = np.asarray(sorted(float(a) for a in np.atleast_1d(aum)))
    grid = _grid(search)
    evaluate([*aum, *grid])  # 表里的规模和搜索网格一次算完
    sims = evaluate(aum)
    rows_out = []
    for size, sim in zip(aum, sims, strict=True):
        annual, pct, top, over, turnover = _summary(sim, participation_cap)
        wiped = _wiped(sim)
        if wiped:
            # 账户被打穿：成本率和扣成本后超额没有意义，记 NaN，并在 wiped_out 列标出。
            annual = np.full(len(COST_ITEMS), np.nan)
        net_final = sim.final_equity / size
        net_annual = net_final ** (1 / years) - 1 if net_final > 0 else np.nan
        rows_out.append(
            [
                *pct,
                top,
                over,
                *annual,
                annual.sum(),
                2 * turnover,
                excess,
                excess - annual.sum(),
                net_annual - bench_annual,
                float(wiped),
            ]
        )
    columns = (
        ("participation_p50", "participation_p90", "participation_p99")
        + ("participation_max", "over_cap_share")
        + tuple(f"cost_{c}" for c in COST_ITEMS)
        + ("cost_total", "annual_turnover", "gross_excess", "net_excess")
        + ("net_excess_path", "wiped_out")
    )
    table = Table(aum, columns, np.array(rows_out), "capacity", "aum")
    threshold = cost_fraction * excess if excess > 0 else np.nan
    by_cost, by_part = _capacities(
        evaluate, grid, cost_of, threshold, breach_of, breach_share, rtol, points
    )
    liquidation = _liquidation(days, rows, panel, aum, participation_cap, adv_window)
    rerun_table = None
    if rerun:
        rerun_table = _rerun(
            targets,
            days,
            rows,
            panel,
            aum,
            costs,
            k,
            impact_exponent,
            sims,
            market,
            config,
            threads,
            bench_annual,
            years,
            exec_price,
            participation_cap,
        )
    return Capacity(
        table,
        {
            "cost_fraction": cost_fraction,
            **by_cost,
            "participation_cap": participation_cap,
            "breach_share": breach_share,
            "by_participation": by_part["value"],
            "by_participation_status": by_part["status"],
            "search_range": tuple(float(x) for x in search),
            "gross_annual": gross_annual,
            "benchmark_annual": bench_annual,
            "gross_excess": excess,
        },
        liquidation,
        rerun_table,
        sims,
        {
            "impact": "volatility_scaled" if impact_k is None else "given_k",
            "impact_coefficient": impact_coefficient if impact_k is None else None,
            "impact_exponent": impact_exponent,
            "vol_window": vol_window,
            "exec_price": exec_price if isinstance(exec_price, str) else "array",
            "years": years,
            "periods_per_year": 252,
            "turnover": "two_sided",
            "blocked_attempts": int(sims[0].blocked) if sims else 0,
            "rtol": rtol,
            "points": points,
            # 解析模拟的批数与每批规模数（表与网格一批，之后每轮多分法一批）。
            "simulation_batches": list(batches),
        },
    )


def _grid(search, per_decade=5):
    lo, hi = (float(x) for x in search)
    if not (0 < lo < hi):
        raise ValueError("search: 需要 0 < 下限 < 上限")
    count = max(2, int(round(np.log10(hi / lo) * per_decade)) + 1)
    return np.logspace(np.log10(lo), np.log10(hi), count)


def _wiped(sim):
    """最低佣金等固定费用把账户打穿：某个成交日前的权益或期末权益 ≤ 0。"""
    return not sim.final_equity > 0 or (len(sim.equity) > 0 and sim.equity.min() <= 0)


def _multisect(evaluate, brackets, rtol, points):
    """成批的多分法：每个区间 (ok, over) 满足 fn(ok) < 0 ≤ fn(over)。

    每一轮在所有未收敛区间里各取 points 个对数等距的内点，一次 `evaluate` 批量算完，
    再把每个区间缩到「第一个越过的点」与它前一个点之间；|ln(over / ok)| ≤ ln(1 + rtol)
    时收敛，返回 over（已知越过门槛的最小规模一侧）。ok 可以大于 over（往小规模方向找）。
    """
    state = [dict(b) for b in brackets]
    tol = np.log1p(rtol)
    while True:
        live = [b for b in state if abs(np.log(b["over"] / b["ok"])) > tol]
        if not live:
            break
        cands = [
            np.exp(np.linspace(np.log(b["ok"]), np.log(b["over"]), points + 2)[1:-1])
            for b in live
        ]
        sims = evaluate(np.concatenate(cands))
        at = 0
        for b, xs in zip(live, cands, strict=True):
            values = [b["fn"](sim) for sim in sims[at : at + len(xs)]]
            at += len(xs)
            hit = next((j for j, v in enumerate(values) if v >= 0), None)
            if hit is None:
                b["ok"] = float(xs[-1])
            else:
                b["over"] = float(xs[hit])
                if hit > 0:
                    b["ok"] = float(xs[hit - 1])
    return [b["over"] for b in state]


def _capacities(
    evaluate, grid, cost_of, threshold, breach_of, breach_share, rtol, points
):
    """两种容量：网格上定位交点所在的格子，再把三个交点的多分法合在同几批里算。"""
    by_cost = {
        "by_cost": np.nan,
        "by_cost_status": "no_gross_excess",
        "min_viable_aum": np.nan,
        "min_viable_status": "no_gross_excess",
        "min_cost_aum": np.nan,
        "min_cost": np.nan,
    }
    by_part = {"value": np.nan, "status": "below_search_floor"}
    sims = evaluate(grid)
    brackets = []
    if np.isfinite(threshold) and threshold > 0:
        costs = np.array([cost_of(sim) for sim in sims])
        m = int(np.argmin(costs))
        by_cost["min_cost_aum"], by_cost["min_cost"] = float(grid[m]), float(costs[m])
        gap = costs - threshold

        def fn(sim):
            return cost_of(sim) - threshold

        if gap[m] >= 0:
            by_cost["by_cost_status"] = by_cost["min_viable_status"] = "never_viable"
        else:
            up = next((i for i in range(m + 1, len(grid)) if gap[i] >= 0), None)
            if up is None:
                by_cost["by_cost"] = float("inf")
                by_cost["by_cost_status"] = "beyond_search_range"
            else:
                brackets.append(
                    {"key": "by_cost", "fn": fn, "ok": grid[up - 1], "over": grid[up]}
                )
            down = next((i for i in range(m - 1, -1, -1) if gap[i] >= 0), None)
            if down is None:
                by_cost["min_viable_status"] = "below_search_floor"
            else:
                brackets.append(
                    {
                        "key": "min_viable_aum",
                        "fn": fn,
                        "ok": grid[down + 1],
                        "over": grid[down],
                    }
                )

    def crossed(sim):
        # 严格大于记为越过（breach_share=0 表示「出现任何超限成交」）。
        return 1.0 if breach_of(sim) > breach_share else -1.0

    flags = np.array([crossed(sim) for sim in sims])
    if flags[0] < 0:
        idx = next((i for i in range(1, len(grid)) if flags[i] >= 0), None)
        if idx is None:
            by_part = {"value": float("inf"), "status": "beyond_search_range"}
        else:
            brackets.append(
                {"key": "part", "fn": crossed, "ok": grid[idx - 1], "over": grid[idx]}
            )
    for b, value in zip(
        brackets, _multisect(evaluate, brackets, rtol, points), strict=True
    ):
        if b["key"] == "part":
            by_part = {"value": value, "status": "found"}
        else:
            by_cost[b["key"]] = value
            by_cost[b["key"].replace("_aum", "") + "_status"] = "found"
    return by_cost, by_part


def _liquidation(days, rows, panel, aum, cap, window):
    """变现天数：每次调仓后的持仓，按 cap × 此前 window 日平均成交额每天卖出需要几天。

    此前 window 日没有成交额的持仓（长期停牌）卖不出去，单列为 `share_no_volume`，
    其余各列只在有成交额的持仓上算。"""
    amount = panel_field(panel, "amount")
    x = np.where(np.isfinite(amount), amount, 0.0)
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    out = []
    for size in aum:
        weighted, worst, within1, within5, stuck = [], [], [], [], []
        for j, day in enumerate(days):
            lo = max(0, day - window)
            adv = (c[day] - c[lo]) / max(day - lo, 1)
            w = rows[j]
            held = w > 0
            if not held.any():
                continue
            share = w[held] / w[held].sum()
            liquid = adv[held] > 0
            need = np.full(held.sum(), np.inf)
            need[liquid] = w[held][liquid] * size / (cap * adv[held][liquid])
            stuck.append(float(share[~liquid].sum()))
            if liquid.any():
                part = share[liquid] / share[liquid].sum()
                weighted.append(float((part * need[liquid]).sum()))
                worst.append(float(need[liquid].max()))
            within1.append(float(share[need <= 1].sum()))
            within5.append(float(share[need <= 5].sum()))
        out.append(
            [
                np.mean(weighted),
                np.median(worst),
                np.max(worst),
                np.mean(within1),
                np.mean(within5),
                np.mean(stuck),
            ]
        )
    return Table(
        aum,
        (
            "days_weighted_mean",
            "days_worst_median",
            "days_worst_max",
            "share_1day",
            "share_5day",
            "share_no_volume",
        ),
        np.array(out),
        "liquidation",
        "aum",
    )


def _rates(res, size, years):
    """panelbt 单策略（批量 K=1）结果 → (年化冲击率, 年化总成本率, 冲击元, 总成本元, 日收益,
    被拦下的买卖单数, 换手合计)。"""

    def one(name):
        v = np.asarray(getattr(res, name))
        return v[0] if v.ndim == 2 else v

    equity = one("equity")
    denominator = np.r_[size, equity[:-1]]
    impact = one("impact_cost")
    total = sum(
        one(a)
        for a in ("commission", "tax", "other_fee", "slippage_cost", "impact_cost")
    )
    return (
        float((impact / denominator).sum() / years),
        float((total / denominator).sum() / years),
        float(impact.sum()),
        float(total.sum()),
        one("returns"),
        float(one("n_blocked_buy").sum() + one("n_blocked_sell").sum()),
        float(one("turnover").sum()),
    )


def _rerun(
    targets,
    days,
    rows,
    panel,
    aum,
    costs,
    k,
    e,
    sims,
    market,
    config,
    threads,
    bench_annual,
    years,
    exec_price,
    cap,
):
    """每个规模跑两次 panelbt：与解析模拟同规则（不取整、无成交量上限），以及真实 A 股规则。

    `real_fill_ratio`：真实规则下的换手合计 ÷ 宽松规则下的换手合计，即目标交易里真正成交了
    多少（两次运行的权益路径不同，是近似；被成交量上限、整手拦下的部分会在之后的日子重试，
    所以它衡量的是「全期最终成交了多少」）。`real_blocked_orders`：真实规则下被拦下的买单
    与卖单数（`n_blocked_buy + n_blocked_sell`，每只证券每天算一次，挂单重试会重复计数），
    不是天数。"""
    import panelbt

    if market is None:
        market = market_from_panel(panel, exec_price)
    real = config if config is not None else panel_config(panel, max_participation=cap)
    relaxed = panelbt.Config(min_qty=0.0, qty_step=0.0, max_participation=0.0)
    if isinstance(targets, panelbt.Targets):
        plan = targets
    else:
        nz = rows != 0
        plan = panelbt.Targets(
            days, np.r_[0, np.cumsum(nz.sum(axis=1))], np.nonzero(nz)[1], rows[nz]
        )
    run_costs = replace(
        costs, impact_k=np.ascontiguousarray(k, dtype=float), impact_exponent=e
    )
    span = slice(int(days[0]), None)
    out = []
    for size, sim in zip(aum, sims, strict=True):
        a_imp = float(sim.costs[:, 4].sum())
        a_tot = float(sim.costs.sum())
        a_rate_imp = float((sim.costs[:, 4] / sim.equity).sum() / years)
        a_rate_tot = float((sim.costs.sum(axis=1) / sim.equity).sum() / years)
        row = [a_rate_imp, a_rate_tot]
        nets, turnovers = [], []
        for cfg in (relaxed, real):
            res = panelbt.run_batch(
                market,
                [plan],
                run_costs,
                replace(cfg, initial_cash=float(size)),
                threads=threads,
            )
            r_imp, r_tot, y_imp, y_tot, ret, blocked, turn = _rates(res, size, years)
            turnovers.append(turn)
            net = float(np.prod(1 + ret[span]) ** (1 / years) - 1)
            row += [
                r_imp,
                r_tot,
                y_imp / a_imp if a_imp else np.nan,
                y_tot / a_tot if a_tot else np.nan,
            ]
            nets.append(net - bench_annual)
            last_blocked = blocked
        net_analytic = (sim.final_equity / size) ** (1 / years) - 1 - bench_annual
        fill = turnovers[1] / turnovers[0] if turnovers[0] > 0 else np.nan
        out.append(row + [net_analytic, *nets, fill, last_blocked])
    return Table(
        aum,
        (
            "analytic_impact",
            "analytic_cost_total",
            "relaxed_impact",
            "relaxed_cost_total",
            "relaxed_impact_ratio",
            "relaxed_cost_ratio",
            "real_impact",
            "real_cost_total",
            "real_impact_ratio",
            "real_cost_ratio",
            "analytic_net_excess",
            "relaxed_net_excess",
            "real_net_excess",
            "real_fill_ratio",
            "real_blocked_orders",
        ),
        np.array(out),
        "capacity_rerun",
        "aum",
    )
