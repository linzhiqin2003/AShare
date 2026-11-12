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
1. 成本吃掉毛超额收益的 `cost_fraction`（默认一半）时的规模；
2. 成交量占比超过 `participation_cap`（默认 10%）的成交额占全部成交额的比例达到
   `breach_share`（默认 5%）时的规模。按成交额加权而不是按笔数，因为笔数会被大量小额
   调整主导。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .adapters import ashare_config, market_from_panel
from .inputs import benchmark_held, carry_forward, total_returns
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


def default_costs(dates):
    """A 股默认成本：佣金万 2.5（最低 5 元）、过户费双边十万分之一、滑点万 5、
    印花税按日期（2008-04-24 前 0.3%，之后 0.1%，2008-09-19 起取 panelbt 的单边税率表）。"""
    import panelbt
    from panelbt.markets.ashare import stamp_tax_sell

    days = np.asarray(
        [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in np.asarray(dates).astype(str)],
        dtype="datetime64[D]",
    )
    tax = np.where(days < np.datetime64("2008-04-24"), 0.003, 0.001)
    late = days >= np.datetime64("2008-09-19")
    if late.any():
        tax[late] = stamp_tax_sell(days[late])
    return panelbt.Costs(tax_sell=tax)


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
        self.close_cf = carry_forward(self.close)
        split = np.where(np.isfinite(self.split), self.split, 1.0)
        # S_t = Π_{u≤t} split_u；第 s 日派现按送转前股数：q_last·S_{s-1}/S_last·div_s。
        self.split_cum = np.cumprod(split, axis=0)
        before = np.vstack([np.ones((1, n)), self.split_cum[:-1]])
        self.div_cum = np.cumsum(before * np.nan_to_num(self.cash_div), axis=0)
        d = np.asarray(self.delist, bool)
        first = np.where(d.any(axis=0), d.argmax(axis=0), t)
        self.first_delist = first

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
    commission = np.maximum(costs.commission_rate * amount, costs.commission_min or 0.0)
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
    return {
        "amount": amount,
        "commission": commission,
        "tax": tax * amount,
        "fee": fee * amount,
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
    """解析模拟的数值核心：按股数记账的现金账户，规则与 panelbt 相同，只是
    不做整手取整、不设成交量占比上限。

    - 目标股数 = 目标权重 × 开盘估值的权益 × (1 − cash_buffer) / 成交价；先卖后买。
    - 买单总花费超过现金时所有买单同比例缩小（panelbt 的比例缩放）。
    - 不能买卖（掩码、无成交价、无成交量）的证券挂起，之后每天用原目标与当日权益重试，
      直到下一次调仓（panelbt 的 carry_blocked）。
    - 送转、派现、退市按面板事件处理；只在有成交的日子结算，日子之间用累计量一次补上。
    """
    m = market if isinstance(market, MarketArrays) else MarketArrays.from_market(market)
    t_len, n = m.exec_price.shape
    q = np.zeros(n)
    cash = float(aum)
    last = 0  # q 对应的送转基准日（q 是第 last 日送转之后的股数）
    plan = None
    pending = np.zeros(n, bool)
    rebalance = {int(d): j for j, d in enumerate(rebalance_days)}
    out_day, out_eq, out_open, out_cost, out_turn = [], [], [], [], []
    parts, notionals = [], []
    blocked = 0

    def settle(day):
        nonlocal q, cash, last
        held = np.flatnonzero(q)
        if len(held) and day > last:
            base = q[held] / m.split_cum[last, held]
            gone = (m.first_delist[held] > last) & (m.first_delist[held] <= day)
            stop = np.where(gone, m.first_delist[held], day)
            div = m.div_cum[stop, held] - m.div_cum[last, held]
            # 退市日的派现在变现之前：取到退市日（含）为止。
            cash += float((base * div).sum())
            if gone.any():
                g = held[gone]
                d = m.first_delist[g]
                shares = base[gone] * m.split_cum[d, g]
                # panelbt：按退市日之前最近的有效收盘价全额变现。
                cash += float((shares * m.close_cf[np.maximum(d - 1, 0), g]).sum())
                q[g] = 0.0
            keep = held[~gone]
            q[keep] = base[~gone] * m.split_cum[day, keep]
        last = day

    days = sorted(set(int(d) for d in rebalance_days))
    queue = list(days)
    visited = set()
    while queue:
        day = queue.pop(0)
        if day in visited or day >= t_len:
            continue
        visited.add(day)
        # 成本率的分母取前一日收盘的权益（与 panelbt 收益和报告模块的口径相同）。
        if day > 0:
            settle(day - 1)
        previous = cash + float(np.nansum(q * m.close_cf[day - 1])) if day > 0 else cash
        settle(day)
        if day in rebalance:
            plan = target_rows[rebalance[day]]
            # 新调仓行清掉挂单，所有列都按新目标处理（不在计划里的目标为 0）。
            names = np.flatnonzero((plan != 0) | (q != 0))
        else:
            names = np.flatnonzero(pending)
        if plan is None or not len(names):
            continue
        p = m.exec_price[day]
        prev_close = m.close_cf[day - 1] if day > 0 else np.full(n, np.nan)
        mark = np.where(np.isfinite(p), p, prev_close)
        value = cash + float(np.nansum(q * mark))
        alive = m.first_delist[names] > day
        names = names[alive]
        good_mark = np.isfinite(mark[names]) & (mark[names] > 0)
        target = np.where(
            good_mark,
            plan[names]
            * max(value, 0.0)
            * (1 - cash_buffer)
            / np.where(good_mark, mark[names], 1.0),
            0.0,
        )
        delta = target - q[names]
        quote_ok = (
            np.isfinite(p[names])
            & (p[names] > 0)
            & np.isfinite(m.volume[day, names])
            & (m.volume[day, names] > 0)
        )
        sell = (delta < 0) & m.can_sell[day, names] & quote_ok
        buy = (delta > 0) & m.can_buy[day, names] & quote_ok
        stuck = (delta != 0) & ~sell & ~buy
        blocked += int(stuck.sum())
        new_pending = np.zeros(n, bool)
        new_pending[names[stuck]] = True
        rows = []
        if sell.any():
            i = names[sell]
            f = _fills(-delta[sell], -1.0, day, i, m, costs, impact_k, impact_exponent)
            cash += float((f["amount"] - f["commission"] - f["tax"] - f["fee"]).sum())
            q[i] += delta[sell]
            rows.append(f)
        if buy.any():
            i = names[buy]
            want = delta[buy]
            f = _fills(want, 1.0, day, i, m, costs, impact_k, impact_exponent)
            spend = float((f["amount"] + f["commission"] + f["tax"] + f["fee"]).sum())
            if spend > cash:
                want = want * (max(cash, 0.0) / spend)
                f = _fills(want, 1.0, day, i, m, costs, impact_k, impact_exponent)
                spend = float(
                    (f["amount"] + f["commission"] + f["tax"] + f["fee"]).sum()
                )
            cash -= spend
            q[i] += want
            rows.append(f)
        pending = new_pending if carry else np.zeros(n, bool)
        if pending.any():
            nxt = day + 1
            upcoming = min((d for d in days if d > day), default=t_len)
            if nxt < upcoming:
                queue.insert(0, nxt)
        if rows:
            item = np.array([sum(float(r[c].sum()) for r in rows) for c in COST_ITEMS])
            traded = sum(float(r["amount"].sum()) for r in rows)
            out_day.append(day)
            out_eq.append(previous)
            out_open.append(value)
            out_cost.append(item)
            out_turn.append(
                traded / 2 / value if value > 0 else 0.0
            )  # panelbt：交易前权益
            parts.extend(r["participation"] for r in rows)
            notionals.extend(r["notional"] for r in rows)
    settle(t_len - 1)
    final = cash + float(np.nansum(q * m.close_cf[-1]))
    first = days[0] if days else 0
    return Simulation(
        float(aum),
        np.asarray(out_day, dtype=np.int64),
        np.asarray(out_eq),
        np.asarray(out_open),
        np.asarray(out_cost).reshape(-1, len(COST_ITEMS)),
        np.asarray(out_turn),
        np.concatenate(parts) if parts else np.empty(0),
        np.concatenate(notionals) if notionals else np.empty(0),
        float(final),
        (t_len - first) / 252.0,
        blocked,
    )


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
):
    """一行估计容量与成本。

    ```python
    cap = an.capacity(targets, panel, aum=[1e7, 1e8, 1e9, 1e10], benchmark="000300.SH")
    cap.table.to_dict()          # 每个规模的占比分布、年化成本、扣成本后的超额
    cap.capacity                 # 两种定义下的容量（元）
    ```
    targets：`panelbt.Targets`（行号是成交日）、`ashare.portfolio.SparseTargets` 或 T×N 稠密权重。
    panel：与回测同日期同证券的面板，需要 open（或 exec_price 指定的字段）、close、
    adj_factor、volume、amount。
    gross_excess：年化毛超额收益；默认用目标权重的不扣成本收益减基准收益（无基准时为绝对收益）。
    costs：`panelbt.Costs`；默认 `default_costs(panel.dates)`。其中的 impact 字段会被
    impact_k / impact_coefficient 覆盖。
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
    costs = default_costs(dates) if costs is None else costs
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

    def cost_total(size):
        annual, *_ = _summary(run(size), participation_cap)
        return float(annual.sum())

    def breach(size):
        return _summary(run(size), participation_cap)[3]

    aum = np.asarray(sorted(float(a) for a in np.atleast_1d(aum)))
    rows_out, sims = [], []
    for size in aum:
        sim = run(size)
        sims.append(sim)
        annual, pct, top, over, turnover = _summary(sim, participation_cap)
        net_final = sim.final_equity / size
        net_annual = net_final ** (1 / years) - 1 if net_final > 0 else -1.0
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
            ]
        )
    columns = (
        ("participation_p50", "participation_p90", "participation_p99")
        + ("participation_max", "over_cap_share")
        + tuple(f"cost_{c}" for c in COST_ITEMS)
        + ("cost_total", "annual_turnover", "gross_excess", "net_excess")
        + ("net_excess_path",)
    )
    table = Table(aum, columns, np.array(rows_out), "capacity", "aum")
    capacity_cost = _solve(lambda s: cost_total(s) - cost_fraction * excess, excess > 0)
    capacity_volume = _solve(lambda s: breach(s) - breach_share, True)
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
            "by_cost": capacity_cost,
            "participation_cap": participation_cap,
            "breach_share": breach_share,
            "by_participation": capacity_volume,
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
        },
    )


def _solve(excess_at, feasible, lo=1e4, hi=1e13, steps=40):
    """在对数刻度上找 excess_at(规模) 第一次 ≥ 0 的规模；从未达到返回 inf，起点就达到返回 lo。"""
    if not feasible:
        return 0.0
    grid = np.logspace(np.log10(lo), np.log10(hi), 19)
    values = [excess_at(g) for g in grid]
    if values[0] >= 0:
        return float(lo)
    idx = next((i for i, v in enumerate(values) if v >= 0), None)
    if idx is None:
        return float("inf")
    a, b = np.log(grid[idx - 1]), np.log(grid[idx])
    for _ in range(steps):
        m = 0.5 * (a + b)
        if excess_at(np.exp(m)) >= 0:
            b = m
        else:
            a = m
    return float(np.exp(b))


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
    """panelbt 单策略（批量 K=1）结果 → (年化冲击率, 年化总成本率, 冲击元, 总成本元, 日收益)。"""

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
    """每个规模跑两次 panelbt：与解析模拟同规则（不取整、无成交量上限），以及真实 A 股规则。"""
    import panelbt

    if market is None:
        market = market_from_panel(panel, exec_price)
    real = (
        config
        if config is not None
        else ashare_config(panel.symbols, max_participation=cap)
    )
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
        nets = []
        for cfg in (relaxed, real):
            res = panelbt.run_batch(
                market,
                [plan],
                run_costs,
                replace(cfg, initial_cash=float(size)),
                threads=threads,
            )
            r_imp, r_tot, y_imp, y_tot, ret, blocked = _rates(res, size, years)
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
        out.append(row + [net_analytic, *nets, last_blocked])
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
            "real_blocked_days",
        ),
        np.array(out),
        "capacity_rerun",
        "aum",
    )
