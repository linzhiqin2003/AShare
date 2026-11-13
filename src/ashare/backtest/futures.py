"""纯期货组合回测：数据层面板 → ``panelbt.FuturesBook``，品种层面的目标 → 合约层面的目标。

最短例子（两个品种按 60 日动量做多做空，名义市值各占权益的一半，主力合约、真实换月）::

    import numpy as np, ashare.data as ad
    from ashare.backtest import execution_targets
    from ashare.backtest.futures import futures_portfolio

    fut = ad.open("/path/to/root").futures()
    pf = futures_portfolio(fut, ["RB", "CU"], start="2020-01-02", end="2024-12-31")
    signal = np.sign(pf.signal_prices / np.roll(pf.signal_prices, 60, axis=0) - 1)
    signal[:60] = np.nan                          # 前 60 行没有信号
    w = np.full(signal.shape, np.nan); w[::21] = 0.5 * signal[::21]
    result = pf.run(execution_targets(w), kind="notional")   # 收盘出信号，下一行开盘成交

约定（详见 ``docs/FUTURES_BACKTEST.md``）：

- 持有的是具体合约。每个品种每行持有哪个合约由 ``rule``（默认 ``"open_interest"``，数据层的
  ``fut.held_columns``：第 t 行只用第 t−1 行的持仓量）决定，换月当行开盘平旧开新，各付一次手续费和滑点。
- 个人客户最后持仓日（``personal=True``，默认）：合约过了这一天还在持仓表里时提前换到下一个合约；
  回测引擎在这一天的下一行把仍有的持仓按当天结算价强平（计手续费和滑点）。
- 信号用 ``signal_prices``（同一换月规则下的比例后复权连续价格），盈亏只按具体合约的结算价算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import panelbt as bt
import polars as pl

# 期货公司账户条款（假设，见 docs/markets/futures_rules.md 第三部分）：保证金 = 交易所 + 5 个百分点；
# 手续费 = 交易所 × 3（期货公司公示标准约为交易所 3 倍，取悲观的一端）。
CN_RETAIL = bt.FuturesAccount(margin_add=0.05, fee_scale=3.0)

# 个人客户最后持仓日规则。值：("before_last_trade", k) = 最后交易日前第 k 个交易日收市后须为 0；
# "delivery_month" = 不得持仓进入交割月；None = 不限制。
PERSONAL_RULES = {
    ("INE", "SC"): ("before_last_trade", 8),  # 官方：INE 风险控制管理细则第六十六条
    ("INE", "LU"): ("before_last_trade", 5),  # 官方：同上
    ("INE", "NR"): ("before_last_trade", 5),  # 官方：同上
}
EXCHANGE_DEFAULT = {
    # 大商所 / 郑商所 / 广期所：个人客户不得进入交割月（旁证，未核实官方原文）。
    "DCE": "delivery_month",
    "CZCE": "delivery_month",
    "GFEX": "delivery_month",
    # 上期所：品种细则未读到；按「不得进入交割月」处理（推断，偏保守）。
    "SHFE": "delivery_month",
    # 能源中心其他品种（BC、EC）：规则未读到；BC 按交割月处理（推断），EC 为现金交割指数，不限制。
    "INE": "delivery_month",
    # 中金所：没有个人客户持仓期限。
    "CFFEX": None,
}
UNLIMITED_PRODUCTS = {("INE", "EC")}


def personal_rule(exchange, product):
    """某品种的个人客户持仓期限规则（见 ``PERSONAL_RULES``）。"""
    if (exchange, product) in UNLIMITED_PRODUCTS:
        return None
    return PERSONAL_RULES.get((exchange, product), EXCHANGE_DEFAULT.get(exchange))


def last_hold_dates(contracts, calendar, lead=0):
    """每个合约的个人客户「最后一个可以收盘持仓的交易日」（YYYYMMDD；不限制为 None）。

    ``contracts``：合约表（symbol、exchange、product、last_trade_date、delivery_month）；
    ``calendar``：升序交易日（YYYYMMDD 字符串数组），应覆盖到最后交易日（数据层的
    ``fut.calendar()`` 带未来一年多的交易所日历）。没覆盖到的日子用周一到周五补上并发出警告：
    节假日被当成交易日，往回数会落在**更晚**的日子，算出的期限偏晚、偏乐观。
    ``lead``：再提前的交易日数（期货公司常比交易所规定更早要求平仓；默认 0）。
    """
    if lead < 0:
        raise ValueError("lead 不能为负")
    calendar = np.asarray(calendar, dtype="U8")
    lasts = [d for d in contracts["last_trade_date"].to_list() if d]
    if lasts and calendar.size and max(lasts) > calendar[-1]:
        import warnings

        warnings.warn(
            f"交易日历只到 {calendar[-1]}，之后用周一到周五补齐：节假日被当成交易日，"
            "个人持仓期限会偏晚（偏乐观）；请传覆盖到最后交易日的日历",
            stacklevel=2,
        )

        def iso(d):
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"

        extra = np.arange(
            np.datetime64(iso(calendar[-1]), "D") + np.timedelta64(1, "D"),
            np.datetime64(iso(max(lasts)), "D") + np.timedelta64(40, "D"),
            dtype="datetime64[D]",
        )
        extra = extra[np.is_busday(extra)]
        calendar = np.concatenate(
            [calendar, np.char.replace(extra.astype("U10"), "-", "").astype("U8")]
        )
    out = {}
    for row in contracts.iter_rows(named=True):
        rule = personal_rule(row["exchange"], row["product"])
        last = row["last_trade_date"]
        if rule is None or not last:
            out[row["symbol"]] = None
            continue
        if rule == "delivery_month":
            # 交割月前一个月最后一个交易日收市前须平仓（与 SC 一类同一口径：那一天收市后为 0），
            # 最后一个可以持仓过夜的是它的前一个交易日。
            month = row.get("delivery_month") or last[:6]
            k = int(np.searchsorted(calendar, month + "01")) - 1 - 1
        else:
            # 第 k 个交易日收市后须为 0：最后一个可以持仓过夜的是第 k+1 个交易日。
            k = int(np.searchsorted(calendar, last, side="right")) - 1 - rule[1] - 1
        k -= lead
        out[row["symbol"]] = str(calendar[k]) if k >= 0 else "00000000"
    return out


def _exchange_calendar(fut, contracts):
    """这些合约所在交易所的交易日（数据层的交易所日历，含未来日期），合并成一份升序数组。"""
    exchanges = sorted(set(contracts["exchange"].to_list()))
    days = (
        fut.calendar()
        .filter(pl.col("exchange").is_in(exchanges) & pl.col("is_open"))
        .select("date")
        .unique()
        .sort("date")
        .collect()["date"]
    )
    return np.asarray(days.to_list(), dtype="U8")


def _rows(dates, day):
    """日期 → 「最后一个不晚于它的行」；早于样本为 -1，晚于末行时为 T（不限制）。

    正好是末行的日期返回 T − 1：最后交易日等于样本最后一天时，那一行照常了结。
    """
    if day is None or day > dates[-1]:
        return len(dates)
    return int(np.searchsorted(dates, day, side="right")) - 1


@dataclass
class FuturesPortfolio:
    """一组品种的合约层面回测输入。``held[t, p]`` 是第 t 行品种 p 持有的列（-1：没有）。"""

    book: bt.FuturesBook
    dates: np.ndarray
    symbols: np.ndarray
    products: list
    held: np.ndarray
    signal_prices: np.ndarray
    contracts: pl.DataFrame
    splits: list
    rolls: pl.DataFrame
    last_hold: dict = field(repr=False)

    def _weights(self, weights):
        t, p = self.held.shape
        if isinstance(weights, bt.Targets):
            if weights.kind != "weights":
                raise ValueError(
                    "传 kind='weights' 的 Targets，手数在这里用 kind='lots' 表示"
                )
            weights = weights.to_dense(t, p)
        w = np.asarray(weights, dtype=float)
        if w.shape != (t, p):
            raise ValueError(f"weights 应为 T×P = {(t, p)}")
        return w

    def simulate(self, weights, kind="notional"):
        """品种层面的 T×P 目标（稠密数组，或 ``execution_targets`` 返回的 ``Targets``）逐行推进回测。

        - 调用方给了目标的行（不全为 NaN）整体调仓：目标落在当行持有的合约上，其余 NaN 当 0，
          不在目标里的合约（含旧合约）平掉。``kind="lots"`` 是当行持有合约的整手数。
        - 品种在目标行还没有可交易的合约（样本第一行、刚上市、合约还没有成交价、换不到下一个合约）时，
          三种写法都挂起，等它第一次落到可交易合约上的那一行开仓：手数照给的开；名义和保证金按
          落地那一行的权益和价格定手数（开盘前已知，不是前视）。
        - 只有换月的行（换月、个人持仓期限、交易单位分界）只动正在换月的品种：平旧开新，
          **保持「手数 × 乘数」**，向零取整（零头计入诊断）；其他品种的持仓不动、不成交，
          引擎里顺延中的单子原样带上。被引擎在结束行了结的持仓（交易单位分界、换不过去的合约）
          同样按手数接到下一个合约。
        - 其余行不交易（引擎照常盯市、追保、顺延、到期了结）。

        返回 ``panelbt.Result``；``diagnostics["futures_portfolio"]`` 记换月取整的次数和名义零头、
        换月当行旧合约没平掉（持仓加倍一天）的次数、挂起目标落地的次数。逐行推进用
        ``FuturesBook.simulator``，实现与不变量见 ``futures_step``。
        """
        if kind not in bt.futures_book.KINDS:
            raise ValueError(f"kind 只能是 {bt.futures_book.KINDS}")
        from .futures_step import simulate

        return simulate(self, self._weights(weights), kind)

    def run(self, weights, kind="notional"):
        """同 ``simulate``。"""
        return self.simulate(weights, kind)

    def run_batch(self, weights_list, kind="notional", threads=0):
        """多组目标各自 ``simulate``；``threads`` > 1 时用线程并行（模拟器推进时释放 GIL）。"""
        weights_list = list(weights_list)
        if threads == 1 or len(weights_list) < 2:
            return [self.simulate(w, kind) for w in weights_list]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=threads or None) as pool:
            return list(pool.map(lambda w: self.simulate(w, kind), weights_list))


def _held(fut, product, rule, start, end, dates, last_hold_rows_of):
    """某品种每行持有的合约代码；过了个人持仓期限的合约换成下一个合约。"""
    panel, held = fut.held_columns(product, rule=rule, start=start, end=end)
    if not np.array_equal(panel.dates, dates):
        raise ValueError(f"{product}: 品种面板的日期与样本不一致")
    symbols = np.asarray(panel.symbols)
    info = (
        fut.contracts(products=[product]).select("symbol", "last_trade_date").collect()
    )
    last = dict(zip(info["symbol"], info["last_trade_date"], strict=True))
    order = sorted(range(len(symbols)), key=lambda j: last[symbols[j]] or "99999999")
    quote = np.asarray(panel["has_quote"])
    t = len(dates)
    out = held.copy()
    for day in range(t):
        h = out[day]
        if h < 0 or last_hold_rows_of(symbols[h]) >= day:
            continue
        # 当前合约在这一行已经不能持有：换到下一个仍可持有、前一行有行情的更晚合约。
        rank = order.index(h)
        nxt = -1
        for j in order[rank + 1 :]:
            if last_hold_rows_of(symbols[j]) >= day and day > 0 and quote[day - 1, j]:
                nxt = j
                break
        out[day] = nxt
    # 只向后换：一旦换到更晚的合约，不回到更早的合约。
    rank_of = {j: r for r, j in enumerate(order)}
    best = -1
    for day in range(t):
        h = out[day]
        if (
            h >= 0
            and best >= 0
            and rank_of[h] < rank_of[best]
            and (last_hold_rows_of(symbols[best]) >= day)
        ):
            out[day] = best
        elif h >= 0:
            best = h if best < 0 or rank_of[h] >= rank_of[best] else best
    names = np.array([symbols[h] if h >= 0 else "" for h in out])
    raw, adjusted, _ = fut.adjust_prices(out, np.asarray(panel["settlement"]), "ratio")
    return names, adjusted


def futures_portfolio(
    fut,
    products,
    *,
    start=None,
    end=None,
    rule="open_interest",
    account=CN_RETAIL,
    personal=True,
    personal_lead=0,
    infer_locked=True,
    slippage=2e-4,
    costs=None,
    config=None,
    initial_cash=1e7,
    delivery_fee_rate=0.0,
    delivery_fee_per_contract=0.0,
    missing=None,
    param_fill="previous",
    use_volume=True,
):
    """按换月规则选出每个品种每行持有的合约，只用这些合约建 ``panelbt.FuturesBook``。

    - ``rule``：数据层的换月规则（``"open_interest"`` / ``"volume"`` / ``"source"``）。
    - ``account``：期货公司条款，默认 ``CN_RETAIL``（假设：保证金 +5 个百分点、手续费 ×3）。
    - ``personal``：个人客户持仓期限（默认开启，规则见 ``PERSONAL_RULES``）；``personal_lead``：
      再提前几个交易日换月（默认 0，见 ``docs/FUTURES_BACKTEST.md`` 第 7 条）。
    - ``infer_locked``：开盘 = 最高 = 最低且有成交的行，按相对前一结算价的方向禁止买入（涨停）或卖出（跌停）。
    - ``slippage``：每笔成交的价格滑点比例（强平也按它计一次）。
    - ``missing`` / ``param_fill``：缺失的保证金、手续费如何处理，原样交给 ``fut.panelbt_inputs``（默认缺值报错）。
    """
    products = list(products)
    if not products:
        raise ValueError("products 不能为空")
    info = fut.contracts(products=products).collect()
    dates = fut.panel(["is_listed"], products=products[:1], start=start, end=end).dates
    holds = {}
    if personal:
        import warnings

        calendar = _exchange_calendar(fut, info)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            holds = last_hold_dates(info, calendar, lead=personal_lead)
        # 交易所日历之后才到期的合约（远月）补齐出来的期限都在样本之后，不影响回测，不警告。
        late = [
            s
            for s, last in zip(info["symbol"], info["last_trade_date"], strict=True)
            if last and last > calendar[-1] and holds.get(s) and holds[s] <= dates[-1]
        ]
        if caught and late:
            warnings.warn(str(caught[0].message), stacklevel=2)

    def hold_row(symbol):
        return _rows(dates, holds.get(symbol)) if personal else len(dates)

    names, signals = [], []
    for product in products:
        held, adjusted = _held(fut, product, rule, start, end, dates, hold_row)
        names.append(held)
        signals.append(adjusted)
    names = np.column_stack(names)
    chosen = sorted({s for s in names.ravel() if s})
    if not chosen:
        raise ValueError("样本区间内这些品种没有可持有的合约")
    inputs = fut.panelbt_inputs(
        symbols=chosen,
        start=start,
        end=end,
        fee_scale=1.0,
        use_volume=use_volume,
        param_fill=param_fill,
        missing=missing,
        multiplier_change="split",
    )
    f = inputs["futures"]
    labels = np.asarray(inputs["symbols"])
    base = np.array([s.split("@")[0] for s in labels])
    t = len(dates)
    contracts = inputs["contracts"]
    kind = dict(zip(contracts["symbol"], contracts["settlement_type"], strict=True))
    exchange = dict(zip(contracts["symbol"], contracts["exchange"], strict=True))
    cash_settled = np.array(
        [kind.get(s) == "cash" and exchange.get(s) == "CFFEX" for s in labels]
    )
    bars = fut.panel(
        ["open", "high", "low", "volume"],
        symbols=list(dict.fromkeys(base)),
        start=start,
        end=end,
    )
    # 拆列后的两列共用同一合约的行情；各自的可交易段由 fut.panelbt_inputs 的掩码决定。
    order = {s: j for j, s in enumerate(bars.symbols)}
    take = np.array([order[s] for s in base])
    market = inputs["market"]
    split_end = {d["before_column"] for d in inputs["splits"]}
    last_hold = np.array(
        [
            t if (j in split_end or not personal) else hold_row(base[j])
            for j in range(len(labels))
        ],
        dtype=np.int64,
    )
    if costs is None:
        costs = bt.Costs(
            commission_rate=0.0,
            commission_min=None,
            fee_rate_buy=0.0,
            fee_rate_sell=0.0,
            slippage=slippage,
        )
    config = config or bt.Config(initial_cash=initial_cash)
    book = bt.FuturesBook(
        settlement=f.settlement,
        close=market["close"],
        exec_price=market["exec_price"],
        tradable=market["can_buy"],
        open=np.asarray(bars["open"])[:, take],
        high=np.asarray(bars["high"])[:, take],
        low=np.asarray(bars["low"])[:, take],
        volume=market["volume"],
        multiplier=f.multiplier,
        expiry=f.expiry,
        margin_rate=f.margin_rate,
        fee_rate=f.fee_rate,
        fee_per_contract=f.fee_per_contract,
        infer_locked=infer_locked and use_volume,
        last_hold=last_hold,
        cash_settled=cash_settled,
        delivery_fee_rate=delivery_fee_rate,
        delivery_fee_per_contract=delivery_fee_per_contract,
        account=account,
        costs=costs,
        config=config,
    )
    # 品种层面的持有合约 → 列号；交易单位分界日起用后一列。
    column = {}
    for j in range(len(labels)):
        column.setdefault(base[j], []).append(j)
    split_date = {d["after_column"]: d["date"] for d in inputs["splits"]}
    held = np.full(names.shape, -1, dtype=np.int64)
    for day in range(t):
        for k in range(len(products)):
            s = names[day, k]
            if not s:
                continue
            cols = column[s]
            pick = cols[0]
            for j in cols[1:]:
                if dates[day] >= split_date[j]:
                    pick = j
            held[day, k] = pick
    rolls = [
        dict(
            date=dates[day],
            product=products[k],
            from_symbol=str(labels[held[day - 1, k]])
            if held[day - 1, k] >= 0
            else None,
            to_symbol=str(labels[held[day, k]]) if held[day, k] >= 0 else None,
        )
        for day in range(1, t)
        for k in range(len(products))
        if held[day, k] != held[day - 1, k]
    ]
    return FuturesPortfolio(
        book=book,
        dates=dates,
        symbols=labels,
        products=products,
        held=held,
        signal_prices=np.column_stack(signals),
        contracts=contracts,
        splits=inputs["splits"],
        rolls=pl.DataFrame(
            rolls,
            schema=dict(
                date=pl.String,
                product=pl.String,
                from_symbol=pl.String,
                to_symbol=pl.String,
            ),
        ),
        last_hold={s: holds.get(s) for s in chosen},
    )


@dataclass
class ExternalBook:
    """``book_from_table`` 的结果：``book`` 加上行（日期）和列（合约）标签。"""

    book: bt.FuturesBook
    dates: np.ndarray
    symbols: np.ndarray
    expiry: np.ndarray

    def column(self, symbol):
        return int(np.flatnonzero(self.symbols == symbol)[0])


def _frame(x, name):
    if x is None:
        return None
    if isinstance(x, pl.LazyFrame):
        return x.collect()
    if isinstance(x, pl.DataFrame):
        return x
    try:
        return pl.DataFrame(x)
    except Exception as exc:  # pandas 等：交给 polars 转
        raise ValueError(
            f"{name}: 需要 polars / pandas 表或「列名 → 数组」字典"
        ) from exc


def book_from_table(
    daily,
    contracts,
    params=None,
    *,
    margin_rate=None,
    fee_rate=None,
    fee_per_contract=None,
    account=None,
    slippage=0.0,
    costs=None,
    config=None,
    initial_cash=1e7,
    exec_field="open",
    infer_locked=False,
    missing="error",
    delivery_fee_rate=0.0,
    delivery_fee_per_contract=0.0,
):
    """外部行情的最薄入口：长表 → ``panelbt.FuturesBook``。列名与 ``ashare.data.futures_schema`` 一致。

    - ``daily``（必填列）：``date``（YYYYMMDD 字符串，交易所交易日）、``symbol``、``settlement``
      （当日结算价，报价单位）、``close``；可选 ``open``、``high``、``low``（报价单位）、``volume``（手，单边）。
      执行价取 ``exec_field``（默认开盘价）；没有这一列时用收盘价并请自己确认时点。
    - ``contracts``（必填列）：``symbol``、``multiplier``（每手的报价单位数量，如 WTI 1000 桶）、
      ``last_trade_date``（YYYYMMDD；最后交易日的结算价应是交割结算价，现金交割品种在
      ``settlement_type`` 写 ``"cash"``，其余按普通结算价强平）；可选 ``settlement_type``、
      ``last_hold_date``（最后一个可以收盘持仓的交易日，过了就在下一行按结算价强平）。
    - ``params``（可选）：``date``、``symbol``、``margin_long``、``margin_short``（比例，取两者较大）、
      ``fee_open_rate``（成交金额的比例）、``fee_open_per_contract``（每手，报价货币）。
      也可以直接给标量 ``margin_rate`` / ``fee_rate`` / ``fee_per_contract``（覆盖 params）。
    - 行 = ``daily`` 里出现过的全部日期（升序），列 = ``contracts`` 的顺序。某合约某天没有行就是没有行情、
      不成交；结算价在合约内向前填。

    例（合成的 WTI 式月度合约链，见 ``docs/FUTURES_BACKTEST.md``）::

        ext = book_from_table(daily, contracts, margin_rate=0.1, fee_per_contract=1.5)
        result = ext.book.run(plan, kind="lots")
    """
    daily = _frame(daily, "daily")
    contracts = _frame(contracts, "contracts")
    params = _frame(params, "params")
    for col in ("date", "symbol", "settlement", "close"):
        if col not in daily.columns:
            raise ValueError(f"daily 缺少列 {col}")
    for col in ("symbol", "multiplier", "last_trade_date"):
        if col not in contracts.columns:
            raise ValueError(f"contracts 缺少列 {col}")
    symbols = np.asarray(contracts["symbol"].to_list())
    if len(set(symbols)) != len(symbols):
        raise ValueError("contracts.symbol 有重复")
    unknown = set(daily["symbol"].unique().to_list()) - set(symbols)
    if unknown:
        raise ValueError(f"daily 里有合约表没有的合约 {sorted(unknown)[:10]}")
    dates = np.asarray(sorted(set(daily["date"].cast(pl.String).to_list())), dtype="U8")
    t, n = len(dates), len(symbols)
    row = {d: i for i, d in enumerate(dates)}
    col = {s: j for j, s in enumerate(symbols)}

    def panel(frame, field):
        out = np.full((t, n), np.nan)
        if field not in frame.columns:
            return None
        r = np.array([row[str(d)] for d in frame["date"].to_list()], dtype=np.int64)
        c = np.array([col[s] for s in frame["symbol"].to_list()], dtype=np.int64)
        out[r, c] = frame[field].cast(pl.Float64).to_numpy()
        return out

    if daily.select(pl.struct("date", "symbol").is_duplicated().any()).item():
        raise ValueError("daily 里同一合约同一天有多行")
    bars = {
        k: panel(daily, k)
        for k in ("open", "high", "low", "close", "settlement", "volume")
    }
    if exec_field not in daily.columns:
        raise ValueError(f"daily 没有执行价列 {exec_field}")
    last = np.asarray(
        contracts["last_trade_date"].cast(pl.String).to_list(), dtype="U8"
    )

    def rows_of(days):
        return np.array(
            [_rows(dates, d) if d and d != "None" else t for d in days], dtype=np.int64
        )

    expiry = rows_of(last)
    hold = (
        rows_of(contracts["last_hold_date"].cast(pl.String).to_list())
        if "last_hold_date" in contracts.columns
        else None
    )
    settled = (
        np.asarray([x == "cash" for x in contracts["settlement_type"].to_list()])
        if "settlement_type" in contracts.columns
        else None
    )

    def rate(name, scalar, fields):
        if scalar is not None:
            return scalar
        if params is None or not all(f in params.columns for f in fields):
            if name == "margin_rate":
                raise ValueError(
                    "需要 params（margin_long、margin_short）或标量 margin_rate"
                )
            return 0.0
        values = [panel(params, f) for f in fields]
        return np.fmax(*values) if len(values) == 2 else values[0]

    book = bt.FuturesBook(
        settlement=bars["settlement"],
        close=bars["close"],
        exec_price=bars[exec_field],
        open=bars["open"],
        high=bars["high"],
        low=bars["low"],
        volume=bars["volume"],
        multiplier=contracts["multiplier"].cast(pl.Float64).to_numpy(),
        expiry=expiry,
        margin_rate=rate("margin_rate", margin_rate, ("margin_long", "margin_short")),
        fee_rate=rate("fee_rate", fee_rate, ("fee_open_rate",)),
        fee_per_contract=rate(
            "fee_per_contract", fee_per_contract, ("fee_open_per_contract",)
        ),
        infer_locked=infer_locked,
        last_hold=hold,
        cash_settled=settled,
        delivery_fee_rate=delivery_fee_rate,
        delivery_fee_per_contract=delivery_fee_per_contract,
        account=account,
        costs=costs
        or bt.Costs(
            commission_rate=0.0,
            commission_min=None,
            fee_rate_buy=0.0,
            fee_rate_sell=0.0,
            slippage=slippage,
        ),
        config=config or bt.Config(initial_cash=initial_cash),
        missing=missing,
    )
    return ExternalBook(book=book, dates=dates, symbols=symbols, expiry=expiry)


__all__ = [
    "CN_RETAIL",
    "ExternalBook",
    "FuturesPortfolio",
    "PERSONAL_RULES",
    "book_from_table",
    "futures_portfolio",
    "last_hold_dates",
    "personal_rule",
]
