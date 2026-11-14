"""回测页的计算：信号 → 选股目标 → `panelbt` 成交模式 → 报告数据。

全部数值来自已有模块，这里只做接线：
- 信号：平台表达式经 `analysis.evaluate_signal` 求值（截到验证期末，TEST 不进入计算）；
- 选股：`ashare.portfolio.top_equal` 取信号最高的前 N 只等权，`to_targets` 转稀疏目标；
- 错位：`ashare.backtest.execution_targets`，信号日收盘后的目标在下一交易日成交；
- 行情与规则：`ashare.backtest.market_inputs`、`boards`，A 股费用 `ASHARE.costs`（历年印花税、
  过户费）、整手 `ASHARE.config`，涨跌停与停牌由面板可买可卖掩码交给引擎；
- 统计：`ashare.report.build_tearsheet`。

样本内与验证期各自从窗口首日起、以同样的初始资金独立回测（与研究台的窗口纪律一致）。
"""

import logging
import math
import time

import numpy as np
import panelbt
from panelbt.markets import ASHARE

from ashare.backtest import boards, execution_targets, iso_dates, market_inputs
from ashare.backtest.market import MASKS, RULES
from ashare.factor import groupbt
from ashare.factor.groupbt.api import UNIVERSES
from ashare.portfolio import to_targets, top_equal
from ashare.report import build_tearsheet
from ashare.report.context import index_benchmark, industry_switch_date

from ..analysis import evaluate_signal

LOG = logging.getLogger("ashare.platform")
WINDOWS = {"is": groupbt.IS, "validation": groupbt.VALIDATION}
INDEX_POOLS = {
    "csi300": "member_csi300",
    "csi500": "member_csi500",
    "csi1000": "member_csi1000",
}
BASE_FIELDS = ("close", "volume", "split", "cash_dividend", "delist", "is_listed")
CONTEXT_FIELDS = ("industry", "is_suspended", "has_quote")


class WindowPanel:
    """面板的一段连续行；缺字段时抛 KeyError（`market_inputs` 据此判断有没有掩码）。"""

    def __init__(self, panel, start, stop):
        self.panel, self.start, self.stop = panel, start, stop
        self.dates = np.asarray(panel.dates)[start:stop]
        self.symbols = panel.symbols

    def __getitem__(self, name):
        if name not in self.panel.arrays:
            raise KeyError(name)
        return self.panel.arrays[name][self.start : self.stop]


def fields(settings):
    """这次回测要读的面板字段；开盘成交优先用面板算好的可买可卖掩码。"""
    universe = settings["universe"]
    pools = (INDEX_POOLS[universe],) if universe in INDEX_POOLS else UNIVERSES[universe]
    price = (settings["trade_at"],)
    rules = MASKS if settings["trade_at"] == "open" else RULES
    return set(BASE_FIELDS + CONTEXT_FIELDS + price + rules + pools)


def ensure_fields(panel, settings):
    wanted = fields(settings)
    try:
        panel.ensure(wanted)
    except FileNotFoundError:
        # 旧面板没有开盘可买可卖掩码：按同一规则现算（`market_inputs` 的退路）。
        if settings["trade_at"] != "open":
            raise
        panel.ensure((wanted - set(MASKS)) | set(RULES))


def universe_mask(view, name):
    """股票池：单一指数的时点成分，或 groupbt 的 TOP800 / TOP1800 / ALL。"""
    if name not in INDEX_POOLS:
        return groupbt.universe_mask(view, name)
    listed = np.asarray(view["is_listed"], dtype=bool)
    member = np.asarray(view[INDEX_POOLS[name]], dtype=bool)
    # 成分数据还没开始的日子整行为空，不把全市场当成分。
    return member & listed & member.any(axis=1)[:, None]


def period_keys(dates, rebalance):
    days = np.asarray(dates).astype(str)
    if rebalance == "day":
        return np.arange(len(days))
    if rebalance == "year":
        return np.array([d[:4] for d in days])
    if rebalance == "month":
        return np.array([d[:6] for d in days])
    if rebalance == "quarter":
        return np.array([f"{d[:4]}{(int(d[4:6]) - 1) // 3}" for d in days])
    # 周：自 1970-01-05（周一）起的周序号；节假日不影响「本周最后一个交易日」。
    stamp = iso_dates(days).astype("datetime64[D]").astype(np.int64)
    return (stamp - 4) // 7


def signal_days(panel, start, stop, rebalance):
    """窗口内的调仓信号日（相对行号）：窗口首日，以及每个周期在面板里的最后一个交易日。

    用整张截断面板的下一行判断周期是否结束，只用日历、不看价格；窗口最后一行的信号
    在窗口内没有成交日，由 `execution_targets` 丢弃。
    """
    keys = period_keys(panel.dates, rebalance)
    rows = np.arange(start, stop)
    nxt = np.minimum(rows + 1, len(keys) - 1)
    last = (keys[rows] != keys[nxt]) | (rows + 1 >= len(keys))
    # 窗口首日也产生信号：第二个交易日建仓，不空等到第一个周期末。
    last[0] = True
    return np.flatnonzero(last)


def build_targets(alpha, pool, days, top_n, length):
    """每个信号日取池内信号有限值最高的 N 只等权，返回成交日上的 `panelbt.Targets`。"""
    rows, chosen = [], set()
    for d in days:
        w = top_equal(alpha[d], top_n, eligible=pool[d])
        rows.append(w)
        chosen.update(np.flatnonzero(w).tolist())
    columns = np.array(sorted(chosen), dtype=np.int64)
    dense = (
        np.array([w[columns] for w in rows])
        if len(rows)
        else np.zeros((0, len(columns)))
    )
    sparse = to_targets(np.asarray(days, dtype=np.int64), dense, columns)
    return execution_targets(sparse, t=length, lag=1)


def securities_context(data):
    """证券简称与申万一级行业名（与报告上下文同一取法）。"""
    import polars as pl

    sec = data.securities().select("symbol", "name").collect()
    names = dict(zip(sec["symbol"].to_list(), sec["name"].to_list(), strict=True))
    table = data.table("index_classify").collect()
    if "level" in table.columns:
        table = table.filter(pl.col("level") == "L1")
    source = (
        table["classification"].to_list()
        if "classification" in table.columns
        else [None] * table.height
    )
    industry_names = {}
    for code, name, src in zip(
        table["industry_code"].to_list(),
        table["industry_name"].to_list(),
        source,
        strict=True,
    ):
        head = str(code).split(".")[0]
        if head.isdigit() and (int(head) not in industry_names or src == "SW2021"):
            industry_names[int(head)] = name
    return names, industry_names


def calendar(data):
    # 日历只用来判断末月、末年是否完整；读不到时报告按工作日推断。
    try:
        frame = data.calendar().collect()
    except Exception:  # noqa: BLE001 - 可选上下文，缺失不影响回测
        LOG.warning("回测读取交易日历失败，末月完整性按工作日推断")
        return None
    import polars as pl

    return frame.filter(pl.col("is_open"))["date"].to_list()


def run_window(panel, alpha, settings, window, context):
    """一个窗口的完整回测，返回 (panelbt 结果, 行情, 目标, 报告数据)。"""
    dates = np.asarray(panel.dates)
    start = int(np.searchsorted(dates, window[0]))
    stop = int(np.searchsorted(dates, window[1]))
    if stop - start < 2:
        raise ValueError("窗口内交易日不足")
    view = WindowPanel(panel, start, stop)
    pool = universe_mask(view, settings["universe"])
    days = signal_days(panel, start, stop, settings["rebalance"])
    targets = build_targets(
        np.asarray(alpha)[start:stop], pool, days, settings["top_n"], stop - start
    )
    market = panelbt.Market(**market_inputs(view, settings["trade_at"]))
    symbols = np.asarray(panel.symbols)
    window_dates = iso_dates(view.dates)
    costs = ASHARE.costs(
        window_dates,
        commission_rate=settings["commission"],
        commission_min=settings["commission_min"],
        slippage=settings["slippage_bps"] / 1e4,
    )
    options = dict(
        initial_cash=settings["capital"] * 1e4,
        record_trades=True,
        carry_blocked=settings["carry_blocked"],
        dividend_tax_rate=settings["dividend_tax"],
    )
    if settings["lots"] == "board":
        config = ASHARE.config(board=context["boards"], **options)
    else:
        config = ASHARE.config(lot_sizes=np.zeros(len(symbols)), **options)
    result = panelbt.run(market, targets, costs, config)
    industry = np.asarray(view["industry"])
    suspended = np.asarray(view["is_suspended"], dtype=bool) | ~np.asarray(
        view["has_quote"], dtype=bool
    )
    benchmark = None
    if context.get("benchmark"):
        try:
            benchmark = index_benchmark(
                context["data"],
                view.dates,
                context["benchmark"],
                "price",
                context.get("benchmark_name"),
            )
        except Exception:  # noqa: BLE001 - 基准只影响相对指标，缺失时照常出结果
            LOG.warning("回测基准没有行情 symbol=%s", context["benchmark"])
    switch = None
    if "industry_source" in panel.arrays:
        switch = industry_switch_date(view["industry_source"], view.dates)
    report = build_tearsheet(
        result,
        market,
        targets,
        view.dates,
        symbols,
        names=context["names"],
        industry=industry,
        industry_names=context["industry_names"],
        industry_switch=switch,
        benchmark=benchmark,
        suspended=suspended,
        costs=costs,
        config=config,
        initial_capital=settings["capital"] * 1e4,
        calendar=context.get("calendar"),
        top_drawdowns=10,
        extras=False,
    )
    return result, market, targets, report


def _num(x):
    return float(x) if x is not None and math.isfinite(x) else None


def _round(x):
    # 与报告数据同样保留 6 位，页面只显示到 0.01%。
    x = _num(x)
    return None if x is None else round(x, 6)


def page_document(report, result, symbols, names):
    """报告数据里页面画得到的字段；调仓记录、成交另存，按页读取。"""
    s, daily = report["summary"], report["daily"]
    attribution = {row["year"]: row for row in report["attribution"]}
    yearly = []
    for row in report["yearly"]:
        a = attribution.get(row["year"], {})
        b = row["benchmark_return"]
        yearly.append(
            dict(
                year=row["year"],
                partial=row["partial"],
                strategy=row["return"],
                benchmark=b,
                excess=None
                if b is None or row["return"] is None
                else row["return"] - b,
                volatility=row["volatility"],
                drawdown=row["max_drawdown"],
                sharpe=row["sharpe"],
                price=a.get("price"),
                dividend=a.get("dividend_net"),
                fees=a.get("fees"),
                slippage=a.get("slippage"),
            )
        )
    reasons = {"limit_up": 0, "limit_down": 0, "suspended": 0}
    unfilled = 0
    for rb in report["rebalances"]:
        unfilled += rb["unfilled"]
        for key, value in rb["unfilled_by_reason"].items():
            reasons[key] += value
    industry = []
    weights = report.get("industry_weights")
    if weights and weights["dates"]:
        for name, values in zip(weights["industries"], weights["weights"], strict=True):
            v = np.array([np.nan if x is None else x for x in values], dtype=float)
            industry.append(
                dict(
                    name=name,
                    latest=_num(v[-1]),
                    mean=_round(np.nanmean(v)) if np.isfinite(v).any() else None,
                    max=_num(np.nanmax(v)) if np.isfinite(v).any() else None,
                )
            )
        industry.sort(key=lambda r: (-(r["latest"] or 0), -(r["mean"] or 0)))
    meta_benchmark = report["meta"]["benchmark"]
    keys = (
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "final_nav",
        "benchmark_annual_return",
        "benchmark_annual_volatility",
        "benchmark_sharpe",
        "benchmark_max_drawdown",
        "benchmark_cumulative_return",
        "excess_annual_return",
        "tracking_error",
        "information_ratio",
        "beta",
        "calmar",
        "monthly_win_rate",
        "annual_turnover_one_sided",
        "total_fees_to_initial",
        "total_slippage_to_initial",
        "rebalances",
    )
    return dict(
        start=s["start"],
        end=s["end"],
        benchmark=None
        if meta_benchmark is None
        else {"symbol": meta_benchmark["symbol"], "name": meta_benchmark["name"]},
        summary={k: s[k] for k in keys},
        constraints=dict(rebalances=s["rebalances"], unfilled=unfilled, **reasons),
        series=dict(
            dates=daily["date"],
            nav=daily["nav"],
            benchmark=daily["benchmark_nav"],
            drawdown=daily["drawdown"],
        ),
        yearly=yearly,
        drawdowns=report["drawdowns"],
        industry=industry,
        industry_sampling=None if not weights else weights["sampling"],
        holdings=[
            dict(
                symbol=h["symbol"],
                name=h["name"],
                industry=h["industry"],
                weight=h["weight"],
            )
            for h in report["holdings_latest"]
        ],
        rebalance_count=len(report["rebalances"]),
        trade_count=len(result.trades or ()),
    )


def rebalance_rows(report):
    keys = (
        "date",
        "targets",
        "buys",
        "sells",
        "new_names",
        "dropped_names",
        "turnover",
        "fees",
        "slippage",
        "unfilled",
    )
    return [{k: rb[k] for k in keys} for rb in report["rebalances"]]


def trade_arrays(result, symbols):
    """逐笔成交；证券代码随结果保存，数据更新改变证券轴后旧结果仍对得上。"""
    trades = result.trades or ()
    symbols = np.asarray(symbols).astype(str)
    return dict(
        day=np.array([t.day for t in trades], dtype=np.int32),
        symbol=symbols[np.array([t.col for t in trades], dtype=np.int64)],
        qty=np.array([t.qty for t in trades], dtype=np.float64),
        price=np.array([t.price for t in trades], dtype=np.float64),
        commission=np.array([t.commission for t in trades], dtype=np.float64),
        tax=np.array([t.tax for t in trades], dtype=np.float64),
        other=np.array([t.other_fee for t in trades], dtype=np.float64),
        slippage=np.array(
            [t.slippage_cost + t.impact_cost for t in trades], dtype=np.float64
        ),
    )


def run_backtest(
    panel, expression, settings, benchmark_name=None, progress=lambda phase: None
):
    """两个窗口的回测；返回 {part: (页面文档, 调仓记录, 成交数组, 日期)}。"""
    ensure_fields(panel, settings)
    panel.ensure(set(expression.fields))
    try:
        panel.ensure({"industry_source"})
    except (FileNotFoundError, KeyError):
        pass
    progress("signal")
    alpha = evaluate_signal(expression, panel)
    names, industry_names = securities_context(panel.data)
    context = dict(
        data=panel.data,
        names=names,
        industry_names=industry_names,
        benchmark=settings.get("benchmark"),
        benchmark_name=benchmark_name,
        calendar=calendar(panel.data),
        boards=boards(panel.data.securities(), panel.symbols)
        if settings["lots"] == "board"
        else None,
    )
    out = {}
    for part, window in WINDOWS.items():
        progress(part)
        started = time.perf_counter()
        result, _, _, report = run_window(panel, alpha, settings, window, context)
        out[part] = dict(
            document=page_document(report, result, panel.symbols, names),
            rebalances=rebalance_rows(report),
            trades=trade_arrays(result, panel.symbols),
            dates=np.asarray(report["daily"]["date"]),
        )
        LOG.info(
            "回测窗口完成 part=%s days=%d trades=%d seconds=%.3f",
            part,
            len(report["daily"]["date"]),
            out[part]["document"]["trade_count"],
            time.perf_counter() - started,
        )
    return out
