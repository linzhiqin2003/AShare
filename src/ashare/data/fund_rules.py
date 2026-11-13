"""场内基金（ETF、LOF、REITs）的回测规则：费用表、交易单位、涨跌停、T+0 / T+1，以及
「从基金面板建 `panelbt.Market`」。

依据与未核实处（详见 docs/FUND_CB_DATA.md「回测规则」）：

- 印花税：基金买卖免征（财税〔1998〕55 号起；A 股印花税从未对基金征收）。**已核实**。
- 过户费：中国结算对 ETF、LOF 场内交易不收过户费（股票的过户费按「A 股」收）。按 0 处理，**未逐条核实** LOF 历史收费。
- 经手费、证管费：交易所按成交额收，通常并入券商佣金报价；和股票一样由调用方用佣金参数给。
- 交易单位：申报数量 100 份或其整数倍（交易所交易规则 3.3 节），卖出余额不足 100 份可一次卖出。**已核实**。
- 涨跌停：一般为前收盘的 ±10%；跟踪创业板、科创板指数的 ETF 自相应板块注册制改革起为 ±20%
  （深市创业板 ETF 2020-08-24、沪市科创板 ETF 上市即 20%）；REITs 上市首日 ±30%、之后 ±10%。
  创业板 / 科创板的认定按名称关键词，**未逐只核实**；校验用实际涨跌幅反查（docs 里有结果）。
- 交收：股票类 ETF、LOF 为 T+1；跨境、债券、黄金、商品期货、货币 ETF 可当日回转（T+0）。日频回测里
  每天只有一次成交机会，T+0 / T+1 只作为元数据（`t0` 列），不影响结果。分类**按基金类型和名称推断、
  未逐只核实**；已知的错分都偏乐观（把 T+1 的判成 T+0）：通过港股通投资的「沪深港」ETF 按股票算 T+1
  （未核实）、分级基金的债券类子份额判为 T+0（未核实）。名称含「有色」「大宗商品」的是股票 ETF，
  「MSCI 中国 A」是 A 股 ETF，已按股票处理。
- 价格最小变动单位 0.001 元，涨跌停价按此四舍五入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from panelbt.markets import FeeSchedule, MarketProfile

from .fund_panel import fund_class

TICK = 0.001
LOT = 100.0

# 基金买卖没有印花税、过户费：费用表全为 0（佣金、经手费由调用方给）。
FUND = MarketProfile(
    "China exchange-traded funds",
    "CNY",
    252,
    FeeSchedule((("1998-01-01", {}),)),
    lot_rule="direct",
    price_limits=True,
    sell_lock=1,
    regulatory_notes=("no stamp duty on fund trades", "no transfer fee on ETF/LOF"),
)

CROSS_BORDER = (
    "纳斯达克",
    "纳指",
    "标普",
    "道琼斯",
    "恒生",
    "恒指",
    "港股",
    "香港",
    "H股",
    "日经",
    "日本",
    "德国",
    "法国",
    "欧洲",
    "中概",
    "海外",
    "全球",
    "美国",
    "美股",
    "亚太",
    "东南亚",
    "沙特",
    "巴西",
    "印度",
    "越南",
    "QDII",
    "美元",
    "中韩",
    "新加坡",
    "互联网50",
)
PRECIOUS = (
    "黄金",
    "上海金",
    "白银",
)  # 名称里没有「股」才算商品（「黄金产业股票ETF」是股票）
GROWTH_BOARD = ("创业板", "创业", "科创", "双创")
GROWTH_BOARD_FROM = {"SZ": "20200824", "SH": "20190722"}


def classify(securities):
    """证券表 → 每只基金的大类（stock / bond / gold_commodity / money / cross_border / reits / other）、
    是否 T+0、涨跌停档（0.1 或 0.2）及其生效日、证券类别 fund_class（证券表里没有就按名称推断）。返回 polars DataFrame（按 symbol 排序）。"""
    rows = []
    for r in securities.iter_rows(named=True):
        name = r.get("name") or ""
        ftype = r.get("fund_type") or ""
        if ftype == "REITs" or "REIT" in name:
            kind = "reits"
        elif ftype == "货币型" or "货币" in name:
            kind = "money"
        elif any(k in name for k in CROSS_BORDER) or (
            "MSCI中国" in name and "MSCI中国A" not in name
        ):
            kind = "cross_border"
        elif "期货" in name or (any(k in name for k in PRECIOUS) and "股" not in name):
            # 商品类只认商品期货基金和贵金属现货基金；「有色金属」「大宗商品股票」是股票 ETF。
            kind = "gold_commodity"
        elif ftype == "债券型" or "债" in name:
            kind = "bond"
        elif ftype in ("股票型", "混合型"):
            kind = "stock"
        else:
            kind = "other"
        exchange = r["symbol"].rsplit(".", 1)[-1]
        growth = kind == "stock" and any(k in name for k in GROWTH_BOARD)
        rows.append(
            dict(
                symbol=r["symbol"],
                asset_class=kind,
                t0=kind in ("cross_border", "gold_commodity", "bond", "money"),
                limit=0.2 if growth else 0.1,
                limit_from=GROWTH_BOARD_FROM.get(exchange) if growth else None,
                fund_class=r.get("fund_class") or fund_class(r["symbol"], name, ftype),
            )
        )
    return pl.DataFrame(
        rows,
        schema=dict(
            symbol=pl.String,
            asset_class=pl.String,
            t0=pl.Boolean,
            limit=pl.Float64,
            limit_from=pl.String,
            fund_class=pl.String,
        ),
    ).sort("symbol")


def price_limits(dates, previous_close, classes, listing_rows=None):
    """涨跌停价（T×N）。classes 为 `classify` 的结果（行顺序与列一致）。

    20% 档在 limit_from 之前按 10%；REITs 上市首日 30%；上市首日（listing_rows）其他基金按 10%
    （基金上市首日没有放开涨跌幅）。前收盘缺失时涨跌停为 NaN（引擎视为当天无涨跌停）。
    """
    pc = np.asarray(previous_close, dtype=float)
    t, n = pc.shape
    pct = np.full((t, n), 0.1)
    for j, r in enumerate(classes.iter_rows(named=True)):
        if r["limit"] > 0.1:
            start = r["limit_from"] or "00000000"
            pct[np.asarray(dates) >= start, j] = r["limit"]
        if (
            r["asset_class"] == "reits"
            and listing_rows is not None
            and listing_rows[j] < t
        ):
            pct[listing_rows[j], j] = 0.3
    up = np.round(pc * (1 + pct) / TICK) * TICK
    down = np.round(pc * (1 - pct) / TICK) * TICK
    return up, down


@dataclass(frozen=True)
class FundMarket:
    """一份基金面板对应的行情与规则；`config()`、`costs()` 补齐账户和费用。"""

    market: object
    dates: np.ndarray
    symbols: np.ndarray
    classes: pl.DataFrame
    panel: object = field(repr=False)

    @property
    def shape(self):
        return self.market.shape

    def config(self, **overrides):
        """每只 100 份起、按 100 份递增；其他参数交给 `panelbt.Config`。"""
        return FUND.config(lot_sizes=np.full(len(self.symbols), LOT), **overrides)

    def costs(
        self, *, commission_rate=1e-4, commission_min=0.0, slippage=5e-4, **overrides
    ):
        """免印花税、免过户费；佣金（含经手费）默认万分之一、无最低收费，滑点默认万分之五。

        佣金按券商协议差别很大（常见万 0.5 到万 3，部分券商 ETF 有 5 元最低），请按实际给。
        """
        days = np.asarray(
            [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in self.dates], "datetime64[D]"
        )
        return FUND.costs(
            days,
            commission_rate=commission_rate,
            commission_min=commission_min,
            slippage=slippage,
            **overrides,
        )


FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "previous_close",
    "volume",
    "split",
    "cash_dividend",
    "delist",
    "has_quote",
    "is_listed",
)


def fund_market(
    funds,
    symbols=None,
    start=None,
    end=None,
    *,
    exec="open",
    graded=False,
    validate=True,
):
    """从基金面板建回测行情。

    funds：`ashare.data.open(root).funds()`。exec："open"（默认）或 "close"。
    可买：成交价有效、当天有行情、在上市期内、成交价不在涨停价及以上（一字涨停也不能买）；
    可卖同理看跌停。拆分、分红、退市直接用面板的 split、cash_dividend、delist。

    分级基金（证券表 ``fund_class == "graded"``）默认不能买（graded=False，发告警列出被排除的代码）：
    A 份额的定期折算不在复权因子里，持有者每次折算会记一笔约 −10% 的假亏损，回测结果不可用。
    graded=True 时照常交易，同样发告警。

    最短例子::

        import ashare.data as ad
        import panelbt as bt
        from ashare.data.fund_rules import fund_market

        fm = fund_market(ad.open(root).funds(), ["510300.SH", "511010.SH"], "2020-01-02")
        result = bt.run(fm.market, targets, fm.costs(), fm.config())
    """
    import panelbt as bt
    from panelbt.markets.ashare import tradability

    if exec not in ("open", "close"):
        raise ValueError("exec: open 或 close")
    p = funds.panel("fund", list(FIELDS), start=start, end=end, symbols=symbols)
    securities = (
        funds.securities("fund")
        .filter(pl.col("symbol").is_in(list(p.symbols)))
        .collect()
    )
    order = {s: i for i, s in enumerate(p.symbols)}
    securities = securities.sort(
        pl.col("symbol").replace_strict(order, return_dtype=pl.Int64)
    )
    classes = classify(securities)
    classes = classes.sort(
        pl.col("symbol").replace_strict(order, return_dtype=pl.Int64)
    )
    if classes["symbol"].to_list() != list(p.symbols):
        raise ValueError("证券表与面板代码对不上")
    dates = np.asarray(p.dates)
    listing = [
        int(np.searchsorted(dates, d)) if d else len(dates)
        for d in securities["listing_date"].to_list()
    ]
    up, down = price_limits(dates, p["previous_close"], classes, listing)
    price = np.asarray(p[exec], dtype=float)
    ok = np.asarray(p["has_quote"]) & np.asarray(p["is_listed"])
    can_buy, can_sell = tradability(price, p["high"], p["low"], up, down, ~ok)
    flagged = [
        s
        for s, c in zip(p.symbols, classes["fund_class"].to_list(), strict=True)
        if c == "graded"
    ]
    if flagged:
        import warnings

        if graded:
            warnings.warn(
                f"包含分级基金 {flagged[:10]}：定期折算不在复权因子里，回测收益不可靠",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            can_buy = np.array(can_buy, copy=True)
            can_buy[:, np.isin(p.symbols, flagged)] = False
            warnings.warn(
                f"分级基金不能买（graded=False）：{flagged[:10]}",
                RuntimeWarning,
                stacklevel=2,
            )
    close = np.where(np.asarray(p["has_quote"]), p["close"], np.nan)
    market = bt.Market(
        np.where(ok, price, np.nan),
        close,
        can_buy,
        can_sell,
        volume=np.nan_to_num(np.asarray(p["volume"], dtype=float)),
        split=np.asarray(p["split"], dtype=float),
        cash_div=np.asarray(p["cash_dividend"], dtype=float),
        delist=np.asarray(p["delist"], dtype=bool),
        validate=validate,
    )
    return FundMarket(market, dates, np.asarray(p.symbols), classes, p)


__all__ = ["FUND", "FundMarket", "classify", "fund_market", "price_limits"]
