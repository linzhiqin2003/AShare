"""分组回测的入口：全市场因子矩阵 → 股票池 → 权重 → 分组与多空 → 绩效、风险暴露和图。

口径见 `docs/GROUP_BACKTEST.md`。时点：因子第 t 行只用到 t 日收盘及以前的数据，t+1 日成交
（`trade_at="open"` 按开盘价，`"close"` 按收盘价），持有到 t+2 日同一时点。结果按成交日记账。
"""

from dataclasses import dataclass, field

import numpy as np

from . import metrics, plot, portfolio, weights

UNIVERSES = {
    "TOP800": ("member_csi300", "member_csi500"),
    "TOP1800": ("member_csi300", "member_csi500", "member_csi1000"),
    "ALL": (),
}
# 用大池时一并检测的小池。
NESTED = {"ALL": ("TOP1800", "TOP800"), "TOP1800": ("TOP800",), "TOP800": ()}
FIELDS = (
    "open",
    "close",
    "adj_factor",
    "upper_limit",
    "lower_limit",
    "total_market_cap",
    "industry",
    "is_listed",
    "is_suspended",
    "has_quote",
    "can_buy_open",
    "can_sell_open",
    "member_csi300",
    "member_csi500",
    "member_csi1000",
)
PRICE_TOL = 0.005  # 与数据层开盘口径相同
PERIODS = 240  # 一年按 240 个交易日


def universe_mask(panel, name):
    """股票池的 (T, N) 布尔面板。TOP1800 在中证1000 有成分数据之前整行为空。"""
    if name not in UNIVERSES:
        raise ValueError(f"universe 只能是 {tuple(UNIVERSES)}，得到 {name!r}")
    listed = np.asarray(panel["is_listed"], dtype=bool)
    if name == "ALL":
        return listed.copy()
    parts = [np.asarray(panel[f], dtype=bool) for f in UNIVERSES[name]]
    mask = np.logical_or.reduce(parts)
    covered = np.logical_and.reduce([p.any(axis=1) for p in parts])
    return mask & listed & covered[:, None]


def industry_groups(panel):
    """申万一级行业编码 → 0 起的连续组号；没有行业的为 -1。"""
    code = np.asarray(panel["industry"]).astype(np.int64)
    known = np.unique(code[code > 0])
    out = np.searchsorted(known, code)
    return np.where(code > 0, out, -1)


def _adjusted(panel, name):
    """复权价；停牌或缺价的日子沿用上一个有效价（持仓在停牌期间收益为 0，复牌时一次体现）。"""
    px = np.asarray(panel[name], dtype=np.float64) * np.asarray(
        panel["adj_factor"], dtype=np.float64
    )
    px = np.where(np.isfinite(px) & (px > 0), px, np.nan)
    idx = np.where(np.isnan(px), 0, np.arange(px.shape[0])[:, None])
    np.maximum.accumulate(idx, axis=0, out=idx)
    return np.take_along_axis(px, idx, axis=0)


def _forward_return(price):
    """第 s 行：s 日到 s+1 日同一时点的收益；最后一行为 NaN。"""
    out = np.full(price.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:-1] = price[1:] / price[:-1] - 1.0
    return out


def close_masks(panel):
    """收盘价成交的可买、可卖：未停牌且有行情；收盘在涨停价不能买，在跌停价不能卖。"""
    close = np.asarray(panel["close"], dtype=np.float64)
    up = np.asarray(panel["upper_limit"], dtype=np.float64)
    down = np.asarray(panel["lower_limit"], dtype=np.float64)
    stop = np.asarray(panel["is_suspended"], dtype=bool) | ~np.asarray(
        panel["has_quote"], dtype=bool
    )
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(close) & (close > 0) & ~stop
        no_buy = np.isfinite(up) & (close >= up - PRICE_TOL)
        no_sell = np.isfinite(down) & (close <= down + PRICE_TOL)
    return valid & ~no_buy, valid & ~no_sell


def _shift(x):
    """信号行 → 成交行：整体下移一行，第一行为空。"""
    out = np.full(x.shape, np.nan)
    out[1:] = x[:-1]
    return out


@dataclass
class GroupBacktest:
    """一个股票池的结果。`subs` 是嵌套小池的同样结果。"""

    universe: str
    trade_at: str
    n_groups: int
    dates: np.ndarray
    group_pnl: np.ndarray  # (n_groups, T)，第 0 行是 top 组
    pnl: dict  # 名字 → 日盈亏：long_short、top，可选 weights
    turnover: dict
    exposure: dict  # 风险因子 → 多空组合的逐日暴露
    by: str = "year"
    subs: dict = field(default_factory=dict)

    def table(self, name="long_short", by=None):
        """绩效表：行是年份或季度加全历史，列是 return、sharpe、turnover、maxdrawdown、margin。"""
        return metrics.summary_table(
            self.pnl[name], self.turnover[name], self.dates, by or self.by, PERIODS
        )

    def curves(self, name="long_short"):
        return metrics.curves(self.pnl[name])

    def exposure_mean(self):
        return {k: float(np.nanmean(v)) for k, v in self.exposure.items()}

    def chart(self, title=None, weights_line=None):
        """分组累计盈亏线加多空线的 SVG。`weights_line` 默认在算了全池权重组合时画出。"""
        extra = None
        show = ("weights" in self.pnl) if weights_line is None else weights_line
        if show:
            if "weights" not in self.pnl:
                raise ValueError("没有全池权重组合：运行时传 weights_portfolio=True")
            extra = {"weights": np.cumsum(np.nan_to_num(self.pnl["weights"]))}
        return plot.pnl_chart(
            [np.cumsum(np.nan_to_num(g)) for g in self.group_pnl],
            np.cumsum(np.nan_to_num(self.pnl["long_short"])),
            self.dates,
            title=self.universe if title is None else title,
            extra=extra,
        )

    def to_dict(self):
        out = {
            "universe": self.universe,
            "trade_at": self.trade_at,
            "n_groups": self.n_groups,
            "long_short": self.table("long_short").to_dicts(),
            "top": self.table("top").to_dicts(),
            "exposure_mean": self.exposure_mean(),
        }
        if self.subs:
            out["subs"] = {k: v.to_dict() for k, v in self.subs.items()}
        return out

    def text(self):
        """两张绩效表和风险暴露均值的纯文本。"""
        lines = [f"{self.universe}  trade_at={self.trade_at}  n_groups={self.n_groups}"]
        for name in ("long_short", "top"):
            lines += ["", name, _format(self.table(name))]
        lines += ["", "risk exposure (mean)"]
        lines += [f"  {k:<12}{v:>10.4f}" for k, v in self.exposure_mean().items()]
        for sub in self.subs.values():
            lines += ["", "-" * 60, sub.text()]
        return "\n".join(lines)

    def html(self):
        """所有股票池的图和表拼成一页（自包含，不依赖外部文件）。"""
        parts = [
            "<meta charset='utf-8'><style>body{font:14px -apple-system,Helvetica,Arial,sans-serif;"
            "margin:24px;color:#222}table{border-collapse:collapse;margin:8px 32px 24px 0}"
            "td,th{padding:3px 10px;text-align:right;border-bottom:1px solid #eee}"
            "th{color:#666;font-weight:500}.t{display:inline-block;vertical-align:top}</style>"
        ]
        for res in (self, *self.subs.values()):
            parts.append(res.chart())
            for name in ("long_short", "top"):
                parts.append(
                    f"<div class='t'><b>{name}</b>{_html_table(res.table(name))}</div>"
                )
        return "\n".join(parts)


def _cells(frame):
    rows = [list(frame.columns)]
    for r in frame.iter_rows():
        period, ret, sharpe, turn, dd, margin = r
        rows.append(
            [
                period,
                "" if ret is None else f"{ret:.2%}",
                "" if sharpe is None else f"{sharpe:.2f}",
                "" if turn is None else f"{turn:.2%}",
                "" if dd is None else f"{dd:.2%}",
                "" if margin is None else f"{margin:.1f}",
            ]
        )
    return rows


def _format(frame):
    return "\n".join(
        "  " + r[0].ljust(8) + "".join(c.rjust(13) for c in r[1:])
        for r in _cells(frame)
    )


def _html_table(frame):
    head, *body = _cells(frame)
    cell = "".join
    return (
        "<table><tr>"
        + cell(f"<th>{c}</th>" for c in head)
        + "</tr>"
        + cell("<tr>" + cell(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
        + "</table>"
    )


def _one(name, alpha_all, inputs, options):
    """一个股票池：抽池、池内去均值、分组、推进、统计。"""
    panel, fwd, can_buy, can_sell, risk = inputs
    mask = universe_mask(panel, name)
    built = weights.build_weights(alpha_all, mask, decay=options["decay"])
    n = options["n_groups"]
    groups = portfolio.assign_groups(built["alpha"], n)

    def run(target, keep=False):
        return portfolio.simulate(
            _shift(target), fwd, can_buy, can_sell, keep_weights=keep
        )

    group_runs = [run(portfolio.group_targets(groups, g)) for g in range(n)]
    long_short = run(portfolio.long_short_targets(groups, n), keep=True)
    pnl = {"long_short": long_short["pnl"], "top": group_runs[0]["pnl"]}
    turnover = {"long_short": long_short["turnover"], "top": group_runs[0]["turnover"]}
    if options["weights_portfolio"]:
        full = run(built["weights"])
        pnl["weights"], turnover["weights"] = full["pnl"], full["turnover"]
    # 持仓在成交日 s 生效，对应的风险因子和股票池是信号日 s-1 的。
    known = {k: _shift(v) for k, v in risk.items()}
    pool = np.zeros(mask.shape, dtype=bool)
    pool[1:] = mask[:-1]
    exposure = metrics.risk_exposure(long_short["weights"], known, pool)
    # 第一个有持仓的成交日之前不计入统计。
    live = np.flatnonzero(np.abs(long_short["weights"]).sum(axis=1) > 0)
    if live.size:
        for series in (*pnl.values(), *turnover.values()):
            series[: live[0]] = np.nan
    group_pnl = np.array([g["pnl"] for g in group_runs])
    if live.size:
        group_pnl[:, : live[0]] = np.nan
    return GroupBacktest(
        universe=name,
        trade_at=options["trade_at"],
        n_groups=n,
        dates=np.asarray(panel.dates),
        group_pnl=group_pnl,
        pnl=pnl,
        turnover=turnover,
        exposure=exposure,
        by=options["by"],
    )


def run(
    factor,
    panel,
    universe,
    *,
    trade_at="open",
    trade_limitation=True,
    n_groups=10,
    group_neutralize=None,
    vector_neut=None,
    decay=0,
    by="year",
    weights_portfolio=False,
    sub_universes=True,
):
    """分组回测。`factor` 是在全市场上算好的 (T, N) 因子矩阵，行列与 `panel` 对齐，值越大越做多。

    - universe：`"TOP800"`（沪深300 + 中证500）、`"TOP1800"`（再加中证1000）、`"ALL"`。
    - trade_at：`"open"` 或 `"close"`，都是信号日的下一个交易日成交。
    - trade_limitation：停牌不交易；涨停只能卖、跌停只能买；冻结的持仓占着额度。
    - group_neutralize：`None`、`"industry"`、`"cap"`、`"cap+industry"`，在全市场上做。
    - vector_neut：一个或一组 (T, N) 风险向量，依次做截面正交。
    - weights_portfolio：另算一个直接按全池归一化权重持仓的多空组合。
    - sub_universes：用大池时把嵌套的小池各跑一遍，放在结果的 `subs` 里。

    `panel` 需要 `FIELDS` 里的字段（`ashare.data.open().panel(FIELDS)`）。分不出 `n_groups`
    组的交易日会抛 `GroupingError`。
    """
    if trade_at not in ("open", "close"):
        raise ValueError("trade_at 只能是 'open' 或 'close'")
    if n_groups not in (5, 10):
        raise ValueError("n_groups 只能是 5 或 10")
    if by not in ("year", "quarter"):
        raise ValueError("by 只能是 'year' 或 'quarter'")
    x = np.asarray(factor, dtype=np.float64)
    if x.shape != panel.shape:
        raise ValueError(f"factor 形状 {x.shape} 与面板 {panel.shape} 不一致")
    x = np.where(np.isfinite(x), x, np.nan)
    cap = np.asarray(panel["total_market_cap"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        logcap = np.log(np.where(cap > 0, cap, np.nan))
    # 全市场上的中性化；抽池、池内去均值在 _one 里做。
    everything = np.ones(x.shape, dtype=bool)
    alpha_all = weights.build_weights(
        x,
        everything,
        industry=industry_groups(panel) if group_neutralize else None,
        logcap=logcap,
        group_neutralize=group_neutralize,
        vector=vector_neut,
    )["alpha"]
    price = _adjusted(panel, trade_at)
    fwd = _forward_return(price)
    if not trade_limitation:
        can_buy = can_sell = None
    elif trade_at == "open":
        can_buy = np.asarray(panel["can_buy_open"], dtype=bool)
        can_sell = np.asarray(panel["can_sell_open"], dtype=bool)
    else:
        can_buy, can_sell = close_masks(panel)
    risk = metrics.risk_factors(_adjusted(panel, "close"), cap)
    inputs = (panel, fwd, can_buy, can_sell, risk)
    options = dict(
        trade_at=trade_at,
        n_groups=n_groups,
        decay=decay,
        by=by,
        weights_portfolio=weights_portfolio,
    )
    result = _one(universe, alpha_all, inputs, options)
    if sub_universes:
        for name in NESTED[universe]:
            result.subs[name] = _one(name, alpha_all, inputs, options)
    return result
