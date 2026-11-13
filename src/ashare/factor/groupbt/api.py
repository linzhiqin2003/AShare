"""分组回测的入口：全市场因子矩阵 → 股票池 → 权重 → 分组与多空 → 绩效、风险暴露和图。

口径见 `docs/GROUP_BACKTEST.md`。时点：因子第 t 行只用到 t 日收盘及以前的数据，t+1 日成交
（`trade_at="open"` 按开盘价，`"close"` 按收盘价），持有到 t+2 日同一时点。结果按成交日记账。
"""

from dataclasses import dataclass, field

import numpy as np
import panelbt
import panelops
from panelbt.markets.ashare import price_permissions

from ..windows import IS, TEST, VALIDATION  # noqa: F401
from . import metrics, plot, portfolio

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
# 中证1000 的成分数据从 2014 年 11 月才有；涉及 TOP1800 的回测统一从这一天起算。
TOP1800_START = "20150101"
PERIODS = 240  # 一年按 240 个交易日


def universe_mask(panel, name, start=None):
    """股票池的 (T, N) 布尔面板。`start`（YYYYMMDD）之前的行为空；TOP1800 固定从
    `TOP1800_START` 起，且在中证1000 有成分数据之前为空。"""
    mask = _members(panel, name)
    if name == "TOP1800":
        start = max(start or "", TOP1800_START)
    if start:
        mask[np.asarray(panel.dates) < start] = False
    return mask


def _members(panel, name):
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
    # ufunc 以 float64 运算，但不先把两份输入各展开成完整的 float64 副本。
    px = np.multiply(panel[name], panel["adj_factor"], dtype=np.float64)
    px[~(np.isfinite(px) & (px > 0))] = np.nan
    return panelops.ts_ffill(np.ascontiguousarray(px))


def _forward_return(price):
    """第 s 行：s 日到 s+1 日同一时点的收益；最后一行为 NaN。"""
    out = np.full(price.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(price[1:], price[:-1], out=out[:-1])
        out[:-1] -= 1.0
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
        return price_permissions(close, up, down, stop)


class _Until:
    """面板截到第 stop 行为止的只读视图：之后的行在回测里读不到。"""

    def __init__(self, panel, stop, first=0):
        self._panel, self._stop, self._first = panel, stop, first
        self.dates = np.asarray(panel.dates)[first:stop]
        self.symbols = panel.symbols

    def __getitem__(self, name):
        return self._panel[name][self._first : self._stop]

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)


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
    window=IS,
):
    """分组回测。`factor` 是在全市场上算好的 (T, N) 因子矩阵，行列与 `panel` 对齐，值越大越做多。

    - universe：`"TOP800"`（沪深300 + 中证500）、`"TOP1800"`（再加中证1000）、`"ALL"`。
      `TOP1800` 从 2015-01-01 起回测（带的 `TOP800` 小池同样）。
    - trade_at：`"open"` 或 `"close"`，都是信号日的下一个交易日成交。
    - trade_limitation：停牌不交易；涨停只能卖、跌停只能买；冻结的持仓占着额度。
    - group_neutralize：`None`、`"industry"`、`"cap"`、`"cap+industry"`，在全市场上做。
    - vector_neut：一个或一组 (T, N) 风险向量，依次做截面正交。
    - weights_portfolio：另算一个直接按全池归一化权重持仓的多空组合。
    - sub_universes：用大池时把嵌套的小池各跑一遍，放在结果的 `subs` 里。

    - window：`(起, 止)`，YYYYMMDD，含起不含止，默认样本内 `IS`（2016-01-01 至 2025-01-01）。
      止日及以后的行情、因子一概不读；起日之前的数据只用来算滚动窗口，不产生持仓。
      样本外窗口 `VALIDATION`、`TEST` 要显式传入才会用到。

    `panel` 需要 `FIELDS` 里的字段（`ashare.data.open().panel(FIELDS)`）。分不出 `n_groups`
    组的交易日会抛 `GroupingError`。调用期间不要修改任何输入（包括共享底层内存的
    其他视图）；内核会释放 GIL。有效窗口的因子独立复制，其他传入内核的视图临时只读。
    """
    if trade_at not in ("open", "close"):
        raise ValueError("trade_at 只能是 'open' 或 'close'")
    if n_groups not in (5, 10):
        raise ValueError("n_groups 只能是 5 或 10")
    if by not in ("year", "quarter"):
        raise ValueError("by 只能是 'year' 或 'quarter'")
    if universe not in UNIVERSES:
        raise ValueError(f"universe 只能是 {tuple(UNIVERSES)}，得到 {universe!r}")
    x = np.asarray(factor, dtype=np.float64)
    if x.shape != panel.shape:
        raise ValueError(f"factor 形状 {x.shape} 与面板 {panel.shape} 不一致")
    start, stop = (str(d) for d in window)
    if not start < stop:
        raise ValueError(f"window 的起日 {start} 要早于止日 {stop}")
    history = _Until(panel, int(np.searchsorted(np.asarray(panel.dates), stop)))
    dates = history.dates
    # 窗口之前不可能持仓；因子的中性化均为逐截面运算，不需要历史行。
    # 价格仍从全历史前填，风险因子单独保留所需的 240 行历史。
    first = int(np.searchsorted(dates, start))
    first = min(first, max(len(dates) - 1, 0))
    panel = _Until(history, len(dates), first)
    x = x[first : len(dates)].copy(order="C")
    if vector_neut is not None:
        many = isinstance(vector_neut, (list, tuple))
        cut = [
            np.asarray(v)[first : len(dates)]
            for v in (vector_neut if many else [vector_neut])
        ]
        vector_neut = cut if many else cut[0]
    if group_neutralize not in (None, "industry", "cap", "cap+industry"):
        raise ValueError(f"无法识别的 group_neutralize 取值：{group_neutralize!r}")
    logcap = np.array(panel["total_market_cap"], dtype=np.float64, copy=True)
    logcap[~(logcap > 0)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        np.log(logcap, out=logcap)
    industry = None
    if group_neutralize in ("industry", "cap+industry"):
        industry = panel["industry"]
    if not trade_limitation:
        can_buy = can_sell = None
    elif trade_at == "open":
        can_buy, can_sell = panel["can_buy_open"], panel["can_sell_open"]
    else:
        can_buy, can_sell = close_masks(panel)
    names = [universe, *(NESTED[universe] if sub_universes else ())]
    pool_start = max(start, TOP1800_START) if universe == "TOP1800" else start
    masks = [universe_mask(panel, name, pool_start) for name in names]
    vectors = (
        []
        if vector_neut is None
        else (vector_neut if isinstance(vector_neut, (list, tuple)) else [vector_neut])
    )
    # ashare 显式指定信号行和成交行的错位；引擎不推断交易时点。
    runs = panelbt.run_group_weights(
        x,
        history[trade_at],
        history["close"],
        history["adj_factor"],
        logcap,
        masks,
        signal_start=first,
        execution_start=first + 1,
        industry=industry,
        industry_zero_missing=True,
        vectors=vectors,
        can_buy=can_buy,
        can_sell=can_sell,
        n_groups=n_groups,
        neutralize=group_neutralize,
        decay=0 if decay <= 1 else decay,
        weighted=weights_portfolio,
    )
    results = []
    for name, out in zip(names, runs, strict=True):
        if out["bad_rows"]:
            raise portfolio.GroupingError(out["bad_rows"])
        p, turn = out["pnl"], out["turnover"]
        for arr in (p, turn):
            arr[:, : out["first_live"]] = np.nan
            arr[:, -1:] = np.nan
        p = np.pad(p, ((0, 0), (first, 0)), constant_values=np.nan)
        turn = np.pad(turn, ((0, 0), (first, 0)), constant_values=np.nan)
        exp = np.pad(out["exposure"], ((0, 0), (first, 0)), constant_values=np.nan)
        pnl = {"long_short": p[n_groups], "top": p[0]}
        turnover = {"long_short": turn[n_groups], "top": turn[0]}
        if weights_portfolio:
            pnl["weights"], turnover["weights"] = p[-1], turn[-1]
        results.append(
            GroupBacktest(
                universe=name,
                trade_at=trade_at,
                n_groups=n_groups,
                dates=dates,
                group_pnl=p[:n_groups].copy(),
                pnl=pnl,
                turnover=turnover,
                exposure=dict(
                    zip(
                        ("size", "reversal_20", "reversal_60", "momentum"),
                        exp,
                        strict=True,
                    )
                ),
                by=by,
            )
        )
    results[0].subs = dict(zip(names[1:], results[1:], strict=True))
    return results[0]
