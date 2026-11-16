"""确认期冻结组合：挖掘期标准化、等权、±1 仓位；统一接入加密回测适配器。"""

import numpy as np

from ashare.backtest.crypto import crypto_market, daily_targets
from ashare.data.crypto.timeutil import DAY


def combine(series, normalizations, trade_hour):
    calendars = []
    for at, x in series:
        use = at % DAY == trade_hour * 3_600_000
        calendars.append((at[use], x[use]))
    times = np.unique(np.concatenate([a for a, _ in calendars]))
    total = np.zeros(len(times))
    for (at, x), normal in zip(calendars, normalizations, strict=True):
        scale = normal["std"]
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(normal["mean"]):
            raise ValueError("挖掘期标准化参数退化")
        z = (x - normal["mean"]) / scale
        total[np.searchsorted(times, at)] += np.where(np.isfinite(z), z, 0)
    return times - DAY, np.clip(total / len(series), -1, 1)[:, None]


def trend_targets(dataset, symbol, trade_hour, end):
    """C3 第一批趋势规则：20/60/120 日投票，30 日波动，20% 年化预算。"""
    frame = dataset.daily(
        trade_hour=trade_hour,
        instruments=["usdm:" + symbol],
        invalid_prices="carry",
        end=end,
    ).sort("day")
    px = frame["close"].to_numpy()[:, None]
    votes = np.zeros_like(px)
    for lookback in (20, 60, 120):
        votes[lookback:] += np.sign(px[lookback:] / px[:-lookback] - 1) / 3
    ret = np.full_like(px, np.nan)
    ret[1:] = px[1:] / px[:-1] - 1
    vol = np.full_like(px, np.nan)
    for i in range(30, len(px)):
        vol[i] = np.std(ret[i - 29 : i + 1], axis=0, ddof=1) * np.sqrt(365)
    w = np.divide(
        votes * 0.20, vol, out=np.zeros_like(px), where=np.isfinite(vol) & (vol > 0)
    )
    w[:120] = 0
    w[~np.isfinite(px)] = 0
    # 原 C3 永续示例总名义上限为 2；本入口统一一倍，规则相同、上限明确为 1。
    return frame["day"].to_numpy(), np.clip(w, -1, 1)


def summarize(result, hours):
    nav = result.nav
    return dict(
        final_nav=float(nav[-1]),
        annualized=float(nav[-1] ** (8760 / hours) - 1),
        max_drawdown=float(
            np.max(1 - np.r_[1.0, nav] / np.maximum.accumulate(np.r_[1.0, nav]))
        ),
        funding_paid=float(np.sum(result.funding)),
        fees=float(np.sum(result.commission)),
        valid=bool(result.diagnostics["crypto"]["valid"]),
    )


def run(dataset, series, normalizations, settings):
    cfg = settings.value
    start, end = settings.bounds("confirmation")
    market = crypto_market(
        dataset,
        [cfg["symbol"]],
        start,
        end,
        kind="usdm",
        leverage=1.0,
        initial_cash=cfg["initial_cash"],
        assume=cfg["assume"],
    )
    combined = combine(series, normalizations, cfg["trade_hour"])
    trend = trend_targets(dataset, cfg["symbol"], cfg["trade_hour"], end)
    # 同一成交钟点、同一区间、同费用和风险档案。
    first = start // DAY * DAY + cfg["trade_hour"] * 3_600_000
    if first < start:
        first += DAY
    hold = (
        np.arange(first, end, DAY) - DAY,
        np.ones((len(np.arange(first, end, DAY)), 1)),
    )
    out = {}
    for name, target in (("combined", combined), ("hold", hold), ("trend", trend)):
        result = market.run(daily_targets(target, trade_hour=cfg["trade_hour"]))
        out[name] = summarize(result, len(market.times))
    return out
