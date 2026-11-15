"""资金费与基差时序因子；时间标签一律是信息可得时刻（UTC 毫秒）。"""

from __future__ import annotations

import numpy as np
import polars as pl

from ashare.data.crypto.signals import BAD_PRICE, QUARTER_BAD_PRICE
from ashare.data.crypto.timeutil import DAY, HOUR, millis

FACTOR_DEFINITION = "crypto-factors-v1-design4.18v5.1"
FACTOR_NAMES = (
    "funding_last",
    "funding_mean_3d",
    "funding_mean_7d",
    "funding_mean_30d",
    "funding_z_90d",
    "premium_1d",
    "premium_7d",
    "mark_premium_1d",
    "mark_premium_7d",
    "basis_q",
    "basis_q_chg_7d",
    "carry_spread",
)


class CryptoFactors(dict):
    """因子字典；identity 为缓存键，selected_contract/roll 同样带 available_at。"""

    def __init__(self, values, *, identity, symbols, selected_contract, roll):
        super().__init__(values)
        self.identity = identity
        self.symbols = tuple(symbols)
        self.selected_contract = selected_contract
        self.roll = roll


def rolling_sum(values, window):
    """固定行数求和；调用者保证规则小时/日轴，缺值视零并另数有效数。"""
    v = np.asarray(values)
    c = np.concatenate(
        [np.zeros((1, *v.shape[1:])), np.cumsum(np.nan_to_num(v, nan=0), axis=0)]
    )
    return c[1:] - c[np.maximum(0, np.arange(1, len(v) + 1) - window)]


def rolling_mean(values, window):
    count = rolling_sum(np.isfinite(values).astype(float), window)
    total = rolling_sum(values, window)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan),
        where=count >= np.ceil(window * 2 / 3),
    )


def daily_factor(values, trade_hour=0):
    """available_at=τ → 日线左端 τ−24h；daily_targets 做唯一一次前移。"""
    if not isinstance(trade_hour, int) or not 0 <= trade_hour < 24:
        raise ValueError("trade_hour 必须是 0..23")
    at, x = values
    at = np.asarray([millis(t) for t in at], dtype=np.int64)
    x = np.asarray(x)
    if len(at) != len(x) or np.any(np.diff(at) <= 0) or np.any(at % HOUR):
        raise ValueError("因子时间必须为唯一递增整点且与值对齐")
    use = at % DAY == trade_hour * HOUR
    return at[use] - DAY, x[use]


def crypto_factors(ds, signals, symbols, start, end, *, names=None):
    """读取全部已覆盖历史预热，最后按 available_at 的 [start,end) 裁剪。"""
    names = tuple(FACTOR_NAMES if names is None else names)
    if set(names) - set(FACTOR_NAMES) or len(set(names)) != len(names):
        raise ValueError("未知或重复因子")
    symbols = tuple(symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("symbols 不能为空或重复")
    fields = [
        "close",
        "mark_close",
        "index_close",
        "quality",
        "funding_rate",
        "funding_interval_hours",
    ]
    instruments = [prefix + s for s in symbols for prefix in ("spot:", "usdm:")]
    a = ds.arrays(fields, instruments=instruments)
    times = a["time"]
    if len(times) == 0 or np.any(np.diff(times) != HOUR):
        raise ValueError("因子需要完整小时轴，缺口保留质量占位")
    at = times + HOUR
    n, k = len(times), len(symbols)
    spot, perp = a["close"][:, 0::2], a["close"][:, 1::2]
    spot_ok = ((a["quality"][:, 0::2].astype(int) & BAD_PRICE) == 0) & (spot > 0)
    perp_ok = ((a["quality"][:, 1::2].astype(int) & BAD_PRICE) == 0) & (perp > 0)
    rates = a["funding_rate"][:, 1::2]
    intervals = a["funding_interval_hours"][:, 1::2]
    result = {}
    annual = np.divide(
        rates * 8760,
        intervals,
        out=np.full_like(rates, np.nan),
        where=np.isfinite(rates) & (intervals > 0),
    )
    last = np.full_like(annual, np.nan)
    raw_inputs = ds.inputs().get("data", {}) if hasattr(ds, "inputs") else {}
    active = (a["quality"][:, 1::2].astype(int) & 128) == 0
    for j in range(k):
        idx = np.maximum.accumulate(
            np.where(np.isfinite(annual[:, j]), np.arange(n), -1)
        )
        good = idx >= 0
        last[good, j] = annual[idx[good], j]
        initial = [
            r
            for r in raw_inputs.get("usdm:" + symbols[j], {}).get("fundingRate", [])
            if r["time"] <= times[0]
        ]
        if initial:
            event = max(initial, key=lambda r: r["time"])
            last[~good, j] = event["rate"] * 8760 / event["interval"]
    last[~active] = np.nan
    result["funding_last"] = last
    for days in (3, 7, 30):
        width = days * 24
        value = (
            np.column_stack(
                [
                    np.convolve(np.nan_to_num(rates[:, j]), np.ones(width), "full")[:n]
                    for j in range(k)
                ]
            )
            * 8760
            / width
        )
        value[at - days * DAY < times[0]] = np.nan
        value[rolling_sum(active.astype(float), width) < width] = np.nan
        result[f"funding_mean_{days}d"] = value
    m = result["funding_mean_7d"]
    z = np.full_like(m, np.nan)
    # 同钟点 90 个过去值；窗口排除当前值，不借未来补足 60 个。
    for hour in range(24):
        idx = np.flatnonzero(at % DAY == hour * HOUR)
        v = m[idx]
        lag = np.vstack([np.full((1, k), np.nan), v[:-1]])
        windows = np.lib.stride_tricks.sliding_window_view(
            np.vstack([np.full((89, k), np.nan), lag]), 90, axis=0
        )
        finite = np.isfinite(windows)
        count = finite.sum(axis=-1)
        mean = np.divide(
            np.nansum(windows, axis=-1), count, out=np.zeros_like(v), where=count > 0
        )
        squares = np.nansum((windows - mean[:, :, None]) ** 2, axis=-1)
        sd = np.sqrt(
            np.divide(squares, count - 1, out=np.zeros_like(v), where=count > 1)
        )
        low = np.min(np.where(finite, windows, np.inf), axis=-1)
        high = np.max(np.where(finite, windows, -np.inf), axis=-1)
        z[idx] = np.divide(
            v - mean,
            sd,
            out=np.full_like(v, np.nan),
            where=(count >= 60) & (high > low) & (sd > 0),
        )
    result["funding_z_90d"] = z
    premium = np.where(spot_ok & perp_ok, perp / spot - 1, np.nan)
    mark, index = a["mark_close"][:, 1::2], a["index_close"][:, 1::2]
    mark_premium = np.where(
        perp_ok & (mark > 0) & (index > 0), mark / index - 1, np.nan
    )
    for days in (1, 7):
        result[f"premium_{days}d"] = rolling_mean(premium, days * 24)
        result[f"mark_premium_{days}d"] = rolling_mean(mark_premium, days * 24)
    basis, change = np.full((n, k), np.nan), np.full((n, k), np.nan)
    selected = np.full((n, k), "", dtype=object)
    rolled = np.zeros((n, k), dtype=bool)
    if signals is not None:
        frame = signals.read()
        metadata = signals.metadata
        for j, symbol in enumerate(symbols):
            for meta in sorted(
                (
                    q
                    for q in metadata
                    if q["pair"] == symbol and q["listed_at"] is not None
                ),
                key=lambda q: q["delivery_at"],
            ):
                remaining = (meta["delivery_at"] - at) / HOUR
                eligible = (at - meta["listed_at"] >= 7 * DAY) & (remaining > 336)
                choose = eligible & (selected[:, j] == "")
                # 选约先于读取有效性，近月缺一小时仍保留近月代码。
                selected[choose, j] = meta["symbol"]
                part = frame.filter(pl.col("symbol") == meta["symbol"])
                price = np.full(n, np.nan)
                loc = np.searchsorted(times, part["time"].to_numpy())
                valid = (
                    (loc < n)
                    & (part["time"].to_numpy() >= times[0])
                    & ((part["quality"].to_numpy() & QUARTER_BAD_PRICE) == 0)
                )
                loc, vals = loc[valid], part["close"].to_numpy()[valid]
                price[loc] = vals
                own = np.divide(
                    (price / spot[:, j] - 1) * 8760,
                    remaining,
                    out=np.full(n, np.nan),
                    where=eligible & spot_ok[:, j] & (price > 0),
                )
                basis[choose, j] = own[choose]
                lag = np.full(n, np.nan)
                lag[168:] = own[:-168]
                change[choose, j] = (own - lag)[choose]
            previous = ""
            for i in range(n):
                current = selected[i, j]
                rolled[i, j] = bool(current and previous and current != previous)
                if current:
                    previous = current
    result.update(basis_q=basis, basis_q_chg_7d=change, carry_spread=m - basis)
    use = (at >= millis(start)) & (at < millis(end))
    identity = (
        ds.version,
        signals.version if signals is not None else None,
        FACTOR_DEFINITION,
    )
    return CryptoFactors(
        {name: (at[use], result[name][use]) for name in names},
        identity=identity,
        symbols=symbols,
        selected_contract=(at[use], selected[use]),
        roll=(at[use], rolled[use]),
    )
