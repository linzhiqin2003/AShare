"""加密货币的日频落点、时点选币和季度链；不重新聚合日线。"""

import re
from dataclasses import dataclass

import numpy as np
import polars as pl

from ashare.data.crypto.events import asof
from ashare.data.crypto.timeutil import DAY, HOUR, millis


def duration(value):
    if isinstance(value, (int, np.integer)) and value >= 0:
        return int(value)
    match = re.fullmatch(r"(\d+)(D|h)", str(value))
    if not match:
        raise ValueError("时长必须为毫秒整数或 '30D'/'8h'")
    return int(match[1]) * {"D": DAY, "h": HOUR}[match[2]]


def hour(value):
    if not isinstance(value, int) or not 0 <= value < 24:
        raise ValueError("trade_hour 必须为 UTC 0..23")
    return value * HOUR


@dataclass(frozen=True)
class DailyTargets:
    times: np.ndarray
    weights: np.ndarray
    symbols: tuple | None
    trade_hour: int

    def align(self, times, symbols):
        times = np.asarray(times, dtype=np.int64)
        w = self.weights
        if self.symbols is not None:
            keys = [s.split(":")[-1] for s in self.symbols]
            expected = [str(s).split(":")[-1] for s in symbols]
            if set(keys) != set(expected):
                raise ValueError("日频目标列与市场列不一致")
            w = w[:, [keys.index(s) for s in expected]]
        if w.shape[1] != len(symbols):
            raise ValueError("日频目标列数与市场列不一致")
        out = np.full((len(times), len(symbols)), np.nan)
        rows = np.searchsorted(times, self.times)
        for k, row in enumerate(rows):
            if row < len(times) and times[row] == self.times[k]:
                out[row] = w[k]
        return out


def daily_targets(w_daily, trade_hour=0):
    """日线左端标签 + 权重 → 下一日 trade_hour 执行，唯一一次移位。

    接受 ``(days, D×N)``、以日期为索引的 pandas 宽表，或含 day/instrument/weight
    的 Polars 长表。day 可为 UTC 日期或该日 trade_hour；不接受已经移位的 available_at。
    """
    offset = hour(trade_hour)
    symbols = None
    if isinstance(w_daily, pl.DataFrame):
        if not {"day", "instrument", "weight"} <= set(w_daily.columns):
            raise ValueError("日频长表需要 day/instrument/weight")
        if w_daily.select(pl.struct("day", "instrument").is_duplicated().any()).item():
            raise ValueError("日频目标重复")
        wide = w_daily.pivot(on="instrument", index="day", values="weight").sort("day")
        days = wide["day"].to_list()
        symbols = tuple(c for c in wide.columns if c != "day")
        w = wide.select(symbols).to_numpy()
    elif isinstance(w_daily, tuple) and len(w_daily) == 2:
        days, w = w_daily
    elif hasattr(w_daily, "index") and hasattr(w_daily, "columns"):
        days = w_daily.index
        symbols = tuple(w_daily.columns)
        w = w_daily.to_numpy()
    else:
        raise ValueError("w_daily 需要带日期标签；用 (days, weights)")
    days = np.array([millis(d) for d in days], dtype=np.int64)
    if np.any((days % DAY != 0) & (days % DAY != offset)):
        raise ValueError("日线标签必须为日期或该日 trade_hour")
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or len(w) != len(days) or np.isinf(w).any() or not len(days):
        raise ValueError("日频目标应为非空 D×N，不能含 infinity")
    times = days // DAY * DAY + offset + DAY
    if np.any(np.diff(times) <= 0):
        raise ValueError("日频标签必须递增且不重复")
    return DailyTargets(times, w, symbols, trade_hour)


@dataclass(frozen=True)
class TopN:
    n: int
    by: str
    lookback: str
    min_age: str

    def select(self, ds, profiles=None, at=None, *, events=None):
        """返回币名；可用 select(ds, at=...)，或 select(frame, profiles, at)。"""
        if at is None:
            raise ValueError("选币必须给调仓时刻 at")
        at = millis(at)
        window = duration(self.lookback)
        age = duration(self.min_age)
        if not isinstance(ds, pl.DataFrame):
            inputs = ds.inputs()
            profiles = inputs["profiles"]
            events = inputs["events"]
            ds = ds.read(start=at - window, end=at)
        events = events or []
        allowed = []
        for p in profiles:
            name = p["instrument"]
            if (
                p["contract_type"] != "PERPETUAL"
                or p.get("underlying_type", "COIN") != "COIN"
                or p["quote_asset"] != "USDT"
                or p.get("margin_asset") != "USDT"
                or at - millis(p["listed_at"]) < age
            ):
                continue
            if any(
                e["kind"] in ("delist", "settlement", "delivery")
                and e["effective_at"] <= at
                for e in asof(events, at, instrument=name)
            ):
                continue
            last = (
                ds.filter(
                    (pl.col("instrument") == name) & (pl.col("time") + HOUR <= at)
                )
                .sort("time")
                .tail(1)
            )
            if (
                last.height
                and last["time"][0] == at - HOUR
                and last["lifecycle"][0] == "active"
                and last["tradable"][0]
            ):
                allowed.append(name)
        return (
            ds.filter(
                pl.col("instrument").is_in(allowed)
                & (pl.col("time") >= at - window)
                & (pl.col("time") + HOUR <= at)
                & (pl.col("lifecycle") == "active")
            )
            .group_by("coin")
            .agg(pl.col(self.by).sum())
            .sort([self.by, "coin"], descending=[True, False])
            .head(self.n)["coin"]
            .to_list()
        )


def top_n(n, by="quote_volume", lookback="30D", min_age="30D"):
    if (
        not isinstance(n, int)
        or n <= 0
        or by != "quote_volume"
        or duration(lookback) <= 0
    ):
        raise ValueError("top_n 需要正整数 n、quote_volume 与正时间窗口")
    duration(min_age)
    return TopN(n, by, lookback, min_age)


def roll_time(expiry, *, roll_before="3D", trade_hour=0):
    offset = hour(trade_hour)
    cutoff = millis(expiry) - duration(roll_before)
    return (cutoff - offset) // DAY * DAY + offset


def quarter_chain(profiles, times, *, roll_before="3D", trade_hour=0):
    """每个季度币链的列号；只向更晚到期日换。输入档案需有 delivery_at。"""
    times = np.asarray(times, dtype=np.int64)
    groups = {}
    for j, p in enumerate(profiles):
        if p["contract_type"] in ("CURRENT_QUARTER", "NEXT_QUARTER"):
            if not p.get("delivery_at"):
                raise ValueError(p["instrument"] + " 季度档案缺 delivery_at")
            groups.setdefault(p["coin"], []).append(j)
    result = np.full((len(times), len(groups)), -1, dtype=int)
    for k, indices in enumerate(groups.values()):
        indices.sort(key=lambda j: millis(profiles[j]["delivery_at"]))
        selected = -1
        for i, t in enumerate(times):
            eligible = [
                j
                for j in indices
                if millis(profiles[j]["listed_at"])
                <= t
                < millis(profiles[j]["delivery_at"])
            ]
            if selected < 0 and eligible:
                selected = eligible[0]
            if t % DAY == hour(trade_hour) and selected >= 0:
                for j in eligible:
                    if millis(profiles[j]["delivery_at"]) > millis(
                        profiles[selected]["delivery_at"]
                    ) and t >= roll_time(
                        profiles[selected]["delivery_at"],
                        roll_before=roll_before,
                        trade_hour=trade_hour,
                    ):
                        selected = j
                        break
            if selected >= 0:
                result[i, k] = selected
    return result
