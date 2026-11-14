"""底部状态栏的收盘行情：主要指数、涨跌家数、申万一级行业涨跌，都取本地数据最近一个交易日。

只读本地面板与指数日线；某一项当天没有数据就不返回该项。按面板版本和清单签名缓存。
"""

import logging
import threading
import time

import numpy as np
from fastapi import APIRouter

LOG = logging.getLogger("ashare.platform")
# 状态栏显示的指数，按此顺序。
TAPE = (
    ("000001.SH", "上证指数"),
    ("399001.SZ", "深证成指"),
    ("399006.SZ", "创业板指"),
    ("000300.SH", "沪深300"),
)


def _round(value, digits=6):
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def index_quotes(frame, day):
    """``frame``：某个指数的日线（date、close，可有 previous_close），返回 ``day`` 的点位与涨跌幅。"""
    frame = frame.sort("date").unique("date", keep="last", maintain_order=True)
    dates = frame.get_column("date").to_list()
    if not dates or dates[-1] != day:
        return None
    close = frame.get_column("close").to_numpy().astype(float)
    previous = np.nan
    if "previous_close" in frame.columns:
        previous = float(frame.get_column("previous_close").to_numpy()[-1] or np.nan)
    if not np.isfinite(previous) or previous <= 0:
        previous = close[-2] if len(close) > 1 else np.nan
    if not np.isfinite(close[-1]):
        return None
    change = (
        close[-1] / previous - 1 if np.isfinite(previous) and previous > 0 else None
    )
    return dict(
        close=_round(close[-1], 4),
        change=_round(change) if change is not None else None,
    )


def breadth(close, previous, quoted):
    """最近一天有行情的股票里，收盘高于、低于前收的家数。"""
    valid = quoted & np.isfinite(close) & np.isfinite(previous) & (previous > 0)
    return dict(
        up=int((valid & (close > previous)).sum()),
        down=int((valid & (close < previous)).sum()),
        flat=int((valid & (close == previous)).sum()),
    )


def industry_moves(codes, close, previous, quoted, weights, names):
    """按申万一级行业算当天涨跌：成分股涨跌幅按流通市值加权（没有市值时等权），从高到低排。"""
    valid = (
        quoted
        & np.isfinite(close)
        & np.isfinite(previous)
        & (previous > 0)
        & (codes != 0)
    )
    change = np.where(valid, close / np.where(previous > 0, previous, 1) - 1, np.nan)
    if weights is None:
        weights = np.ones_like(close, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    rows = []
    for code in np.unique(codes[valid]):
        name = names.get(int(code))
        if name is None:
            continue
        members = valid & (codes == code)
        weight = weights[members]
        values = change[members]
        total = weight.sum()
        value = (values * weight).sum() / total if total > 0 else values.mean()
        rows.append(dict(name=name, change=_round(value), count=int(members.sum())))
    rows.sort(key=lambda row: -(row["change"] if row["change"] is not None else -1))
    return rows


class Market:
    def __init__(self, root=None, *, data=None):
        self.root, self.data = root, data
        self.lock = threading.Lock()
        self.cache = (None, None)
        self.checked = 0.0

    def _open(self):
        import ashare.data

        return self.data or ashare.data.open(self.root)

    def _signature(self):
        """面板版本与清单的文件签名；不打开面板，状态栏轮询时足够便宜。"""
        if self.data is not None:
            return str(self.data.version)
        import ashare.data

        root = ashare.data.market_root(self.root)
        state = []
        for name in (
            "panel/current.json",
            "manifest.sqlite3",
            "manifest.sqlite3-wal",
            "manifest.json",
        ):
            try:
                stat = (root / name).stat()
                state.append((name, stat.st_mtime_ns, stat.st_size))
            except OSError:
                state.append((name, None))
        return tuple(state)

    def summary(self):
        with self.lock:
            cached, signature = self.cache
            if cached is not None and time.monotonic() - self.checked < 5:
                return cached
            self.checked = time.monotonic()
            current = self._signature()
            if cached is not None and current == signature:
                return cached
            started = time.perf_counter()
            try:
                result = self._build(self._open())
            except (FileNotFoundError, ValueError, RuntimeError, OSError):
                LOG.warning("状态栏行情不可用", exc_info=True)
                result = dict(date=None, indexes=[], breadth=None, industries=[])
            self.cache = (result, current)
            LOG.info("状态栏行情重建 elapsed=%.3fs", time.perf_counter() - started)
            return result

    def _build(self, data):
        import polars as pl

        result = dict(date=None, indexes=[], breadth=None, industries=[])
        try:
            dates = data.panel([]).dates
        except (FileNotFoundError, ValueError, RuntimeError):
            return result
        if not len(dates):
            return result
        day = str(dates[-1])
        result["date"] = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        start = str(dates[max(0, len(dates) - 15)])
        for symbol, name in TAPE:
            try:
                frame = (
                    data.table("index_daily", start=start, symbols=[symbol])
                    .collect()
                    .filter(pl.col("symbol") == symbol)
                )
            except (KeyError, ValueError, OSError, pl.exceptions.PolarsError):
                continue
            quote = index_quotes(frame, day)
            if quote is not None:
                result["indexes"].append(dict(symbol=symbol, name=name, **quote))
        latest = {}
        for name in (
            "close",
            "previous_close",
            "has_quote",
            "industry",
            "float_market_cap",
        ):
            try:
                latest[name] = np.asarray(data.panel([name], start=day)[name][-1])
            except (FileNotFoundError, KeyError):
                continue
        if "close" not in latest or "previous_close" not in latest:
            return result
        close = latest["close"].astype(float)
        previous = latest["previous_close"].astype(float)
        quoted = (
            latest["has_quote"].astype(bool)
            if "has_quote" in latest
            else np.isfinite(close)
        )
        result["breadth"] = breadth(close, previous, quoted)
        if "industry" in latest:
            weights = latest.get("float_market_cap")
            result["industries"] = industry_moves(
                latest["industry"].astype(np.int64),
                close,
                previous,
                quoted,
                None if weights is None else weights.astype(float),
                _industry_names(data),
            )
        return result


def _industry_names(data):
    """申万一级行业代码（数字部分）→ 名称；SW2021 优先，旧代码用 SW2014 补。"""
    import polars as pl

    try:
        frame = data.table("index_classify").collect()
    except (KeyError, ValueError, OSError, pl.exceptions.PolarsError):
        return {}
    names = {}
    for preferred in ("SW2014", "SW2021"):
        for row in frame.filter(
            (pl.col("classification") == preferred) & (pl.col("level") == "L1")
        ).iter_rows(named=True):
            try:
                names[int(str(row["industry_code"]).split(".")[0])] = row[
                    "industry_name"
                ]
            except ValueError:
                continue
    return names


def routes(data_root=None):
    """``create_app`` 用一行 ``include_router`` 注册。"""
    market = Market(data_root)
    router = APIRouter()

    @router.get("/api/market")
    def summary():
        return market.summary()

    router.market = market
    return router
