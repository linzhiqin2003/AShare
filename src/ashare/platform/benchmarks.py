"""本地指数目录与展示指标；不修改回测设置或结果身份。"""

import hashlib
import threading
from functools import lru_cache

import numpy as np

from ashare.factor import groupbt

from .analysis import finite, packed

# 与数据目录的默认值一致；访问名单通过公开 Data.indexes，不反向依赖 catalog 内部模块。
DEFAULT_BENCHMARK = "000001.SH"


class Benchmarks:
    def __init__(self, root=None, *, data=None):
        self.root, self.data = root, data
        self.items = None
        self.lock = threading.RLock()
        self.daily = lru_cache(maxsize=24)(self._daily)

    def _load(self):
        if self.items is not None:
            return
        if self.data is None:
            import ashare.data

            self.data = ashare.data.open(self.root)
        # 数据层负责取主流名单与本地日线的交集，平台只发送控件需要的列。
        items = self.data.indexes().select("symbol", "name").to_dicts()
        self.items = sorted(items, key=lambda row: (row["name"], row["symbol"]))
        self.version = hashlib.sha256(str(self.data.version).encode()).hexdigest()[:16]

    def selection(self, symbol=None):
        with self.lock:
            self._load()
            requested = symbol or DEFAULT_BENCHMARK
            selected = next(
                (item for item in self.items if item["symbol"] == requested), None
            )
            if selected is None and self.items:
                selected = self.items[0]
            return dict(
                selected or {"symbol": None, "name": "无本地指数"},
                fallback=selected is not None and selected["symbol"] != requested,
                requested=requested,
            )

    def search(self, search="", selected=None):
        with self.lock:
            self._load()
            query = search.casefold()
            rows = [
                item
                for item in self.items
                if query in item["symbol"].casefold()
                or query in item["name"].casefold()
            ]
            return {
                "items": rows,
                "total": len(rows),
                "selected": self.selection(selected),
                "default": self.selection(),
            }

    def _daily(self, symbol, trade_at):
        with self.lock:
            self._load()
        frame = (
            self.data.table("index_daily", end="20251231", symbols=[symbol])
            .select("date", trade_at)
            .collect()
            .sort("date")
            .unique("date", keep="last", maintain_order=True)
        )
        dates = frame.get_column("date").to_numpy().astype(str)
        prices = frame.get_column(trade_at).to_numpy().astype(float)
        returns = np.full(len(prices), np.nan)
        if len(prices) > 1:
            np.divide(prices[1:], prices[:-1], out=returns[:-1], where=prices[:-1] > 0)
            returns[:-1] -= 1
        return dict(zip(dates, returns, strict=True))

    def comparison(self, store, runs, symbol=None):
        selected = self.selection(symbol)
        output = dict(selected, parts=[], relatives={})
        if not selected["symbol"]:
            return output
        for run in runs:
            if run["status"] != "complete":
                continue
            arrays = store.series(run["id"])
            daily = self.daily(selected["symbol"], run["settings"]["trade_at"])
            relative, parts = {}, []
            for part, _ in (("is", groupbt.IS), ("validation", groupbt.VALIDATION)):
                dates = arrays[f"{part}.dates"]
                baseline = np.array([daily.get(str(day), np.nan) for day in dates])
                top = arrays[f"{part}.groups"][0]
                keep = np.isfinite(top) & np.isfinite(baseline)
                excess = top[keep] - baseline[keep]
                volatility = np.std(excess, ddof=1) if len(excess) > 1 else np.nan
                relative[part] = {
                    "excess": finite(np.mean(excess) * 240) if len(excess) else None,
                    "information_ratio": finite(
                        np.mean(excess) / volatility * np.sqrt(240)
                    )
                    if volatility > 0
                    else None,
                    "n": int(keep.sum()),
                }
                parts.append(
                    {
                        "part": part,
                        "dates": dates.tolist(),
                        "series": {"benchmark": packed(baseline)},
                    }
                )
            output["relatives"][run["id"]] = relative
            if not output["parts"]:
                output["parts"] = parts
        return output
