"""港股、美股的公开读取接口：``ashare.data.open(root, market="hk" | "us")``。

最短例子::

    import ashare.data as ad

    hk = ad.open("/path/to/root", market="hk")                  # 近 3 年全部港股
    p = hk.panel(["close", "adj_factor", "connect_eligible"], start="2025-01-02")
    p.dates, p.symbols, p["close"]                               # T×N 只读数组（港元）
    hk.securities().collect()                                    # 证券表（含当前每手）
    hk.fx("USDHKD").collect()                                    # 汇率日线

    us = ad.open("/path/to/root", market="us", universe="sample")  # 2010 年起的 200 只样本
    us.panel(["close", "split"], symbols=["AAPL"])["split"]
"""

import json

import numpy as np
import polars as pl

from .overseas_fields import fields_for
from .overseas_sync import spec_for
from .reader import Panel
from .store import Store
from .sync import compact


class OverseasData:
    def __init__(self, root=None, market="hk", universe="all"):
        spec_for(market)
        if universe not in ("all", "sample"):
            raise ValueError(
                "universe 只能是 all（按交易日的全部证券）或 sample（2010 年起的样本）"
            )
        self.market, self.universe = market, universe
        self._store = Store(root, market, readonly=True).snapshot()
        current = self._store.root / f"panel_{universe}" / "current.json"
        self._path = self._meta = None
        if current.exists():
            self._path = current.parent / json.loads(current.read_text())["version"]
            self._meta = json.loads((self._path / "metadata.json").read_text())

    def _require(self):
        if self._meta is None:
            raise FileNotFoundError(
                f"{self.market} 面板（{self.universe}）尚未构建；请运行 "
                f"python -m ashare.data --market {self.market} build-overseas --universe {self.universe}"
            )
        if not (self._path / "metadata.json").exists():
            raise RuntimeError("面板已更新且旧代已清理；请重新调用 ashare.data.open()")

    @property
    def version(self):
        self._require()
        return self._meta["version"]

    @property
    def currency(self):
        return spec_for(self.market)["currency"]

    def fields(self):
        return tuple(fields_for(self.market))

    def field(self, name):
        specs = fields_for(self.market)
        if name not in specs:
            raise KeyError(f"{self.market} 没有字段 {name!r}；可用：{', '.join(specs)}")
        f = specs[name]
        return dict(
            name=name,
            unit=f.unit,
            dtype=f.dtype,
            known_at=f.known_at,
            source="Tushare Pro",
            currency=self.currency,
        )

    def diagnostics(self):
        self._require()
        return self._meta["diagnostics"]

    def panel(self, fields, start=None, end=None, symbols=None):
        """返回 ``Panel``：dates × symbols 的只读数组；数值缺失为 NaN，布尔缺失为 False。"""
        self._require()
        if isinstance(fields, str):
            fields = [fields]
        for name in fields:
            self.field(name)
        dates = np.array(self._meta["dates"], dtype="U8")
        codes = np.array(self._meta["symbols"])
        first = compact(start) if start else dates[0]
        last = compact(end) if end else dates[-1]
        if first > last:
            raise ValueError("start 必须不晚于 end")
        if first < dates[0] or last > dates[-1]:
            raise ValueError(f"请求 {first}..{last} 超出面板 {dates[0]}..{dates[-1]}")
        rows = slice(
            int(np.searchsorted(dates, first)),
            int(np.searchsorted(dates, last, side="right")),
        )
        if symbols is None:
            columns = None
            chosen = codes
        else:
            chosen = np.array(list(symbols))
            unknown = sorted(set(chosen.tolist()) - set(codes.tolist()))
            if unknown:
                raise KeyError(f"面板里没有证券 {unknown[:10]}")
            if len(set(chosen.tolist())) != len(chosen):
                raise ValueError("symbols 不允许重复")
            columns = np.searchsorted(codes, chosen)
        arrays = {}
        for name in fields:
            full = np.load(self._path / f"{name}.npy", mmap_mode="r")
            if columns is None:
                arrays[name] = full[rows]
            else:
                out = np.ascontiguousarray(full[rows][:, columns])
                out.flags.writeable = False
                arrays[name] = out
        return Panel(dates[rows].copy(), chosen.copy(), arrays, self.version)

    def calendar(self):
        self._require()
        return (
            pl.scan_parquet(self._path / "calendar.parquet")
            .select(
                pl.col("cal_date").alias("date"),
                (pl.col("is_open").cast(pl.Int64) == 1).alias("is_open"),
            )
            .sort("date")
        )

    def securities(self):
        """证券表：symbol、名称、上市退市日、币种、首末行情日；港股另有当前每手 board_lot_current。"""
        self._require()
        return pl.scan_parquet(self._path / "securities.parquet").rename(
            {"ts_code": "symbol"}
        )

    def fx(self, pair):
        """汇率日线（统一字段）。

        - FXCM 报价（``USDHKD``、``USDCNH``）：date 为格林尼治日期，bid/ask 收盘及其中间价 ``close``；
        - 人民币中间价（``CNYHKD``、``CNYUSD``）：``rate`` 是 1 单位外币兑人民币（源值 ÷ 100）。
        """
        m = spec_for(self.market)
        if pair + ".FXCM" in m["fx_daily"]:
            frame = self._store.scan("overseas_fx_daily").filter(
                pl.col("ts_code") == pair + ".FXCM"
            )
            return frame.select(
                pl.col("trade_date").alias("date"),
                pl.lit(pair).alias("pair"),
                ((pl.col("bid_close") + pl.col("ask_close")) / 2).alias("close"),
                pl.col("bid_close"),
                pl.col("ask_close"),
                pl.col("tick_qty").alias("ticks"),
            ).sort("date")
        if pair in m["fx_midrate"]:
            frame = self._store.scan("overseas_fx_midrate").filter(
                pl.col("ts_code") == pair
            )
            return frame.select(
                pl.col("trade_date").alias("date"),
                pl.lit(pair).alias("pair"),
                (pl.col("mid_rate") / 100).alias("rate"),
            ).sort("date")
        pairs = [p.split(".")[0] for p in m["fx_daily"]] + list(m["fx_midrate"])
        raise KeyError(f"{self.market} 没有汇率 {pair!r}；可用：{pairs}")


__all__ = ["OverseasData"]
