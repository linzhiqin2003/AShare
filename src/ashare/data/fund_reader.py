"""场内基金与可转债的读取入口：``ashare.data.open(root).funds()``。

最短例子::

    import ashare.data as ad

    f = ad.open("/path/to/root").funds()
    p = f.panel("fund", ["close", "adj_factor"], start="2020-01-02", symbols=["510300.SH"])
    p.dates, p.symbols, p["close"]            # T×N 只读数组
    f.securities("cb").collect()              # 可转债证券表（正股、到期日、初始转股价……）
    f.table("cb", "calls").collect()          # 强赎 / 到期赎回公告
    f.field("cb", "conversion_premium")       # 单位与时点
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from .fund_schema import KINDS
from .reader import Panel
from .sync import compact


class FundData:
    """只读；面板按 kind（``"fund"`` 场内基金、``"cb"`` 可转债）分开存储。"""

    def __init__(self, root):
        self._root = Path(root)
        self._meta = {}
        self._path = {}
        for kind in KINDS:
            current = self._root / "fund_panel" / kind / "current.json"
            if current.exists():
                path = current.parent / json.loads(current.read_text())["version"]
                self._path[kind] = path
                self._meta[kind] = json.loads((path / "metadata.json").read_text())

    def _require(self, kind):
        if kind not in KINDS:
            raise KeyError(f"kind 只能是 {sorted(KINDS)}")
        if kind not in self._meta:
            raise FileNotFoundError(
                f"{kind} 面板尚未构建；请运行 python -m ashare.data sync-funds 与 build-funds"
            )
        return self._meta[kind]

    def version(self, kind):
        return self._require(kind)["version"]

    def fields(self, kind):
        return tuple(KINDS[kind])

    def field(self, kind, name):
        """字段说明：单位、类型、时点。"""
        if kind not in KINDS or name not in KINDS[kind]:
            raise KeyError(
                f"{kind} 没有字段 {name!r}；可用：{', '.join(KINDS.get(kind, {}))}"
            )
        spec = KINDS[kind][name]
        return dict(
            name=name,
            unit=spec.unit,
            dtype=spec.dtype,
            known_at=spec.known_at,
            source="Tushare Pro",
        )

    def diagnostics(self, kind):
        return self._require(kind)["diagnostics"]

    def panel(self, kind, fields, start=None, end=None, symbols=None):
        meta = self._require(kind)
        if isinstance(fields, str):
            fields = [fields]
        for name in fields:
            self.field(kind, name)
        dates = np.array(meta["dates"], dtype="U8")
        codes = np.array(meta["symbols"])
        first = compact(start) if start else dates[0]
        last = compact(end) if end else dates[-1]
        if first > last:
            raise ValueError("start 必须不晚于 end")
        if first < dates[0] or last > dates[-1]:
            raise ValueError(
                f"请求 {first}..{last} 超出 {kind} 面板 {dates[0]}..{dates[-1]}"
            )
        rows = slice(
            np.searchsorted(dates, first), np.searchsorted(dates, last, side="right")
        )
        requested = list(codes) if symbols is None else list(symbols)
        unknown = sorted(set(requested) - set(codes))
        if unknown:
            raise KeyError(f"{kind} 面板里没有 {unknown[:10]}")
        index = [int(np.searchsorted(codes, c)) for c in requested]
        if len(set(index)) != len(index):
            raise ValueError("symbols 不允许重复")
        arrays, temporary = {}, None
        for name in fields:
            full = np.load(self._path[kind] / f"{name}.npy", mmap_mode="r")
            if symbols is None:
                arrays[name] = full[rows]
            else:
                if temporary is None:
                    temporary = tempfile.TemporaryDirectory(prefix="ashare-funds-")
                path = Path(temporary.name) / f"{name}.npy"
                np.save(path, np.ascontiguousarray(full[rows][:, index]))
                arrays[name] = np.load(path, mmap_mode="r")
        return Panel(
            dates[rows], np.array(requested), arrays, meta["version"], temporary
        )

    def securities(self, kind):
        self._require(kind)
        return pl.scan_parquet(self._path[kind] / "securities.parquet")

    def table(self, kind, name):
        meta = self._require(kind)
        if name not in meta["tables"]:
            raise KeyError(f"{kind} 没有表 {name!r}；可用：{meta['tables']}")
        return pl.scan_parquet(self._path[kind] / f"table_{name}.parquet")

    def dates(self, kind):
        return np.array(self._require(kind)["dates"], dtype="U8")


__all__ = ["FundData"]
