"""期货的公开读取接口：``ashare.data.open(root).futures()``。

最短例子::

    import ashare.data as ad

    fut = ad.open("/path/to/root").futures()
    fut.contracts(products=["SC"]).collect()               # 合约表（统一字段）
    p = fut.panel(["settlement", "volume"], products=["SC"], start="2024-01-02")
    p.dates, p.symbols, p["settlement"]                    # T×N 只读数组
    sc = fut.continuous("SC", rule="open_interest", adjust="ratio")
    rolls = fut.roll_schedule("SC", rule="open_interest")  # 换月日程表
"""

import json

import numpy as np
import polars as pl

from .futures_schema import PANEL_FIELDS
from .reader import Panel
from .sync import compact


class FuturesData:
    def __init__(self, root):
        current = root / "futures_panel/current.json"
        if not current.exists():
            raise FileNotFoundError(
                "期货面板尚未构建；请运行 python -m ashare.data sync-futures，然后 build-futures"
            )
        self._path = current.parent / json.loads(current.read_text())["version"]
        self._meta = json.loads((self._path / "metadata.json").read_text())
        self._dates = np.array(self._meta["dates"], dtype="U8")
        self._dates.flags.writeable = False
        self._owner = {
            symbol: key
            for key, block in self._meta["blocks"].items()
            for symbol in block["symbols"]
        }

    @property
    def version(self):
        return self._meta["version"]

    @property
    def source(self):
        return self._meta["source"]

    def fields(self):
        return tuple(PANEL_FIELDS)

    def field(self, name):
        if name not in PANEL_FIELDS:
            raise KeyError(f"期货字段 {name!r} 不存在；可用：{', '.join(PANEL_FIELDS)}")
        return dict(name=name, **self._meta["fields"][name], source=self.source)

    def products(self):
        """(交易所, 品种, 合约数) 列表。"""
        return pl.DataFrame(
            [
                dict(
                    exchange=b["exchange"],
                    product=b["product"],
                    contracts=len(b["symbols"]),
                )
                for b in self._meta["blocks"].values()
            ]
        ).sort("exchange", "product")

    def _filter(
        self, frame, products=None, symbols=None, start=None, end=None, day="date"
    ):
        if products is not None:
            frame = frame.filter(pl.col("product").is_in(list(products)))
        if symbols is not None:
            frame = frame.filter(pl.col("symbol").is_in(list(symbols)))
        if start:
            frame = frame.filter(pl.col(day) >= compact(start))
        if end:
            frame = frame.filter(pl.col(day) <= compact(end))
        return frame

    def contracts(self, products=None, exchange=None):
        frame = pl.scan_parquet(self._path / "contracts.parquet")
        if exchange is not None:
            frame = frame.filter(pl.col("exchange") == exchange)
        return self._filter(frame, products)

    def calendar(self, exchange=None):
        frame = pl.scan_parquet(self._path / "calendar.parquet")
        return (
            frame if exchange is None else frame.filter(pl.col("exchange") == exchange)
        )

    def table(self, name, start=None, end=None, symbols=None, products=None):
        """长表：daily（行情）、params（保证金和费用）、mapping（来源的主力/连续映射）。"""
        if name not in ("daily", "params", "mapping"):
            raise KeyError(f"期货长表 {name!r} 不存在；可用：daily, params, mapping")
        path = self._path / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"来源 {self.source} 没有提供 {name}")
        frame = pl.scan_parquet(path)
        if products is not None:
            frame = frame.filter(
                pl.col("symbol").str.extract(r"^([A-Z_]+)\d").is_in(list(products))
            )
        return self._filter(frame, None, symbols, start, end)

    def symbols(self, products=None):
        blocks = [
            b
            for b in self._meta["blocks"].values()
            if products is None or b["product"] in products
        ]
        return [s for b in blocks for s in b["symbols"]]

    def panel(self, fields, products=None, symbols=None, start=None, end=None):
        """返回 ``Panel``：dates × symbols 的只读数组（内存，按品种块拼出）。

        给 products 取这些品种的全部合约（按最后交易日排序）；给 symbols 按给定顺序。
        数值字段缺失为 NaN，布尔字段缺失为 False。
        """
        if isinstance(fields, str):
            fields = [fields]
        for name in fields:
            self.field(name)
        if symbols is None:
            chosen = self.symbols(products)
            if products is not None and not chosen:
                raise KeyError(
                    f"没有品种 {list(products)}；可用：{self.products()['product'].to_list()}"
                )
        else:
            chosen = list(symbols)
            unknown = [s for s in chosen if s not in self._owner]
            if unknown:
                raise KeyError(f"未知合约 {unknown[:10]}")
            if len(set(chosen)) != len(chosen):
                raise ValueError("symbols 不允许重复")
        dates = self._dates
        first = compact(start) if start else dates[0]
        last = compact(end) if end else dates[-1]
        if first > last:
            raise ValueError("start 必须不晚于 end")
        lo = int(np.searchsorted(dates, first))
        hi = int(np.searchsorted(dates, last, side="right"))
        rows = dates[lo:hi]
        position = {s: j for j, s in enumerate(chosen)}
        arrays = {
            name: (
                np.zeros((len(rows), len(chosen)), dtype=bool)
                if self._meta["fields"][name]["dtype"] == "bool"
                else np.full((len(rows), len(chosen)), np.nan)
            )
            for name in fields
        }
        by_block = {}
        for symbol in chosen:
            by_block.setdefault(self._owner[symbol], []).append(symbol)
        for key, members in by_block.items():
            block = self._meta["blocks"][key]
            local = {s: j for j, s in enumerate(block["symbols"])}
            sources = {
                name: np.load(
                    self._path / "blocks" / key / f"{name}.npy", mmap_mode="r"
                )
                for name in fields
            }
            for symbol in members:
                j = local[symbol]
                start, length, offset = (
                    block["starts"][j],
                    block["lengths"][j],
                    block["offsets"][j],
                )
                a, b = max(lo, start), min(hi, start + length)
                if a >= b:
                    continue
                column = position[symbol]
                for name in fields:
                    arrays[name][a - lo : b - lo, column] = sources[name][
                        offset + a - start : offset + b - start
                    ]
        for array in arrays.values():
            array.flags.writeable = False
        return Panel(rows.copy(), np.array(chosen), arrays, self.version)

    def source_continuous(self):
        """来源自带的主力连续日线（Tushare 的 ``RB.SHF`` 等），只用于对账。"""
        path = self._path / "source_continuous.parquet"
        if not path.exists():
            raise FileNotFoundError(f"来源 {self.source} 没有自带连续合约")
        return pl.scan_parquet(path)

    def continuous(self, product, **kwargs):
        """主力 / 次主力连续价格，参数见 ``futures_continuous.continuous``。"""
        from .futures_continuous import continuous

        return continuous(self, product, **kwargs)

    def roll_schedule(self, product, **kwargs):
        """换月日程表，参数见 ``futures_continuous.roll_schedule``。"""
        from .futures_continuous import roll_schedule

        return roll_schedule(self, product, **kwargs)


__all__ = ["FuturesData"]
