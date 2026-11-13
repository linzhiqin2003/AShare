"""市场无关的公开读取接口：统一字段、单位和只读数组。"""

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from .aliases import CanonicalStore
from .catalog import FIELDS, PUBLIC_TABLES, SYMBOL_ALIASES, TABLE_DATES
from .store import Store
from .sync import compact


class Panel:
    def __init__(self, dates, symbols, arrays, version, temporary=None):
        self._temporary = temporary
        self.dates, self.symbols = dates, symbols
        self.dates.flags.writeable = False
        self.symbols.flags.writeable = False
        self._arrays, self.version = arrays, version

    def __getitem__(self, name):
        return self._arrays[name]

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)


class Data:
    def __init__(self, root=None, market="cn"):
        if market in ("hk", "us"):
            # 港股、美股的面板由 OverseasData 读取（ashare.data.open(root, market="hk") 会自动分派）。
            raise ValueError(
                f"Data 只读 A 股；{market} 请用 ashare.data.open(root, market={market!r})"
            )
        self._store = Store(root, market, readonly=True).snapshot()
        current = self._store.root / "panel/current.json"
        self._path = None
        self._meta = None
        if current.exists():
            self._path = current.parent / json.loads(current.read_text())["version"]
            self._meta = json.loads((self._path / "metadata.json").read_text())
        self._store = CanonicalStore(
            self._store, self._meta.get("aliases", []) if self._meta else []
        )

    def _require(self):
        if self._meta is None:
            raise FileNotFoundError(
                "面板尚未构建；请运行 python -m ashare.data build-panel"
            )

        if not (self._path / "metadata.json").exists():
            raise RuntimeError(
                "面板已更新且旧代已清理；请重新调用 ashare.data.open() 获取新句柄。已经取得的数组仍可读取。"
            )

    @property
    def version(self):
        return self._meta["version"] if self._meta else self._store.version

    def fields(self):
        return tuple(FIELDS)

    def field(self, name):
        if name not in FIELDS:
            raise KeyError(
                f"字段 {name!r} 不存在；可用字段：{', '.join(self.fields())}；采集后请运行 build-panel"
            )
        spec = FIELDS[name]
        return {
            "name": name,
            "unit": spec.unit,
            "dtype": spec.dtype,
            "known_at": self._meta["fields"]
            .get(name, {})
            .get("known_at", spec.known_at)
            if self._meta
            else spec.known_at,
            "source": "Tushare Pro",
            "historical_revisions": True,
            "coverage": self._meta["fields"].get(name, {}).get("coverage")
            if self._meta
            else None,
        }

    def panel(self, fields, start=None, end=None, symbols=None):
        self._require()
        if isinstance(fields, str):
            fields = [fields]
        for name in fields:
            self.field(name)
            if name not in self._meta["fields"]:
                raise FileNotFoundError(
                    f"当前面板未构建字段 {name}；请运行 python -m ashare.data build-panel"
                )
        dates = np.array(self._meta["dates"], dtype="U8")
        codes = np.array(self._meta["symbols"])
        first, last = (
            compact(start) if start else dates[0],
            compact(end) if end else dates[-1],
        )
        if first < dates[0] or last > dates[-1]:
            raise ValueError(
                f"请求 {first}..{last} 超出面板 {dates[0]}..{dates[-1]}；请运行 sync --start {first} --end {last}，然后 build-panel"
            )
        if first > last:
            raise ValueError("start 必须不晚于 end")
        row_slice = slice(
            np.searchsorted(dates, first), np.searchsorted(dates, last, side="right")
        )
        requested = list(symbols) if symbols is not None else list(codes)
        unknown = set(requested) - set(codes)
        if unknown:
            raise KeyError(
                f"缺少证券 {sorted(unknown)}；请运行 sync --datasets stock_basic 并 build-panel"
            )
        indices = [int(np.searchsorted(codes, code)) for code in requested]
        if len(indices) != len(set(indices)):
            raise ValueError("symbols 不允许重复")
        arrays = {}
        temporary = None
        for name in fields:
            full = np.load(self._path / f"{name}.npy", mmap_mode="r")
            if requested == list(codes):
                arrays[name] = full[row_slice]
            else:
                # 任意列投影需要复制；只复制所请求日期，缓存位于系统临时目录。
                if temporary is None:
                    temporary = tempfile.TemporaryDirectory(prefix="ashare-projection-")
                path = Path(temporary.name) / f"{name}.npy"
                np.save(path, np.ascontiguousarray(full[row_slice][:, indices]))
                arrays[name] = np.load(path, mmap_mode="r")

        return Panel(
            dates[row_slice], np.array(requested), arrays, self.version, temporary
        )

    def futures(self):
        """境内期货（全部交易所）的读取入口，见 ``FuturesData``；不依赖股票面板。"""
        from .futures_reader import FuturesData

        return FuturesData(self._store.root)

    def calendar(self):
        self._require()
        return (
            pl.scan_parquet(self._path / "calendar.parquet")
            .select(
                pl.col("cal_date").alias("date"),
                pl.col("is_open").cast(pl.Boolean).alias("is_open"),
            )
            .sort("date")
        )

    def securities(self):
        self._require()
        return pl.scan_parquet(self._path / "securities.parquet").select(
            pl.col("ts_code").alias("symbol"),
            "name",
            pl.col("market").alias("board"),
            "exchange",
            pl.col("list_date").alias("listing_date"),
            pl.col("delist_date").alias("delisting_date"),
        )

    def table(self, name, start=None, end=None, symbols=None):
        if name == "symbol_aliases":
            return pl.DataFrame(SYMBOL_ALIASES).lazy()
        if name not in PUBLIC_TABLES:
            raise KeyError(f"长表 {name!r} 不存在；可用：{list(PUBLIC_TABLES)}")
        frame = self._store.scan(name)
        schema = frame.collect_schema()
        columns = []
        for raw, (public, scale) in PUBLIC_TABLES[name].items():
            if raw in schema:
                expr = pl.col(raw) * scale if scale != 1 else pl.col(raw)
                columns.append(expr.alias(public))
        frame = frame.select(columns)
        day = TABLE_DATES.get(name, "date")
        if (start or end) and day is None:
            raise ValueError(f"{name} 是静态分类表，不支持日期过滤")
        if start:
            frame = frame.filter(pl.col(day) >= compact(start))
        if end:
            frame = frame.filter(pl.col(day) <= compact(end))
        if symbols is not None:
            frame = frame.filter(pl.col("symbol").is_in(symbols))
        return frame
