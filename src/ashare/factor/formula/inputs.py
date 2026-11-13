"""公式型因子的取数：在数据层面板上补基准指数和行业组号，并给参考实现准备 DataFrame。

字段口径与 ``dsl`` 一致。这里只做两件事：

- ``PublicSource``：给求值器用的取数协议。数据层面板的字段原样透传，另加三个派生字段：
  ``benchmark_open``、``benchmark_close``（长度 T 的基准指数点位，按日期对齐，缺失为 NaN）
  和 ``industry_l1``（int64，申万一级编码，缺失为 −1）。
- ``reference_frames``：把同一份数据换算成参考实现要的 ``{名字: DataFrame}``，换算与
  ``dsl`` 里的表达式逐位相同（同样的 numpy 乘除）。
"""

from dataclasses import dataclass, field

import numpy as np

DEFAULT_BENCHMARK = "000300.SH"

# 参考实现和三套翻译用到的数据层字段。
PANEL_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "amount",
    "adj_factor",
    "previous_close",
    "total_market_cap",
    "is_listed",
    "listing_age",
    "industry",
)


@dataclass
class PublicSource:
    """求值器的取数协议实现；数组只读，不复制数据层的内存映射。"""

    dates: np.ndarray
    symbols: np.ndarray
    arrays: dict
    version: str
    benchmark: str = DEFAULT_BENCHMARK
    benchmark_open: np.ndarray = field(default=None, repr=False)
    benchmark_close: np.ndarray = field(default=None, repr=False)

    def __getitem__(self, name):
        if name == "benchmark_open":
            return _required(self.benchmark_open, name)
        if name == "benchmark_close":
            return _required(self.benchmark_close, name)
        if name == "industry_l1":
            # 数据层的行业编码是 uint32：先转 int64 再放 −1（uint32 里放不下负数）。
            codes = np.asarray(self.arrays["industry"]).astype(np.int64)
            return np.where(codes > 0, codes, -1)
        return self.arrays[name]

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)

    def tail(self, rows):
        """最后 rows 行的切片（核对时去掉预热段）。"""
        if rows > len(self.dates):
            raise ValueError("切片行数超过数据长度")
        cut = slice(len(self.dates) - rows, None)
        return PublicSource(
            self.dates[cut],
            self.symbols,
            {k: v[cut] for k, v in self.arrays.items()},
            self.version,
            self.benchmark,
            None if self.benchmark_open is None else self.benchmark_open[cut],
            None if self.benchmark_close is None else self.benchmark_close[cut],
        )


def _required(value, name):
    if value is None:
        raise KeyError(f"{name} 需要用 load_inputs(benchmark=...) 取基准指数")
    return value


def load_inputs(
    root=None,
    *,
    days=750,
    warmup=260,
    symbols=None,
    benchmark=DEFAULT_BENCHMARK,
    end=None,
):
    """从数据层取最近 ``days + warmup`` 个交易日（``end`` 为止，默认面板最后一天）。

    ``symbols`` 为 None 时取全部证券；也可以给整数 n，取前 n 个（按代码排序，只用于测试和
    快速试跑）或一个代码列表。返回 ``PublicSource``。
    """
    import polars as pl

    import ashare.data as ad

    data = ad.open(root=root)
    calendar = np.asarray(data.panel(["close"], end=end).dates)
    total = days + warmup
    if total > len(calendar):
        raise ValueError(f"面板只有 {len(calendar)} 个交易日，不够 {total}")
    start = calendar[-total]
    pick = None
    if isinstance(symbols, int):
        everything = np.asarray(data.panel(["close"], end=end).symbols)
        pick = list(everything[:symbols])
    elif symbols is not None:
        pick = list(symbols)
    panel = data.panel(list(PANEL_FIELDS), start=start, end=end, symbols=pick)
    dates = np.asarray(panel.dates)
    arrays = {name: panel[name] for name in PANEL_FIELDS}
    opens = closes = None
    if benchmark is not None:
        frame = (
            data.table(
                "index_daily",
                start=str(dates[0]),
                end=str(dates[-1]),
                symbols=[benchmark],
            )
            .filter(pl.col("symbol") == benchmark)
            .select("date", "open", "close")
            .collect()
        )
        if frame.height == 0:
            raise KeyError(f"数据层没有基准指数 {benchmark} 的日线")
        lookup = dict(
            zip(
                frame["date"].to_list(),
                zip(frame["open"].to_list(), frame["close"].to_list(), strict=True),
                strict=True,
            )
        )
        opens = np.array([lookup.get(d, (np.nan, np.nan))[0] for d in dates], float)
        closes = np.array([lookup.get(d, (np.nan, np.nan))[1] for d in dates], float)
    version = f"{panel.version}+public:{benchmark}:{dates[0]}"
    return PublicSource(
        dates, np.asarray(panel.symbols), arrays, version, benchmark, opens, closes
    )


def reference_frames(source):
    """参考实现的输入：``{名字: pandas.DataFrame}``（行为日期、列为证券），口径见文档。

    价格后复权（乘复权因子），成交量除以复权因子，成交额不复权，收益为
    ``close / previous_close − 1``（上市首日为 NaN）；``listed`` 为布尔；``industry`` 为浮点编码，缺失为 NaN；
    ``benchmark_open``、``benchmark_close`` 为按日期索引的 Series。
    """
    import pandas as pd

    def frame(values):
        return pd.DataFrame(
            np.array(values, dtype=float), index=source.dates, columns=source.symbols
        )

    a = {k: np.asarray(v) for k, v in source.arrays.items()}
    factor = a["adj_factor"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            name: frame(a[name] * factor) for name in ("open", "high", "low", "close")
        }
        out["vwap"] = frame(a["vwap"] * factor)
        out["volume"] = frame(a["volume"] / factor)
        out["amount"] = frame(a["amount"])
        # 上市首日（listing_age == 1）的 previous_close 是发行价：收益为 NaN（D-03）。
        first_day = np.asarray(a["listing_age"]) == 1
        out["returns"] = frame(
            np.where(first_day, np.nan, a["close"] / a["previous_close"] - 1)
        )
        out["cap"] = frame(a["total_market_cap"])
    out["listed"] = pd.DataFrame(
        a["is_listed"].astype(bool), index=source.dates, columns=source.symbols
    )
    industry = a["industry"].astype(float)
    out["industry"] = frame(np.where(industry > 0, industry, np.nan))
    for name in ("benchmark_open", "benchmark_close"):
        values = getattr(source, name)
        if values is not None:
            out[name] = pd.Series(np.asarray(values, float), index=source.dates)
    return out
