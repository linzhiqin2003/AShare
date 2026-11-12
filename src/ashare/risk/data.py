"""独立于真实数据层的最小读取协议及统一字段表。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

FIELDS = {
    "close": "未复权收盘价，元，收盘可知",
    "adj_factor": "当日可知的累计复权因子，不能用未来修订值",
    "total_mv": "总市值，元",
    "circ_mv": "流通市值，元",
    "turnover_rate": "日换手率，小数（1% 输入 0.01）",
    "pb": "市净率，倍；允许负值，零视为缺失",
    "pe_ttm": "滚动市盈率，倍；允许负值，零视为缺失",
    "industry": "申万一级行业整数编码，未知使用 -1",
    "is_suspended": "当天是否停牌，布尔",
    "is_listed": "当天是否上市，布尔；包含退市证券",
    "risk_free": "可选，无风险日收益，小数；缺省为零",
}


class PanelSource(Protocol):
    dates: np.ndarray
    symbols: tuple[str, ...]
    version: str

    def __getitem__(self, field: str) -> np.ndarray: ...


@dataclass
class MemoryPanel:
    dates: np.ndarray
    symbols: tuple[str, ...]
    version: str
    fields: dict[str, np.ndarray]

    def __post_init__(self):
        self.dates = np.asarray(self.dates).astype(str)
        self.symbols = tuple(map(str, self.symbols))
        if self.dates.ndim != 1 or np.any(self.dates[1:] <= self.dates[:-1]):
            raise ValueError("dates: 必须严格升序")
        if len(set(self.symbols)) != len(self.symbols) or not self.symbols:
            raise ValueError("symbols: 必须非空且唯一")
        self.fields = {k: np.array(v, copy=True) for k, v in self.fields.items()}
        for key, a in self.fields.items():
            if a.shape != (len(self.dates), len(self.symbols)):
                raise ValueError(f"{key}: 必须为 T×N")
            a.flags.writeable = False

    def __getitem__(self, field):
        return self.fields[field]


def adjusted_returns(source, end):
    """只读取截至 end 的前缀，不跨停牌缺价填收益。"""
    close = np.asarray(source["close"][: end + 1], dtype=float)
    adj = np.asarray(source["adj_factor"][: end + 1], dtype=float)
    price = close * adj
    r = np.full_like(price, np.nan)
    good = (
        (price[:-1] > 0)
        & (price[1:] > 0)
        & np.isfinite(price[:-1])
        & np.isfinite(price[1:])
    )
    r[1:] = (
        np.divide(
            price[1:], price[:-1], out=np.full_like(price[1:], np.nan), where=good
        )
        - 1
    )
    listed = np.asarray(source["is_listed"][: end + 1], dtype=bool)
    r[~listed] = np.nan
    r[1:][~listed[:-1]] = np.nan
    return r
