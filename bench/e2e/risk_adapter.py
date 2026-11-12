"""只供剖析用的适配：把真实面板接到 ashare.risk 的读取协议。

主干还没有正式接口，这里按 ashare.risk.data.FIELDS 的口径改名：申万一级行业
代码按升序映射到 0..30（0 即缺失记为 -1），无风险利率取 0（不提供该字段）。
上市日数由 model_series 自己按 is_listed 累计，与面板 listing_age 的口径另行核对。
"""

import numpy as np

import ashare.data
from ashare.risk import FundamentalConfig, MemoryPanel

NAMES = {
    "close": "close",
    "adj_factor": "adj_factor",
    "total_mv": "total_market_cap",
    "circ_mv": "float_market_cap",
    "turnover_rate": "turnover",
    "pb": "pb",
    "pe_ttm": "pe_ttm",
    "is_suspended": "is_suspended",
    "is_listed": "is_listed",
}


def industry_codes(raw):
    codes = np.unique(raw[raw > 0])
    lookup = np.full(int(codes.max()) + 1, -1, dtype=np.int64)
    lookup[codes] = np.arange(len(codes))
    return lookup[raw], len(codes)


def load(root, days=None, symbols=None):
    """最近 days 个交易日（None 为全历史）、前 symbols 只（None 为全部）。"""
    data = ashare.data.open(root)
    panel = data.panel([*NAMES.values(), "industry"])
    rows = slice(None if days is None else len(panel.dates) - days, None)
    cols = slice(None if symbols is None else symbols)
    fields = {}
    for target, name in NAMES.items():
        value = panel[name][rows, cols]
        fields[target] = (
            value.astype(bool)
            if target.startswith("is_")
            else np.asarray(value, dtype=np.float64)
        )
    industry, count = industry_codes(
        np.asarray(panel["industry"][rows, cols], dtype=np.int64)
    )
    fields["industry"] = industry
    source = MemoryPanel(
        panel.dates[rows], tuple(panel.symbols[cols]), panel.version, fields
    )
    return source, FundamentalConfig(industry_codes=tuple(range(count)))
