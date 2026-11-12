"""数据层面板 → 风险模型输入：字段改名、类型、行业重编号和申万行业名。

风险模型的读取协议（`ashare.risk.FIELDS`）有自己的字段名；这里是唯一的转接处：

| 风险模型 | 面板 |
|---|---|
| total_mv | total_market_cap |
| circ_mv | float_market_cap |
| turnover_rate | turnover |
| 其余（close、adj_factor、pb、pe_ttm、is_suspended、is_listed） | 同名 |

行业：面板的 `industry` 是申万一级指数代码（如 801780），0 或 NaN 为缺失。所选区间里
出现过的代码按升序编成 0..K-1（缺失为 -1），行业因子用申万一级名称命名
（`industry_银行`）。2021 年申万改版前后的代码混在面板里：同一代码两版都有时用
SW2021 的名称（如 801030 叫「基础化工」），只在旧版出现的用旧版名称（801020 采掘）。
无风险利率不提供（模型按零处理）。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl

from .data import MemoryPanel
from .fundamental import FundamentalConfig

PANEL_FIELDS = {
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
# 名称优先级：新版在前。
CLASSIFICATIONS = ("SW2021", "SW2014")


def industry_codes(raw):
    """申万代码面板 → (0..K-1 编号面板，缺失 -1；升序的原始代码)。"""
    raw = np.asarray(raw, dtype=np.float64)
    present = np.isfinite(raw) & (raw > 0)
    values = np.where(present, raw, 0).astype(np.int64)
    codes = np.unique(values[present])
    out = np.full(raw.shape, -1, dtype=np.int64)
    out[present] = np.searchsorted(codes, values[present])
    return out, codes


def industry_names(data, codes):
    """申万一级代码 → 名称；data 读不到分类表或缺某个代码时，用代码本身作名称。"""
    labels = {}
    try:
        table = (
            data.table("index_classify")
            .filter(pl.col("level") == "L1")
            .select("industry_code", "industry_name", "classification")
            .collect()
        )
        for scheme in reversed(CLASSIFICATIONS):
            for code, name, cls in table.iter_rows():
                if cls == scheme:
                    labels[int(str(code).split(".")[0])] = name
    except (AttributeError, KeyError, FileNotFoundError, pl.exceptions.PolarsError):
        pass
    names = [labels.get(int(code), str(int(code))) for code in codes]
    # 两个代码同名时（极少见）补上代码区分，保证因子名唯一。
    seen = {n: names.count(n) for n in names}
    return tuple(
        n if seen[n] == 1 else f"{n}_{int(c)}"
        for n, c in zip(names, codes, strict=True)
    )


def risk_inputs(panel, *, names=None, config=None, **overrides):
    """已读好的面板 → (MemoryPanel, FundamentalConfig)。

    panel：有 dates、symbols、version，按字段名取 T×N 数组（`ashare.data.Panel`）。
    names：行业代码 → 名称的函数（输入升序代码，返回同长名称）；None 时用代码作名称。
    config：基础配置，行业编码和名称会被替换；overrides 是其他 FundamentalConfig 参数。
    """
    fields = {}
    for target, name in PANEL_FIELDS.items():
        value = panel[name]
        fields[target] = (
            np.asarray(value).astype(bool)
            if target.startswith("is_")
            else np.asarray(value, dtype=np.float64)
        )
    fields["industry"], codes = industry_codes(panel["industry"])
    labels = tuple(str(int(c)) for c in codes) if names is None else tuple(names(codes))
    source = MemoryPanel(panel.dates, tuple(panel.symbols), panel.version, fields)
    base = FundamentalConfig() if config is None else config
    return source, replace(
        base,
        industry_codes=tuple(range(len(codes))),
        industry_names=labels,
        **overrides,
    )


def from_panel(
    data, start=None, end=None, *, symbols=None, panel=None, config=None, **overrides
):
    """一步把数据层接到风险模型：`source, config = from_panel(ad.open(), start=...)`。

    返回的 source、config 直接交给 `model_series(source, config=config)`。
    panel：已经读好的面板（含 PANEL_FIELDS 的面板字段和 industry）；给了就不再读，
    start/end/symbols 必须为空。config、overrides 同 `risk_inputs`。
    """
    if panel is None:
        panel = data.panel(
            [*PANEL_FIELDS.values(), "industry"], start=start, end=end, symbols=symbols
        )
    elif start is not None or end is not None or symbols is not None:
        raise ValueError("给了 panel 时 start/end/symbols 不起作用，请在读面板时指定")
    return risk_inputs(
        panel,
        names=lambda codes: industry_names(data, codes),
        config=config,
        **overrides,
    )
