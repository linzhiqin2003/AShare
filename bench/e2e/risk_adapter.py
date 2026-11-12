"""剖析用的取数：真实面板 → ashare.risk 的读取协议，走正式入口 `ashare.risk.from_panel`。

只多做一件事：按「最近 days 天、前 symbols 只」在内存映射上切片（不复制），
保持剖析的取数开销与改动前相同。
"""

import ashare.data
from ashare.data.reader import Panel
from ashare.risk import from_panel
from ashare.risk.panel import PANEL_FIELDS


def load(root, days=None, symbols=None):
    """最近 days 个交易日（None 为全历史）、前 symbols 只（None 为全部）。"""
    data = ashare.data.open(root)
    panel = data.panel([*PANEL_FIELDS.values(), "industry"])
    rows = slice(None if days is None else len(panel.dates) - days, None)
    cols = slice(None if symbols is None else symbols)
    view = Panel(
        panel.dates[rows],
        panel.symbols[cols],
        {name: panel[name][rows, cols] for name in panel._arrays},
        panel.version,
    )
    return from_panel(data, panel=view)
