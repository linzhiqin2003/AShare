"""开盘价成交的可买、可卖掩码：建面板时逐行算好，回测直接读，不再每次从行情现算。

口径与 `panelbt.markets.ashare.tradability` 逐格相同（成交价取开盘价，停牌取
`is_suspended | ~has_quote`）。这里另写一遍是为了让构建只依赖数据层自己的代码
（代码哈希进入面板版本）；`verify` 用 panelbt 的函数独立核对，两边口径一旦分叉就会报出。
"""

import numpy as np

# 与 panelbt.markets.ashare.PRICE_TOL 相同；测试守住两者相等。
PRICE_TOL = 0.005
# 逐块计算，临时数组只占几十 MB。
ROWS = 256


def open_masks(open_, high, low, upper, lower, suspended, has_quote):
    """一块行的 (can_buy_open, can_sell_open)；输入为同形二维数组。"""
    p, h, lo = (np.asarray(x, dtype=np.float64) for x in (open_, high, low))
    up, down = (np.asarray(x, dtype=np.float64) for x in (upper, lower))
    stop = np.asarray(suspended, dtype=np.bool_) | ~np.asarray(has_quote, dtype=bool)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(p) & (p > 0) & ~stop
        has_up, has_down = np.isfinite(up), np.isfinite(down)
        one_price = np.isfinite(h) & np.isfinite(lo) & (np.abs(h - lo) <= PRICE_TOL)
        # 开盘即在涨停（跌停）价，或全天一字涨停（跌停）。
        no_buy = has_up & ((p >= up - PRICE_TOL) | (one_price & (h >= up - PRICE_TOL)))
        no_sell = has_down & (
            (p <= down + PRICE_TOL) | (one_price & (lo <= down + PRICE_TOL))
        )
    return valid & ~no_buy, valid & ~no_sell


def fill(source, buy, sell, start):
    """把 source 的 [start:] 行算进 buy/sell（长度为 rows-start）；按块推进。"""
    names = (
        "open",
        "high",
        "low",
        "upper_limit",
        "lower_limit",
        "is_suspended",
        "has_quote",
    )
    total = len(buy)
    for i in range(0, total, ROWS):
        j = min(total, i + ROWS)
        blocks = [np.asarray(source(name)[start + i : start + j]) for name in names]
        buy[i:j], sell[i:j] = open_masks(*blocks)
    return buy, sell
