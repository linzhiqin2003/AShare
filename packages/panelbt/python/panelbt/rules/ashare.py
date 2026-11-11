"""A 股基础规则；调用方负责历史证券主表和真实涨跌停价。"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike

from ..models import BoolArray, FloatArray, IntArray

# 半分钱作为绝对容差，不随股价扩大；边界也算触及涨跌停。
PRICE_TOL = 0.005


class Board(IntEnum):
    MAIN = 0
    CHINEXT = 1
    STAR = 2
    BSE = 3


def tradability(
    exec_price: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    up_limit: ArrayLike,
    down_limit: ArrayLike,
    suspended: ArrayLike,
) -> tuple[BoolArray, BoolArray]:
    p, h, lo, up, down = [
        np.asarray(x, dtype=np.float64)
        for x in (exec_price, high, low, up_limit, down_limit)
    ]
    stop = np.asarray(suspended, dtype=np.bool_)
    if any(x.shape != p.shape for x in (h, lo, up, down, stop)):
        raise ValueError("tradability: 所有输入必须形状相同")
    valid = np.isfinite(p) & (p > 0) & ~stop
    # NaN 涨跌停价表示当天无该方向价格限制。
    at_up = np.isfinite(up) & (p >= up - PRICE_TOL)
    at_down = np.isfinite(down) & (p <= down + PRICE_TOL)
    one_price = np.isfinite(h) & np.isfinite(lo) & (np.abs(h - lo) <= PRICE_TOL)
    locked_up = one_price & np.isfinite(up) & (h >= up - PRICE_TOL)
    locked_down = one_price & np.isfinite(down) & (lo <= down + PRICE_TOL)
    return valid & ~at_up & ~locked_up, valid & ~at_down & ~locked_down


def stamp_tax_sell(dates: ArrayLike) -> FloatArray:
    try:
        days = np.asarray(dates, dtype="datetime64[D]")
    except (ValueError, TypeError) as exc:
        raise ValueError("dates: 期望可转换为日期的数组") from exc
    if (
        days.ndim != 1
        or np.any(np.isnat(days))
        or np.any(days < np.datetime64("2008-09-19"))
    ):
        raise ValueError("dates: 期望一维有效日期，支持范围从 2008-09-19 起")
    return np.where(days >= np.datetime64("2023-08-28"), 0.0005, 0.001)


def lot_rules(board: ArrayLike) -> tuple[FloatArray, FloatArray]:
    b = np.asarray(board)
    if not np.all(np.isin(b, [int(x) for x in Board])):
        raise ValueError("board: 期望 Board.MAIN/CHINEXT/STAR/BSE")
    return np.where(b == Board.STAR, 200.0, 100.0), np.where(
        (b == Board.MAIN) | (b == Board.CHINEXT), 100.0, 1.0
    )


def infer_board(codes: ArrayLike) -> IntArray:
    """按六位字符串前缀推断；北交所旧码仅为候选，需调用方主表确认。"""
    c = np.asarray(codes, dtype=np.str_)
    valid = (np.char.str_len(c) == 6) & np.char.isdecimal(c)
    out = np.full(c.shape, -1, dtype=np.int64)
    for prefixes, board in [
        (("600", "601", "603", "605", "000", "001", "002", "003"), Board.MAIN),
        (("300", "301"), Board.CHINEXT),
        (("688", "689"), Board.STAR),
        (("43", "83", "87", "88", "920"), Board.BSE),
    ]:
        mask = np.zeros(c.shape, dtype=np.bool_)
        for prefix in prefixes:
            mask |= np.char.startswith(c, prefix)
        out[mask & valid] = int(board)
    if np.any(out < 0):
        raise ValueError("codes: 期望受支持的六位 A 股代码字符串前缀")
    return out
