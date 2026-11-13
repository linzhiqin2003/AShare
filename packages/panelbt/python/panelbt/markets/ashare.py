"""A-share adapters; the caller supplies point-in-time metadata and limits."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike

from ..models import BoolArray, FloatArray, IntArray

# Half-cent absolute tolerance, including the boundary.
PRICE_TOL = 0.005


class Board(IntEnum):
    MAIN = 0
    CHINEXT = 1
    STAR = 2
    BSE = 3


def price_permissions(exec_price, up_limit, down_limit, suspended):
    """Directional price limits and suspensions, without intraday range rules."""
    p, up, down = [
        np.asarray(x, dtype=np.float64) for x in (exec_price, up_limit, down_limit)
    ]
    stop = np.asarray(suspended, dtype=np.bool_)
    if any(x.shape != p.shape for x in (up, down, stop)):
        raise ValueError("price_permissions: all inputs must have the same shape")
    valid = np.isfinite(p) & (p > 0) & ~stop
    at_up = np.isfinite(up) & (p >= up - PRICE_TOL)
    at_down = np.isfinite(down) & (p <= down + PRICE_TOL)
    return valid & ~at_up, valid & ~at_down


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
        raise ValueError("tradability: all inputs must have the same shape")
    buy, sell = price_permissions(p, up, down, stop)
    one_price = np.isfinite(h) & np.isfinite(lo) & (np.abs(h - lo) <= PRICE_TOL)
    locked_up = one_price & np.isfinite(up) & (h >= up - PRICE_TOL)
    locked_down = one_price & np.isfinite(down) & (lo <= down + PRICE_TOL)
    return buy & ~locked_up, sell & ~locked_down


def stamp_tax_sell(dates: ArrayLike) -> FloatArray:
    try:
        days = np.asarray(dates, dtype="datetime64[D]")
    except (ValueError, TypeError) as exc:
        raise ValueError("dates: expected an array convertible to dates") from exc
    if (
        days.ndim != 1
        or np.any(np.isnat(days))
        or np.any(days < np.datetime64("2008-09-19"))
    ):
        raise ValueError(
            "dates: expected one-dimensional valid dates on or after 2008-09-19"
        )
    return np.where(days >= np.datetime64("2023-08-28"), 0.0005, 0.001)


def lot_rules(board: ArrayLike) -> tuple[FloatArray, FloatArray]:
    b = np.asarray(board)
    if not np.all(np.isin(b, [int(x) for x in Board])):
        raise ValueError("board: expected Board.MAIN/CHINEXT/STAR/BSE")
    return np.where(b == Board.STAR, 200.0, 100.0), np.where(
        (b == Board.MAIN) | (b == Board.CHINEXT), 100.0, 1.0
    )


def infer_board(codes: ArrayLike) -> IntArray:
    """Infer board candidates from code prefixes; verify historical membership.

    Accepts six-digit codes, optionally with an exchange suffix (``.SH``,
    ``.SZ``, ``.BJ``). Prefer an authoritative board column where one exists:
    prefixes are a heuristic and new code ranges appear over time (ChiNext
    ``302`` from 2025). Unrecognised codes raise and are listed in the error.
    """
    raw = np.asarray(codes, dtype=np.str_)
    stripped = [
        x[:6] if len(x) == 9 and x[6:] in (".SH", ".SZ", ".BJ") else x
        for x in raw.ravel().tolist()
    ]
    c = np.asarray(stripped, dtype="U16").reshape(raw.shape)
    valid = (np.char.str_len(c) == 6) & np.char.isdecimal(c)
    out = np.full(c.shape, -1, dtype=np.int64)
    for prefixes, board in [
        (("600", "601", "603", "605", "000", "001", "002", "003"), Board.MAIN),
        (("300", "301", "302"), Board.CHINEXT),
        (("688", "689"), Board.STAR),
        (("43", "83", "87", "88", "920"), Board.BSE),
    ]:
        mask = np.zeros(c.shape, dtype=np.bool_)
        for prefix in prefixes:
            mask |= np.char.startswith(c, prefix)
        out[mask & valid] = int(board)
    if np.any(out < 0):
        bad = raw[out < 0]
        raise ValueError(
            "codes: unsupported A-share code prefix for "
            + ", ".join(bad[:10].tolist())
            + (f" (+{bad.size - 10} more)" if bad.size > 10 else "")
        )
    return out
