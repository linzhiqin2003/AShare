"""Compatibility import for the A-share rule adapters now owned by markets."""

from ..markets.ashare import (
    PRICE_TOL,
    Board,
    infer_board,
    lot_rules,
    stamp_tax_sell,
    tradability,
)

__all__ = [
    "PRICE_TOL",
    "Board",
    "infer_board",
    "lot_rules",
    "stamp_tax_sell",
    "tradability",
]
