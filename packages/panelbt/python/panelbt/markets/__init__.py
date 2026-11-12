"""Market profiles expand dated rules into engine-neutral NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from ..models import Config, Costs, Futures, MarginModel, Market
from .ashare import Board, infer_board, lot_rules, stamp_tax_sell
from .ashare import tradability as ashare_tradability


@dataclass(frozen=True)
class FeeSchedule:
    """Effective-date patches, applied cumulatively; dates before inception fail."""

    periods: tuple
    required_overrides_before: tuple = ()

    def __post_init__(self):
        periods = tuple(
            (np.datetime64(date, "D"), MappingProxyType(dict(terms)))
            for date, terms in self.periods
        )
        if (
            not periods
            or any(np.isnat(d) for d, _ in periods)
            or any(a[0] >= b[0] for a, b in zip(periods, periods[1:], strict=False))
        ):
            raise ValueError(
                "FeeSchedule requires strictly increasing valid effective dates"
            )
        allowed = {
            "tax_buy",
            "tax_sell",
            "fee_rate_buy",
            "fee_rate_sell",
            "fee_per_share_buy",
            "fee_per_share_sell",
        }
        if any(set(terms) - allowed for _, terms in periods):
            raise ValueError(
                "FeeSchedule accepts directional taxes/fees; broker commission belongs in overrides"
            )
        object.__setattr__(self, "periods", periods)

    def expand(self, dates):
        dates = np.asarray(dates, dtype="datetime64[D]")
        if (
            dates.ndim != 1
            or np.isnat(dates).any()
            or (dates < self.periods[0][0]).any()
        ):
            raise ValueError(
                "dates precede schedule or are not valid one-dimensional dates"
            )
        output = {
            k: np.zeros(len(dates))
            for k in (
                "tax_buy",
                "tax_sell",
                "fee_rate_buy",
                "fee_rate_sell",
                "fee_per_share_buy",
                "fee_per_share_sell",
            )
        }
        for start, terms in self.periods:
            for key, value in terms.items():
                output[key][dates >= start] = value
        return output


@dataclass(frozen=True)
class MarketProfile:
    """Immutable market identity and rule adapters; no market logic in Rust."""

    name: str
    currency: str
    periods_per_year: float
    fees: FeeSchedule
    lot_rule: str = "direct"
    price_limits: bool = False
    sell_lock: int = 0
    margin: MarginModel = field(default_factory=MarginModel)
    regulatory_notes: tuple = ()

    def __post_init__(self):
        if (
            self.periods_per_year <= 0
            or self.sell_lock not in (0, 1)
            or self.lot_rule not in ("direct", "ashare")
        ):
            raise ValueError("invalid market profile")

    def costs(self, dates, **overrides):
        """Expand the fee table; caller-specific broker charges override defaults."""
        for before, fields in self.fees.required_overrides_before:
            if (
                np.any(np.asarray(dates, dtype="datetime64[D]") < np.datetime64(before))
                and not set(fields) <= overrides.keys()
            ):
                raise ValueError(
                    f"dates before {before} require explicit overrides: {', '.join(fields)}"
                )
        values = self.fees.expand(dates)
        values.update(overrides)
        return Costs(**values)

    def quantities(self, *, board=None, lot_sizes=None, n=None):
        """Use board attributes or explicit per-security board-lot sizes."""
        if lot_sizes is not None:
            a = np.asarray(lot_sizes, dtype=float)
            if a.ndim != 1 or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError("lot_sizes: expected finite nonnegative length N")
            if n is not None and len(a) != n:
                raise ValueError("lot_sizes: expected length N")
            return a.copy(), a.copy()
        if self.lot_rule == "ashare" and board is not None:
            return lot_rules(board)
        if n is None:
            raise ValueError("provide lot_sizes, board, or n for fractional securities")
        return np.zeros(n), np.zeros(n)

    def tradability(
        self,
        exec_price,
        *,
        suspended=None,
        high=None,
        low=None,
        up_limit=None,
        down_limit=None,
    ):
        p = np.asarray(exec_price, dtype=float)
        stop = (
            np.zeros(p.shape, bool)
            if suspended is None
            else np.asarray(suspended, dtype=bool)
        )
        if stop.shape != p.shape:
            raise ValueError("suspended shape mismatch")
        if self.price_limits:
            if any(x is None for x in (high, low, up_limit, down_limit)):
                raise ValueError(
                    "price-limit profile requires high/low/up_limit/down_limit"
                )
            return ashare_tradability(p, high, low, up_limit, down_limit, stop)
        valid = np.isfinite(p) & (p > 0) & ~stop
        return valid, valid.copy()

    def market(self, exec_price, close, **kwargs):
        """Convert prices and trading rules to a validated Market."""
        keys = ("suspended", "high", "low", "up_limit", "down_limit")
        rules = {k: kwargs.pop(k) for k in keys if k in kwargs}
        buy, sell = self.tradability(exec_price, **rules)
        return Market(exec_price, close, buy, sell, **kwargs)

    def config(self, *, board=None, lot_sizes=None, n=None, **overrides):
        minimum, step = self.quantities(board=board, lot_sizes=lot_sizes, n=n)
        values = dict(min_qty=minimum, qty_step=step, margin=self.margin)
        values.update(overrides)
        return Config(**values)


ASHARE = MarketProfile(
    "A shares",
    "CNY",
    252,
    FeeSchedule(
        (
            ("2008-09-19", {"tax_sell": 0.001}),
            ("2022-04-29", {"fee_rate_buy": 0.00001, "fee_rate_sell": 0.00001}),
            ("2023-08-28", {"tax_sell": 0.0005}),
        ),
        required_overrides_before=(("2022-04-29", ("fee_rate_buy", "fee_rate_sell")),),
    ),
    lot_rule="ashare",
    price_limits=True,
    sell_lock=1,
)


def cffex_index_futures(
    products,
    settlement,
    expiry,
    *,
    margin_rate,
    fee_rate,
    fee_per_contract,
    delivery_fee_rate,
    delivery_fee_per_contract,
):
    """Build IF/IH/IC/IM column metadata with caller-verified cost terms.

    A None product denotes a spot column. Multipliers follow the CFFEX
    contract rules; broker margin and all fee terms must be supplied explicitly.
    """
    names = tuple(products)
    mapping = {"IF": 300.0, "IH": 300.0, "IC": 200.0, "IM": 200.0}
    if not names or any(x is not None and x not in mapping for x in names):
        raise ValueError("products: expected IF, IH, IC, IM or None per column")
    return Futures(
        is_future=[x is not None for x in names],
        multiplier=[mapping.get(x, 1.0) for x in names],
        margin_rate=margin_rate,
        settlement=settlement,
        expiry=expiry,
        fee_rate=fee_rate,
        fee_per_contract=fee_per_contract,
        delivery_fee_rate=delivery_fee_rate,
        delivery_fee_per_contract=delivery_fee_per_contract,
    )


def ashare_margin(
    *,
    long_initial,
    long_maintenance,
    short_maintenance,
    collateral_haircut,
    short_initial,
    allow_financing=False,
    financing_rate=0.0,
    borrow_rate=0.0,
    margin_call="liquidate",
):
    """Construct an explicitly calibrated equity-margin A-share profile.

    Exchange collateral/debt ratios are NOT interchangeable with this engine's
    equity ratios. All regulatory-to-model coefficients must be supplied by the
    caller; no unverified or mathematically mismatched regulatory default is used.
    `collateral_haircut` is recorded calibration metadata, not an extra charge.
    """
    h = float(collateral_haircut)
    if not np.isfinite(h) or not 0 <= h <= 1:
        raise ValueError("collateral_haircut: expected [0, 1]")
    margin = MarginModel(
        allow_short=True,
        long_initial=long_initial,
        short_initial=short_initial,
        long_maintenance=long_maintenance,
        short_maintenance=short_maintenance,
        allow_financing=allow_financing,
        short_proceeds_usable=False,
        financing_rate=financing_rate,
        borrow_rate=borrow_rate,
        margin_call=margin_call,
    )
    return replace(
        ASHARE,
        name="A shares margin (caller calibrated)",
        margin=margin,
        regulatory_notes=(("collateral_haircut", h),),
    )


__all__ = [
    "ASHARE",
    "Board",
    "FeeSchedule",
    "MarketProfile",
    "ashare_margin",
    "cffex_index_futures",
    "infer_board",
    "lot_rules",
    "stamp_tax_sell",
]
