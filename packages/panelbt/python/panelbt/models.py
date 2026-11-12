"""Public array models and NumPy input validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def array(value: ArrayLike, name: str, dtype: Any, ndim: int) -> NDArray:
    try:
        a = np.asarray(value)
        if dtype == np.int64 and (
            a.dtype.kind not in "iu" or (a.size and np.any(a > np.iinfo(np.int64).max))
        ):
            raise ValueError(f"{name}: expected int64 integer indices")
        a = np.ascontiguousarray(a, dtype=dtype)
    except (TypeError, OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name}: expected an array convertible to {np.dtype(dtype)}; {exc}"
        ) from exc
    if a.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions, got shape {a.shape}")
    # Do not change the caller's writable flag; the caller must not mutate aliases.
    a = a.view()
    a.flags.writeable = False
    return a


def nonnegative(
    a: NDArray, name: str, *, nan: bool = False, positive: bool = False
) -> None:
    bad = np.isinf(a) | ((a <= 0) if positive else (a < 0))
    if not nan:
        bad |= np.isnan(a)
    if np.any(bad):
        raise ValueError(
            f"{name}: expected finite {'positive' if positive else 'nonnegative'} values"
            + (" or NaN" if nan else "")
        )


def vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name}: expected a numeric scalar or length {size} vector"
        ) from exc
    if a.ndim == 0:
        value = a.item()
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name}: expected finite nonnegative values")
        result = np.full(size, value)
        result.flags.writeable = False
        return result
    a = array(a, name, np.float64, 1)
    if a.shape != (size,):
        raise ValueError(
            f"{name}: expected scalar or length {size}, got shape {a.shape}"
        )
    nonnegative(a, name)
    return a


def future_rate(value: ArrayLike, shape: tuple[int, int], name: str) -> FloatArray:
    """Scalar, (1, N) per contract, (T, 1) per day, or T×N.

    One-dimensional input is rejected: a length-N and a length-T vector are
    indistinguishable when T equals N, so the axis must be explicit.
    """
    t, n = shape
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name}: expected a numeric scalar, (1, N), (T, 1) or T×N"
        ) from exc
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        raise ValueError(
            f"{name}: a 1-D rate is ambiguous; pass shape (1, N) per contract "
            "or (T, 1) per day"
        )
    if a.ndim == 2 and a.shape == (t, 1) and n != 1:
        # Per-day rates are expanded once; the kernel reads 1, N or T×N.
        a = np.repeat(a, n, axis=1)
    if a.shape not in ((1, 1), (1, n), (t, n)):
        raise ValueError(f"{name}: expected scalar, (1, N), (T, 1) or T×N")
    # Values are checked in the kernel (`_core.validate_futures`).
    return array(a, name, np.float64, 2)


@dataclass(frozen=True)
class Futures:
    """Contract columns, settlement, expiry and explicitly supplied exchange costs.

    `is_future` marks each column. Expiry is the inclusive panel row on which
    cash settlement occurs; T means the contract expires after the panel.
    """

    is_future: ArrayLike
    multiplier: ArrayLike
    margin_rate: ArrayLike
    settlement: ArrayLike
    expiry: ArrayLike
    fee_rate: ArrayLike = 0.0
    fee_per_contract: ArrayLike = 0.0
    delivery_fee_rate: ArrayLike = 0.0
    delivery_fee_per_contract: ArrayLike = 0.0

    def validated(
        self, shape: tuple[int, int], market: Market, validate: bool = True
    ) -> Futures:
        """Convert and shape-check; with `validate`, check values in the kernel."""
        from . import _core

        t, n = shape
        kind = array(self.is_future, "is_future", np.bool_, 1)
        multiplier = array(self.multiplier, "multiplier", np.float64, 1)
        expiry = array(self.expiry, "expiry", np.int64, 1)
        settlement = array(self.settlement, "settlement", np.float64, 2)
        if kind.shape != (n,) or multiplier.shape != (n,) or expiry.shape != (n,):
            raise ValueError(
                "futures: is_future, multiplier and expiry require length N"
            )
        if settlement.shape != shape:
            raise ValueError("settlement: expected T×N matching market prices")
        if not kind.any():
            raise ValueError("is_future: expected at least one futures column")
        names = (
            "margin_rate",
            "fee_rate",
            "fee_per_contract",
            "delivery_fee_rate",
            "delivery_fee_per_contract",
        )
        rates = {name: future_rate(getattr(self, name), shape, name) for name in names}
        if validate:
            empty, none = np.empty((0, 0)), np.empty((0, 0), np.uint8)
            _core.validate_futures(
                kind.view(np.uint8),
                multiplier,
                expiry,
                settlement,
                empty if market.split is None else market.split,
                empty if market.cash_div is None else market.cash_div,
                none if market.delist is None else market.delist.view(np.uint8),
                list(rates.items()),
            )
        return Futures(
            kind, multiplier, rates.pop("margin_rate"), settlement, expiry, **rates
        )


@dataclass(frozen=True)
class Market:
    """Aligned, validated, borrowed read-only market panels."""

    exec_price: ArrayLike
    close: ArrayLike
    can_buy: ArrayLike
    can_sell: ArrayLike
    volume: ArrayLike | None = None
    split: ArrayLike | None = None
    cash_div: ArrayLike | None = None
    delist: ArrayLike | None = None
    can_short: ArrayLike | None = None
    futures: Futures | None = None
    # Daily bars for declarative rules and limit targets (see `Rules`).
    open: ArrayLike | None = None
    high: ArrayLike | None = None
    low: ArrayLike | None = None
    validate: bool = field(default=True, kw_only=True)
    first_delist: IntArray = field(init=False, repr=False)
    action_indptr: IntArray = field(init=False, repr=False)
    action_cols: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from . import _core

        # Thin layer: dtype, dimensions, memory layout and matching shapes are
        # always checked here; per-cell values are checked in the kernel.
        shape = None
        for name in (
            "exec_price",
            "close",
            "can_buy",
            "can_sell",
            "volume",
            "split",
            "cash_div",
            "delist",
            "can_short",
            "open",
            "high",
            "low",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            dtype = (
                np.bool_
                if name in ("can_buy", "can_sell", "delist", "can_short")
                else np.float64
            )
            a = array(value, name, dtype, 2)
            if shape is None:
                shape = a.shape
                if 0 in shape:
                    raise ValueError("exec_price: expected a nonempty T×N panel")
            if a.shape != shape:
                raise ValueError(
                    f"{name}: expected shape {shape} matching exec_price, got {a.shape}"
                )
            object.__setattr__(self, name, a)
        if any(
            getattr(self, key) is None
            for key in ("exec_price", "close", "can_buy", "can_sell")
        ):
            raise ValueError(
                "Market: exec_price, close, can_buy and can_sell are required"
            )
        if not isinstance(self.validate, bool):
            raise ValueError("validate: expected bool")
        bars = [getattr(self, k) for k in ("open", "high", "low")]
        if self.validate and any(b is not None for b in bars):
            for name, b in zip(("open", "high", "low"), bars, strict=True):
                if b is not None:
                    nonnegative(b, name, nan=True, positive=True)
            if all(b is not None for b in bars):
                o, h, lo = bars
                with np.errstate(invalid="ignore"):
                    bad = (lo > h) | (lo > o) | (o > h)
                if np.any(bad):
                    raise ValueError("open/high/low: expected low <= open <= high")
        empty, none = np.empty((0, 0)), np.empty((0, 0), np.uint8)

        def floats(name):
            value = getattr(self, name)
            return empty if value is None else value

        def masks(name):
            value = getattr(self, name)
            return none if value is None else value.view(np.uint8)

        # Content checks (if requested), the first delist row and the sparse
        # daily corporate-action index are computed in parallel in Rust.
        first, pointers, columns = _core.prepare_market(
            self.exec_price,
            self.close,
            masks("can_buy"),
            masks("can_sell"),
            floats("volume"),
            floats("split"),
            floats("cash_div"),
            masks("delist"),
            masks("can_short"),
            self.validate,
        )
        for a in (first, pointers, columns):
            a.flags.writeable = False
        object.__setattr__(self, "first_delist", first)
        object.__setattr__(self, "action_indptr", pointers)
        object.__setattr__(self, "action_cols", columns)
        if self.futures is not None:
            if not isinstance(self.futures, Futures):
                raise ValueError("futures: expected a Futures specification")
            object.__setattr__(
                self,
                "futures",
                self.futures.validated(self.shape, self, self.validate),
            )

    @property
    def validated(self) -> bool:
        """Whether every cell passed the content checks at construction."""
        return self.validate

    @property
    def shape(self) -> tuple[int, int]:
        return self.exec_price.shape  # type: ignore[union-attr]


@dataclass(frozen=True)
class Targets:
    """Sparse rebalance plan; an empty scheduled row closes all positions."""

    days: ArrayLike
    indptr: ArrayLike
    cols: ArrayLike
    weights: ArrayLike
    kind: str = "weights"
    # Limit price per entry; NaN is an ordinary target (needs `Config.rules`).
    limits: ArrayLike | None = None
    validate: bool = field(default=True, kw_only=True)
    has_negative: bool = field(init=False, repr=False)
    max_row_sum: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from . import _core

        for name in ("days", "indptr", "cols", "weights"):
            dtype = np.float64 if name == "weights" else np.int64
            value = getattr(self, name)
            # Empty lists have no dtype; create integer indices explicitly.
            if dtype == np.int64 and np.asarray(value).size == 0:
                value = np.asarray(value, dtype=np.int64)
            object.__setattr__(self, name, array(value, name, dtype, 1))
        if self.kind not in ("weights", "shares"):
            raise ValueError("kind: expected weights or shares")
        if not isinstance(self.validate, bool):
            raise ValueError("validate: expected bool")
        if self.limits is not None:
            limits = array(self.limits, "limits", np.float64, 1)
            if limits.shape != self.weights.shape:
                raise ValueError("limits: expected one limit per target entry")
            if np.any(np.isinf(limits) | (limits <= 0)):
                raise ValueError("limits: expected positive limit prices or NaN")
            object.__setattr__(self, "limits", limits)
        d, p, c, w = self.days, self.indptr, self.cols, self.weights
        # Pointer and index structure is always checked; day order, finite
        # weights and duplicate columns only with `validate`.
        ordered, negative, row_max = _core.check_plan(d, p, c, w, self.validate)
        if not ordered:
            rows = np.repeat(np.arange(len(d)), np.diff(p))
            order = np.lexsort((c, rows))
            object.__setattr__(self, "cols", array(c[order], "cols", np.int64, 1))
            object.__setattr__(
                self, "weights", array(w[order], "weights", np.float64, 1)
            )
            if self.limits is not None:
                object.__setattr__(
                    self, "limits", array(self.limits[order], "limits", np.float64, 1)
                )
            ordered, negative, row_max = _core.check_plan(
                d, p, self.cols, self.weights, False
            )
            if not ordered and self.validate:
                raise ValueError(
                    "cols: duplicate columns in a rebalance row are not allowed"
                )
        # Immutable plans can cache context-independent validation summaries.
        object.__setattr__(self, "has_negative", negative)
        object.__setattr__(self, "max_row_sum", row_max)

    @property
    def validated(self) -> bool:
        """Whether the plan passed the content checks at construction."""
        return self.validate

    def validate_shape(self, t: int, n: int) -> None:
        if len(self.days) and self.days[-1] >= t:
            raise ValueError(f"days: expected index < T={t}")
        if len(self.cols) and np.max(self.cols) >= n:
            raise ValueError(f"cols: expected index < N={n}")

    @classmethod
    def from_dense(
        cls,
        weights: ArrayLike,
        *,
        kind: str = "weights",
        limits: ArrayLike | None = None,
        validate: bool = True,
    ) -> Targets:
        """Dense rows: all-NaN = no target; NaN elsewhere = 0.

        `limits` (T×N, NaN = ordinary) attaches a limit price to a cell; a
        cell with a limit is kept even when its weight is 0 (a limit exit).
        """
        w = array(weights, "weights", np.float64, 2)
        days = np.flatnonzero(~np.all(np.isnan(w), axis=1))
        keep = (w != 0) & ~np.isnan(w)
        lim = None
        if limits is not None:
            lim = array(limits, "limits", np.float64, 2)
            if lim.shape != w.shape:
                raise ValueError("limits: expected the shape of weights")
            row_on = ~np.all(np.isnan(w), axis=1)[:, None]
            keep = keep | (~np.isnan(lim) & row_on)
        rows, cols = np.nonzero(keep)
        counts = np.bincount(rows, minlength=w.shape[0])[days]
        indptr = np.concatenate(([0], np.cumsum(counts)))
        values = np.nan_to_num(w[rows, cols], nan=0.0)
        return cls(
            days,
            indptr,
            cols,
            values,
            kind,
            None if lim is None else lim[rows, cols],
            validate=validate,
        )

    def to_dense(self, t: int, n: int) -> FloatArray:
        if not isinstance(t, int) or not isinstance(n, int) or t < 0 or n < 0:
            raise ValueError("T/N: expected nonnegative integers")
        self.validate_shape(t, n)
        w = np.full((t, n), np.nan)
        w[self.days] = 0
        rows = np.repeat(self.days, np.diff(self.indptr))
        w[rows, self.cols] = self.weights
        return w


@dataclass(frozen=True, kw_only=True)
class Costs:
    """Per-fill commission, directional taxes/fees and execution costs."""

    commission_rate: float = 2.5e-4
    commission_min: float | None = 5.0
    tax_sell: ArrayLike = 0.0
    tax_buy: ArrayLike = 0.0
    fee_rate_buy: ArrayLike = 1e-5
    slippage: ArrayLike = 5e-4
    fee_rate_sell: ArrayLike = 1e-5
    commission_per_share: float = 0.0
    commission_max_rate: float | None = None
    fee_per_share_buy: ArrayLike = 0.0
    fee_per_share_sell: ArrayLike = 0.0
    impact_k: ArrayLike | None = None
    impact_exponent: float = 0.5


@dataclass(frozen=True)
class MarginModel:
    """Equity-based spot margin model; all rates are per period, not annual."""

    allow_short: bool = False
    long_initial: ArrayLike = 1.0
    short_initial: ArrayLike = 1.0
    long_maintenance: ArrayLike = 0.0
    short_maintenance: ArrayLike = 0.0
    allow_financing: bool = False
    short_proceeds_usable: bool = False
    financing_rate: ArrayLike = 0.0
    borrow_rate: ArrayLike = 0.0
    margin_call: str = "liquidate"


@dataclass(frozen=True, kw_only=True)
class Rules:
    """Declarative exits and limit orders executed inside the kernel.

    `fill_time` says where `exec_price` sits in the day ("open", "close" or
    "average") and decides whether rules act after or before the day's
    rebalance. Percentages are positive fractions: a scalar, a length-N
    vector or a T×N panel; NaN disables a cell. Requires `Market` open, high
    and low panels; not available with futures columns. See SPEC.md.

    `limit_fill="cross"` (default) fills a limit buy only when the day's low
    is strictly below the limit (a sell: the high strictly above); "touch"
    also fills when the low (high) only equals it, which is optimistic.
    `sell_lock=True` (A-share T+1) is the default for every market; set it to
    False elsewhere.
    """

    fill_time: str
    stop_loss: ArrayLike | None = None
    take_profit: ArrayLike | None = None
    trailing_stop: ArrayLike | None = None
    max_hold: int | None = None
    both_hit: str = "stop"
    cooldown: int | None = None
    dividend_adjusts_cost: bool = False
    sell_lock: bool = True
    limit_valid: str = "day"
    limit_fill: str = "cross"


@dataclass(frozen=True)
class Config:
    initial_cash: float = 1e7
    min_qty: ArrayLike = 100.0
    qty_step: ArrayLike = 100.0
    max_participation: float = 0.0
    carry_blocked: bool = True
    delist_recovery: float = 1.0
    cash_buffer: float = 0.0
    record_trades: bool = False
    trade_band: float = 0.0
    dividend_tax_rate: ArrayLike = 0.0
    cash_flow: ArrayLike = 0.0
    cash_rate: ArrayLike = 0.0
    margin: MarginModel = field(default_factory=MarginModel)
    rules: Rules | None = None


class TradeReason(IntEnum):
    """Why a fill happened: target rows (including carried and forced fills) or a rule."""

    REBALANCE = 0
    STOP_LOSS = 1
    TAKE_PROFIT = 2
    TRAILING_STOP = 3
    MAX_HOLD = 4
    LIMIT = 5


@dataclass(frozen=True)
class Trade:
    day: int
    col: int
    qty: float
    price: float
    commission: float
    tax: float
    other_fee: float
    slippage_cost: float
    impact_cost: float = 0.0
    reason: TradeReason = TradeReason.REBALANCE


@dataclass(frozen=True)
class DelistEvent:
    day: int
    col: int
    qty: float
    price: float
    proceeds: float


@dataclass(frozen=True)
class LiquidationEvent:
    day: int
    col: int
    qty: float


@dataclass(frozen=True)
class ExpiryEvent:
    day: int
    col: int
    qty: float
    settlement: float
    fee: float


METRICS = (
    "nav",
    "cash",
    "turnover",
    "commission",
    "tax",
    "other_fee",
    "slippage_cost",
    "n_blocked_buy",
    "n_blocked_sell",
    "impact_cost",
    "dividend_gross",
    "dividend_tax",
    "cash_flow",
    "withdrawal_shortfall",
    "cash_interest",
    "returns",
    "unit_nav",
    "long_value",
    "short_value",
    "borrow_cost",
    "financing_interest",
    "margin_usage",
    "margin_call",
    "equity",
    "n_reduction_clipped",
    "n_invalid_quote",
    "futures_pnl",
    "futures_margin",
    "delivery_fee",
    "n_expired_target",
)
# The original counters stay separate so the unchanged long-only generator
# continues to assert that every original branch was actually exercised.
COUNTERS = (
    "trades",
    "blocked_buy",
    "blocked_sell",
    "cash_scaled",
    "min_commission",
    "delist",
    "dividend",
    "split",
    "carry_trade",
)
EXTRA_COUNTERS = (
    "shares_target",
    "trade_band",
    "dividend_tax",
    "cash_flow",
    "withdrawal_shortfall",
    "cash_interest",
    "impact",
    "per_share_fee",
    "commission_cap",
    "open_short",
    "cover_short",
    "margin_scaled",
    "borrow_cost",
    "short_dividend",
    "margin_call",
    "liquidation",
    "bankruptcy",
    "financing_interest",
    "reduction_clipped",
    "invalid_quote",
    "genuine_bankruptcy",
    "reversal",
    "futures_open",
    "futures_close",
    "futures_margin_scaled",
    "futures_margin_call",
    "futures_liquidation",
    "futures_expiry",
    "expired_target",
    "futures_rounded_to_zero",
)
# Present in `Result.diagnostics` only when `Config.rules` is set.
RULE_COUNTERS = (
    "stop_loss",
    "take_profit",
    "trailing_stop",
    "max_hold",
    "rule_deferred",
    "reentry_blocked",
    "limit_filled",
    "limit_unfilled",
    "limit_blocked",
)


@dataclass
class Result:
    nav: FloatArray
    cash: FloatArray
    turnover: FloatArray
    commission: FloatArray
    tax: FloatArray
    other_fee: FloatArray
    slippage_cost: FloatArray
    n_blocked_buy: IntArray
    n_blocked_sell: IntArray
    delist_events: list[DelistEvent] | list[list[DelistEvent]]
    trades: list[Trade] | list[list[Trade]] | None = None
    positions: FloatArray | None = None
    diagnostics: dict[str, int | IntArray] = field(default_factory=dict)
    impact_cost: FloatArray | None = None
    dividend_gross: FloatArray | None = None
    dividend_tax: FloatArray | None = None
    cash_flow: FloatArray | None = None
    withdrawal_shortfall: FloatArray | None = None
    cash_interest: FloatArray | None = None
    returns: FloatArray | None = None
    unit_nav: FloatArray | None = None
    long_value: FloatArray | None = None
    short_value: FloatArray | None = None
    borrow_cost: FloatArray | None = None
    financing_interest: FloatArray | None = None
    margin_usage: FloatArray | None = None
    margin_call: BoolArray | None = None
    equity: FloatArray | None = None
    n_reduction_clipped: IntArray | None = None
    n_invalid_quote: IntArray | None = None
    futures_pnl: FloatArray | None = None
    futures_margin: FloatArray | None = None
    delivery_fee: FloatArray | None = None
    n_expired_target: IntArray | None = None
    contribution: FloatArray | None = None
    liquidation_events: list = field(default_factory=list)
    bankruptcy_events: list = field(default_factory=list)
    expiry_events: list[ExpiryEvent] | list[list[ExpiryEvent]] = field(
        default_factory=list
    )
    coverage: dict = field(default_factory=dict)
    # True when the market and every plan of this run passed content checks.
    inputs_validated: bool = True

    def to_npz(self, path) -> None:
        """Save arrays and JSON metadata without pickle or executable payloads."""
        import json
        from dataclasses import asdict

        arrays = {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}

        def encode(value):
            if hasattr(value, "__dataclass_fields__"):
                return asdict(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)

        meta = {k: v for k, v in vars(self).items() if k not in arrays}
        arrays["metadata"] = np.array(json.dumps(meta, default=encode))
        arrays["format_version"] = np.array(1)
        np.savez_compressed(path, **arrays)

    @classmethod
    def from_npz(cls, path) -> Result:
        """Load a versioned archive with pickle explicitly disabled."""
        import json

        with np.load(path, allow_pickle=False) as archive:
            if archive["format_version"].item() != 1:
                raise ValueError("unsupported Result archive version")
            values = json.loads(archive["metadata"].item())
            values.update(
                {
                    k: archive[k]
                    for k in archive.files
                    if k not in ("metadata", "format_version")
                }
            )

        def restore(items, kind):
            return [
                restore(x, kind) if isinstance(x, list) else kind(**x) for x in items
            ]

        for key, kind in (
            ("trades", Trade),
            ("delist_events", DelistEvent),
            ("liquidation_events", LiquidationEvent),
            ("expiry_events", ExpiryEvent),
        ):
            if key in values and values[key] is not None:
                values[key] = restore(values[key], kind)
        # Version-1 archives created before futures support have no future fields.
        for key in ("futures_pnl", "futures_margin", "delivery_fee"):
            values.setdefault(key, np.zeros_like(values["nav"]))
        values.setdefault("n_expired_target", np.zeros_like(values["n_blocked_buy"]))
        for key in ("diagnostics", "coverage"):
            values[key] = {
                k: np.asarray(v, dtype=np.int64) if isinstance(v, list) else v
                for k, v in values[key].items()
            }
        return cls(**values)

    def to_frame(self, index=None):
        """Return daily fields as a pandas DataFrame (optional dependency)."""
        import pandas as pd

        if self.nav.ndim != 1:
            raise ValueError("to_frame requires a single-strategy Result")
        return pd.DataFrame({k: getattr(self, k) for k in METRICS}, index=index)
