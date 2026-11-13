"""Stand-alone futures portfolios on top of the daily-settled futures kernel.

`FuturesBook` turns per-contract, per-day exchange inputs (settlement, prices,
multiplier, margin and fee rates, last trading row) into a ready `Market`,
`Costs` and `Config`, applies account terms (broker margin add-on and fee
multiple), forced closes (last trading row of physically settled contracts,
optional last holding row of retail accounts), price-limit masks, and
converts three kinds of targets (contracts, notional, margin) into kernel
plans. The kernel loop is the unchanged `run` of SPEC "Futures"; this module
only prepares arrays, so results are ordinary `Result` objects.

Minimal example (one contract, 10 contracts long for the whole sample)::

    import numpy as np, panelbt as bt
    s = np.array([[100.0], [101.0], [99.0]])
    book = bt.FuturesBook(settlement=s, close=s, exec_price=s, multiplier=[10.0],
                          expiry=[3], margin_rate=0.1, initial_cash=1e5)
    r = book.run(np.array([[10.0], [np.nan], [np.nan]]), kind="lots")
    r.equity  # [100000, 100100, 99900]: 10 lots x 10 x settlement change

See SPEC.md "Stand-alone futures portfolios" for every rule and its reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Config, Costs, Futures, Market, Targets

KINDS = ("lots", "notional", "margin")


@dataclass(frozen=True)
class FuturesAccount:
    """Broker terms applied on top of exchange rates (all assumptions).

    Effective margin rate = exchange rate × `margin_scale` + `margin_add`;
    effective trading fees (rate and per contract) = exchange fees ×
    `fee_scale`. Delivery fees are not scaled. The neutral defaults reproduce
    exchange minimums; callers state their broker's terms explicitly.
    """

    margin_add: float = 0.0
    margin_scale: float = 1.0
    fee_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("margin_add", "margin_scale", "fee_scale"):
            value = getattr(self, name)
            if not (isinstance(value, int | float) and np.isfinite(value)) or value < 0:
                raise ValueError(f"{name}: expected a finite nonnegative number")
        if self.margin_scale == 0 and self.margin_add == 0:
            raise ValueError("margin_scale/margin_add: margin would be zero")


def _panel(value, shape, name, *, dtype=np.float64, allow_none=False):
    """Broadcast a scalar, (1, N), (T, 1) or T×N input to a fresh T×N array."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name}: required")
    a = np.asarray(value, dtype=dtype)
    t, n = shape
    if a.ndim == 1:
        raise ValueError(
            f"{name}: a 1-D panel is ambiguous; pass a scalar, (1, N), (T, 1) or T×N"
        )
    if a.ndim == 0:
        a = a.reshape(1, 1)
    if a.ndim != 2 or a.shape[0] not in (1, t) or a.shape[1] not in (1, n):
        raise ValueError(f"{name}: expected a scalar, (1, N), (T, 1) or T×N")
    return np.array(np.broadcast_to(a, shape), dtype=dtype)


def _vector(value, n, name, dtype):
    a = np.asarray(value, dtype=dtype)
    if a.ndim == 0:
        a = np.full(n, a, dtype=dtype)
    if a.shape != (n,):
        raise ValueError(f"{name}: expected a scalar or length N")
    return a


def _ffill(matrix):
    """Forward fill each column (earlier values only)."""
    out = matrix.copy()
    t = out.shape[0]
    index = np.where(np.isfinite(out), np.arange(t)[:, None], -1)
    np.maximum.accumulate(index, axis=0, out=index)
    seen = index >= 0
    rows, cols = np.nonzero(seen)
    out[rows, cols] = matrix[index[rows, cols], cols]
    return out


class FuturesBook:
    """A futures (or mixed) market plus costs and account, ready to run.

    Required (T×N unless noted; rates may be scalar, (1, N), (T, 1) or T×N):

    - `settlement`: daily settlement price (the variation mark). On a cash
      settled contract's last trading row it must be the delivery settlement
      price. NaN rows of a futures column are filled from the last valid
      value (before the first one: the first valid value, a placeholder the
      kernel only uses to size an untradable target).
    - `close`: closing price, reported but never used for futures P&L.
    - `exec_price`: the fill price (default: `open`); NaN never trades.
    - `multiplier[N]`: units per contract.
    - `expiry[N]`: last trading row (inclusive); T means after the sample.
    - `margin_rate`: exchange margin rate.

    Optional: `fee_rate`, `fee_per_contract` (exchange trading fees: a rate
    of notional and a fixed amount per contract, both charged on every fill),
    `volume`, `tradable` (bool, default finite `exec_price`), `open`,
    `high`, `low`, `no_buy`/`no_sell` (bool masks), `limit_up`/`limit_down`
    (prices: a fill at or beyond the limit is refused in that direction),
    `infer_locked` (refuse the direction of the move on one-price days, see
    SPEC), `last_hold[N]` (last row a position may still be held at the
    close; any position still held at the close of the next row is
    force-closed at that row's settlement, paying the trading fee and the
    slippage; that row still trades normally, so a target can close the
    position at the execution price first), `cash_settled[N]` (True: a position held at expiry is cash
    settled at the delivery settlement price with the delivery fee; False,
    the default: it is closed at the last settlement like a forced close),
    `delivery_fee_rate`, `delivery_fee_per_contract`, `is_future[N]` (spot
    columns: `settlement`, rates and fees are ignored), `market_extra`
    (other `Market` keyword arguments, e.g. spot `split`), `account`,
    `costs` (default: slippage only), `config` (or `initial_cash`),
    `missing` ("error", "previous" or a fill value for NaN margin/fee cells
    inside a contract's trading window) and `validate`.
    """

    def __init__(
        self,
        *,
        settlement,
        close,
        multiplier,
        expiry,
        margin_rate,
        exec_price=None,
        fee_rate=0.0,
        fee_per_contract=0.0,
        volume=None,
        tradable=None,
        open=None,  # noqa: A002 - the bar field name
        high=None,
        low=None,
        no_buy=None,
        no_sell=None,
        limit_up=None,
        limit_down=None,
        infer_locked=False,
        last_hold=None,
        cash_settled=None,
        delivery_fee_rate=0.0,
        delivery_fee_per_contract=0.0,
        is_future=None,
        market_extra=None,
        account=None,
        costs=None,
        config=None,
        initial_cash=None,
        missing="error",
        validate=True,
    ):
        account = account or FuturesAccount()
        if not isinstance(account, FuturesAccount):
            raise ValueError("account: expected FuturesAccount")
        if exec_price is None:
            if open is None:
                raise ValueError("exec_price: required (or pass open)")
            exec_price = open
        price = np.array(exec_price, dtype=np.float64)
        if price.ndim != 2 or 0 in price.shape:
            raise ValueError("exec_price: expected a nonempty T×N panel")
        shape = t, n = price.shape
        if config is not None and initial_cash is not None:
            raise ValueError("initial_cash: pass it inside config instead")
        config = config or Config(
            initial_cash=1e7 if initial_cash is None else float(initial_cash)
        )
        costs = costs or Costs(
            commission_rate=0.0,
            commission_min=None,
            fee_rate_buy=0.0,
            fee_rate_sell=0.0,
            slippage=0.0,
        )
        future = (
            np.ones(n, bool)
            if is_future is None
            else _vector(is_future, n, "is_future", bool)
        )
        mult = _vector(multiplier, n, "multiplier", np.float64)
        expiry = _vector(expiry, n, "expiry", np.int64)
        if np.any(expiry < 0):
            raise ValueError("expiry: expected nonnegative row indices")
        expiry = np.minimum(expiry, t)
        hold = (
            np.full(n, t, np.int64)
            if last_hold is None
            else np.minimum(_vector(last_hold, n, "last_hold", np.int64), t)
        )
        if np.any(hold < -1):
            raise ValueError("last_hold: expected row indices >= -1 (T: no limit)")
        settled = (
            np.zeros(n, bool)
            if cash_settled is None
            else _vector(cash_settled, n, "cash_settled", bool)
        )
        # The row on which a futures position ends: the last trading row, or
        # the first row after the last holding row, whichever is earlier.
        end = np.where(future, np.minimum(expiry, hold + 1), expiry)
        personal = future & (end < expiry)

        settle = _panel(settlement, shape, "settlement")
        # Checked whatever `validate` says: a nonpositive price does not make
        # the run slower but wrong (negative collateral, no margin call).
        close_panel = _panel(close, shape, "close")
        for name, panel in (
            ("settlement", settle),
            ("exec_price", price),
            ("close", close_panel),
            ("open", None if open is None else _panel(open, shape, "open")),
        ):
            if panel is None:
                continue
            bad = future[None, :] & np.isfinite(panel) & (panel <= 0)
            if bad.any():
                r, c = map(int, np.argwhere(bad)[0])
                raise ValueError(
                    f"{name}: zero or negative futures prices are not supported "
                    f"(row {r}, column {c}); negative prices such as WTI on "
                    "2020-04-20 cannot be simulated, validate=False does not "
                    "change this"
                )
        tradable = (
            np.isfinite(price)
            if tradable is None
            else _panel(tradable, shape, "tradable", dtype=bool) & np.isfinite(price)
        )
        filled = _ffill(settle)
        for j in np.flatnonzero(future):
            valid = np.isfinite(filled[:, j])
            if not valid.any():
                raise ValueError(f"settlement: futures column {j} has no settlement")
            first = int(np.argmax(valid))
            filled[:first, j] = filled[first, j]
        filled[:, ~future] = 1.0
        # Trading window of each futures column: first tradable row .. end.
        rows = np.arange(t)[:, None]
        any_trade = tradable.any(axis=0)
        start = np.where(any_trade, np.argmax(tradable, axis=0), t)
        window = future[None, :] & (rows >= start[None, :]) & (rows <= end[None, :])

        def rate(value, name):
            a = _panel(value, shape, name)
            bad = ~np.isfinite(a) & window
            if bad.any():
                if isinstance(missing, str) and missing == "previous":
                    a = np.where(window, _ffill(np.where(window, a, np.nan)), a)
                elif not isinstance(missing, str):
                    a = np.where(bad, float(missing), a)
                elif missing != "error":
                    raise ValueError(
                        "missing: expected 'error', 'previous' or a number"
                    )
                bad = ~np.isfinite(a) & window
                if bad.any():
                    cols = sorted(set(np.nonzero(bad)[1].tolist()))[:10]
                    raise ValueError(
                        f"{name}: missing values inside the trading window of "
                        f"columns {cols}; pass missing='previous' or a fill value"
                    )
            if np.any(a[window] < 0):
                raise ValueError(f"{name}: expected nonnegative values")
            return np.where(window, a, 0.0)

        margin = rate(margin_rate, "margin_rate")
        margin = np.where(
            window, margin * account.margin_scale + account.margin_add, 0.0
        )
        fee = rate(fee_rate, "fee_rate") * account.fee_scale
        fee_fixed = rate(fee_per_contract, "fee_per_contract") * account.fee_scale
        slip = _panel(costs.slippage, shape, "costs.slippage")
        d_rate = _panel(delivery_fee_rate, shape, "delivery_fee_rate")
        d_fixed = _panel(delivery_fee_per_contract, shape, "delivery_fee_per_contract")

        # The fee charged on the end row: delivery fee when a cash-settled
        # contract reaches its own expiry, otherwise a forced close.
        delivery = np.zeros(shape)
        delivery_fixed = np.zeros(shape)
        forced = np.zeros(n, bool)
        for j in np.flatnonzero(future & (end < t)):
            r = int(end[j])
            if settled[j] and not personal[j]:
                delivery[r, j] = d_rate[r, j]
                delivery_fixed[r, j] = d_fixed[r, j]
            else:
                forced[j] = True
                delivery[r, j] = fee[r, j] + slip[r, j]
                delivery_fixed[r, j] = fee_fixed[r, j]

        can_buy = tradable.copy()
        can_sell = tradable.copy()
        for mask, target, name in (
            (no_buy, can_buy, "no_buy"),
            (no_sell, can_sell, "no_sell"),
        ):
            if mask is not None:
                target &= ~_panel(mask, shape, name, dtype=bool)
        with np.errstate(invalid="ignore"):
            if limit_up is not None:
                can_buy &= ~(price >= _panel(limit_up, shape, "limit_up"))
            if limit_down is not None:
                can_sell &= ~(price <= _panel(limit_down, shape, "limit_down"))
        self.locked_up = np.zeros(shape, bool)
        self.locked_down = np.zeros(shape, bool)
        if infer_locked:
            bars = [open, high, low, volume]
            if any(b is None for b in bars):
                raise ValueError("infer_locked: needs open, high, low and volume")
            o, h, lo, v = (
                _panel(b, shape, k) for b, k in zip(bars, "ohlv", strict=True)
            )
            with np.errstate(invalid="ignore"):
                one_price = (o == h) & (h == lo) & (v > 0) & future[None, :]
                previous = np.full(shape, np.nan)
                previous[1:] = filled[:-1]
                started = rows > start[None, :]
                self.locked_up = one_price & started & (o > previous)
                self.locked_down = one_price & started & (o < previous)
            can_buy &= ~self.locked_up
            can_sell &= ~self.locked_down

        futures = Futures(
            is_future=future,
            multiplier=np.where(future, mult, 1.0),
            margin_rate=margin,
            settlement=filled,
            expiry=np.where(future, end, t),
            fee_rate=fee,
            fee_per_contract=fee_fixed,
            delivery_fee_rate=delivery,
            delivery_fee_per_contract=delivery_fixed,
        )
        extra = dict(market_extra or {})
        bars = {
            k: _panel(v, shape, k, allow_none=True)
            for k, v in (("open", open), ("high", high), ("low", low))
        }
        self.market = Market(
            exec_price=np.where(tradable, price, np.nan),
            close=_panel(close, shape, "close"),
            can_buy=can_buy,
            can_sell=can_sell,
            volume=None
            if volume is None
            else np.nan_to_num(_panel(volume, shape, "volume")),
            futures=futures,
            validate=validate,
            **bars,
            **extra,
        )
        self.costs = costs
        self.config = config
        self.account = account
        self.margin_rate = margin
        self.end_row = np.where(future, end, t)
        self.forced_close = forced
        self.personal = personal
        self.is_future = future
        self.shape = shape

    # ------------------------------------------------------------ targets
    def targets(self, plan, kind: str = "notional") -> Targets:
        """Convert a plan to kernel `Targets`.

        `plan` is a T×N array (all-NaN row: no rebalance; NaN cell: 0) or a
        `Targets` with kind "weights". `kind`:

        - "lots": signed whole contracts (spot columns: shares);
        - "notional": signed notional value / pretrade equity (spot: weight);
        - "margin": signed margin used / pretrade equity, i.e. notional =
          weight / effective margin rate of that row (spot: weight).
        """
        if kind not in KINDS:
            raise ValueError(f"kind: expected one of {KINDS}")
        if isinstance(plan, Targets):
            if plan.kind != "weights":
                raise ValueError(
                    "plan: pass Targets with kind='weights' and choose kind here"
                )
            days, indptr, cols = plan.days, plan.indptr, plan.cols
            values = np.array(plan.weights, dtype=np.float64)
        else:
            dense = Targets.from_dense(plan)
            days, indptr, cols = dense.days, dense.indptr, dense.cols
            values = np.array(dense.weights, dtype=np.float64)
        if kind == "margin" and len(cols):
            rows = np.repeat(days, np.diff(indptr))
            fut = self.is_future[cols]
            rate = self.margin_rate[rows, cols]
            if np.any(fut & (rate <= 0)):
                bad = sorted(set(cols[fut & (rate <= 0)].tolist()))[:10]
                raise ValueError(
                    f"margin targets on columns {bad} outside their trading window"
                )
            values = np.where(fut, values / np.where(fut, rate, 1.0), values)
        return Targets(
            days,
            indptr,
            cols,
            values,
            kind="shares" if kind == "lots" else "weights",
        )

    # ------------------------------------------------------------ running
    def run(self, plan, kind: str = "notional"):
        from .api import run

        return run(self.market, self.targets(plan, kind), self.costs, self.config)

    def run_batch(self, plans, kind: str = "notional", threads: int = 0):
        from .api import run_batch

        return run_batch(
            self.market,
            [self.targets(p, kind) for p in plans],
            self.costs,
            self.config,
            threads,
        )

    def simulator(self, kind: str = "notional", **kwargs):
        """A `Simulator` over this book; kind "lots" or "notional".

        For margin targets convert each row with `margin_to_notional`.
        """
        from .step import Simulator

        if kind not in ("lots", "notional"):
            raise ValueError("kind: expected lots or notional (see margin_to_notional)")
        return Simulator(
            self.market,
            self.costs,
            self.config,
            kind="shares" if kind == "lots" else "weights",
            **kwargs,
        )

    def margin_to_notional(self, day: int, cols, values):
        """Margin weights of one row -> notional weights (for `Simulator`)."""
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        rate = self.margin_rate[day, cols]
        fut = self.is_future[cols]
        if np.any(fut & (rate <= 0)):
            raise ValueError("margin targets outside the trading window")
        return cols, np.where(fut, values / np.where(fut, rate, 1.0), values)


__all__ = ["FuturesAccount", "FuturesBook"]
