"""Row-by-row stepping interface over the same kernel as `run`."""

from __future__ import annotations

import hashlib
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _core
from .api import _prepare, _result, _rules
from .models import (
    CRYPTO_METRICS,
    METRICS,
    Config,
    Costs,
    FloatArray,
    IntArray,
    Market,
    Result,
    _LegacyWalletResult,
    _MissingWallet,
)

DayRecord = namedtuple("DayRecord", ("day", *METRICS))
CryptoDayRecord = namedtuple("CryptoDayRecord", ("day", *METRICS, *CRYPTO_METRICS))
WalletDayRecord = namedtuple(
    "WalletDayRecord",
    ("day", *METRICS, *CRYPTO_METRICS, "wallet_balance", "unrealized"),
)
DayRecord.__doc__ = "Daily metrics of one stepped row (the same fields as Result)."


def _view(a: np.ndarray) -> np.ndarray:
    # A view over a read-only buffer export: numpy refuses to make it
    # writeable again, so callers cannot modify the simulator's state.
    return np.frombuffer(memoryview(a).toreadonly(), dtype=a.dtype).reshape(a.shape)


_SAMPLE = 65536


def _market_digest(arrays, n: int, full: bool) -> bytes:
    """Digest of the market panels for the snapshot fingerprint.

    "sample" hashes each array's shape and dtype, its first and last rows and
    a fixed-stride sample of about 64K cells; "full" hashes every byte.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(b"full" if full else b"sample")
    for a in arrays:
        flat = np.ascontiguousarray(a).reshape(-1)
        h.update(f"{np.shape(a)}{flat.dtype.str}".encode())
        if full:
            h.update(flat.tobytes())
            continue
        h.update(flat[:n].tobytes())
        h.update(flat[-n:].tobytes())
        h.update(flat[:: max(1, flat.size // _SAMPLE)].tobytes())
    return h.digest()


@dataclass(frozen=True)
class State:
    """What is known before row `day` trades.

    Scalars are fixed when the state is read. The arrays are zero-copy,
    read-only *live* views of the simulator (writing through internals that
    bypass the read-only buffer is undefined behaviour): they change as it steps, so
    `.copy()` anything you want to keep.

    - `day`: the row the next `step` executes.
    - `cash`: cash after row `day`'s static external flow, dividends and delisting
      proceeds, but before the next `step(flow=...)` (known before the open).
    - `equity`: equity at the close of row `day - 1` (initial cash on row 0).
    - `shares`: holdings after row `day`'s splits and delistings.
    - `cost`: quantity-weighted raw execution price of each open position
      (split adjusted; dividends do not change it). 0 when flat.
    - `entry_day`: row on which each open position was entered; -1 when flat.
    - `last`: valuation mark of held columns: the latest valid close through
      row `day - 1`, adjusted for row `day`'s dividends and splits.
      Not meaningful for columns that are not held.
    - `futures_basis`: settlement basis of held futures (last settlement).
    - `pending_cols`, `pending_targets`: price-, mask- or volume-blocked
      orders still being retried (`carry_blocked`).
    - `margin_call`: a maintenance call at the last close; row `day` will be
      a forced-liquidation row. `bankrupt`: the account has failed.
    - `wallet_balance`, `unrealized`: exchange futures W and previous-close U;
      None without exchange liquidation. W includes prepared static flows.
    - `entry_price`: effective opening averages, excluding fees, distinct from
      `cost`; a read-only copy (empty for non-wallet accounts).
    - `notional`, `initial_margin`: absolute futures notional at the previous
      settlement, and margin at that price with the next row's effective rate.
    - `transferable`: dynamic withdrawal limit at this prepared boundary;
      cash for cash-only spot, max(0, W + min(U, 0) - IM) for exchange futures.
    """

    day: int
    cash: float
    equity: float
    margin_call: bool
    bankrupt: bool
    shares: FloatArray
    cost: FloatArray
    entry_day: IntArray
    last: FloatArray
    futures_basis: FloatArray
    pending_cols: IntArray
    pending_targets: FloatArray
    wallet_balance: float | None = None
    unrealized: float | None = None
    transferable: float = 0.0
    entry_price: FloatArray | None = None
    notional: float = 0.0
    initial_margin: float = 0.0


class _LegacyWalletState(_MissingWallet, State):
    pass


class _LegacyWalletDay(_MissingWallet, CryptoDayRecord):
    pass


class Simulator:
    """Advance one backtest a row at a time; the same result as `bt.run`.

    ```python
    sim = bt.Simulator(market, costs, config)
    for t in range(sim.n_days):
        target = strategy(t, sim.state)   # None, a length-N row, or (cols, values)
        sim.step(target)
    result = sim.result()                 # identical to bt.run with those targets
    ```

    Row `t`'s target is what to hold after row `t`'s trading; nothing is
    shifted. Targets are weights or, with `kind="shares"`, quantities. A dense
    row treats NaN as zero; an all-NaN dense row (or `None`) gives no new
    target. With `Config(rules=...)`, `(cols, values, limits)` attaches limit
    prices (NaN: ordinary target). Prefer `(cols, values)` for large universes: a dense row costs
    O(N) per step. `step` releases the GIL, so independent simulators can
    advance in parallel threads.
    """

    def __init__(
        self,
        market: Market,
        costs: Costs | None = None,
        config: Config | None = None,
        *,
        kind: str = "weights",
        fingerprint: str = "sample",
    ) -> None:
        if kind not in ("weights", "shares"):
            raise ValueError("kind: expected weights or shares")
        if fingerprint not in ("sample", "full"):
            raise ValueError("fingerprint: expected sample or full")
        config = config or Config()
        costs = costs or Costs()
        floats, masks, _, vectors, scalars, flags, first, actions, futures = _prepare(
            market, [], costs, config
        )
        margin = config.margin
        rules = _rules(market, config.rules, False)
        panels = [*floats, *masks, *futures[3]] + (list(rules[0][:3]) if rules else [])
        digest = _market_digest(panels, market.shape[1], fingerprint == "full")
        self._args = (
            floats,
            masks,
            vectors,
            scalars,
            flags,
            first,
            actions,
            futures,
            kind == "shares",
            (margin.allow_short, not margin.allow_financing and not margin.allow_short),
            market.validated,
            rules,
            digest,
        )
        self._legacy_wallet = False
        self._core = _core.Simulator(*self._args)
        self._market, self._costs, self._config, self.kind = market, costs, config, kind
        self.n_days, self.n_securities = market.shape
        self._views = tuple(_view(a) for a in self._core.arrays())

    @property
    def day(self) -> int:
        """The next row to step (equals `n_days` when finished)."""
        return self._core.scalars()[0]

    def opening_feasible(self, target, prices, *, flow=0.0) -> bool:
        """Preview collateral at supplied reference prices using Rust fill checks.

        Dense targets are contract counts. The prepared cash plus hypothetical
        flow funds reductions and then openings at the supplied prices, using
        effective fee/margin terms. This is read-only, does not clip the flow,
        and excludes liquidity, filters, price gaps, slippage and liquidation.
        Only available with a current exchange-liquidation wallet ledger.
        """
        return self._core.opening_feasible(
            np.ascontiguousarray(target, dtype=float).reshape(-1),
            np.ascontiguousarray(prices, dtype=float).reshape(-1),
            float(flow),
        )

    @property
    def state(self) -> State:
        day, cash, equity, call, bankrupt = self._core.scalars()
        shares, last, basis, cost, _peak, entry = self._views
        cols, targets = self._core.pending()
        wallet, unrealized, transferable, entry_price, notional, initial_margin = (
            (None, None, 0.0, np.empty(0), 0.0, 0.0)
            if self._legacy_wallet
            else self._core.wallet()
        )
        state_type = _LegacyWalletState if self._legacy_wallet else State
        return state_type(
            day,
            cash,
            equity,
            call,
            bankrupt,
            shares,
            cost,
            entry,
            last,
            basis,
            cols,
            targets,
            wallet,
            unrealized,
            transferable,
            _view(entry_price),
            notional,
            initial_margin,
        )

    def _row(self, target: Any):
        if target is None:
            return None, None, None
        if isinstance(target, tuple):
            if len(target) not in (2, 3):
                raise ValueError(
                    "target: expected (cols, values) or (cols, values, limits)"
                )
            cols = np.asarray(target[0])
            if cols.size == 0:
                cols = cols.astype(np.int64)
            if cols.dtype.kind not in "iu":
                raise ValueError("cols: expected integer column indices")
            cols = np.ascontiguousarray(cols, dtype=np.int64).ravel()
            values = np.ascontiguousarray(target[1], dtype=np.float64).ravel()
            limits = (
                None
                if len(target) == 2 or target[2] is None
                else np.ascontiguousarray(target[2], dtype=np.float64).ravel()
            )
            if cols.shape != values.shape:
                raise ValueError("cols/values: expected equal lengths")
            if limits is not None and limits.shape != values.shape:
                raise ValueError("limits: expected one limit per target entry")
            if (
                limits is not None
                and self._market.futures is not None
                and np.isfinite(limits).any()
            ):
                raise ValueError(
                    "limits: limit targets are not supported in markets with futures columns"
                )
            if cols.size > 1 and np.any(cols[1:] <= cols[:-1]):
                order = np.argsort(cols, kind="stable")
                cols, values = cols[order], values[order]
                if limits is not None:
                    limits = limits[order]
                if np.any(cols[1:] == cols[:-1]):
                    raise ValueError("cols: duplicate columns in a target row")
            return cols, values, limits
        row = np.asarray(target, dtype=np.float64)
        if row.shape != (self.n_securities,):
            raise ValueError(f"target: expected length N={self.n_securities} row")
        nan = np.isnan(row)
        if nan.all():
            return None, None, None
        cols = np.flatnonzero((row != 0) & ~nan)
        return cols.astype(np.int64), np.ascontiguousarray(row[cols]), None

    def step(self, target: Any = None, *, flow: float | None = None) -> DayRecord:
        """Apply optional flow before opening checks, trade, close, prepare next row.

        Positive flow deposits, negative flow withdraws up to `transferable`.
        Any explicit flow conflicts with a nonzero static Config.cash_flow.
        Bankrupt exchange wallets reject both directions. The returned
        cash_flow is actual; withdrawal_shortfall is actual minus requested.
        Omitting flow preserves the existing static-flow path.
        """
        cols, values, limits = self._row(target)
        day = self.day
        values = self._core.step(cols, values, limits, flow)
        if self._legacy_wallet:
            return _LegacyWalletDay(day, *values)
        if (
            self._market.futures is not None
            and self._market.futures.liquidation is not None
        ):
            return WalletDayRecord(day, *values)
        if self._market.futures is not None and self._market.futures.has_crypto:
            return CryptoDayRecord(day, *values)
        return DayRecord(day, *values[: len(METRICS)])

    def run(self, strategy: Callable[[int, State], Any]) -> Result:
        """Step every remaining row with `strategy(day, state)`; return the result."""
        while self.day < self.n_days:
            self.step(strategy(self.day, self.state))
        return self.result()

    def result(self) -> Result:
        """The finished Result; every row must have been stepped."""
        result = _result(
            self._core.result(),
            self._market.validated,
            bool(self._market.futures and self._market.futures.has_crypto),
        )

        if self._legacy_wallet:
            result.__class__ = _LegacyWalletResult
        return result

    def snapshot(self) -> bytes:
        """Serialize the simulation; restore it with `Simulator.restore`."""
        return bytes(self._core.snapshot())

    @classmethod
    def restore(
        cls,
        snapshot: bytes,
        market: Market,
        costs: Costs | None = None,
        config: Config | None = None,
        *,
        kind: str = "weights",
        fingerprint: str = "sample",
    ) -> Simulator:
        """Resume from `snapshot` against the same market, costs and config.

        Shapes, account type, target kind, costs, configuration, rule
        parameters and a digest of the market panels are fingerprinted and must
        match. `fingerprint="sample"` (default) digests each panel's shape,
        first and last rows and a fixed-stride sample of about 64K cells;
        `"full"` hashes every cell (slower); use the same choice as when the
        snapshot was taken.
        """
        sim = cls(market, costs, config, kind=kind, fingerprint=fingerprint)
        sim._core.load(bytes(snapshot))
        sim._legacy_wallet = bool(
            market.futures
            and market.futures.liquidation is not None
            and not snapshot.startswith(b"PBTSTEP4")
        )
        return sim
