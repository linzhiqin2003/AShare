"""Row-by-row stepping interface over the same kernel as `run`."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _core
from .api import _prepare, _result, _rules
from .models import METRICS, Config, Costs, FloatArray, IntArray, Market, Result

DayRecord = namedtuple("DayRecord", ("day", *METRICS))
DayRecord.__doc__ = "Daily metrics of one stepped row (the same fields as Result)."


def _view(a: np.ndarray) -> np.ndarray:
    v = a.view()
    v.flags.writeable = False
    return v


@dataclass(frozen=True)
class State:
    """What is known before row `day` trades.

    Scalars are fixed when the state is read. The arrays are zero-copy,
    read-only *live* views of the simulator: they change as it steps, so
    `.copy()` anything you want to keep.

    - `day`: the row the next `step` executes.
    - `cash`: cash after row `day`'s external flow, dividends and delisting
      proceeds (known before the open).
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
    ) -> None:
        if kind not in ("weights", "shares"):
            raise ValueError("kind: expected weights or shares")
        config = config or Config()
        costs = costs or Costs()
        floats, masks, _, vectors, scalars, flags, first, actions, futures = _prepare(
            market, [], costs, config
        )
        margin = config.margin
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
            _rules(market, config.rules, False),
        )
        self._core = _core.Simulator(*self._args)
        self._market, self._costs, self._config, self.kind = market, costs, config, kind
        self.n_days, self.n_securities = market.shape
        self._views = tuple(_view(a) for a in self._core.arrays())

    @property
    def day(self) -> int:
        """The next row to step (equals `n_days` when finished)."""
        return self._core.scalars()[0]

    @property
    def state(self) -> State:
        day, cash, equity, call, bankrupt = self._core.scalars()
        shares, last, basis, cost, _peak, entry = self._views
        cols, targets = self._core.pending()
        return State(
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

    def step(self, target: Any = None) -> DayRecord:
        """Trade and close the current row, then open the next one."""
        cols, values, limits = self._row(target)
        day = self.day
        return DayRecord(day, *self._core.step(cols, values, limits))

    def run(self, strategy: Callable[[int, State], Any]) -> Result:
        """Step every remaining row with `strategy(day, state)`; return the result."""
        while self.day < self.n_days:
            self.step(strategy(self.day, self.state))
        return self.result()

    def result(self) -> Result:
        """The finished Result; every row must have been stepped."""
        return _result(self._core.result(), self._market.validated)

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
    ) -> Simulator:
        """Resume from `snapshot` against the same market, costs and config.

        Shapes, account type, target kind, costs and configuration are
        fingerprinted and must match; the market panels are not, so pass the
        same market the snapshot was taken on.
        """
        sim = cls(market, costs, config, kind=kind)
        sim._core.load(bytes(snapshot))
        return sim
