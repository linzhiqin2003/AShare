"""Typed public API. Monetary integers use 1/10,000 currency units; time is epoch ms."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._core import BrokerCore

if TYPE_CHECKING:
    from .journal import QuoteStore

SCALE = 10_000
COLUMNS = (
    "symbol",
    "time",
    "last",
    "open",
    "high",
    "low",
    "previous",
    "volume",
    "amount",
    "ask",
    "ask_size",
    "bid",
    "bid_size",
    "upper",
    "lower",
    "halted",
    "auction_volume",
    "auction_kind",
)
QuoteArray = NDArray[np.int64]
Side = Literal["buy", "sell"]
Phase = Literal[
    "preopen", "open_auction", "continuous", "break", "close_auction", "closed"
]
Event = dict[str, Any]


def money(value: float | str | Decimal) -> int:
    """Convert currency to fixed point, with decimal half-up rounding."""
    return int(
        (Decimal(str(value)) * SCALE).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    )


@dataclass(frozen=True)
class Costs:
    commission_ppm: int = 250
    minimum: int = 50_000
    sell_tax_ppm: int = 500


@dataclass(frozen=True)
class Lots:
    step: int = 100
    minimum: int = 0


@dataclass(frozen=True)
class Instrument:
    """Versioned per-security rule profile; amounts are fixed-point integers."""

    tick: int = 100
    t_plus_one: bool = True
    costs: Costs = field(default_factory=Costs)
    lots: Lots = field(default_factory=Lots)
    profile_id: str = "cash-equity-v1"
    cost_model: str = "simple-v1"
    lot_model: str = "minimum-step-v1"


@dataclass(frozen=True)
class MarketConfig:
    tick: int = 100
    participation_ppm: int = 100_000
    t_plus_one: bool = True
    enforce_bands: bool = True
    slippage_ticks: int = 0
    slippage_ppm: int = 0
    costs: Costs = field(default_factory=Costs)
    lots: Lots = field(default_factory=Lots)
    margin_model: str = "cash"
    utc_offset_seconds: int = 28_800
    instruments: dict[str, Instrument] = field(default_factory=dict)

    @classmethod
    def ashare(cls, **kwargs: Any) -> MarketConfig:
        """A-share cash profile with deliberately simple provisional costs/lots."""
        return cls(**kwargs)


class Broker:
    """Own a native engine, optionally with a synchronous write-ahead journal."""

    def __init__(
        self,
        symbols: list[str],
        initial_cash: float = 1_000_000,
        config: MarketConfig | None = None,
        *,
        journal: str | Path | None = None,
        quote_store: QuoteStore | None = None,
        read_only: bool = False,
        snapshot_retention: int = 3,
        quote_retention_days: int = 20,
    ) -> None:
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be nonempty and unique")
        if snapshot_retention < 1 or quote_retention_days < 0:
            raise ValueError("invalid retention policy")
        self.snapshot_retention = snapshot_retention
        self.quote_retention_days = quote_retention_days
        requested = list(symbols)
        self.config = config or MarketConfig()
        self.symbols = list(symbols)
        settings = asdict(self.config) | {
            "symbols": len(symbols),
            "initial_cash": money(initial_cash),
            "instruments": [self._instrument(s) for s in symbols],
        }
        if journal is not None and (Path(journal) / "manifest.json").exists():
            manifest = json.loads((Path(journal) / "manifest.json").read_bytes())
            if manifest.get("version") != 3:
                raise ValueError(
                    "unsupported journal version; use an explicit offline migration"
                )
            original = manifest["config"]

            def stable(c: dict) -> dict:
                return {
                    k: v for k, v in c.items() if k not in ("symbols", "instruments")
                }

            if stable(original) != stable(settings):
                raise ValueError("journal configuration mismatch")
            for profile in original["instruments"]:
                if profile != self._instrument(profile["name"]):
                    raise ValueError("journal instrument profile mismatch")
            settings = original
            self.symbols = list(manifest["symbols"])
        self.symbol_ids = {s: i for i, s in enumerate(self.symbols)}
        self._core = BrokerCore(json.dumps(settings))
        self._pending: list[Event] = []
        self.history: list[Event] = []
        self.request_id = 0
        self.daily_states: dict[str, Event] = {}
        self.journal = None
        if journal is not None:
            from .journal import Journal

            self.journal = Journal(
                Path(journal), self, quote_store, read_only=read_only
            )
            self._sync_symbols()
            additions = [s for s in requested if s not in self.symbol_ids]
            if additions and not read_only:
                self.append_symbols(additions)

    def _instrument(self, symbol: str) -> Event:
        profile = self.config.instruments.get(symbol) or Instrument(
            tick=self.config.tick,
            t_plus_one=self.config.t_plus_one,
            costs=self.config.costs,
            lots=self.config.lots,
        )
        return asdict(profile) | {"name": symbol}

    def _sync_symbols(self) -> None:
        self.symbols = [
            p["name"] for p in json.loads(self.dump())["config"]["instruments"]
        ]
        self.symbol_ids = {s: i for i, s in enumerate(self.symbols)}

    def append_symbols(self, symbols: list[str], time: int | None = None) -> Event:
        """Append stable IDs through the WAL; retired names are never renumbered."""
        result = self.apply(
            dict(
                type="append",
                time=time if time is not None else json.loads(self.dump())["time"],
                instruments=[self._instrument(s) for s in symbols],
            )
        )
        self._sync_symbols()
        return result

    def _capture(self) -> list[Event]:
        events = json.loads(self._core.events())
        if events:
            day = events[-1]["day"]
            self.history = [e for e in self.history if e["day"] == day]
            self._pending = [e for e in self._pending if e["day"] == day]
            self.daily_states = {
                k: v for k, v in self.daily_states.items() if int(k) == day
            }
        self._pending.extend(events)
        self.history.extend(events)
        if any(e["type"] == "settled" for e in events):
            summary = self.summary()
            self.daily_states = {str(summary["day"]): summary}
            if self.journal is not None and not self.journal.read_only:
                self.journal.archive(summary)
        return events

    def summary(self) -> Event:
        """Compact report data, with positions identified explicitly by symbol ID."""
        state = json.loads(self.dump())
        account = self.account()
        account["positions"] = [
            p
            | dict(
                symbol=self.symbols[i],
                symbol_id=i,
                mark=state["marks"][i],
                market_value=p["total"] * state["marks"][i],
                cost_price=p["cost"] / p["total"] if p["total"] else 0,
            )
            for i, p in enumerate(account["positions"])
            if p["total"] or p["frozen"] or p["cost"]
        ]
        return dict(
            version=3,
            day=state["day"],
            money_scale=SCALE,
            account=account,
            fills=[e for e in self.history if e["type"] == "fill"],
            orders=state["orders"],
            events=list(self.history),
        )

    def apply(self, event: Event) -> Event:
        """Apply an event using fixed-point amounts and integer symbol IDs."""
        encoded = json.dumps(
            event, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if self.journal is not None:
            self.journal.apply(encoded)
        else:
            self._core.apply(encoded)
        self.request_id += 1
        events = self._capture()
        if event["type"] == "append":
            self._sync_symbols()
        return {
            "request_id": self.request_id,
            "accepted": not any(e["type"] == "rejected" for e in events),
            "order_id": next(
                (e["id"] for e in events if e["type"] == "accepted"), None
            ),
            "events": events,
        }

    def feed(self, quotes: QuoteArray, *, received_at: int | None = None) -> None:
        """Borrow a C-contiguous int64 N×18 array without copying it to Rust."""
        if (
            quotes.dtype != np.int64
            or quotes.ndim != 2
            or quotes.shape[1] != len(COLUMNS)
            or not quotes.flags.c_contiguous
        ):
            raise ValueError("quotes must be C-contiguous int64 N x 18")
        if self.journal is not None:
            self.journal.feed(quotes, received_at)
        else:
            self._core.feed(quotes, received_at)
        self._capture()

    def session(self, time: int, day: int, phase: Phase) -> Event:
        return self.apply({"type": "session", "time": time, "day": day, "phase": phase})

    def order(
        self,
        time: int,
        symbol: str | int,
        side: Side,
        qty: int,
        price: float,
        kind: Literal["limit", "market"] = "limit",
    ) -> Event:
        """Submit a DAY order. Market orders require an explicit protection price."""
        return self.apply(
            {
                "type": "order",
                "time": time,
                "symbol": self.symbol_ids.get(symbol, len(self.symbols))
                if isinstance(symbol, str)
                else symbol,
                "side": side,
                "kind": kind,
                "qty": qty,
                "price": money(price),
            }
        )

    def cancel(self, time: int, order_id: int) -> Event:
        return self.apply({"type": "cancel", "time": time, "id": order_id})

    def transfer(self, time: int, amount: float) -> Event:
        return self.apply({"type": "transfer", "time": time, "amount": money(amount)})

    def settle(
        self,
        time: int,
        day: int,
        actions: list[Event] | None = None,
        interest: float = 0,
    ) -> Event:
        return self.apply(
            {
                "type": "settle",
                "time": time,
                "day": day,
                "actions": actions or [],
                "interest": money(interest),
            }
        )

    def target_weights(
        self, time: int, weights: dict[str, float], *, protection_ticks: int = 2
    ) -> list[Event]:
        """Reconcile targets, retaining compatible resting orders and sharing buying power.

        Call again after fills. Sales release cash on a later quote, so rotating a
        portfolio intentionally takes multiple reconciliation passes.
        """
        if (
            set(weights) - self.symbol_ids.keys()
            or any(not np.isfinite(w) or w < 0 for w in weights.values())
            or sum((Decimal(str(w)) for w in weights.values()), Decimal(0)) > 1
            or protection_ticks < 0
        ):
            raise ValueError("invalid target weights")
        state = json.loads(self.dump())
        equity = self.account()["equity"]
        plans = []
        results = []
        for i, symbol in enumerate(self.symbols):
            quote = state["quotes"][i]
            if quote is None or quote[15]:
                continue
            profile = state["config"]["instruments"][i]
            step = profile["lots"]["step"]
            minimum = max(step, profile["lots"]["minimum"])
            target = (
                int(Decimal(str(weights.get(symbol, 0))) * equity / quote[2])
                // step
                * step
            )
            delta = target - state["positions"][i]["total"]
            resting = [
                o for o in state["orders"] if o["symbol"] == i and o["status"] == "open"
            ]
            pending = sum(
                o["remaining"] * (1 if o["side"] == "buy" else -1) for o in resting
            )
            if pending and delta * pending > 0 and abs(pending) <= abs(delta) + step:
                delta -= pending
            else:
                for o in resting:
                    results.append(self.cancel(time, o["id"]))
            if not delta:
                continue
            side: Side = "buy" if delta > 0 else "sell"
            tick = profile["tick"]
            touch = quote[9 if side == "buy" else 11]
            protection = (touch if touch > 0 else quote[2]) + (
                1 if side == "buy" else -1
            ) * protection_ticks * tick
            protection = (
                (protection + tick - 1) // tick if side == "buy" else protection // tick
            ) * tick
            if quote[13] > 0:
                protection = min(protection, quote[13])
            if quote[14] > 0:
                protection = max(protection, quote[14])
            qty = abs(delta)
            if side == "sell":
                available = self.account()["positions"][i]["sellable"]
                qty = min(qty, available)
                if step > 1 and qty % step not in (0, available % step):
                    # Preserve the whole residual lot, or use whole lots only.
                    residual = available % step
                    qty = (
                        ((qty - residual) // step * step + residual)
                        if qty >= residual
                        else qty // step * step
                    )
                if step == 1 and qty < minimum and qty != available:
                    qty = 0
            else:
                qty = qty // step * step
            if qty and (
                qty >= minimum
                or side == "sell"
                and (step > 1 or qty == self.account()["positions"][i]["sellable"])
            ):
                plans.append([side, i, qty, protection, profile])

        def required(plan: list) -> int:
            _, _, qty, price, profile = plan
            return qty * price + max(
                profile["costs"]["minimum"],
                (qty * price * profile["costs"]["commission_ppm"] + 999_999)
                // 1_000_000,
            )

        buys = [p for p in plans if p[0] == "buy"]
        cash = self.account()["cash"]
        need = sum(required(p) for p in buys)
        if need > cash:
            # Allocate in proportion to all shortfalls, not symbol iteration order.
            for p in buys:
                step = p[4]["lots"]["step"]
                p[2] = p[2] * cash // need // step * step
        for side, i, qty, protection, profile in sorted(
            plans, key=lambda p: p[0] == "buy"
        ):
            if side == "buy":
                step = profile["lots"]["step"]
                while (
                    qty
                    and required([side, i, qty, protection, profile])
                    > self.account()["cash"]
                ):
                    qty -= step
                if qty < max(step, profile["lots"]["minimum"]):
                    continue
            if qty:
                results.append(
                    self.apply(
                        dict(
                            type="order",
                            time=time,
                            symbol=i,
                            side=side,
                            kind="limit",
                            qty=qty,
                            price=protection,
                        )
                    )
                )
        return results

    def account(self) -> Event:
        """Return exact integers; cost is total cost basis, not price per share."""
        return json.loads(self._core.account())

    def events(self, after: int | None = None) -> list[Event]:
        """Drain today's pending events, or read durable events strictly after a cursor.

        Persist the last consumed ``seq`` to deduplicate after a restart. Archived
        days are read on demand when a journal is configured.
        """
        if after is None:
            events, self._pending = self._pending, []
            return events
        if after < 0:
            raise ValueError("event cursor must be nonnegative")
        events = {}
        if self.journal:
            for path in sorted((self.journal.directory / "days").glob("*.json")):
                for e in json.loads(path.read_bytes())["events"]:
                    if e["seq"] > after:
                        events[e["seq"]] = e
        for e in self.history:
            if e["seq"] > after:
                events[e["seq"]] = e
        return [events[key] for key in sorted(events)]

    def dump(self) -> bytes:
        """Return canonical deterministic state bytes (excluding the output queue)."""
        return self._core.dump().encode()

    def checkpoint(self) -> None:
        if self.journal is None:
            raise ValueError("no journal configured")
        self.journal.checkpoint()

    def close(self) -> None:
        if self.journal is not None:
            self.journal.close()

    def __enter__(self) -> Broker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
