# paperbroker

Deterministic, event-driven paper trading with a Rust kernel and a typed NumPy API.
Runtime dependency: NumPy only. No network access, filesystem access, wall clock,
or Python dependencies in the kernel. Licensed under Apache-2.0.

## Quick start

```python
import numpy as np
from paperbroker import Broker, COLUMNS, money

broker = Broker(["EXAMPLE"], initial_cash=100_000)
broker.session(0, 20261111, "continuous")
row = [0, 1, money(10), money(10), money(10), money(10), money(10),
       0, 0, money(10), 1000, money(9.99), 1000,
       money(11), money(9), 0, 0, 0]
broker.feed(np.array([row], dtype=np.int64))  # Establish volume baseline.
broker.order(2, "EXAMPLE", "buy", 100, 10.5)
row[1], row[7] = 2, 0
broker.feed(np.array([row], dtype=np.int64))  # Observe volume at submission.
row[1], row[7] = 3, 1000
broker.feed(np.array([row], dtype=np.int64))
print(broker.events())
print(broker.account())
broker.session(4, 20261111, "closed")
broker.settle(5, 20261111)
```

`order(..., kind="market")` requires a protection price. Both order types are DAY
orders; unfilled quantities remain open until cancelled or settled. New orders do
not consume an already observed quote. Order calls return a request ID, acceptance flag, acceptance order ID (if any), and output events. Orders receive increasing acceptance IDs;
`cancel(time, order_id)` releases their remaining reservations.

## Array and monetary contract

`feed()` borrows a C-contiguous `int64[N, 18]` NumPy array without copying it into
Rust. It retains the GIL during the borrow. Do not mutate that array concurrently
from native threads. Input validation completes before any row is processed.
Columns, in order (also exported as `COLUMNS`):

| Index | Field | Unit |
|---|---|---|
| 0 | symbol | index in the constructor's symbol list |
| 1 | time | epoch milliseconds; synthetic monotone integers also work |
| 2–6 | last, open, high, low, previous | 1/10,000 currency unit |
| 7 | volume | cumulative shares for the current trading day |
| 8 | amount | cumulative 1/10,000 currency units |
| 9–12 | ask, ask_size, bid, bid_size | fixed-point price, shares |
| 13–14 | upper, lower | fixed-point bands; zero disables that bound |
| 15 | halted | 0 or 1 |
| 16 | auction_volume | shares in this particular auction |
| 17 | auction_kind | 0 regular; 1 opening; 2 closing |

A book price of **-1 means missing**, enabling last-price fallback. **0 means a
known empty book**, so that side does not execute. At the upper/lower band,
missing or empty opposing depth blocks buys/sells respectively. The first quote establishes a baseline and has no executable continuous volume. An order can only use a volume interval whose previous observation was received at or after submission; an interval straddling submission is conservatively excluded. Older quote times or decreasing cumulative volume are ignored during trading. Equal timestamps are ignored except a one-time regular-to-auction enrichment with identical price, book and cumulative volume, in the corresponding auction phase. In `closed`, a newer authoritative daily bar may correct volume and valuation without matching. An explicit new-day
session resets quote baselines while preserving valuation marks.

`money(10.23)` converts currency to fixed point using decimal half-up rounding.
All amounts returned by `account()`, `events()` and `dump()` remain exact integers.
`Position.cost` is total cost basis, including purchase fees; divide it by total
shares for average cost. `cash` is available cash; `frozen` is separate reserved
cash. Equity is `cash + frozen + market_value`. Position `sellable` excludes
frozen sell orders and same-day buys under T+1.

The provisional cost model uses cumulative order notional: commission is charged
up to the order minimum on the first partial fill, with only incremental amounts
on later fills. Tax is charged only on sells. Fees round up to one monetary unit.
The default lot rule is 100-share buy quantities. A sell may contain the entire residual lot plus any whole lots: with 250 available shares, selling 50, 100, 150, 200 or 250 is valid; 30 or 130 is not. Step-one profiles permit a below-minimum sell only when it exhausts the available balance. `Instrument` overrides tick, settlement lag, fees, minimum quantity and quantity step per security; a minimum of 200 with step 1 expresses STAR-style declarations, and minimum 100 / step 1 expresses BSE-style declarations. An ETF tick of 10 expresses 0.001 currency units. Executions may be smaller than an order lot. These are intentionally **not**
a complete exchange fee or order-rule implementation.

One quote's participation budget is shared by **all** orders and both directions;
each direction also shares its observed depth. Orders match in acceptance order.
Slippage moves prices against the order, rounds to the configured tick, and must
still satisfy both protection price and bands. Auctions use their explicit auction
price/volume, not continuous-book depth. Only orders predating the auction quote
can participate, and each symbol's auction executes at most once per day.

For production ingestion, pass `feed(quotes, received_at=...)`: this event clock
orders commands independently of the per-row exchange time. Quote times determine
freshness, trading date and whether an order predates an auction; receipt times
determine which observed volume interval an order may consume. A quote stamped
in the future cannot advance the receipt clock. If `received_at` is omitted, the current event time is retained; no exchange timestamp is promoted to an event clock. Epoch quotes
from another trading date are ignored. Small synthetic clocks are supported for
tests and have no civil date.

## Events, configuration and settlement

`MarketConfig.ashare()` supplies the provisional T+1 cash profile. Tick size,
participation (parts per million), price-band enforcement, slippage, T+0/T+1, fees
and lot step are configurable. Sessions are external events, so the caller owns
the exchange calendar and session table. A fixed UTC offset (default +08:00) maps epoch quote dates to the event-supplied trading day; it never reads a timezone database or wall clock.

Public operations: `feed`, `order`, `cancel`, `session`, `target_weights`,
`transfer`, `settle`, `account`, `events`, `dump`, `checkpoint`, `close`.
`apply()` additionally accepts raw fixed-point event dictionaries. `target_weights`
retains compatible resting orders, submits sells before buys, and distributes available buying power proportionally across shortfalls. Protection defaults to the visible touch plus/minus two ticks and is configurable.
Unlisted symbols target zero; buys use currently available cash and never assume
that outstanding sells have filled. Re-evaluate targets after sells fill if needed.

Output types: `appended`, `accepted`, `rejected`, `fill`, `ended`, `account`, `session`,
`corporate`, `settled`. Rejection reasons: `session`, `unknown_symbol`, `no_quote`,
`suspended`, `quantity`, `tick`, `price_band`, `cash`, `sellable`, `unknown_order`,
`time`, `already_settled`, `invalid_action`.

Settlement requires `closed` and the current day, expires DAY orders, applies any
explicit corporate actions, unlocks positions and emits an account snapshot. Call
it once per trading day before advancing the day. The raw `corporate` event is
available in `preopen`, before outstanding orders, to apply ex-date actions to
overnight holdings. An optional `day` on that event identifies the historical ex-date; process dates in ascending order before trading. The persisted `actions_through` cursor and per-security action dates make outage catch-up idempotent. Action fields:

```python
{"symbol": 0, "cash_per_share": money(0.1), "split_num": 3,
 "split_den": 2, "reference_price": None}
```

Cash entitlement uses pre-action shares. Integer share multiplication rounds down;
fractional compensation and delayed payment/listing dates are not modeled. The
mark defaults to `(previous_mark - cash_per_share) / split_ratio`; a known
post-action reference price can be supplied. Each symbol permits one action per
ex-date; combine multiple entitlements upstream. Do not pass the same ex-date action
at both preopen and settlement. Signed positions and a reserved `margin_model`
configuration field permit later extensions, but only `"cash"` is accepted today.
Rust `CostModel` and `LotRule` receive a symbol ID and its optional instrument profile. Profiles persist model IDs (`simple-v1`, `minimum-step-v1`) and parameters, so P3 can explicitly version any replacement. No complete exchange fee or margin schedule is implemented.

## Durability and replay

```python
with Broker(["EXAMPLE"], journal="run-directory") as broker:
    # Use normal operations; every state-changing input is write-ahead logged.
    broker.checkpoint()
```

The journal first trial-runs each input against a disposable engine, so malformed or overflowing inputs cannot poison recovery. It then writes immutable quote recordings, flushes them (`F_FULLFSYNC` on macOS), writes and flushes
checksummed JSON lines referring to those recordings, then invokes the kernel.
It repairs only an incomplete trailing line. Complete malformed lines, hash
mismatches and missing recordings fail recovery. A process lock excludes a second
writer. State snapshots are atomic, checksummed and anchored to a journal prefix;
recovery uses the latest matching valid snapshot and replays later inputs. A
truncated log cannot accidentally load a snapshot from its discarded future.
The manifest fixes the original identity and configuration. `append_symbols` journals new profiles and stable IDs; reopening with a changed or reordered requested universe preserves every previous ID, including retired names. Rule changes require a new account or an explicit migration. After any append failure,
close and recover before continuing. Snapshots contain the current state, today's outputs and today's compact report summary. Settled reports are archived once in `days/`; history and terminal orders are pruned on rollover. Recovery regenerates outputs after the snapshot and repairs missing daily archives. Every output has a monotonic `seq`, persisted in state. `events()` drains today's pending events with at-least-once delivery after recovery. For restart-safe consumers, persist the last acknowledged sequence and use `events(after=sequence)` (strictly greater); historical days are loaded on demand. Without a journal only today's events are available. This cursor does not make a consumer's external side effects exactly-once.

The standalone package uses compressed `.npz` quote recordings to keep NumPy as its only
runtime dependency. `QuoteStore` is an injectable protocol; an application can
provide Parquet (the AShare adapter does). The text log never embeds full quotes.
`paperbroker.replay(broker, recordings, script)` accepts ordered
`(receipt_time, quote_array)` pairs and event dictionaries, applying commands before
quotes at equal receipt times. Its result is the full output event list; account
and canonical state remain available on the supplied broker.

WAL, manifest and snapshot formats are version 3. Earlier versions are explicitly rejected; there is no automatic financial-history migration. `read_only=True` replays complete lines without locks, tail truncation or filesystem writes. Durable file locks and directory fsync currently target POSIX systems. Windows
journal support is not claimed. The pure matching engine itself has no OS IO.

## Build and verification

Build with `python -m pip install .` or `maturin develop --release`. A source
archive created by maturin includes the path-dependent `paperbroker-core` crate;
it does not require AShare or the other workspace crates. No machine-specific
paths or credentials belong in the package. Synthetic tests live in `tests/`;
`paperbroker.reference` independently implements the state machine in plain Python.

## Deliberate limits

Market orders are protected DAY orders, not exchange-specific IOC/FOK variants.
The core accepts cancellations in every phase; it rejects new orders during
breaks. Exchange-specific no-cancel windows and lunch queuing are not implemented.
`snapshot_retention=3` keeps the latest three validated snapshots; `quote_retention_days=20` keeps recordings for the latest twenty observed trading days (0 disables quote cleanup). `checkpoint()` prunes old snapshots only after publishing and validating the replacement. `broker.journal.gc()` additionally removes old quote files behind every retained recovery point. It never deletes the WAL or settled daily reports. After quote GC, startup needs a retained valid snapshot; full replay from inception requires an independently retained recording or backup. Losing all post-GC snapshots is an explicit recovery error. Report archives still grow with meaningful audit history; snapshot size and current-day memory do not grow with account age.

Workspace release builds enable overflow checks for the generated binding code as well as the kernel. The Python extension catches arithmetic panics, restores its prior state and raises `OverflowError`. Journal preflight keeps such inputs out of both the WAL and quote store. Investigate the rejected input before retrying; it is not silently saturated or wrapped. POSIX durability and crash recovery are tested with injected failures, not physical power loss.

The Rust cdylib is named `paperbroker_native`; its Python module remains
`paperbroker._core`. This avoids shared Cargo target collisions with other packages.
