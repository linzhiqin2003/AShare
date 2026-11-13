# panelbt

A deterministic, market-agnostic daily portfolio engine. NumPy is the only
Python runtime dependency; the Rust kernel shares per-fill primitives with
`tradecore`. Licensed under Apache-2.0.

## Minimal example

```python
import numpy as np
import panelbt as bt

prices = np.array([[10.0, 20.0], [11.0, 19.0], [12.0, 21.0]])
market = bt.Market(prices, prices, prices > 0, prices > 0)
targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
costs = bt.Costs(commission_rate=0, commission_min=0,
                 fee_rate_buy=0, fee_rate_sell=0, slippage=0)
config = bt.Config(initial_cash=1000, min_qty=0, qty_step=0)
result = bt.run(market, targets, costs, config)
np.testing.assert_allclose(result.nav, [1.0, 1.025, 1.125])
```

Targets execute on the supplied row: **the engine never shifts signals**.
An all-NaN row means no rebalance; an all-zero row closes the portfolio.
`Targets.from_dense(..., kind="shares")` accepts exact signed share targets,
including fractional/odd shares. Negative targets require
`Config(margin=MarginModel(allow_short=True, ...))`.

## Pure weight accounting

`run_weights(target, fwd_ret, can_buy=None, can_sell=None)` and
`run_weights_batch(fwd_ret, targets, ..., keep_weights=True)` use a separate
accounting mode in the same Rust engine. They accept execution-day targets;
the caller aligns signals to execution days. There is no cash, quantity, lot
rounding, fee, dividend ledger, or price-driven weight drift. Each day the engine
resets to the target, freezes blocked increases/decreases at the previous weight,
and rescales free long and short positions to their remaining budgets. PnL is
held weights dotted with holding-period returns; turnover is the sum of absolute
weight changes (both sides). NaNs are zero and infinities follow `nan_to_num`.

```python
result = bt.run_weights(
    np.array([[0.5, -0.5], [0.0, 0.0]]),
    np.array([[0.02, -0.01], [0.0, 0.0]]),
    keep_weights=False,
)
```

`GroupWeightTargets(labels, long_group, short_group=-1, long_budget=1,
short_budget=0)` shares integer labels across plans without expanding a dense
target panel per group. `keep_weights` can also be a per-plan list in batch mode.
The GIL is released during computation; concurrent callers must keep inputs
unchanged. The independent NumPy oracle lives in `panelbt.reference_weights`.

`run_group_weights` additionally streams neutralization, stable grouping and
style exposures through the same weight ledger. It requires explicit
`signal_start` and `execution_start` indices, full raw price/adjustment history,
and active-window factors, log market caps, industry labels and pool masks.
It returns daily series, not dense held-weight panels. Cross-sectional kernels
are statically shared with `panelops-core`; the Python runtime still requires
only NumPy. NumPy-order reductions and its installed float64 dot kernel preserve
bitwise results, including signed zero. This research mode does not approximate
the cash/share accounting performed by `run` or `run_batch`.

## Market adapters

```python
from panelbt.markets import ASHARE, Board

market = ASHARE.market(prices, prices, high=prices, low=prices,
                      up_limit=prices * 1.1, down_limit=prices * 0.9)
costs = ASHARE.costs(["2025-01-02", "2025-01-03", "2025-01-06"],
                      commission_rate=0.00025, commission_min=5)
config = ASHARE.config(board=[Board.MAIN, Board.STAR])
```

`ASHARE.costs(dates)` expands the dated A-share stamp duty and ChinaClear
transfer fee from 2005-01-24 onwards, so a full-history backtest needs no
manual fee arrays:

| Effective | Stamp duty | Transfer fee (both sides) |
|---|---|---|
| 2005-01-24 | 0.1% both sides | 0.001 per share + 0.0255‰ of amount (unverified, conservative; see below) |
| 2007-05-30 | 0.3% both sides | |
| 2008-04-24 | 0.1% both sides | |
| 2008-09-19 | 0.1% sell only | |
| 2012-09-01 | | 0.0003 per share + 0.0255‰ of amount |
| 2015-08-01 | | 0.02‰ of amount |
| 2021-11-15 | | 0.025‰ of amount (BSE rate, conservative) |
| 2022-04-29 | | 0.01‰ of amount |
| 2023-08-28 | 0.05% sell only | |

Before 2015-08-01 Shanghai charged the transfer fee on par value (per share)
and Shenzhen on amount. Engine fees vary by date, not by security, so both are
charged to every security: an upper bound. Sources and verification status for
every row are in the comment above `ASHARE` in `panelbt/markets/__init__.py`.
Broker commission is not market data; pass it explicitly. Overriding one form
of a directional transfer fee (`fee_rate_buy` or `fee_per_share_buy`) zeroes
the other scheduled form, so a fixed-rate setup is explicit:
`ASHARE.costs(dates, tax_buy=0, tax_sell=0.001, fee_rate_buy=1e-5, fee_rate_sell=1e-5)`.
`infer_board` accepts six-digit codes with or without `.SH`/`.SZ`/`.BJ`; prefer
an authoritative board column where you have one.

To build a custom profile, use `MarketProfile` directly (the built-in Hong Kong
and US profiles are in the next section). The following **illustrative rates
are not verified market defaults; verify them before use**. These examples reuse
`prices` and `targets` above.

```python
from panelbt.markets import FeeSchedule, MarketProfile

hk_style = MarketProfile("HK-style illustration", "HKD", 252,
    FeeSchedule((("2025-01-01", {"tax_buy": 0.001, "tax_sell": 0.001}),)))
dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
hk_config = hk_style.config(lot_sizes=[100, 500], initial_cash=100000)
hk_costs = hk_style.costs(dates, commission_rate=0.001,
                         commission_min=0, slippage=0)
hk_result = bt.run(hk_style.market(prices, prices), targets, hk_costs, hk_config)

us_style = MarketProfile("US-style illustration", "USD", 252,
    FeeSchedule((("2025-01-01", {"fee_per_share_sell": 0.001}),)))
us_config = us_style.config(lot_sizes=[0, 0], initial_cash=1000)
us_costs = us_style.costs(dates, commission_rate=0, slippage=0,
    commission_per_share=0.005, commission_min=1, commission_max_rate=0.01)
us_result = bt.run(us_style.market(prices, prices), targets, us_costs, us_config)
```

## Hong Kong and US profiles

Three built-in profiles cover a local Hong Kong account, Southbound Stock
Connect (mainland investors buying Hong Kong shares) and US equities. Pick one
by name; `costs`, `config` and `market` work as for `ASHARE`:

```python
import numpy as np
import panelbt as bt
from panelbt import markets

prices = np.array([[50.0, 8.0], [52.0, 8.2], [55.0, 7.9]])
dates = ["2026-03-02", "2026-03-03", "2026-03-04"]
targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
hk = markets.get("hk_direct")             # local HK account, in HKD
config = hk.config(lot_sizes=[500, 2000], h_share=[True, True], initial_cash=1e6)
result = bt.run(hk.market(prices, prices), targets, hk.costs(dates), config)
```

```python
import numpy as np
import panelbt as bt
from panelbt import markets

prices = np.array([[50.0, 8.0], [52.0, 8.2], [55.0, 7.9]])
dates = ["2026-03-02", "2026-03-03", "2026-03-04"]
targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
sb = markets.get("hk_southbound")         # Stock Connect, backtested in HKD
eligible = np.array([[True, True], [True, False], [True, False]])  # leaves Connect
config = sb.config(lot_sizes=[500, 2000], initial_cash=1e6)
market = sb.market(prices, prices, connect_eligible=eligible)
result = bt.run(market, targets, sb.costs(dates), config)
cny = markets.connect_home_equity(result.equity, result.cash, [0.92, 0.921, 0.919], 1e6)
```

```python
import numpy as np
import panelbt as bt
from panelbt import markets

prices = np.array([[50.0, 8.0], [52.0, 8.2], [55.0, 7.9]])
dates = ["2026-03-02", "2026-03-03", "2026-03-04"]
targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
us = markets.get("us")                    # whole shares, 10% treaty withholding
config = us.config(n=2, initial_cash=1e5)
result = bt.run(us.market(prices, prices), targets, us.costs(dates), config)
```

- Fee tables are dated from 2010 (Southbound from 2014-11-17): Hong Kong stamp
  duty, SFC and AFRC levies, HKEX trading fee, HKSCC settlement fee and the
  trading tariff; US SEC Section 31 fee and FINRA TAF (sell side only).
  `profile.charges(dates)` lists every statutory item by name.
- The kernel has no per-fill fixed amounts, per-charge floors or round-ups.
  They are added as `amount / fill_value` (default HK$20,000 and US$10,000).
  At that fill size the result equals the per-fill rules' upper bound; fixed
  charges and floors are overcharged on larger fills and undercharged on
  smaller ones, while the stamp round-up allowance exceeds its average
  (about HK$0.5) for fills above half of `fill_value`. Set it to your typical
  fill, or `None`. Per-charge caps are ignored (an overcharge).
- Default commissions are assumptions: HK$15 per fill (local), 0.03% (Connect),
  IBKR Fixed $0.005/share, $1 minimum, 1% cap (US). Override any of them.
- Dividend withholding: local HK 0% non-H, 10% H shares; optional flags for
  foreign-invested-enterprise H shares (20% from 2026-09-01) and red chips of
  Chinese resident enterprises (`resident_enterprise`, 10%; off by default,
  which is optimistic if they are withheld). Without `h_share` every security
  is withheld at 10% with a warning. Connect 20%; US 10% (`US_WITHHOLDING` has
  30% and 0%). The kernel takes one rate per day: pass the market's
  `cash_div` to `config` so only securities paying that day count; otherwise
  the day's highest rate is used with a warning. `net_dividends` is the exact
  alternative for accounts that cannot short.
- Delisting: prices freeze at the last close until the delisting row.
  `delisting="zero"` (long-only default) settles at 0: pessimistic for longs,
  optimistic for shorts; `"last"` settles at the last close: optimistic for
  longs, neutral for shorts. When the margin model allows shorting you must
  choose explicitly. No price limits and no sell lock: pass
  `Rules(sell_lock=False)`. Connect cannot short or borrow; its `fx_cost` is
  kept when you override `fee_rate_buy/sell` (pass `fx_cost=0` to drop it).
- Margin accounts: `markets.reg_t_margin(financing_rate=...)` (Reg T) and
  `markets.hk_margin(...)` (ratios from your broker).
- What is approximated, not modelled, or needs kernel support, and the
  defaults awaiting confirmation, are in `docs/markets/PROFILES.md` of the
  AShare repository.

## Accounts and results

`MarginModel` provides dated initial/maintenance equity margin, short eligibility,
financing and borrow charges, restricted short proceeds, and next-period
liquidation. It is a simplified equity-based account, **not a regulatory broker
ledger**. `markets.ashare_margin` requires caller-calibrated coefficients because
exchange collateral/debt formulas are not identical to this model.

`Config` also accepts `trade_band`, `dividend_tax_rate`, `cash_flow`, and
`cash_rate` (rates are per period). `Costs` accepts directional daily taxes and
fees, per-share commission, a commission floor/cap, panel slippage, and optional
volume impact. All panels must be aligned and immutable while in use.

`result.nav` remains equity / initial capital. Use `result.returns` and
`result.unit_nav` for flow-adjusted performance. `result.contribution` attributes
security P&L including trading and borrow fees; cash interest and financing
interest remain separate. `panelbt.metrics.summarize(result)` provides analytics.
`result.to_npz(path)` and `Result.from_npz(path)` use a versioned pickle-free
archive. `result.to_frame(index=dates)` imports optional pandas lazily.

`run_batch(market, targets_list, costs, config, threads=0)` shares market panels;
zero threads means all available CPUs. Thread counts do not change per-strategy
floating-point execution order. Recording positions/trades explicitly allocates
T×N (or K×T×N) output; leave `record_trades=False` for large batches.

## Futures hedge columns

Attach `Futures` to a `Market` to mix daily settled futures and cash equities in
one panel. Futures weights are signed notional divided by equity; share targets
are integer contracts. Contract multipliers, settlement prices, expiry rows,
dated margin rates, trading fees, and delivery fees are explicit inputs.

```python
hedge_prices = np.array([[10., 100.], [11., 101.], [12., 99.]])
futures = bt.Futures(
    is_future=[False, True], multiplier=[1, 10], margin_rate=0.12,
    settlement=hedge_prices,
    expiry=[3, 3], fee_rate=0.0002, fee_per_contract=0,
    delivery_fee_rate=0, delivery_fee_per_contract=0,
)
hedge_market = bt.Market(hedge_prices, hedge_prices,
                         hedge_prices > 0, hedge_prices > 0,
                         futures=futures)
hedge_targets = bt.Targets.from_dense(
    [[0.5, -1.0], [np.nan, np.nan], [0, 0]]
)
hedge_result = bt.run(hedge_market, hedge_targets, costs, config)
```

The prices, rates, and margin above are illustrative; verify contract terms
before using them. `panelbt.futures.roll_targets` creates a deterministic
pre-expiry roll schedule, and `hedge_contracts` converts equity value or beta
exposure into a signed contract count. `panelbt.markets.cffex_index_futures`
supplies verified IF/IH/IC/IM multipliers but requires fee and margin inputs.

`hedge_contracts` rounds toward zero, like the engine, and warns
(`HedgeRoundingWarning`) when an exposure is smaller than one contract. A cash
account keeps its exact cash-account behaviour for stocks when futures are
attached: contract collateral is reserved from cash first, and maintenance
calls apply only after a contract has been held. Only collateral, not
notional, is limited by cash: at a 12% margin rate a cash account can hold
about eight times its equity in contract notional. If a futures loss leaves
cash negative, stock sales still execute as long as their proceeds cover their
own fees. Rates are scalars, `(1, N)`
per contract, `(T, 1)` per day or `T×N`; 1-D rates are rejected as ambiguous.
On an expiry row, `settlement` must be the exchange delivery settlement price.
The engine does not distinguish close-today fees, which CFFEX charges at a much
higher rate; same-day open-and-close rolls are therefore under-costed.

## Stand-alone futures portfolios

`FuturesBook` builds a futures-only (or mixed) market from per-contract,
per-row exchange inputs and runs it with the same kernel. It adds broker
terms (`FuturesAccount(margin_add, margin_scale, fee_scale)`), forced closes
on a physical contract's last trading row or after an optional last holding
row (at that row's settlement, paying the fee and the slippage), price-limit
masks and three target kinds: contracts (`"lots"`), notional / equity
(`"notional"`) and margin / equity (`"margin"`).

```python
s = np.array([[100.0], [101.0], [99.0]])
book = bt.FuturesBook(
    settlement=s, close=s, exec_price=s, multiplier=[10.0], expiry=[3],
    margin_rate=0.1, fee_rate=1e-4,
    account=bt.FuturesAccount(margin_add=0.05, fee_scale=2.0),
    initial_cash=1e5,
)
book_result = book.run(np.array([[0.5], [np.nan], [np.nan]]), kind="margin")
book_result.equity, book_result.futures_margin
```

P&L always comes from each contract's settlement prices; build signals from a
continuous series if you like, but never compute returns from it (a roll gap
would become profit or loss). Semantics and reasons: SPEC "Stand-alone futures
portfolios".

## Stepping one row at a time

`bt.Simulator` advances the same kernel one row per call, so a strategy can
decide each target from the account state. Row `t`'s target is what to hold
after row `t`'s trading; `sim.state` before step `t` only holds what is known
before row `t` trades. The result is bit-for-bit what `bt.run` returns for the
same targets.

```python
import numpy as np
import panelbt as bt

prices = np.array([[10.0, 20.0], [11.0, 19.0], [12.0, 21.0]])
market = bt.Market(prices, prices, prices > 0, prices > 0)
costs = bt.Costs(commission_rate=0, commission_min=0,
                 fee_rate_buy=0, fee_rate_sell=0, slippage=0)
config = bt.Config(initial_cash=1000, min_qty=0, qty_step=0)

sim = bt.Simulator(market, costs, config)
for t in range(sim.n_days):
    state = sim.state                     # cash, equity, shares, cost, entry_day, ...
    if t == 0:
        target = ([0, 1], [0.5, 0.5])     # (cols, weights); or a length-N row
    elif state.shares[0] and prices[t - 1, 0] < 0.9 * state.cost[0]:
        target = ([1], [0.5])             # exit column 0 after a 10% loss
    else:
        target = None                     # no new target today
    sim.step(target)
result = sim.result()

blob = sim.snapshot()                     # bytes; resume with Simulator.restore
```

State arrays are read-only live views (copy them to keep history; numpy
refuses to make them writeable, and the public interface cannot change the
state). `step` releases the GIL, so independent
simulators can run in separate threads; one simulator must not be stepped or
read from two threads at once (that raises `RuntimeError`). The
per-step cost is a few microseconds and does not grow with the number of
securities. Fixed patterns such as stops are faster as declarative rules (below).

## Declarative exit rules and limit targets

Stops, take-profits, trailing stops, holding periods and limit prices run
inside the kernel, in `run`, `run_batch` and `Simulator` alike. Daily bars do
not tell the order of prices within a day, so every ambiguous case is resolved
against the strategy: a gap through a stop fills at the open, a day that hits
both the stop and the target takes the stop, and the trailing line uses the
peak up to the previous day only. `fill_time` says where `exec_price` sits in
the day and must be given.

```python
import numpy as np
import panelbt as bt

o = np.array([[10.0], [10.4], [9.0], [9.2]])
h = np.array([[10.5], [10.9], [9.3], [9.6]])
l = np.array([[9.9], [10.1], [8.7], [9.0]])
c = np.array([[10.3], [10.2], [9.1], [9.5]])
ok = np.ones_like(o, dtype=bool)
bars = bt.Market(o, c, ok, ok, open=o, high=h, low=l)  # exec_price = open

rules = bt.Rules(
    fill_time="open",        # exec_price is the open
    stop_loss=0.08,          # scalar, per-security vector or T×N panel
    trailing_stop=0.12,
    take_profit=0.30,
    max_hold=20,             # rows
)
exit_config = bt.Config(initial_cash=1e5, rules=rules, record_trades=True)
exit_result = bt.run(bars, bt.Targets.from_dense([[1.0], [np.nan], [np.nan], [np.nan]]),
                     bt.Costs(), exit_config)
print([(t.day, t.reason.name, t.price) for t in exit_result.trades])
print(exit_result.diagnostics["stop_loss"])   # rule counters sit in diagnostics

# A limit target: buy only if the price trades below 9.1 that day (a low that
# only equals the limit does not fill; Rules(limit_fill="touch") allows it).
limits = np.array([[np.nan], [np.nan], [9.1], [np.nan]])
weights = np.array([[np.nan], [np.nan], [1.0], [np.nan]])
limit_result = bt.run(bars, bt.Targets.from_dense(weights, limits=limits), bt.Costs(),
                      bt.Config(initial_cash=1e5, rules=bt.Rules(fill_time="open"),
                                record_trades=True))
```

After a rule exit the column cannot be reopened until a later target row
(`cooldown=None`, default) or for `cooldown=k` rows. With `sell_lock=True`
(default for every market, A-share T+1; set False elsewhere) shares bought
today are not sold by a rule today; the exit waits for the next row. A limit
buy fills only when the day's low is strictly below the limit (a sell: the
high strictly above): a low that just equals the limit is often an order
resting at the day's extreme that never traded, and counting those days as
fills overstates buy-the-dip strategies considerably. An exit that cannot trade (limit-down, suspension)
retries at the next rows' `exec_price`. Without rules, results are bit-for-bit
those of a build without the feature. In a market with futures columns, exit
rules act on the spot columns only (a scalar percentage applies to spot
columns; per-security or T×N values must be NaN on futures columns) and limit
targets are rejected. The exact semantics are in [SPEC.md](SPEC.md).

## Input validation

`Market(...)` and `Targets(...)` check every cell by default (finite positive
prices or NaN, valid masks, positive splits, nonnegative dividends, futures
settlement and rates, increasing rebalance days, finite weights, unique
columns), in parallel in the Rust core. Pass `validate=False` to skip only these
per-cell content checks:

```python
fast_market = bt.Market(prices, prices, prices > 0, prices > 0, validate=False)
fast_targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]],
                                     validate=False)
fast = bt.run(fast_market, fast_targets, costs, config)
np.testing.assert_allclose(fast.nav, [1.0, 1.025, 1.125])
assert not fast.inputs_validated
```

Use it when the same market has already been validated once and is rebuilt
repeatedly, or when inputs come from an upstream pipeline that already enforces
these rules. Shapes, dtypes, memory layout, matching panel shapes, plan pointers
and row/column bounds are always checked, so unchecked inputs cannot cause
out-of-bounds access; invalid values then simply produce meaningless results.
Context checks that need the account (short permission, integer contracts,
cash-account weight sums) are skipped for unchecked plans as well. Unchecked
rebalance days that repeat or decrease are not rejected: the later rows are
silently never executed.
`result.inputs_validated` tells whether the market and every plan of a run
(all plans of a batch) were checked.

See [SPEC.md](SPEC.md) for exact ordering, accounting conventions and limitations.

## Build

Use Python 3.13+ and a current stable Rust compiler. A source distribution bundles
the Rust path dependencies; it can be built outside this repository with
`pip wheel .` (or `uv build`). Wheels require only NumPy at runtime. No credentials,
data sources, project modules, or local absolute paths are required.
