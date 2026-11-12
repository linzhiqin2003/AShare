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

## Market adapters

```python
from panelbt.markets import ASHARE, Board

market = ASHARE.market(prices, prices, high=prices, low=prices,
                      up_limit=prices * 1.1, down_limit=prices * 0.9)
costs = ASHARE.costs(["2025-01-02", "2025-01-03", "2025-01-06"],
                      commission_rate=0.00025, commission_min=5)
config = ASHARE.config(board=[Board.MAIN, Board.STAR])
```

Only A shares have a built-in profile. The following **illustrative rates are
not verified market defaults; verify them before use**. These examples reuse
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

## Stepping one row at a time

`bt.Simulator` advances the same kernel one row per call, so a strategy can
decide each target from the account state. Row `t`'s target is what to hold
after row `t`'s trading; `sim.state` before step `t` only holds what is known
before row `t` trades. The result is bit-for-bit what `bt.run` returns for the
same targets.

```python
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

State arrays are read-only live views (copy them to keep history). `step`
releases the GIL, so independent simulators can run in separate threads. The
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

# A limit target: buy only if the price trades at 9.1 or below that day.
limits = np.array([[np.nan], [np.nan], [9.1], [np.nan]])
weights = np.array([[np.nan], [np.nan], [1.0], [np.nan]])
limit_result = bt.run(bars, bt.Targets.from_dense(weights, limits=limits), bt.Costs(),
                      bt.Config(initial_cash=1e5, rules=bt.Rules(fill_time="open"),
                                record_trades=True))
```

After a rule exit the column cannot be reopened until a later target row
(`cooldown=None`, default) or for `cooldown=k` rows. With `sell_lock=True`
(default, T+1) shares bought today are not sold by a rule today; the exit
waits for the next row. An exit that cannot trade (limit-down, suspension)
retries at the next rows' `exec_price`. Without rules, results are bit-for-bit
those of a build without the feature. Rules are not available with futures
columns. The exact semantics are in [SPEC.md](SPEC.md).

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
