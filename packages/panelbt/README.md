# panelbt

A deterministic, market-agnostic daily portfolio engine. NumPy is the only
Python runtime dependency; the Rust kernel shares per-fill primitives with
`tradecore`. License pending (许可证待定). No publication license is granted yet.

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

See [SPEC.md](SPEC.md) for exact ordering, accounting conventions and limitations.

## Build

Use Python 3.13+ and a current stable Rust compiler. A source distribution bundles
the Rust path dependencies; it can be built outside this repository with
`pip wheel .` (or `uv build`). Wheels require only NumPy at runtime. No credentials,
data sources, project modules, or local absolute paths are required.
