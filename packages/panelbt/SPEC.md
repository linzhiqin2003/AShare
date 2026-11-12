# panelbt 0.1 execution specification (round 1b)

## Scope and alignment

A deterministic, daily, single-currency spot portfolio simulator. Each row is
one execution opportunity, followed by a closing mark. The caller aligns signals;
no signal shifting occurs inside the engine. Prices and quantities use float64.
The default is the original long-only, unlevered cash account. Signed holdings
are enabled only by `MarginModel.allow_short`.

Market panels share nonempty T×N shapes. Required: `exec_price`, `close`,
`can_buy`, `can_sell`. Optional: `volume`, `split`, `cash_div`, `delist`,
`can_short`. Prices are positive finite values or NaN; volumes are nonnegative
finite values or NaN; splits are positive finite multipliers; dividends are
nonnegative finite amounts per pre-split share. NaN execution prices never trade.
NaN closing marks use the most recent valid close adjusted for intervening
corporate actions: `mark = max(mark - cash_div, 0) / split`. This price series
also applies on reentry and to delisting recovery. Before any valid close,
default cash accounts retain the legacy zero valuation without bankruptcy or
forced liquidation. Margin accounts use a valid execution price when no close
mark exists; bankruptcy is deferred while a held security has no valid mark.
Delisting uses the previous valid close, never that day's future closing mark.
The first delist row permanently disables a column. Short delisting buys back
at prior close × recovery; negative event proceeds debit cash.

Construction validates and converts arrays to C-contiguous float64/bool once.
Arrays are borrowed through read-only views; writable aliases must not be
mutated during the lifetime of the market. `first_delist[N]` is computed once,
avoiding repeated historical scans for newly targeted securities.

## Targets, lots and limits

`Targets(days, indptr, cols, weights, kind="weights")` is a CSR rebalance plan.
Days are strictly increasing nonnegative int64; pointers start at zero and end
at nnz; columns are unique within each row and are sorted with their values.
All target values must be finite. Context checks enforce row/column bounds,
short permission and, for an unlevered cash account, weight sums ≤ 1+1e-9.
An all-NaN dense row means no rebalance; other NaNs become zero. Scheduled empty
rows close positions. `to_dense` preserves that distinction.

For weights, target quantity is signed floor-to-lot of
`abs(weight × max(pretrade equity,0) × (1-cash_buffer) / valuation_price)`.
For an existing position, a same-sign nonzero residual below min_qty is retained
for sizing reductions. The execution price values a held position when available,
otherwise the corporate-action-adjusted prior mark.
With step>0, opening lot rounding is zero below min_qty, otherwise
`min_qty + floor((q-min_qty)/qty_step)×qty_step`. Step=0 allows fractional shares.
Partial closes round down by step and must meet the minimum. Complete closes permit the actual odd-lot
quantity. `kind="shares"` bypasses quantity rounding altogether, but still obeys
eligibility, volume, cash and margin constraints; it does not guarantee execution.

For weight targets, a nonzero current position and nonzero target are left alone
when `abs(target_weight-current_weight)<trade_band`. Zero-target liquidation and
opening from zero are exempt. The band compares the requested weight before the
cash buffer. Share targets ignore the band.

Buys/short covers require can_buy; long sales require can_sell; short openings
also require can_short when supplied (omission permits all sellable securities).
Volume participation is disabled when max_participation=0. Otherwise a fill is
capped by volume×participation, rounded down by step. Zero/NaN volume prevents
participation-limited fills. Sign reversals share one day's participation budget.

New rebalance rows replace pending orders. With carry_blocked, only price/mask/
volume-blocked names are retried using the original target and current equity.
Cash/margin shortfalls and lot-rounding residuals are not carried. Delisted names
never reenter. Forced liquidation suspends the old plan until a subsequent
rebalance row. On a liquidation-day rebalance, increases are discarded but
risk-reducing targets are honored: the executed reduction is the larger of the
liquidation plan and the target's reduction. A reversal's new exposure is discarded.

## Fees and execution prices

For signed fill direction s, positive quantity q and effective price p_eff:

- A=q×p_eff.
- Commission=max(commission_rate×A+commission_per_share×q, commission_min).
  If commission_max_rate is provided, take min of that commission and
  commission_max_rate×A. The cap wins over the floor. Zero fills incur no charge.
- Tax=tax_buy[t]×A or tax_sell[t]×A.
- Other fee=fee_rate_buy/sell[t]×A+fee_per_share_buy/sell[t]×q.
- The old transfer_fee input/output name is removed; transfer fees belong in the
  directional fee_rate arrays and are reported in other_fee.

Slippage is scalar or T×N. Slipped price is p×(1+s×slippage).
Optional impact_k is T×N and requires volume; impact exponent defaults to 0.5.
Effective price is slipped_price×(1+s×impact_k×(q/volume)^exponent).
Slippage cost=q×p×slippage; impact cost=q×abs(p_eff-slipped_price).
Both are embedded in cash execution and reported separately, never debited twice.
An undefined/nonpositive impact quote is rejected. Volume impact applies even
when max_participation is disabled. Nonlinear buy affordability uses bisection.
All fees are nonnegative; scalar/daily/panel domains are validated before Rust.

## Daily sequence and cash flows

1. Apply external cash_flow before corporate actions. Withdrawals use available
   positive own cash and never increase debt. Record actual signed flow and
   nonnegative requested-minus-available withdrawal shortfall.
2. Pay gross dividends using pre-split signed quantities. Positive dividends are
   taxed by dividend_tax_rate; shorts pay the gross dividend without tax.
   Apply split multipliers, then delisting recovery.
3. Resolve forced liquidation, scheduled targets or eligible carried targets.
4. Compute target quantities at execution marks. Risk reductions precede all
   openings: sell longs, cover shorts, then open shorts and buy longs. Original
   cash-account order and floating-point arithmetic are preserved.
5. Increasing-risk baskets are scaled by a common factor to fit initial margin
   and cash, then rounded. The margin path also verifies each fill prefix; it
   may reduce a prefix further to avoid consuming unavailable cash. Long-only
   cash scaling preserves the original proportional-then-affordable algorithm.
6. Mark at close. Charge borrow fees on closing short values. Charge financing
   interest on negative own cash, and credit cash_rate on positive own cash.
   Both are per-period rates. Closing fees and adverse price moves can create a
   deficit even if every opening was affordable; do not clip account debt away.
7. Record equity, maintenance status, bankruptcy and flow-adjusted returns.

Own cash is C-S when short proceeds are restricted, otherwise C. C includes
short-sale proceeds, L is long market value, S is positive short market value,
and E=C+L-S. Initial margin is long_initial[t]×L+short_initial[t]×S≤E.
Without financing, an opening must additionally leave own cash≥0. Borrow rate
accepts scalar, T or T×N; other margin rates accept scalar or T.

A healthy nonfinancing account cannot use a risk reduction to create a cash
shortfall; reductions are trimmed when necessary to pay fees. An already
insolvent/deficient account may close risk despite remaining debt. It is
mathematically impossible to promise nonnegative own cash at every close under
arbitrary short price gaps, restricted proceeds, fees and blocked trading.
Likewise, a blocked liquidation cannot make initial margin hold immediately.
These deficits are retained and observable, not hidden by truncation.

Maintenance call is `long_maintenance×L+short_maintenance×S>E` at close.
`margin_call="record"` records only. With `"liquidate"`, next-row holdings are
scaled toward initial-margin compliance, accounting for fees, eligibility and
volume. If full feasible liquidation cannot restore compliance, close what is
possible and continue flagging only while planned reductions remain unfilled
(with a relative quantity tolerance). A completed plan does not create a new
call merely because initial margin is breached at the close: maintenance margin
is the closing criterion. The exact trial quantities are reused for execution.
E≤0 records the first bankruptcy row only for a valued margin account with
holdings or debt; an empty debt-free account may resume after a deposit. Subsequent
rows attempt closure and never reopen, even after deposits or new targets.

## Result accounting

Daily arrays include nav=E/initial_cash, equity, cash, long_value, short_value,
turnover, commission, tax, other_fee, slippage_cost, impact_cost, blocked counts,
dividend_gross, dividend_tax, actual cash_flow, withdrawal_shortfall, cash_interest,
borrow_cost, financing_interest, initial margin_usage and maintenance margin_call.
Trade and position recording is optional; delisting, liquidation and bankruptcy
events are always available. Diagnostics count original branches; coverage counts
new branches separately so original coverage tests retain their meaning.

Turnover=(sum of absolute effective execution amounts)/2/pretrade equity.
Corporate actions and external flows do not count as trades. Returns are
`E[t]/(E[t-1]+actual_flow[t])-1`, with E[-1]=initial_cash. Unit NAV is their
cumulative product. When the denominator is effectively zero (≤8 float64 eps ×
max(initial_cash,abs(previous equity),abs(flow))), return is defined as zero;
this avoids a meaningless return from residual roundoff after full withdrawal.

Contribution[N] is cumulative signed security cash flows plus the final mark:
price P&L, net dividends, delisting recovery, all execution costs and borrow fees.
For cash accounts, sum(contribution)+sum(cash_interest)=final equity-initial
cash-net inflows. With financing, subtract financing_interest as a separate
account-level expense. Financing is not arbitrarily attributed to one security.

`to_npz` stores versioned arrays and JSON, never pickle; `from_npz` rejects unknown
versions and loads with allow_pickle=False. Event types, batch axes and counters
round-trip. `to_frame` lazily imports optional pandas and accepts only a single
strategy. Batch results prepend K, preserve input order, and are bitwise identical
across thread counts in the same architecture/compiler environment.

## Analytics

`metrics.calculate` takes simple returns and explicit periods_per_year. Cumulative
and annual returns are geometric; volatility/tracking error use sample standard
deviation (ddof=1). Sharpe annualizes mean period excess over per-period risk-free
rate. Sortino uses the RMS of all negative excess returns, including zero terms
for nonnegative periods. Drawdown starts from wealth 1; duration includes an
unrecovered terminal decline. Calmar=geometric annual return/max drawdown.
Turnover and cost ratios annualize arithmetic means. Zero denominators yield NaN.
Relative annual return/drawdown use relative wealth; information ratio uses
arithmetic active returns. Beta is centered covariance/variance and alpha is the
annualized arithmetic CAPM intercept. Benchmark returns must exceed -1.

## Market profiles and regulatory provenance

Frozen MarketProfile records name, currency, periods_per_year, dated FeeSchedule,
lot/price-limit adapters, and sell_lock metadata (0 or 1). Fees are cumulative
effective-date patches; dates before schedule inception fail. Broker commissions
are overrides. Direct per-security lot sizes support heterogeneous board lots
and zero for fractional shares. No-limit profiles require only valid prices and
no suspension. A-share adapters preserve the original board and price-limit
rules; `rules.ashare` remains an import facade over `markets.ashare`.

ASHARE is the only built-in market. Before 2022-04-29 callers must explicitly
override both other-fee rates: historical transfer fees differed by venue. The
tax adapter still supports dates from 2008-09-19. HK-style bilateral tax/heterogeneous lots
and US-style per-share commission/caps/fractional shares are illustrative tests
and README examples, not trusted market defaults.

`ashare_margin` deliberately requires explicit long_initial, short_initial,
long_maintenance, short_maintenance and collateral_haircut. The haircut is
calibration metadata, not a second kernel deduction. Verified sources:

- SSE notice No.98 (2024), effective **2024-07-22**: new short-sale margin at least
  100%, private funds at least 120%; existing contracts retain old treatment until
  extension. [Official notice](https://big5.sse.com.cn/site/cht/www.sse.com.cn/lawandrules/sselawsrules2025/trade/specific/margin/c/c_20250616_10782021.shtml).
- CSRC statement **2019-08-09**: removed the universal 130% maintenance collateral
  ratio floor; brokers and clients agree it.
  [Official statement](https://www.csrc.gov.cn/csrc/c100028/c1000931/content.shtml).
- SSE **2023 revised implementation rules**, article 36: collateral conversion
  caps include 70% for SSE180 members, 65% for other A shares, and zero for specified
  risk-warning/negative-earnings securities. Article 42 uses collateral/debt;
  available margin separately depends on financing principal, short-sale proceeds
  and discounted unrealized P&L.
  [Official rules](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/specific/margin/c/c_20250616_10782015.shtml).

Those regulatory ratios are not interchangeable with this engine's E-based
coefficients, and individual broker haircuts/agreements cannot be verified from
public maxima. Therefore none is silently installed as a misleading default.
This model does not reproduce the full Chinese securities-credit ledger or
contract-vintage rules. Daily bars also cannot simulate the rule that a short
sale's quoted price must not be below the latest transaction price.

## Currently unsupported: changes to the kernel are required

- Multiple execution opportunities within a day. A target may reverse sign via a
  close and an opening in one opportunity, but never buys and then sells the same
  long shares. Daily long-only T+1 is therefore naturally satisfied. Intraday
  execution needs an explicit unlock inventory/clock; sell_lock is metadata now.
- Multiple currencies, FX conversion, and cross-currency financing.
- Full broker regulatory collateral/debt ledgers and contract-vintage terms.
- Futures (planned next round): instrument kinds, multipliers, settlement marks,
  expiry and cash variation margin are not implemented. `tradecore::Account`
  separates cash from spot L/S, and its margin method accepts additional_margin;
  per-fill fees are separate from principal movement. These are extension points,
  not an advertised futures API. Future settlement must feed cash independently
  of spot notional while contributing additional margin to the same constraint.


## Review clarifications

- A partial close must meet `min_qty` and `qty_step`; a full close may dispose of
  the entire odd balance, including a balance below the minimum. Liquidity caps
  are applied before this final quantity check. Shares targets bypass lot rules.
- A nonzero weight target below the opening minimum does not mean zero holdings.
  Its unrounded residual is retained when deciding reductions; the proposed fill
  itself is checked against the closing minimum. Zero targets still mean exit.
- Target differences within `1e-9 * max(abs(target), abs(held), 1)` are zero.
  Monetary branch diagnostics ignore interest below `1e-10` currency units;
  maintenance comparisons use `1e-10 * max(abs(E), L, S, 1)` tolerance.
- `n_reduction_clipped` counts reductions cut by the own-cash constraint;
  `n_invalid_quote` counts rejected nonpositive/nonfinite effective quotes.
  Both are daily arrays and have matching coverage counters. Such reductions
  do not become ordinary carried orders; an unfilled forced plan stays due.
- Coverage separately reports genuine bankruptcy (short exposure or own-cash
  debt) and same-day reversals. `dividend_gross` remains the signed net distribution
  across long and short positions; the two cash directions can be reconstructed
  from holdings and the dividend panel.
- Setting maintenance requirements explicitly selects the margin path, even for
  long-only exposure. The default cash account has no maintenance enforcement.
- `metrics.summarize` accepts losses below -100%, marks `bankrupt` and
  `geometric_metrics_valid`, and returns NaN for geometric annualization and
  Calmar when compounding crosses zero. Arithmetic risk metrics and drawdowns
  remain available. `calculate` remains strict unless `allow_bankruptcy=True`.
- `Costs` uses keyword-only construction. NumPy >=2.1 is sufficient: no API newer
  than NumPy 2.1 is used, and 2.1 is the first series supporting Python 3.13
  ([release notes](https://numpy.org/doc/2.1/release/2.1.0-notes.html)).
