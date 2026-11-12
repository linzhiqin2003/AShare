# panelbt 0.1 execution specification (round 1c)

## Scope and alignment

A deterministic, daily, single-currency portfolio simulator. Each row is
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

Construction converts arrays to C-contiguous float64/bool once and checks
dimensions and matching shapes in Python. Per-cell content checks (the value
domains above, boolean bytes 0/1, futures columns, finite target weights,
strictly increasing days, unique sorted columns) run in the Rust core, scanning
fixed chunks in parallel and stopping at the first failing field in the
original field order; messages are unchanged. When several inputs are bad at
once, which one is reported may differ from earlier versions; each message is
still the one for its field. `Market(..., validate=False)` and
`Targets(..., validate=False)` skip only those content checks (and the
account-context plan checks: short permission, integer contracts, cash weight
sums). Shapes, dtypes, layout, plan pointers and row/column bounds are always
checked, so unchecked input cannot index out of bounds; masks are read as bytes
(nonzero is true). Objects expose `validated`; `Result.inputs_validated` is true
only if the market and every plan of the run were checked. Arrays are borrowed
through read-only views; writable aliases must not be mutated during the
lifetime of the market. `first_delist[N]` and the sparse corporate-action index
are computed once in Rust, avoiding repeated historical scans.

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

ASHARE was the first built-in market (Hong Kong and US profiles follow below). Its schedule starts on 2005-01-24 and
covers every stamp-duty and transfer-fee change since (sources beside the
`ASHARE` definition). Before 2015-08-01 the transfer fee differed by venue (per
share in Shanghai, per amount in Shenzhen); fees cannot vary by security, so
both forms are charged (an upper bound). `FeeSchedule.components` groups keys
that express one charge: overriding any key in a group zeroes the group's other
scheduled keys, so explicit fixed rates never double count. The standalone
`stamp_tax_sell` adapter still supports dates from 2008-09-19. HK-style bilateral tax/heterogeneous lots
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

### Hong Kong and US profiles

`markets.get(name)` returns a built-in profile: `ashare` (the `ASHARE` object
itself), `hk_direct`, `hk_southbound` or `us`. The three new profiles are
`RegionalProfile` subclasses (`markets/hk.py`, `markets/us.py`) and add no
kernel behaviour; sources, dated fee history, unverified items and open
defaults are in `docs/markets/PROFILES.md` of the AShare repository.

- Each profile keeps an itemised dated table of statutory charges
  (`charges(dates)`), from which the engine-key `FeeSchedule` is derived.
  Hong Kong: stamp duty in `tax_buy/tax_sell`; SFC levy, AFRC levy, HKEX
  trading fee and HKSCC settlement fee summed in `fee_rate_buy/sell`.
  US: SEC Section 31 fee in `fee_rate_sell`, FINRA TAF in `fee_per_share_sell`.
- `costs(dates, fill_value=..., **overrides)`: per-fill fixed charges, per-fill
  floors and round-ups are added as `amount / fill_value` (equal to the per-fill
  rule's upper bound at that fill size; pessimistic above it). Per-fill caps
  are ignored (an overcharge). `fill_value=None` keeps proportional statutory
  charges only. Default broker commissions are assumptions; every keyword
  replaces the final value. `HK_SOUTHBOUND.costs(fx_cost=...)` adds the
  CNY/HKD settlement spread to both directions.
- `config(lot_sizes=..., n=..., dates=..., cash_div=..., delisting="zero"|"last", ...)`:
  Hong Kong lots are per security (length N, or T×N reduced to the largest lot
  with a warning); US uses whole shares (`fractional=True` for zero steps).
  Dividend withholding is computed per security (`h_share`, `fie_h_share`,
  `resident_enterprise`; `withholding`, `adr`, `adr_withholding`). The kernel
  takes one rate per day: with `cash_div` the day's maximum is taken over
  securities paying that day, otherwise over all securities, with a warning
  when paying securities differ. Without `h_share` the local HK profile
  withholds 10% from every security and warns. `delisting="zero"` sets
  `delist_recovery=0` (pessimistic for longs, optimistic for shorts), `"last"`
  sets 1 (optimistic for longs, neutral for shorts); long-only accounts
  default to `"zero"`, and a margin model that allows shorting requires an
  explicit choice. A `Rules` with `sell_lock=True` is rejected (both markets
  allow same-day round trips). Southbound rejects margin models that allow
  shorting or financing.
- `HK_SOUTHBOUND.costs` adds `fx_cost` after keyword overrides, so overriding
  `fee_rate_buy/sell` keeps the exchange spread (`fx_cost=0` drops it).
- `net_dividends(cash_div, rates, account)` refuses an account that may short
  (shorts would pay net instead of gross dividends) unless `allow_short=True`.
- `market(..., connect_eligible=...)` masks Southbound buys only.
- Helpers: `net_dividends`, `home_currency_nav`, `connect_home_equity`,
  `settlement_days`, `hk_margin`, `reg_t_margin`, `US_WITHHOLDING`.

## Currently unsupported: changes to the kernel are required

- Multiple execution opportunities within a day. A target may reverse sign via a
  close and an opening in one opportunity, but never buys and then sells the same
  long shares. Daily long-only T+1 is therefore naturally satisfied. Intraday
  execution needs an explicit unlock inventory/clock; sell_lock is metadata now.
- Multiple currencies, FX conversion, and cross-currency financing.
- Full broker regulatory collateral/debt ledgers and contract-vintage terms.
- Exchange-specific spread offsets and cross-contract collateral netting. The
  futures implementation charges the sum of absolute contract margins, which
  is deliberately conservative unless the caller supplies an adjusted rate.
- Physical delivery, options, currency conversion and intraday margin calls.

## Futures: instrument columns, settlement and expiry

`Market(..., futures=Futures(...))` is opt-in. There is one daily account
loop. It is compiled once without futures and once with futures columns; the
spot rules are the same source in both builds, and the account terms alone
choose between the margin path and the exact cash specialization. A futures
column that is never held or targeted performs no futures arithmetic, so
**adding such a column leaves every spot output bit-for-bit unchanged**, for
cash and margin accounts, including liquidation rows. This is a standing
property test (`test/test_futures_invariance.py`).

A `Futures` object gives `is_future[N]`, `multiplier[N]`, `settlement[T,N]`,
inclusive `expiry[N]` row indices, and margin and fee rates. A rate is a
scalar, a `(1, N)` per-contract row, a `(T, 1)` per-day column or a `T×N`
panel; one-dimensional rates are rejected because a length-N and a length-T
vector are indistinguishable when T equals N. An expiry equal to T denotes an
expiry beyond the backtest. Future columns require a positive finite settlement
on every row, a positive multiplier, and no split, dividend or delisting. Spot
columns ignore futures parameters. On a contract's expiry row, `settlement`
must hold the exchange **delivery settlement price** (for CFFEX index futures,
the average index level of the final two hours), not an ordinary daily
settlement; it is both the last variation mark and the cash-delivery price.

`kind="shares"` means an integer contract count for future columns; fractional
future targets are rejected. `kind="weights"` means signed *notional* divided
by pretrade equity: floor of `abs(weight × max(E,0) × (1-cash_buffer) /
(execution_mark × multiplier))`, with the original sign, i.e. rounded toward
zero. A nonzero weight that rounds to zero contracts is counted in
`coverage["futures_rounded_to_zero"]`; no order is sent. Future shorts are
permitted even when spot `allow_short=False`, since no securities are borrowed.
An unheld contract without a valid execution quote is sized at the previous
row's settlement (zero on row 0); no trade uses that quote, so this mark only
decides whether the target is counted as blocked or as rounded to zero and
what is carried. Rows before a contract lists still need a positive
settlement; such placeholder values are read only for this count.
Future fills use an integer minimum and step of one contract, independent of
spot lot rules. The same buy/sell masks, participation cap, slippage and impact
apply to contracts. No future trade uses a missing or nonpositive execution
quote. Spot commissions, stamp tax and per-share charges never apply to future
fills. Future commission is `abs(contracts) × (multiplier × effective_price ×
fee_rate + fee_per_contract)`; it is included in the normal commission and
turnover fields. Slippage and impact costs are multiplied by the contract
multiplier and are already embedded in the effective price.

At execution, a future fill changes cash only by its commission and realized
P&L on any contracts closed at the effective price relative to their last
settlement basis. Adding contracts updates the basis by quantity-weighted
execution price; a reversal starts the new side at the new execution price.
At each close, the remaining contracts earn `contracts × multiplier ×
(today_settlement - basis)` in cash, then their basis becomes today's settlement.
This gives opening-day `settlement - execution`, later `today - yesterday`, and
correct execution-price P&L for intraday closes. A future has no long or short
spot market value. Equity is `cash + spot_long - spot_short`, with all accrued
future variation already in cash. `futures_pnl` reports realized plus closing
variation; future contribution includes this P&L less all future fees.

Expiry is processed after that day's normal execution and settlement, even if
buy/sell masks are false or execution price is NaN. Any open quantity is cash
settled at that row's supplied delivery settlement price, pays
`abs(contracts) × (multiplier × delivery_price × delivery_fee_rate +
delivery_fee_per_contract)`, and becomes zero. The delivery fee appears in
both `delivery_fee` and the aggregate `other_fee`; it is debited only once.
The event records signed contracts, settlement price and fee. A same-day target
may trade (including opening) before expiry and then settle at that row's
close. Starting the following row, nonzero targets are ignored and counted in
`n_expired_target`/`expired_target` diagnostics; they never reopen a contract.
This also holds on forced-liquidation rows, where the requested row is replaced
by the liquidation plan: its nonzero expired targets are still counted. The caller expresses a roll by making the old
contract zero and the next contract nonzero on a rebalance row. The pure
`roll_targets` helper schedules this at `expiry - roll_days` trading rows and
never moves back to an earlier contract: when no contract expires more than
`roll_days` rows ahead, the latest unexpired contract is held to its expiry.
`hedge_contracts` converts spot value and beta to signed whole contracts,
rounded toward zero like the kernel, and warns (`HedgeRoundingWarning`) when
a positive exposure rounds to zero contracts.

Future collateral is `abs(contracts) × multiplier × mark × margin_rate[day,col]`.
The pretrade mark is the last settlement (or an execution price on a new
contract); today's settlement is not read during trade sizing or affordability.
An executed contract uses its effective execution price for the projected
collateral of that column. At close the mark changes to today's settlement.
The sum of future collateral is added to spot initial and maintenance margin.
Available own cash is `C - restricted_spot_short_value - future_collateral`
for every account. Without financing it must stay nonnegative after openings; withdrawals and
idle-cash interest use the same available balance. Financing interest applies
to its negative part. All opening orders are scaled by a common factor and
rounded down to legal quantities, with a prefix feasibility check.

The mixed account performs all reductions before openings. Future reductions
go first because they release collateral without consuming spot principal;
spot long sales precede spot short covers, preserving the spot risk order.
**Future reductions and closes are never trimmed by the own-cash rule and are
not subject to the cash account's legacy sale check.** A reduction only
releases collateral, and the variation loss is settled at the close whether or
not the contracts are closed, so trimming would only keep requested risk. Only
openings and additions are constrained by cash and margin. If the commission
of a reduction exceeds available own cash, the deficit is retained like any
other account debt: it reduces own cash, pays `financing_rate` on its negative
part at the close and can trigger a maintenance call; it is never clipped.
Spot sales in the cash specialization keep the legacy check
`cash + proceeds >= fees` while cash before the sale is nonnegative. When a
futures loss has left cash negative, a sale need only pay its own fees
(`proceeds >= fees`): it never deepens the debt, so it executes even though it
does not repay the debt in one fill. This applies to requested reductions and
forced liquidation alike, consistent with DESIGN 4.8 (risk-reducing orders
fill first); a cash account that never holds a contract cannot reach negative
cash on validated input, so its results are unchanged.

Future openings reserve hedge collateral before short-sale and cash-funded
spot openings. In a margin account all openings form one basket scaled by a
common factor, futures first in the prefix check. In the cash specialization
(long-only, no financing, long_initial=1, zero maintenance) the spot side keeps
its exact legacy algorithm: futures openings are first scaled so that total
collateral does not exceed cash, then the legacy proportional-then-affordable
spot basket uses cash minus collateral. A cash account records maintenance
calls (collateral > equity), forced liquidation and insolvency only after it
has held a contract; a cash account that never holds one behaves exactly as
without futures.

On a next-day maintenance call, the same proportional liquidation solver
includes both instruments, then applies these reduction orders under quote,
mask and volume constraints. A risk-reducing target on that row sizes futures
with the unrounded `weight × E × (1-cash_buffer) / (mark × multiplier)`, exactly
as spot uses the unrounded share quantity. Unfilled reductions retain the
call; a feasible completed plan does not trigger a new call solely because
initial margin remains above equity. A scheduled risk-reducing target may
increase that day's liquidation, while increases are discarded. Insolvency
records the first valued, nonempty account failure and attempts closure on
following rows. Expiry cash settlement still occurs if forced trading is
blocked. The engine cannot guarantee that an adverse overnight variation or
blocked exit leaves nonnegative cash or initial margin at every close; it
reports the deficit and forced events instead.

`markets.cffex_index_futures` supplies only verified IF/IH 300 and IC/IM 200
CNY-per-index-point multipliers. Margin and both ordinary/delivery fee terms
are required caller inputs because exchange and broker schedules change.
The multiplier sources are the CFFEX IF trading rules (2013-08-30), IH and IC
trading rules (2015-03-27), and IM trading rules (2022-07-18):
[IF](https://www.cffex.com.cn/u/cms/www/202003/27165505w3j5.pdf),
[IH](https://www.cffex.com.cn/u/cms/www/202003/271658063l8q.pdf),
[IC](https://www.cffex.com.cn/u/cms/www/202003/27165856gzzt.pdf),
[IM](https://www.cffex.com.cn/u/cms/www/202207/182020307zry.pdf).
The [CFFEX settlement rules](https://www.cffex.com.cn/u/cms/www/202501/071747590uho.pdf)
describe daily cash settlement. The model does not infer exchange margin
offsets, broker add-ons, special closing fees or current rates from these
contract specifications.


## Stepping interface

`Simulator(market, costs, config, kind="weights"|"shares")` runs the same
daily loop as `run`, one row per `step(target)`. The loop is split into two
phases that `run` also uses: *open* (external cash flow, corporate actions,
delisting) and *trade* (resolve targets, execute, close). Construction runs
the open phase of row 0; `step` runs the trade phase of the current row and
then the open phase of the next row. Therefore, before step `t`, `state`
contains exactly what is known before row `t` trades: the close of row `t-1`
plus row `t`'s cash flow, dividends, splits and delisting. Row `t`'s prices,
masks, volumes and settlements are never read before step `t`.

A target is `None` (no new target row), a length-N row (NaN is zero; an
all-NaN row is `None`; an all-zero row is an empty target row that closes all
positions) or `(cols, values)`. Rows are validated with the same context rules
as plans (short permission, integer contracts, cash-account weight sum). For
any plan, stepping its rows reproduces `run` bit for bit in every field,
event, trade and counter, for cash, margin and futures accounts.

`state` fields: `day`, `cash`, `equity` (last close), `shares`, `cost`
(quantity-weighted raw execution price of the open position, excluding
slippage, impact and fees; adds reweight it, reductions keep it, a new or
reversed position resets it; splits divide it; dividends do not change it),
`entry_day` (row of entry; adds keep it; -1 when flat), `last` (valuation mark
of held columns), `futures_basis`, `pending_cols`/`pending_targets` (carried
blocked orders), `margin_call`, `bankrupt`. `cost` and `entry_day` are kept for
spot columns only: on futures columns they always read 0 and -1, whatever the
position (read `shares`). Arrays are zero-copy live views over read-only
buffers: numpy refuses to make them writeable, and the state cannot be
modified through the public interface. Writing to the simulator's memory by
bypassing that protection (for example through the views' `.base` chain) is
undefined behaviour. `result()` requires every row to be stepped.

`snapshot()` returns a binary blob (format version 2, closed by a CRC-32 of
all preceding bytes); `Simulator.restore(blob, market, costs, config, kind)`
checks the checksum first, then the fingerprint, and continues bit for bit.
The fingerprint covers shapes, account type, target kind, every cost and
configuration vector and scalar, every rule term and every cell of the rule
percentage arrays (stop, take, trail), and a digest of every market panel
the kernel reads (prices, masks, volume, actions, open/high/low, futures
panels). `Simulator(..., fingerprint="sample")` (default) digests each
panel's shape, dtype, first and last rows and a fixed-stride sample of about
64K cells, computed once at construction (about 5–7 ms on a 4000×5500
market); a change confined to unsampled cells of a larger panel is not
detected. `fingerprint="full"` hashes every cell (about 0.8–1.2 s on the same
market). Restore with the choice used when the snapshot was taken. `bt.run`
computes no digest. Any truncated, extended or corrupted blob, a blob of an
older format and any mismatch raise `ValueError`; a failed restore leaves no
half-loaded simulator.

Threads: `step` releases the GIL, so independent simulators may advance in
parallel threads. One simulator must not be used from two threads at once:
a second concurrent `step`, or reading `state` from another thread during a
`step`, raises `RuntimeError` (the state is not damaged).

## Declarative exit rules and limit targets

`Config(rules=Rules(...))` runs fixed state-dependent patterns inside the
kernel, so `run`, `run_batch` and `Simulator` all use them at kernel speed.
Daily bars only give open, high, low and close, not their order inside the day:
every choice below takes the pessimistic side so a rule never gains from an
unknowable intraday path. With futures columns, exit rules act on spot
columns only and limit targets are rejected (see "Exit rules in mixed
panels").

**Inputs.** `Market` accepts optional `open`, `high`, `low` T×N panels
(positive or NaN; with validation, `low <= open <= high` where all three are
finite and `low <= high`). Price rules and limit targets require all three.
`Rules(fill_time, stop_loss=None, take_profit=None, trailing_stop=None,
max_hold=None, both_hit="stop", cooldown=None, dividend_adjusts_cost=False,
sell_lock=True, limit_valid="day", limit_fill="cross")`. Percentages are positive fractions given
as a scalar, a length-N vector (per security) or a T×N panel (read on the
evaluation row); NaN disables a cell. `max_hold` is a positive number of rows.
`fill_time` is required and states where `exec_price` sits in the day:
`"open"`, `"close"` or `"average"` (a VWAP-like price).
`Targets(..., limits=...)` (and `from_dense(..., limits=...)`) gives a positive
limit price per entry; NaN means an ordinary target. Limit targets need a
`Rules` object (for `fill_time`), possibly with no exit rule. `sell_lock`
defaults to True for every market (A-share T+1); set it to False for markets
without a same-day sale restriction. With `Market(validate=False)` the
`low <= open <= high` checks are skipped; on rows that violate them the
guarantee that a trigger fill lies within `[l, h]` no longer holds.

**Daily order.** Pre-open phase (flow, actions, delisting) is unchanged. On a
new target row, limit entries replace all active limit orders and are removed
from the ordinary row; the row also lifts `cooldown=None` re-entry blocks set
on earlier rows. Then: with `fill_time="open"` the ordinary rebalance at
`exec_price` comes first and the intraday phase second; with `"close"` and
`"average"` the intraday phase comes first. `"average"` is treated like
`"close"`: rules act on the positions held at the start of the day over the
full range, and positions bought at the average price are not evaluated until
the next row, because the part of the range before the purchase cannot be
separated (otherwise a take-profit could use a high that came before the buy).
The intraday phase is: (a) rule exits, (b) limit orders. Forced-liquidation
rows (margin call, bankruptcy) skip the intraday phase: due exits wait, limit
orders do not trade (day orders lapse). The close is unchanged. A column that
had a rule fill or a due exit today, or that has a limit entry in today's row,
is skipped entirely by today's ordinary rebalance (the exit or the limit order
takes precedence).

**Position bookkeeping.** `cost` is the quantity-weighted *raw* execution price
(before slippage and impact, fees excluded); adds reweight it, reductions keep
it, a new or reversed position resets it. Splits divide `cost` and `peak`.
Cash dividends do not change them by default (an ex-dividend drop moves the
price toward a long stop); `dividend_adjusts_cost=True` subtracts the dividend
from both (floored at zero). `entry_day` is the entry row (adds keep it).
`peak` is the highest high (long) or lowest low (short) since entry, starting
at the entry's raw price. A position is *fresh* on the row it was entered
unless it was entered by an ordinary fill with `fill_time="open"`: fresh
positions are not evaluated that row and that row's range is not added to
`peak`. At each close, for held non-fresh positions, `peak` takes that row's
high (long) or low (short) when finite: today's high therefore never raises
today's trailing line.

**Exit evaluation** (each held column, ascending, skipping fresh ones and those
with a due exit; `o, h, l` are the row's open, high and low; with any of them
NaN there is no price trigger that row). For a long position:
stop line `S = cost × (1 − stop_loss)`, trailing line `R = peak × (1 −
trailing_stop)`, the effective stop is `max(S, R)` (the one hit first on the
way down; on a tie, stop-loss); target line `P = cost × (1 + take_profit)`.
1. Gap: if `o <= stop` the stop fills at `o`; else if `o >= P` the target fills
   at `o` (the open is the first price of the day, so this is certain).
2. Otherwise stop is hit when `l <= stop`, target when `h >= P`. If both are
   hit their order is unknown: `both_hit="stop"` (default, pessimistic) takes
   the stop at its line, `"target"` the target at its line. If one is hit it
   fills at its line.
Short positions mirror this: `S = cost × (1 + stop_loss)`, `R = peak × (1 +
trailing_stop)`, effective stop `min(S, R)`, `P = cost × (1 − take_profit)`
(no target when `take_profit >= 1`), gap when `o >= stop` or `o <= P`, hits
when `h >= stop` or `l <= P`. The fill price is the line or open *before*
slippage and impact, which then apply as for any fill; therefore a trigger
fill's raw price always lies within `[l, h]`.
3. Holding period: when no price rule fired and `row − entry_day >=
max_hold`, the position exits at `exec_price`. With `fill_time="open"` this
check comes before the price rules (the open fill happens first).

**Exit execution.** An exit closes the whole position. It is an ordinary fill
at its raw price: long sales need `can_sell`, covers need `can_buy`; the
participation cap counts today's earlier fills of that column; quantities round
like closes (`min_qty`/`qty_step`, full close of any odd balance). With
`sell_lock=True` (default, A-share T+1) shares bought on this row cannot be sold
by an exit or a limit sell on the same row. Cash rules are those of
reductions: the cash account's legacy fee check, and the margin account's
own-cash trim (account valued at `last` marks, the traded column at its raw
fill price; the exit is cut to what keeps own cash non-negative). If an exit
cannot complete it stays *due*: on following rows it retries at that row's
`exec_price` (not the old trigger price) until the position is closed, before
any new evaluation, keeping its reason. A due exit ends when the position
reaches zero by any route (a forced liquidation, a delisting, an ordinary
reduction on a forced row). Each row
with an unfinished due exit counts one `rule_deferred`. Triggers are counted
once when they fire (`stop_loss`, `take_profit`, `trailing_stop`, `max_hold`).

**Re-entry.** After a rule exit fires on a column, openings of that column (ordinary,
carried or limit, either direction) are blocked: with `cooldown=None` until a
target row on a later row arrives; with `cooldown=k` through row `exit + k`
(target rows inside the window do not lift it). A target row on the exit row
itself never lifts it. Each blocked opening counts `reentry_blocked`. A due
exit also blocks openings. A target for that column in the exit row itself is
not an opening attempt (the exit takes precedence, see the daily order) and is
not counted.

**Limit targets.** Each limit entry `(col, target, L)` becomes an order for the
rest of the row (`limit_valid="day"`) or until filled or replaced by the next
target row (`"until_next"`). In the intraday phase after exits, orders first
reduce (all columns, ascending), then open (short openings, then long, by
column). Sizing uses `E = cash + Σ shares × last` at the start of the limit
step (no price from the current row) and the limit price: weight targets
become `w × max(E, 0) × (1 − cash_buffer) / L`, rounded like ordinary targets
(no trade band). With `limit_fill="cross"` (default) a buy needs the range to
cross the limit, `l < L`, and a sell `h > L`; a day whose low (high) only
equals the limit does not fill, because a limit order resting at the day's
extreme may well not have traded. `limit_fill="touch"` also fills on equality
(optimistic). A buy fills at `min(o, L)`, a sell at `max(o, L)` (a gap through
the limit fills at the open). Masks, short permission, sell lock,
participation cap, re-entry blocks and bankruptcy apply. Cash account
openings are cut to what cash affords including fees (bisection then lot
rounding, per order in sequence, no common scaling), like an ordinary
purchase; margin account openings to what keeps initial margin and, without
financing, own cash feasible (account at `last` marks). Such a cut counts
`cash_scaled` (cash account) or `margin_scaled` (margin account). It completes
the order only when the shortfall is at most one lot (`qty_step`), or, for
fractional quantities, at most the quantity the order's own slippage, impact
and fees amount to at its fill price; a zero fill never completes an order.
Any other shortfall, from the account or from a mask, participation cap,
block, bankruptcy or a reduction that could not complete, counts
`limit_blocked` and keeps the remainder for `"until_next"` orders (retried on
later rows under the same rules); day orders drop it. An order whose target
is reached is removed. Counters: `limit_filled` (fills: an order that reduces
and then opens on the other side counts two), `limit_unfilled` (rows on which
an active order's price was not reached), `limit_blocked` (reached but left
short of target beyond that allowance).

`cash_scaled` counts rows: a row on which an ordinary purchase basket, a
limit opening or both were cut to what cash affords counts once. Without
limit orders the count is unchanged from earlier versions. For
`margin_scaled` a limit cut likewise counts only on a row that has no margin
scaling yet; ordinary margin baskets keep their existing counting. A limit opening is fresh. Limit fills are not carry
trades and limit orders are never carried by `carry_blocked`.

**Results.** With rules configured, `Result.diagnostics` additionally holds
`stop_loss`, `take_profit`, `trailing_stop`, `max_hold`, `rule_deferred`,
`reentry_blocked`, `limit_filled`, `limit_unfilled` and `limit_blocked`.
`Trade.reason` is a `TradeReason` (an `IntEnum`, so `astuple(trade)` stays
numeric): `REBALANCE` (0) for ordinary, carried and liquidation fills,
otherwise `STOP_LOSS` (1), `TAKE_PROFIT` (2), `TRAILING_STOP` (3), `MAX_HOLD`
(4) or `LIMIT` (5). Without rules and limit targets every output is
bit-for-bit what it was before rules existed: the rule code is compiled out.

**Same-day cash order (approximation).** Within a row, cash released by a rule
exit is available to that row's limit openings (the intraday phase runs exits
before limit orders), and with `fill_time="close"`/`"average"` to the row's
ordinary rebalance. In a real session a limit buy may fill before the stop
that funds it, and with an average price the rebalance trades through the day
while an exit may come late; both are mildly optimistic for cash-constrained
accounts.

## Stand-alone futures portfolios

`FuturesBook` (module `futures_book`) prepares a futures-only (or mixed)
`Market`, `Costs` and `Config` from per-contract, per-row exchange inputs and
runs it with the unchanged kernel loop of "Futures" above; results are the
ordinary `Result`. The kernel is not modified for this: every rule below is
an input transformation, so the existing spot and hedge-leg paths are
untouched. `reference_book.py` is an independent naive loop written from this
section; it reads the raw book inputs cell by cell.

Inputs: `settlement`, `close`, `exec_price` (default `open`), `multiplier[N]`,
`expiry[N]` (last trading row; T = after the sample) and `margin_rate`; rates
are scalar, (1, N), (T, 1) or T×N. A futures column's settlement is forward
filled (rows before its first value take that value as a sizing placeholder).
The trading window of a column runs from its first tradable row to its end
row (below); outside it margin and fees are zero and need no value.

1. **Targets.** `kind="lots"` (signed whole contracts), `"notional"` (signed
   notional / pretrade equity, the kernel's weights) and `"margin"` (signed
   margin used / pretrade equity). A margin weight becomes notional by
   dividing by that row's effective margin rate; the rate of row t is known
   before row t opens (exchanges set it at the previous settlement), so this
   is not look-ahead. A margin target outside the column's window is an
   error. Lots round toward zero like every futures target. Shorts are
   negative and need no borrow permission. `Simulator` takes lots or notional;
   `margin_to_notional` converts one margin row.
2. **Marking and cash.** Unchanged kernel rules: variation at each row's
   settlement, collateral `|lots| × multiplier × settlement × rate`,
   available cash = equity − collateral (pure futures: equity is cash).
   `FuturesAccount(margin_add, margin_scale, fee_scale)` applies broker terms:
   effective rate = exchange rate × margin_scale + margin_add, trading fees ×
   fee_scale. Defaults are neutral (exchange minimums); the domestic
   preset in the research package (`ashare.backtest.futures.CN_RETAIL`:
   +5 percentage points, fees × 3) is an assumption taken from published
   broker schedules, on the pessimistic end.
3. **Fees.** Every fill pays `|lots| × (multiplier × price × fee_rate +
   fee_per_contract)`; both terms may be nonzero and vary by contract and
   row. A daily target never opens and closes the same contract on one row
   (a reversal closes the previous position and opens a new one), so
   close-today fees cannot arise and are not modeled. The stepping interface
   also trades each column once per direction per row, so the same holds; a
   position opened on its own end row is closed at that row's settlement with
   the ordinary fee, which understates exchanges whose close-today fee is
   higher (the reason this is mentioned at all).
4. **Margin calls.** The cash account rules of "Futures" apply: a call when
   collateral exceeds equity at the close (available cash < 0), next-row
   proportional liquidation at the execution price (with slippage) toward the
   initial requirement, insolvency when equity ≤ 0 (close everything, never
   reopen). Counted in `margin_call`, `coverage["futures_margin_call"]`,
   `coverage["futures_liquidation"]`, `liquidation_events` and
   `bankruptcy_events`.
5. **Price limits.** `no_buy`/`no_sell` masks or `limit_up`/`limit_down`
   prices (a fill at or beyond a limit is refused in that direction).
   `infer_locked=True` refuses, on a row whose open, high and low are equal
   with positive volume, buying when the open is above the previous row's
   settlement and selling when below (an open equal to it refuses nothing).
   What it catches and misses: most one-price rows are thin contracts with a
   single small fill, not limit days, so refusing them is pessimistic there;
   a row that opens at the limit and then trades away from it is not caught,
   so a fill at such an open is optimistic. With limit prices available, pass
   `limit_up`/`limit_down` and leave `infer_locked` off.
6. **Rolls.** The book holds individual contracts; the research package
   steps product targets onto contracts on its roll schedule (decision at row
   t−1's close, execution at row t's open, both legs paying fees and
   slippage). A roll row moves only the rolling product (same lots, rescaled
   by the multiplier ratio); other positions do not trade. Continuous prices
   are for signals only.
7. **End rows.** A futures column ends on `min(expiry, last_hold + 1)`.
   `last_hold[N]` is the last row a position may be held at the close (a
   retail position limit); the next row still trades normally, and whatever
   is held at its close is closed at that row's settlement. On a cash-settled
   contract's own expiry (`cash_settled=True`) the position settles at the
   delivery settlement price paying the delivery fee; otherwise (a physical
   contract's last trading row, or a retail end row) the close pays the
   trading fee plus the slippage, at the settlement: one forced exit with
   slippage. The kernel's expiry step performs it (reported as an expiry
   event and in `delivery_fee`/`other_fee`), and later nonzero targets are
   counted as expired.
8. **Unit changes.** A contract whose multiplier changes is two columns: the
   first ends on the row before the change (a forced close as in 7), the
   second trades from the change row (a new opening). Fees and slippage are
   paid on both sides and the position is flat for the night between; the
   exchange's proportional conversion is not modeled.
9. **Missing rates.** A NaN margin or fee inside a trading window raises by
   default. `missing="previous"` forward-fills within the window (never from
   later rows); a number fills the gap with that value.
10. **Execution time.** `exec_price` is whatever the caller passes; with the
   open of an exchange that has a night session it is the previous evening's
   auction. A signal computed at row t's close fills at row t+1's open; row
   t+1's open, high, low or volume must not enter that signal.
11. **No negative prices.** A zero or negative settlement, execution, close
   or open price on a futures column is rejected whatever `validate` says:
   collateral would turn negative and losses would not trigger a margin
   call. Negative-price episodes (WTI on 2020-04-20) cannot be simulated.

Not modeled, with the direction of the error: close-today fees (unclear:
cheaper on some SHFE products, ten times on CFFEX index futures);
exchange-specific margin offsets and the larger-side rule for two-sided
positions (pessimistic: every contract pays full margin); limit expansion
after one-sided markets, the automatic margin increase that goes with it and
the ban on opening in the limit direction after a limit close (optimistic);
queueing at the limit (unclear, see 5); forced position reduction at the limit
(optimistic); daily open-position caps and position limits (optimistic);
physical delivery (unclear: positions never enter it); proportional unit
conversion (mostly pessimistic, see 8); intraday paths (unclear).

## Exit rules in mixed panels

With futures columns, `Rules` act on spot columns only; limit targets are
rejected (`run`, `run_batch`, `Simulator.step`), because a limit opening is
sized against spot cash and margin only and could overbuy beside futures
collateral. A scalar percentage or `max_hold` applies to spot columns; a
length-N or T×N percentage must be NaN on every futures column, otherwise the
input is rejected. Futures columns are never evaluated, keep no cost, entry
row or peak, and their ordinary targets are never blocked by a rule state.

The rule phase runs where it runs without futures (before the rebalance for
`fill_time` "close"/"average", after the openings for "open"). In a margin
account the own-cash trim of a rule reduction subtracts the futures collateral
held at that moment: the collateral at the last settlement and today's rate
before the rebalance, or after the rebalance the running value in which
contracts traded earlier in the row use their execution price. In the cash
specialization rule sales keep the legacy fee check, like ordinary spot sales
beside futures. The kernel compiles a futures + tracking build of the loop
for this case; markets without rules or without futures use the same builds
as before. `reference_rules_futures.py` combines the rule reference with the
mixed-account reference.

## Review clarifications

- A partial close must meet `min_qty` and `qty_step`; a full close may dispose of
  the entire odd balance, including a balance below the minimum. Liquidity caps
  are applied before this final quantity check. Shares targets bypass lot rules.
- A nonzero weight target below the opening minimum does not mean zero holdings.
  Its unrounded residual is retained when deciding reductions; the proposed fill
  itself is checked against the closing minimum. Zero targets still mean exit.
- Target differences within `1e-9 * max(abs(target), abs(held), 1)` are zero.
  The same threshold is checked again after opening-basket and per-fill
  affordability scaling, immediately before execution; a scaled dust order
  does not pay fees, alter holdings or create a trade record.
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
  The exact cash specialization requires no shorting, no financing,
  long_initial equal to 1 on every row, and both maintenance rates equal to 0
  on every row. Positive long_maintenance therefore changes missing-close
  valuation and selects margin_scaled rather than cash_scaled diagnostics.
  The valued guard is reachable after a distribution exceeds the carried mark
  while both current quotes are missing: the mark is floored to zero and the
  bankruptcy decision waits for a valid quote. The historical
  genuine_bankruptcy name additionally requires short exposure or own-cash
  debt larger than the monetary tolerance; it can differ from bankruptcy for
  sub-tolerance negative equity in very small accounts. It is a materiality
  diagnostic, not a second economic definition of insolvency.
- `metrics.summarize` accepts losses below -100%, marks `bankrupt` and
  `geometric_metrics_valid`, and returns NaN for geometric annualization and
  Calmar when compounding crosses zero. Arithmetic risk metrics and drawdowns
  remain available. `calculate` remains strict unless `allow_bankruptcy=True`.
- `Costs` uses keyword-only construction. NumPy >=2.1 is sufficient: no API newer
  than NumPy 2.1 is used, and 2.1 is the first series supporting Python 3.13
  ([release notes](https://numpy.org/doc/2.1/release/2.1.0-notes.html)).
- Futures diagnostics. `futures_rounded_to_zero` counts a nonzero weight
  target whose contract count `0 < |x| < 1` floors to zero, evaluated after the
  trade band; a held position closed this way also counts; with E ≤ 0 the
  quantity is zero by definition and is not counted. `expired_target` counts
  every nonzero target that reaches an expired contract, including carried
  targets and the requested row of a liquidation day. Future fills never count
  `min_commission`, `per_share_fee`, `commission_cap`, `open_short` or
  `cover_short`; a forced contract fill counts both `liquidation` and
  `futures_liquidation`; `futures_margin_call` is a maintenance call while
  contracts remain held at the close. A cash account's futures-opening scaling
  counts only `futures_margin_scaled`. Among several contract reductions on
  one row, a margin account processes longs before shorts (by column), and the
  cash specialization processes them by column.
