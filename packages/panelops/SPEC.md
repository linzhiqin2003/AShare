# panelops specification, version 0.1.0

## Arrays, validity and concurrency

Public panel inputs must be two-dimensional, C-contiguous NumPy float64 arrays.
No implicit conversion: wrong array dtype raises TypeError; wrong layout,
dimensionality or shape raises ValueError. Invalid scalar parameters raise
TypeError or ValueError (integer parameter types are checked explicitly). Zero-sized dimensions are supported.
Optional masks must be C-contiguous bool arrays of identical shape. False cells
are missing both as historical observations and at the output date. NaN and either
infinity are missing. Pair operations use the intersection of valid observations.
Inputs are never modified. Every result is newly allocated and C-contiguous.
There is initially no `out=` contract; allocation reuse must earn its complexity
in benchmarks. Do not mutate inputs concurrently with a call. Native computation
releases the GIL. `set_num_threads(0)` selects available CPUs; positive integers
select a reusable process-wide Rayon pool. Concurrent calls with the same count
share that pool; use separate processes for independent one-core factor workers. Results are bit-identical across thread counts on
the same machine/build: reductions never split one series or cross-section.

## Time series

All windows include today: rows max(0,t-window+1) through t. `window` is a positive
integer. `min_periods=None` means window; otherwise 1 <= min_periods <= window.
Only finite, unmasked observations count. Insufficient counts yield NaN.
A missing current observation need not suppress a rolling aggregate (a false
current mask always does). No operation reads future rows.

| Function | Definition and additional rules |
|---|---|
| ts_delay(x, periods=1) | x[t-periods]; periods >= 0; insufficient history NaN |
| ts_delta(x, periods=1) | x[t]-x[t-periods]; both observations required |
| ts_sum / ts_mean | Sum / arithmetic mean of valid window observations |
| ts_std | sqrt(sum((x-mean)^2)/(count-ddof)); ddof=1 default, nonnegative integer; count <= ddof gives NaN |
| ts_min / ts_max | Minimum / maximum of valid observations |
| ts_rank | Average one-based rank of today's value in valid window; missing today NaN; normalize=True maps (rank-1)/(count-1), singleton 0.5; default False |
| ts_quantile | Linear interpolation at (count-1)*q in sorted valid values, 0 <= q <= 1 |
| ts_argmin / ts_argmax | Days since most recent tied extremum (today=0); distance is in trading-row positions, including missing rows; non-trading dates have no rows |
| ts_product | Product of valid values; overflow yields IEEE infinity, underflow zero; zero times an overflowing product is defined as zero |
| ts_corr / ts_cov(x,y) | Pairwise complete windows; covariance denominator count-ddof (default 1); correlation requires >=2 pairs and positive variance in both series, otherwise NaN |
| ts_decay_linear | Weights 1..window from oldest nominal position to today; ignore missing values and renormalize surviving weights |
| ts_ema | Recursive adjust=False, alpha=2/(window+1) unless supplied; first valid value seeds state, missing rows hold state (ignore_na=True); min_periods counts all valid observations since start, not only window; masked output NaN but state retained; lookback is unbounded |
| ts_skew | Biased population third standardized central moment; >=3 observations, positive variance, else NaN |
| ts_kurt | Biased population fourth standardized central moment minus 3; >=4 observations, positive variance, else NaN |
| ts_slope / ts_residual | Regress x on original time positions with intercept; >=2 valid samples; slope per row / today's residual; residual requires finite today |
| ts_regression_slope / ts_regression_residual(x,y) | Regress dependent x on independent y with intercept using paired observations; constant predictor or <2 pairs gives NaN; residual needs both today's values |

Reference comparisons use rtol=2e-10, atol=2e-10, including finite adversarial
inputs across large dynamic ranges. Constant windows are detected by exact
equality of observations, never by a rounded mean: standard deviation and
variance are exactly zero; correlation, skew, kurtosis, z-score, R-squared and
regression with a constant predictor are NaN. A constant dependent variable
with a nonconstant predictor has slope and residual exactly zero.
The native implementation uses compensated sums and shifted moments, tracking
roundoff accumulated since the last rebuild as well as cancellation in each
central moment, including cross products. An error budget derived from the
comparison tolerance triggers a window rebuild and a new anchor. Rebuilds can
also occur every 512 rows for sum/mean/std/var/zscore and every 128 rows for
paired/higher moments and decay. For the two-moment kernel the propagated error
bound uses `8*eps*(2*rows_since_actual_reset+W)` times the largest shifted
observation and its square, including the cancellation term. Other moment
kernels accumulate per-update absolute error bounds. A budget of `1e-11`,
smaller than the public `2e-10` tolerance, leaves room for final divisions. Exact
constant windows reset their two-moment state without a window scan.
Rare sum/mean/decay windows with severe cancellation are rebuilt using a short
floating expansion, preserving small terms across more than two value scales.
Slicing off leading history can change low bits within
tolerance. Appending or modifying future rows cannot change earlier bits.
Linear decay subtracts the main sum and compensation separately. Very large values
whose intermediate centered powers exceed float64 range may yield NaN/inf;
this is not an arbitrary-precision library. Output nonfinites from arithmetic
are accepted; when used as another operator's input they become missing.

## Cross sections

Each row is independent; missing inputs have missing outputs. `cs_rank` uses the
same average ranks and optional normalization as ts_rank. `cs_zscore(ddof=0)`
subtracts valid mean and divides by std; zero variance yields NaN.
`cs_winsor_quantile(lower=.01, upper=.99)` clips at linear sample quantiles.
`cs_winsor_mad(scale=3)` clips at median +/- scale*median(abs(x-median)); no 1.4826
normal-consistency multiplier, zero MAD clips every valid value to the median.
`cs_group_demean(x, groups)` accepts C-contiguous int64 groups of shape T x N;
negative groups yield NaN, singleton groups yield zero.
`cs_neutralize(x, exposures=None, groups=None, weights=None, rcond=1e-12)` fits
an intercept, one indicator per nonnegative group (if supplied), and K continuous
exposures (sequence of float64 panels). All regressors must be finite; weights
must be finite and positive. Missing/negative-group rows are excluded. A weighted
rank-revealing least-squares projection handles dependent columns and singleton
industries; no arbitrary minimum industry size or ridge is used. Residuals are
orthogonal in the weighted inner product to the retained design column space.
Rank decisions use relative tolerance rcond after column normalization. The native
solver uses pivoted reorthogonalized QR; the independent reference uses SVD.
Near the numerical rank cutoff these criteria can retain different spaces;
reference equivalence is not promised at that ill-conditioned boundary.

## Evaluation kernels

Returns supplied to evaluation functions must already be aligned by the caller;
no implicit return shift is performed. `ic(x, returns)` is daily Pearson
correlation on the valid intersection, length T; >=2 pairs and nonzero variance
required. `rank_ic` ranks each operand on the paired intersection before Pearson.
`factor_autocorr(x, periods=1, rank=True)` compares today's and lagged cross-section
on their common finite intersection (ranks recomputed on that intersection).

`group_returns(scores, returns, q=5, weights=None)` sorts finite scores ascending,
breaks ties by column index, assigns ordinal position i to floor(i*q/count),
and returns T x q. Assignment uses scores only; missing returns/invalid weights
are then excluded within each group, remaining positive weights renormalized.
Empty groups return NaN; q must be positive (q > count may produce empty groups).
Equal weights are the default. This deterministic ordinal partition deliberately
permits tied scores and differs from qcut at tied quantile boundaries.
`group_turnover(scores, q=5, periods=1)` is 1-|members_today intersect members_past| /
|members_today| per group; insufficient history or empty current group is NaN.

`targets(scores, k=None, fraction=None, days=None)` selects largest finite scores,
with ties broken by ascending column, either positive integer k or 0<fraction<=1
(default .2); fraction selects ceil(valid_count*fraction). Returns a SparseTargets
value with int64 days, indptr, cols and float64 weights. days are strictly increasing
valid row indices (default every row), cols are sorted within rows, weights equal
1/selected_count. A row with no valid scores is an explicit empty/liquidation row.
No leverage, negative weights or timing shifts are introduced. No dependency on
any backtest package is present.

## Additional operators

| Function | Semantics |
|---|---|
| ts_var(x, window=20, min_periods=None, ddof=1) | Variance with denominator count-ddof; exact zero for constant windows with sufficient observations |
| ts_zscore(x, window=20, min_periods=None, ddof=1) | Today's centered value divided by rolling standard deviation; missing today or zero variance gives NaN |
| ts_rsquare(x, window=20, min_periods=None) | R-squared of x regressed on original trading-row positions, with intercept; squared Pearson correlation with time; constant x or fewer than two valid observations gives NaN |
| ts_median | ts_quantile with q=0.5 |
| ts_mad | Median absolute deviation from the window median, without consistency multiplier |
| ts_count | Count of finite nonzero observations; min_periods counts all finite observations, including zeros |
| ts_sumif(x, condition, window=20, min_periods=None) | Sum x where finite condition is nonzero; paired validity; min_periods counts finite pairs including false conditions; an eligible all-false window returns zero |
| ts_ffill(x, limit=None) | Last finite unmasked value; optional positive limit is the maximum age in trading rows; None is unlimited; initial gaps and masked output remain NaN |
| ts_idxmin / ts_idxmax | One-based position in the nominal window, including missing/padded leading positions; ties choose latest; idx = window - arg, where arg is ts_argmin/ts_argmax |
| cs_scale | Divide finite values by their absolute sum; zero absolute sum gives NaN |
| cs_demean | Subtract the finite cross-sectional mean; constant rows become exactly zero |

Public operators have independent signatures; irrelevant keywords are rejected.
Layout selection is an internal implementation detail, not a public parameter.
Window=1 product is the finite observation itself (bitwise); longer products multiply
valid observations directly in chronological order.

## Expression elementwise semantics

The factor expression layer supports scalar broadcasting and same-shaped panels.
`gt/ge/lt/le/eq/ne` produce float64 1 or 0; either missing operand gives NaN.
`and_/or_/not_` interpret any finite nonzero value as true, zero as false;
missing propagates, without three-valued short-circuiting. `where(cond,a,b)`
uses the selected branch only; missing cond gives NaN. `maximum/minimum`
propagate either missing operand. `sign`, `power`, `signed_power`, `sqrt`,
`exp` and `clip(x,lower,upper)` follow real float64 arithmetic; signed_power
is sign(x)*abs(x)**power. Invalid domains and nonfinite results become NaN.
`is_nan` returns 1 for missing and 0 otherwise; `fill_nan(x,value)` replaces
missing x with value. The universe mask always wins, including after filling.
Clip bounds are finite scalar parameters with lower<=upper. Parameter types,
arity and irrelevant parameters are validated when constructing an expression;
defaults are normalized before hashing. Structural Expr equality remains separate
from the named comparison expression `eq`.

A market-series leaf is declared with field(name, broadcast=True). The source
returns a float64 vector of length T or a T-by-1 array; the evaluator broadcasts
it over the source's N symbols and applies each security's universe mask. Regular
field leaves still require exactly T-by-N. No time shift is introduced.

`panelops.build_id` is a deterministic build-time SHA-256 of native and Python
source content. Persistent factor caches use this identity, data version, axes,
universe and mask identity, expression digest and evaluator implementation identity.
