# panelops

`panelops` is an independent library for deterministic time-by-security numerical
panels. It does not read market data or place orders. NumPy is its only Python
runtime dependency. The native module ships in the wheel and is built from safe
Rust with Rayon. It imports no research or backtest package.
Licensed under Apache-2.0.

## Installation

From the repository root (Python 3.13+, a Rust compiler and uv):

```sh
make dev
```

For a standalone wheel:

```sh
CARGO_HOME="$PWD/.cache/cargo-home" uv run --no-sync maturin build --release --manifest-path packages/panelops/Cargo.toml
```

The maturin source distribution includes the local `panelops-core` dependency.
NumPy is the sole installed runtime dependency.

## Example

```python
import numpy as np
import panelops as po

close = np.array([[10., 20.], [11., 19.], [12., 21.]], dtype=np.float64)
change = close / po.ts_delay(close) - 1
score = po.cs_rank(po.ts_mean(change, 2, min_periods=1), normalize=True)
target = po.targets(score, k=1)
# target.days / indptr / cols / weights form a CSR target schedule.
po.set_num_threads(1)  # 0 restores the available CPU count
```

## Operators

| Family | Functions |
|---|---|
| Shifts | `ts_delay`, `ts_delta` |
| Rolling statistics | `ts_sum`, `ts_mean`, `ts_std`, `ts_var`, `ts_zscore`, `ts_median`, `ts_mad`, `ts_count`, `ts_sumif`, `ts_min`, `ts_max`, `ts_rank`, `ts_quantile`, `ts_skew`, `ts_kurt`, `ts_product`, `ts_argmin`, `ts_argmax`, `ts_idxmin`, `ts_idxmax` |
| Smoothing | `ts_decay_linear`, `ts_ema`, `ts_ffill` |
| Rolling relationships | `ts_corr`, `ts_cov`, `ts_slope`, `ts_residual`, `ts_rsquare`, `ts_regression_slope`, `ts_regression_residual` |
| Cross sections | `cs_rank`, `cs_zscore`, `cs_winsor_quantile`, `cs_winsor_mad`, `cs_group_demean`, `cs_neutralize`, `cs_scale`, `cs_demean` |
| Elementwise | `elementwise(op, ...)`: gt/ge/lt/le/eq/ne, and_/or_/not_, where, maximum/minimum, sign, power/signed_power, sqrt, exp, clip, is_nan, fill_nan |
| Evaluation | `ic`, `rank_ic`, `rank_ic_many`, `factor_autocorr`, `group_returns`, `group_turnover`, `targets` |

All panels are C-contiguous float64 arrays, T rows by N securities. Masks are
same-shaped C-contiguous bool arrays. NaN and infinity are missing; false mask
cells are excluded historically and in the output. Inputs remain unchanged;
results are new arrays. Invalid types/layouts raise errors instead of converting.
Native computation releases the GIL; callers must not mutate borrowed inputs
concurrently. Thread count never changes the order of an individual reduction.

Windows include today and never look ahead. The default minimum valid count is
the whole window. Average ranks handle ties. `ts_std` defaults to ddof=1;
`cs_zscore` defaults to ddof=0. Zero-variance correlations and z-scores are NaN.
`ts_ema` is recursive `adjust=False, ignore_na=True`; its effective history is
unbounded and min_periods counts all valid observations since the start. Regression includes an intercept and handles dependent exposures.
Read [SPEC.md](SPEC.md) before interpreting missing-data or grouping edge cases.

Evaluation returns must already be aligned to the score dates. Targets contain
**no timing shift**. Industry group identifiers are int64; negative values exclude
an observation. Target score ties break by column index, making allocation stable.

## Validation and performance

`panelops.reference` is an intentionally slow executable specification. Tests
cover randomized comparisons, no-lookahead properties, masks, degeneracy,
thread determinism, input ownership and large dynamic ranges. External comparisons
use pandas and alphalens; actual source mutations verify that tests detect faults.

Measured tables, output-excluded incremental RSS, observed CPU thread counts and environment metadata live in
`bench/panelops/results/` and `docs/PANELOPS_BENCHMARKS.md`. Results are machine-
and workload-specific; other concurrent sessions may affect these measurements.
The benchmark compares NumPy, bottleneck, pandas, both Polars layouts, and Rust
with one/default threads. Libraries without a corresponding native operator are
marked unavailable. pandas rolling.apply callbacks are labeled separately when needed.
Runtime logs and samples go to the ignored `.cache/panelops/` directory; the
repository retains one compact frozen result. Use `bench/panelops/run.py --full`
for both shapes, all four windows and all three missing fractions.
`panelops.build_id` is a build-time SHA-256 of the native and Python sources,
independent of manual version bumps.

```sh
make panelops-bench       # representative configuration, compare frozen baseline
make panelops-crosscheck # pandas and alphalens
make panelops-mutate     # changes source temporarily, restores it, reruns checks
```

**Precision versus bottleneck.** `ts_sum`, `ts_mean`, `ts_std`, `ts_var` and
`ts_sumif` keep compensated sums, event rebuilds and error bounds. With data-dependent
branches replaced by bit selection they now run single-threaded at bottleneck's
speed (4000x5500, W=20, 5% missing, alternating runs on a loaded machine: sum 76.3
vs 76.3 ms, std 89.0 vs 92.0 ms, sumif 94-98 vs 113.6 ms; 12-17 ms with default
threads). On the adversarial corpus of `test/panelops/test_exact_reference.py`,
compared with exact rational values, bottleneck exceeds the 2e-10 tolerance in
7,344 (sum), 12,808 (std), 11,733 (var) and 7,271 (sumif) of about 53,000 cells,
with relative errors up to 1e23 even without the 2^200 column; panelops exceeds it
in none. After 200 rows of 1e6 followed by 0.1, bottleneck reports a standard
deviation of 0.014 on windows of exact 0.1 values; panelops reports 0.

<!-- BENCHMARK_SUMMARY:START -->
## Measured snapshot

4000×5500, window 20, 5% missing; median of three runs, milliseconds. See the benchmark report for thread/memory methodology.

| Operator | Fastest single baseline | Rust 1 | Rust default |
|---|---:|---:|---:|
| ts_mean | bottleneck: 72.34 | 66.60 | 16.82 |
| ts_std | bottleneck: 75.91 | 80.42 | 16.06 |
| ts_quantile | bottleneck: 612.83 | 490.54 | 51.49 |
| cs_neutralize | numpy_1: 3395.70 | 311.76 | 45.98 |
| rank_ic | polars_1: 1129.12 | 517.21 | 55.65 |
| targets | numpy: 514.03 | 254.42 | 29.75 |
<!-- BENCHMARK_SUMMARY:END -->
