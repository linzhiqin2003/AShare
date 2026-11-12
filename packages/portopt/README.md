# portopt

Array-only numerical kernels and direct Clarabel bindings. Runtime dependencies: NumPy.
No imports from ashare, panelbt, or panelops. The Rust core forbids unsafe code.

`ew_variance(x, half_life, lags, threads)` handles finite values and NaN, preserving
trading-day lag positions. `rolling_bias(realized, predicted, window, threads)`
computes rolling sample standard deviations of standardized returns.

`regress` accepts a shared N×K exposure matrix or a T×N×K panel and T×N
returns, market capitalization and estimation masks. Regression weights in the
loss are square roots of capitalization; industry constraints use capitalization.

`solve` accepts CSC matrix tuples `(rows, columns, indptr, indices, values)`,
linear vectors, and `(kind, dimension)` cones (0 equality, 1 nonnegative, 2 SOC).
`solve_many` solves independent problems sharing P and A in parallel. It does not
implement a dependent rebalance sequence. Returned statuses must be checked.

Inputs are validated in Python. Arrays are borrowed read-only during detached
Rust calls; concurrent mutation by the caller is unsupported. `threads=0` uses
the process default selected with `set_num_threads(n)`; `get_num_threads()` reports
the effective default, while `available_threads()` reports hardware availability.
Pools are cached per thread count and reused. Setting zero restores the hardware
default. Explicit positive `threads` values override the process default.
Rank-deficient or nearly collinear regressions use a deterministic SVD in the
industry-constraint null space; full batches and one-day calls share this code.
The Rust library artifact is named `portopt_ext`; the Python module remains
`portopt._core`, so other workspace extensions cannot overwrite its artifact.

From the workspace: `make dev && make check`.
