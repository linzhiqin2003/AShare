"""Configurable thresholds for the test-metric tools in ``bench/testing``.

Change the numbers here; every tool and the metrics page read them from this
module. Times are seconds per test (junit ``time``, call phase plus fixtures).
"""

# Parallel workers for pytest-xdist. The machine is shared: keep it modest.
WORKERS = 4

# A test slower than its category budget is listed as over budget.
CATEGORY_BUDGETS = {
    "unit": 2.0,
    "property": 10.0,
    "differential": 10.0,
    "smoke": 5.0,
    "integration": 30.0,
    "concurrency": 30.0,
    "e2e": 120.0,
    "perf": 120.0,
}

# How many of the slowest tests the runtime report lists.
SLOWEST = 20

# Coverage targets (percent). Files below the target are flagged.
PYTHON_BRANCH_TARGET = 80.0
RUST_LINE_TARGET = 80.0

# Lowest-covered files listed in the report.
LOWEST_FILES = 10

# Default number of runs for flaky detection, and the first random seed.
FLAKY_RUNS = 5
FLAKY_BASE_SEED = 20261114

# Parallel compile jobs for cargo llvm-cov (shared machine).
CARGO_JOBS = 6

# Per-command timeouts (seconds) so a hung run cannot block the machine.
PYTEST_TIMEOUT = 3600
CARGO_TIMEOUT = 3600
# Per-test timeout passed to pytest-timeout during metric runs.
TEST_TIMEOUT = 900
