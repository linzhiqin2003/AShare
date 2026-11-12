ifeq ($(SANDBOX),1)
export CARGO_HOME := $(CURDIR)/.cache/cargo-home
endif
export PYO3_PYTHON := $(CURDIR)/.venv/bin/python

.PHONY: dev check bench bench-pair bench-compare bench-docs spike mutate mutate-1c risk-bench risk-validate risk-mutate

dev:
	uv sync --all-packages --group dev --inexact
	uv run --no-sync python bench/clean_extension.py

check:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test --workspace
	uv run --no-sync pytest

bench:
	uv run --no-sync python bench/self.py
	uv run --no-sync python bench/futures.py

bench-pair:
	uv run --no-sync python bench/pair.py --base-ref "$(BASE_REF)" $(PAIR_ARGS)

bench-compare:
	uv sync --all-packages --group compare
	MPLCONFIGDIR="$(CURDIR)/.cache/mpl" uv run --no-sync python bench/compare/run.py $(COMPARE_ARGS)

spike:
	uv run --no-sync maturin develop --release --manifest-path bench/spike/Cargo.toml
	uv run --no-sync python bench/spike/benchmark.py --batch

mutate:
	uv run --no-sync python bench/mutations.py

mutate-1c:
	uv run --no-sync python bench/mutations_1c.py

bench-docs:
	uv run --no-sync python bench/compare/docs.py

risk-bench:
	uv run --no-sync python bench/risk/run.py

risk-validate:
	uv run --no-sync python -m ashare.risk.validate

risk-mutate:
	uv run --no-sync python bench/risk/mutations.py

.PHONY: panelops-bench panelops-crosscheck panelops-mutate

panelops-bench:
	uv sync --all-packages --group panelops-bench
	uv run --no-sync python bench/panelops/run.py --compare bench/panelops/results/baseline.json

panelops-crosscheck:
	uv sync --all-packages --group panelops-bench --group panelops-crosscheck
	MPLCONFIGDIR="$(CURDIR)/.cache/matplotlib" uv run --no-sync python bench/panelops/crosscheck.py

panelops-mutate:
	uv run --no-sync python bench/panelops/mutations.py

.PHONY: perf-profile perf-compare

perf-profile:
	uv run --no-sync python bench/e2e/run.py $(PERF_ARGS)

perf-compare:
	uv run --no-sync python bench/e2e/compare.py $(PERF_OLD) $(PERF_NEW)
