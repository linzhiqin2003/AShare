export CARGO_HOME := $(CURDIR)/.cache/cargo-home
export PYO3_PYTHON := $(CURDIR)/.venv/bin/python

.PHONY: dev check bench spike mutate

dev:
	uv sync --all-packages

check:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test --workspace
	uv run --no-sync pytest

bench:
	uv run --no-sync python bench/benchmark.py

spike:
	uv run --no-sync maturin develop --release --manifest-path bench/spike/Cargo.toml
	uv run --no-sync python bench/spike/benchmark.py --batch

mutate:
	uv run --no-sync python bench/mutations.py

.PHONY: panelops-bench panelops-crosscheck panelops-mutate

panelops-bench:
	uv sync --all-packages --group panelops-bench
	uv run --no-sync python bench/panelops/run.py --compare bench/panelops/results/baseline.json

panelops-crosscheck:
	uv sync --all-packages --group panelops-bench --group panelops-crosscheck
	MPLCONFIGDIR="$(CURDIR)/.cache/matplotlib" uv run --no-sync python bench/panelops/crosscheck.py

panelops-mutate:
	uv run --no-sync python bench/panelops/mutations.py
