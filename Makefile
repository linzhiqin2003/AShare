export CARGO_HOME := $(CURDIR)/.cache/cargo-home
export PYO3_PYTHON := $(CURDIR)/.venv/bin/python

.PHONY: dev check bench spike mutate risk-bench risk-validate risk-mutate

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

risk-bench:
	uv run --no-sync python bench/risk/run.py

risk-validate:
	uv run --no-sync python -m ashare.risk.validate

risk-mutate:
	uv run --no-sync python bench/risk/mutations.py
