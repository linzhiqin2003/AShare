"""Mutation check for the stand-alone futures layer (Python only, no rebuild).

    python bench/mutations_futures_book.py

Each mutant edits one line of `panelbt/futures_book.py` or
`ashare/backtest/futures.py`, runs the futures-book tests and restores the
file. Every mutant must make at least one test fail. Writes the table to
bench/results/mutations_futures_book.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "packages/panelbt/python/panelbt/futures_book.py"
PORT = ROOT / "src/ashare/backtest/futures.py"
TESTS = ["test/test_futures_book.py", "test/backtest/test_futures_backtest.py"]

MUTANTS = [
    (
        "broker margin add-on ignored",
        BOOK,
        "window, margin * account.margin_scale + account.margin_add, 0.0",
        "window, margin * account.margin_scale, 0.0",
    ),
    (
        "fee multiple not applied to the per-contract fee",
        BOOK,
        'fee_fixed = rate(fee_per_contract, "fee_per_contract") * account.fee_scale',
        'fee_fixed = rate(fee_per_contract, "fee_per_contract")',
    ),
    (
        "forced close without slippage",
        BOOK,
        "delivery[r, j] = fee[r, j] + slip[r, j]",
        "delivery[r, j] = fee[r, j]",
    ),
    (
        "last holding row off by one",
        BOOK,
        "end = np.where(future, np.minimum(expiry, hold + 1), expiry)",
        "end = np.where(future, np.minimum(expiry, hold), expiry)",
    ),
    (
        "locked-day direction swapped",
        BOOK,
        "self.locked_up = one_price & started & (o > previous)",
        "self.locked_up = one_price & started & (o < previous)",
    ),
    (
        "margin targets multiplied by the rate",
        BOOK,
        "values = np.where(fut, values / np.where(fut, rate, 1.0), values)",
        "values = np.where(fut, values * np.where(fut, rate, 1.0), values)",
    ),
    (
        "cash-settled expiry treated as a forced close",
        BOOK,
        "if settled[j] and not personal[j]:",
        "if False:",
    ),
    (
        "limit-up refusal only beyond the limit",
        BOOK,
        'can_buy &= ~(price >= _panel(limit_up, shape, "limit_up"))',
        'can_buy &= ~(price > _panel(limit_up, shape, "limit_up"))',
    ),
    (
        "untradable cells keep their price",
        BOOK,
        "exec_price=np.where(tradable, price, np.nan),",
        "exec_price=price,",
    ),
    (
        "missing='previous' fills from later values",
        BOOK,
        "a = np.where(window, _ffill(np.where(window, a, np.nan)), a)",
        "a = np.where(window, _ffill(np.where(window, a, np.nan)[::-1])[::-1], a)",
    ),
    (
        "lots not rescaled across a unit change",
        PORT,
        "value = np.trunc(value * reference[k] / mult[h])",
        "value = np.trunc(value)",
    ),
    (
        "roll rows not written without a new target",
        PORT,
        "current[k] != 0 and self.held[day, k] != written[k] for k in range(p)",
        "False for k in range(p)",
    ),
    (
        "personal override disabled",
        PORT,
        "if h < 0 or last_hold_rows_of(symbols[h]) >= day:",
        "if True:",
    ),
    (
        "SC personal rule off by one day",
        PORT,
        'k = int(np.searchsorted(calendar, last, side="right")) - 1 - rule[1] - 1',
        'k = int(np.searchsorted(calendar, last, side="right")) - 1 - rule[1]',
    ),
]


def run_tests() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *TESTS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return proc.returncode == 0


def main():
    import panelbt

    import ashare

    assert Path(panelbt.__file__).resolve().is_relative_to(ROOT), panelbt.__file__
    assert Path(ashare.__file__).resolve().is_relative_to(ROOT), ashare.__file__
    assert run_tests(), "tests must pass before mutation"
    rows = []
    for name, path, old, new in MUTANTS:
        source = path.read_text()
        assert source.count(old) == 1, name
        try:
            path.write_text(source.replace(old, new))
            caught = not run_tests()
        finally:
            path.write_text(source)
        rows.append(dict(mutant=name, file=str(path.relative_to(ROOT)), caught=caught))
        print(("caught  " if caught else "MISSED  ") + name, flush=True)
    out = ROOT / "bench/results/mutations_futures_book.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    if not all(r["caught"] for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
