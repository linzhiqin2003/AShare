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
STEP = ROOT / "src/ashare/backtest/futures_step.py"
TESTS = [
    "test/test_futures_book.py",
    "test/backtest/test_futures_backtest.py",
    "test/backtest/test_futures_public_entries.py",
    "test/backtest/test_futures_step.py",
]

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
        "carried lots not rescaled across a unit change",
        STEP,
        'wait[k] = ("units", q0[c] * mult[c], None)',
        'wait[k] = ("units", q0[c], None)',
    ),
    (
        "carried lots rounded to nearest instead of toward zero",
        STEP,
        "return keep_value(day, h, np.trunc(amount / mult[h]), equity, basis)",
        "return keep_value(day, h, np.round(amount / mult[h]), equity, basis)",
    ),
    (
        "roll lots not converted by the multiplier ratio",
        STEP,
        "lots = np.trunc(units / mult[h])",
        "lots = np.trunc(units / mult[own[0]])",
    ),
    (
        "roll lots rounded to nearest instead of toward zero",
        STEP,
        "lots = np.trunc(units / mult[h])",
        "lots = np.round(units / mult[h])",
    ),
    (
        "first-row lots divided by the multiplier",
        STEP,
        "lots = amount if ref is None else np.trunc(amount * ref / mult[h])",
        "lots = np.trunc(amount / mult[h])",
    ),
    (
        "kept positions sized without the half-lot margin",
        STEP,
        "return (lots + 0.5 * np.sign(lots)) * m * mult[col] / (equity * buffer)",
        "return lots * m * mult[col] / (equity * buffer)",
    ),
    (
        "roll rows re-target every product",
        STEP,
        "vals.append(keep_value(day, c, shares[c], equity, basis))",
        "vals.append(goal_value(day, c, goal_now[k]))",
    ),
    (
        "written rows drop the engine's carried orders",
        STEP,
        "vals.append(pend[c])  # 引擎自己的顺延目标原样带上",
        "vals.append(0.0)",
    ),
    (
        "target on a contract's end row treated as waiting",
        STEP,
        "return h >= 0 and first[h] <= day <= end_row[h]",
        "return h >= 0 and first[h] <= day < end_row[h]",
    ),
    (
        "written rows keep a leftover old contract",
        STEP,
        "if k < 0 or h < 0 or c == h:",
        "if True:",
    ),
    (
        "waiting target ignored while a leftover old contract is held",
        STEP,
        "if wait[k] is not None and not pend_by.get(k):",
        "if wait[k] is not None and not own and not pend_by.get(k):",
    ),
    (
        "waiting entries kept through a margin-call liquidation",
        STEP,
        "        if call:\n            wait[:] = [None] * p",
        "        if False:\n            wait[:] = [None] * p",
    ),
    (
        "personal lead ignored",
        PORT,
        "        k -= lead\n",
        "",
    ),
    (
        "last trading day on the last sample row treated as after the sample",
        PORT,
        "if day is None or day > dates[-1]:",
        "if day is None or day >= dates[-1]:",
    ),
    (
        "delivery-month personal rule one day late",
        PORT,
        'k = int(np.searchsorted(calendar, month + "01")) - 1 - 1',
        'k = int(np.searchsorted(calendar, month + "01")) - 1',
    ),
    (
        "personal roll looks at the switch row's quote (look-ahead)",
        PORT,
        "and day > 0 and quote[day - 1, j]",
        "and day > 0 and quote[day, j]",
    ),
    (
        "external params take the smaller side's margin",
        PORT,
        "return np.fmax(*values) if len(values) == 2 else values[0]",
        "return np.fmin(*values) if len(values) == 2 else values[0]",
    ),
    (
        "one-price day at the previous settlement blocks buying",
        BOOK,
        "self.locked_up = one_price & started & (o > previous)",
        "self.locked_up = one_price & started & (o >= previous)",
    ),
    (
        "nonpositive prices accepted with validate=False",
        BOOK,
        "            if bad.any():\n                r, c = map(int, np.argwhere(bad)[0])",
        "            if bad.any() and validate:\n                r, c = map(int, np.argwhere(bad)[0])",
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
