"""Rebuild and test each 1c rule mutation, then restore the exact source bytes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "crates/panelbt-core/src/lib.rs"
FUT = ROOT / "crates/panelbt-core/src/futures.rs"
API = ROOT / "packages/panelbt/python/panelbt/api.py"
REF = ROOT / "packages/panelbt/python/panelbt/reference.py"
FREF = ROOT / "packages/panelbt/python/panelbt/reference_futures.py"
HELP = ROOT / "packages/panelbt/python/panelbt/futures.py"
OUT = ROOT / ".cache/step1c/mutations"


def e(path, old, new, occurrence=1):
    return path, old, new, occurrence


CASES = [
    (
        "N3 action order",
        "test/test_cleanup.py::test_simultaneous_dividend_and_split_order",
        [
            e(
                CORE,
                "st.last[i] /= f;",
                "st.last[i] = st.last[i] / f - m.dividend[base + i];",
            ),
        ],
        [
            e(
                REF,
                "max(marks[i] - div_price, 0) / factor_price",
                "max(marks[i] / factor_price - div_price, 0)",
            )
        ],
    ),
    (
        "N5b classic dispatch",
        "test/test_cleanup.py::test_explicit_maintenance_selects_margin_account",
        [
            e(API, "and np.all(vectors[13] == 0)", "and True"),
        ],
        [e(REF, "and np.all(np.asarray(margin.long_maintenance) == 0)", "and True")],
    ),
    (
        "N14c opening invalid quote",
        "test/test_cleanup.py::test_invalid_opening_quote_is_counted",
        [
            e(
                CORE,
                "out.daily[25 * m.t + day] += 1.0;",
                "out.daily[25 * m.t + day] += 0.0;",
                2,
            ),
        ],
        [
            e(
                REF,
                'data["n_invalid_quote"][day] += 1',
                'data["n_invalid_quote"][day] += 0',
                2,
            )
        ],
    ),
    (
        "N16 unvalued bankruptcy",
        "test/test_cleanup.py::test_unvalued_margin_position_defers_bankruptcy",
        [
            e(CORE, "&& valued\n", "&& true\n"),
        ],
        [e(REF, "and valued\n", "and True\n")],
    ),
    (
        "N18 interest materiality",
        "test/test_cleanup.py::test_interest_coverage_uses_material_threshold",
        [
            e(CORE, "if interest > 1e-10 {", "if interest > 0.0 {"),
        ],
        [e(REF, "int(interest > 1e-10)", "int(interest > 0.0)")],
    ),
    (
        "N19 genuine bankruptcy",
        "test/test_cleanup.py::test_bankruptcy_materiality_counter_boundary",
        [
            e(CORE, "if short > 0.0 || own < -tolerance {", "if true {"),
        ],
        [e(REF, "int(short > 0 or own < -tolerance)", "1")],
    ),
    (
        "N20 margin call tolerance",
        "test/test_cleanup.py::test_maintenance_comparison_has_relative_tolerance",
        [
            e(CORE, "short > equity + tolerance", "short > equity"),
        ],
        [e(REF, "short > equity + tolerance", "short > equity")],
    ),
    (
        "N23 forced long reduction",
        "test/test_cleanup.py::test_forced_day_reduces_long_from_new_row",
        [
            e(CORE, "current - desired.max(0.0)", "0.0 * (current - desired.max(0.0))"),
        ],
        [e(REF, "current - max(desired, 0)", "0 * (current - max(desired, 0))")],
    ),
    (
        "N24 forced weighted row",
        "test/test_cleanup.py::test_forced_day_reduces_long_from_new_row",
        [
            e(CORE, "w * value.max(0.0) * (1.0 - s.scalars[8]) / mark", "current"),
        ],
        [e(REF, "w * max(value, 0) * (1 - config.cash_buffer) / mark", "current")],
    ),
    (
        "F1 settlement basis",
        "test/test_futures.py::test_contract_settlement_through_expiry",
        [
            e(
                FUT,
                "portfolio.basis[i],\n                settlement,",
                "settlement,\n                settlement,",
            ),
        ],
        [e(FREF, "(settle - basis[i])", "(settle - settle)")],
    ),
    (
        "F2 contract multiplier",
        "test/test_futures.py::test_contract_settlement_through_expiry",
        [
            e(
                FUT,
                "portfolio.spot.shares[i],\n                f.multiplier[i],\n                portfolio.basis[i],",
                "portfolio.spot.shares[i],\n                1.0,\n                portfolio.basis[i],",
            ),
        ],
        [
            e(
                FREF,
                "q * spec.multiplier[i] * (settle - basis[i])",
                "q * 1.0 * (settle - basis[i])",
            )
        ],
    ),
    (
        "F3 effective opening price",
        "test/test_futures.py::test_opening_day_uses_effective_execution_price",
        [
            e(
                FUT,
                "before.signum() != after.signum() {\n            x.price",
                "before.signum() != after.signum() {\n            m.price[day * m.n + i]",
            ),
        ],
        [
            e(
                FREF,
                "else price\n                if before == 0",
                "else float(market.exec_price[day, i])\n                if before == 0",
            )
        ],
    ),
    (
        "F4 expiry closure",
        "test/test_futures.py::test_contract_settlement_through_expiry",
        [
            e(FUT, "if day as i64 == f.expiry[i] {", "if false {"),
        ],
        [e(FREF, "if day == spec.expiry[i]:", "if False:")],
    ),
    (
        "F5 delivery charge",
        "test/test_futures.py::test_contract_settlement_through_expiry",
        [
            e(
                FUT,
                "let fee = futures_charge(\n                    qty,",
                "let fee = 0.0 * futures_charge(\n                    qty,",
            ),
        ],
        [
            e(
                FREF,
                "fee = abs(q) * (\n                    spec.multiplier",
                "fee = 0.0 * abs(q) * (\n                    spec.multiplier",
            )
        ],
    ),
    (
        "F6 dated collateral",
        "test/test_futures.py::test_dated_margin_is_checked_at_close_and_before_next_trade",
        [
            e(FUT, "rate(f.margin_rate, day, i, m)", "rate(f.margin_rate, 0, i, m)"),
        ],
        [e(FREF, "fee_at(spec.margin_rate, day, i)", "fee_at(spec.margin_rate, 0, i)")],
    ),
    (
        "F7 collateral in affordability",
        "test/test_futures.py::test_margin_scales_integer_contracts",
        [
            e(
                FUT,
                "a.margin(s.v(11, day), s.v(12, day), fm)",
                "a.margin(s.v(11, day), s.v(12, day), 0.0)",
            ),
            e(
                FUT,
                "a.own_cash(s.flags[4]) - fm >= 0.0",
                "a.own_cash(s.flags[4]) >= 0.0",
            ),
        ],
        [
            e(FREF, "short + fm\n", "short\n"),
            e(FREF, "else short) - fm", "else short)"),
        ],
    ),
    (
        "F8 future-first liquidation",
        "test/test_futures.py::test_mixed_liquidation_releases_futures_collateral_first",
        [
            e(
                FUT,
                "(!f.kind[i], portfolio.spot.shares[i] < 0.0, i)",
                "(f.kind[i], portfolio.spot.shares[i] < 0.0, i)",
            ),
        ],
        [
            e(
                FREF,
                "key=lambda i: (not spec.is_future[i], pos[i] < 0, i)",
                "key=lambda i: (spec.is_future[i], pos[i] < 0, i)",
                2,
            )
        ],
    ),
    (
        "F9 expired target",
        "test/test_futures.py::test_expiry_disables_later_targets",
        [
            e(
                FUT,
                "if f.kind[i] && day as i64 > f.expiry[i] {",
                "if false && f.kind[i] && day as i64 > f.expiry[i] {",
            ),
        ],
        [
            e(
                FREF,
                "if spec.is_future[i] and day > spec.expiry[i]:",
                "if False and spec.is_future[i] and day > spec.expiry[i]:",
            )
        ],
    ),
    (
        "F10 roll boundary",
        "test/test_futures.py::test_roll_changes_on_boundary_and_hedge_contracts",
        [
            e(HELP, "expiry > day + roll_days", "expiry > day + roll_days + 1"),
        ],
        [],
    ),
]


def mutate(data, old, new, occurrence):
    count = data.count(old)
    if count < occurrence:
        raise AssertionError((old, count, occurrence))
    pos = -1
    for _ in range(occurrence):
        pos = data.index(old, pos + 1)
    return data[:pos] + new + data[pos + len(old) :]


def run_command(args, log):
    with (OUT / log).open("w") as stream:
        return subprocess.run(
            args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        ).returncode


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    originals = {
        path: path.read_bytes()
        for case in CASES
        for edits in case[2:]
        for path, *_ in edits
    }
    assert run_command(["make", "dev", "SANDBOX=1"], "baseline-build.log") == 0
    assert run_command(["make", "check", "SANDBOX=1"], "baseline-check.log") == 0
    results = []
    try:
        for name, test, single, dual in CASES:
            for scope in ("single", "dual") if dual else ("single",):
                edits = single + (dual if scope == "dual" else [])
                label = name.split()[0] + "-" + scope
                try:
                    for path, old, new, occurrence in edits:
                        path.write_text(mutate(path.read_text(), old, new, occurrence))
                    built = (
                        run_command(["make", "dev", "SANDBOX=1"], label + "-build.log")
                        == 0
                    )
                    caught = (
                        built
                        and run_command(
                            ["uv", "run", "--no-sync", "pytest", test, "-q"],
                            label + "-test.log",
                        )
                        == 1
                    )
                    if caught:
                        log = (OUT / (label + "-test.log")).read_text()
                        caught = "failed" in log and "extension is stale" not in log
                    results.append(
                        dict(mutation=name, scope=scope, built=built, caught=caught)
                    )
                    print(label, "caught" if caught else "SURVIVED", flush=True)
                finally:
                    for path, data in originals.items():
                        path.write_bytes(data)
                    assert (
                        run_command(
                            ["make", "dev", "SANDBOX=1"], label + "-restore.log"
                        )
                        == 0
                    )
                    assert (
                        run_command(
                            ["uv", "run", "--no-sync", "pytest", test, "-q"],
                            label + "-restore-test.log",
                        )
                        == 0
                    )
    finally:
        for path, data in originals.items():
            path.write_bytes(data)
        assert run_command(["make", "dev", "SANDBOX=1"], "final-build.log") == 0
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert run_command(["make", "check", "SANDBOX=1"], "final-check.log") == 0
    output = dict(mutations=results, all_caught=all(x["caught"] for x in results))
    (ROOT / "bench/results/step1c-mutations.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    if not output["all_caught"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
