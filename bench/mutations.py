"""Real source mutations, fresh build verification, per-mutation restoration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "crates/panelbt-core/src/lib.rs"
TRADE = ROOT / "crates/tradecore/src/lib.rs"
REF = ROOT / "packages/panelbt/python/panelbt/reference.py"
OUT = ROOT / ".cache/rework/mutations"


def old_cases():
    return [
        (
            TRADE,
            "tax omitted",
            "tax: self.tax_rate * amount",
            "tax: 0.0 * amount",
            "test_engine.py::test_fees_and_tax_on_slipped_amount",
        ),
        (
            CORE,
            "can_sell ignored",
            "m.sell[base + i]",
            "true",
            "test_engine.py::test_can_sell_and_carry",
        ),
        (
            CORE,
            "cash scaling omitted",
            "st.cash / total",
            "1.0",
            "test_engine.py::test_proportional_cash_scaling_before_column_order",
        ),
        (
            TRADE,
            "commission floor omitted",
            ".max(self.commission_min)",
            ".max(0.0)",
            "test_engine.py::test_min_commission_and_cash_scaling",
        ),
        (
            CORE,
            "dividend cash omitted",
            "st.cash += d - tax;",
            "st.cash += 0.0 * (d - tax);",
            "test_engine.py::test_dividend_before_split_and_odd_buy",
        ),
        (
            CORE,
            "split omitted",
            "st.shares[i] *= f;",
            "st.shares[i] *= 1.0;",
            "test_engine.py::test_dividend_before_split_and_odd_buy",
        ),
        (
            CORE,
            "carry touches all holdings",
            "std::mem::swap(&mut commands, &mut pending);",
            "std::mem::swap(&mut commands, &mut pending); for &i in &st.held { commands.push((i,weights[i])); }",
            "test_engine.py::test_carry_only_blocked_security",
        ),
        (
            CORE,
            "volume cap omitted",
            "let cap = capacity(m, s, day, i, step);",
            "let cap = f64::INFINITY;",
            "test_engine.py::test_volume_and_odd_lot_liquidation",
        ),
        (
            CORE,
            "sell slippage reversed",
            "p * (1.0 + q.signum() * slip)",
            "p * (1.0 + slip)",
            "test_engine.py::test_fees_and_tax_on_slipped_amount",
        ),
        (
            CORE,
            "delist recovery ignored",
            "st.last[i] * s.scalars[7]",
            "st.last[i]",
            "test_engine.py::test_delist_prior_close_and_no_reentry",
        ),
        (
            CORE,
            "shares rounded",
            "out.counts[9] += 1;\n                w",
            "out.counts[9] += 1;\n                lot(w,s.vectors[6][i],s.vectors[7][i])",
            "test_extended.py::test_shares_bypass_lots_and_trade_band_exceptions",
        ),
        (
            CORE,
            "band omitted",
            "< s.scalars[9]",
            "< 0.0",
            "test_extended.py::test_shares_bypass_lots_and_trade_band_exceptions",
        ),
        (
            CORE,
            "dividend tax omitted",
            "d * s.v(8, day)",
            "0.0 * s.v(8, day)",
            "test_extended.py::test_dividend_flow_interest_and_contribution",
        ),
        (
            CORE,
            "external flow omitted",
            "let requested = s.v(10, day);",
            "let requested = 0.0_f64;",
            "test_extended.py::test_dividend_flow_interest_and_contribution",
        ),
        (
            CORE,
            "cash interest omitted",
            "own.max(0.0) * s.v(9, day)",
            "0.0 * s.v(9, day)",
            "test_extended.py::test_dividend_flow_interest_and_contribution",
        ),
        (
            CORE,
            "impact omitted",
            "slipped * (1.0 + q.signum() * impact)",
            "slipped * (1.0 + 0.0 * impact)",
            "test_extended.py::test_impact_cash_and_panel_slippage",
        ),
        (
            TRADE,
            "per-share commission omitted",
            "self.commission_per_share * quantity",
            "0.0 * quantity",
            "test_extended.py::test_per_share_commission_without_cap",
        ),
        (
            TRADE,
            "commission cap omitted",
            "commission.min(cap * amount)",
            "commission.max(cap * amount)",
            "test_extended.py::test_commission_cap_directional_share_fees",
        ),
        (
            CORE,
            "can_short ignored",
            "(m.can_short.is_empty() || m.can_short[base + i])",
            "true",
            "test_extended.py::test_short_permission_dividend_borrow_cover_and_delist",
        ),
        (
            CORE,
            "short dividends omitted",
            "let d = st.shares[i] * m.dividend[base + i];",
            "let d = st.shares[i].max(0.0) * m.dividend[base + i];",
            "test_extended.py::test_short_permission_dividend_borrow_cover_and_delist",
        ),
        (
            CORE,
            "borrow fee omitted",
            "let fee = -v * cell(s.borrow, day, i, m);",
            "let fee = 0.0 * cell(s.borrow, day, i, m);",
            "test_extended.py::test_short_permission_dividend_borrow_cover_and_delist",
        ),
        (
            CORE,
            "maintenance call omitted",
            "s.v(13, day) * long + s.v(14, day) * short > equity",
            "false && s.v(13, day) * long + s.v(14, day) * short > equity",
            "test_extended.py::test_margin_call_liquidate_record_and_bankruptcy",
        ),
        (
            CORE,
            "forced direction reversed",
            "let mut q = if allowed {\n                    reduction.signum()",
            "let mut q = if allowed {\n                    (if forced {-1.0} else {1.0}) * reduction.signum()",
            "test_extended.py::test_margin_call_liquidate_record_and_bankruptcy",
        ),
        (
            CORE,
            "initial margin uses maintenance",
            "a.margin(s.v(11, day), s.v(12, day), 0.0)",
            "a.margin(s.v(13, day), s.v(14, day), 0.0)",
            "test_extended.py::test_margin_scaling_financing_and_account_identity",
        ),
    ]


def new_cases():
    # Each triple changes one rule in both implementations, where applicable.
    c = []

    def add(name, test, *edits):
        c.append((name, "test_review.py::" + test, edits))

    add(
        "M3 restricted interest base",
        "test_restricted_proceeds_interest_bases",
        (
            CORE,
            "let own = st.cash - if s.flags[4] { 0.0 } else { short };",
            "let own = st.cash;",
        ),
        (
            REF,
            "own = cash - (0 if margin.short_proceeds_usable else short)",
            "own = cash",
        ),
    )
    add(
        "S28 cash interest includes proceeds",
        "test_restricted_proceeds_interest_bases",
        (CORE, "own.max(0.0) * s.v(9, day)", "st.cash.max(0.0) * s.v(9, day)"),
        (REF, "max(own, 0) * v[9][day]", "max(cash, 0) * v[9][day]"),
    )
    add(
        "S29 financing ignores restricted proceeds",
        "test_restricted_proceeds_interest_bases",
        (CORE, "(-own).max(0.0) * s.v(15, day)", "(-st.cash).max(0.0) * s.v(15, day)"),
        (REF, "max(-own, 0) * v[15][day]", "max(-cash, 0) * v[15][day]"),
    )
    add(
        "S11 reversal volume reset",
        "test_reversal_shares_daily_volume",
        (CORE, "(cap - used).max(0.0)", "cap.max(0.0)"),
        (REF, "max(cap - used, 0)", "max(cap, 0)"),
    )
    add(
        "S12 longs open before shorts",
        "test_opening_and_reduction_order",
        (
            CORE,
            "orders.sort_by_key(|&(i, q)| (q > 0.0, i))",
            "orders.sort_by_key(|&(i, q)| (q < 0.0, i))",
        ),
        (
            REF,
            "orders.sort(key=lambda x: (x[1] > 0, x[0]))",
            "orders.sort(key=lambda x: (x[1] < 0, x[0]))",
        ),
    )
    add(
        "S13 covers precede long sales",
        "test_opening_and_reduction_order",
        (
            CORE,
            "commands.sort_by_key(|&(i, _)| (st.shares[i] < 0.0, i))",
            "commands.sort_by_key(|&(i, _)| (st.shares[i] > 0.0, i))",
        ),
        (
            REF,
            "key=lambda i: (quantities[i] < 0, i)",
            "key=lambda i: (quantities[i] > 0, i)",
        ),
    )
    add(
        "S19 unfilled plan forgotten",
        "test_unfilled_liquidation_continues_without_maintenance_breach",
        (CORE, "|| unfilled_liquidation", "|| false"),
        (REF, "or unfilled_liquidation", "or False"),
    )
    add(
        "split stale mark",
        "test_suspended_action_preserves_wealth",
        (CORE, "st.last[i] /= f;", "st.last[i] /= 1.0;"),
        (
            REF,
            "max(marks[i] - div_price, 0) / factor_price",
            "max(marks[i] - div_price, 0)",
        ),
    )
    add(
        "dividend stale mark",
        "test_suspended_action_preserves_wealth",
        (CORE, "(st.last[i] - m.dividend[base + i]).max(0.0)", "st.last[i].max(0.0)"),
        (REF, "max(marks[i] - div_price, 0) / factor_price", "marks[i] / factor_price"),
    )
    add(
        "cash account bankruptcy enabled",
        "test_cash_account_missing_mark_is_not_bankrupt",
        (
            CORE,
            "let valued = !CLASSIC && st.held.iter().all(|&i| st.last[i] > 0.0);",
            "let valued = true;",
        ),
        (CORE, "if !CLASSIC\n            && equity <= 0.0", "if equity <= 0.0"),
    )
    add(
        "empty account bankrupt",
        "test_empty_account_can_resume_after_deposit",
        (CORE, "&& (!st.held.is_empty() || st.cash < 0.0)", "&& true"),
        (REF, "and (any(quantities) or cash < 0)", "and True"),
    )
    add(
        "initial margin repeats liquidation",
        "test_liquidation_stops_at_filled_plan",
        (
            CORE,
            "|| unfilled_liquidation",
            "|| unfilled_liquidation || (forced && required > equity)",
        ),
        (
            REF,
            "or unfilled_liquidation",
            "or unfilled_liquidation or (forced and required > equity)",
        ),
    )
    add(
        "partial sale minimum omitted",
        "test_selling_minimum_and_nonzero_small_target",
        (
            TRADE,
            "} else if quantity < minimum {",
            "} else if quantity < 0.0 * minimum {",
        ),
        (REF, "if q < v[6][i]", "if q < 0 * v[6][i]"),
    )
    add(
        "quantity dust tolerance omitted",
        "test_split_dust_has_no_commission",
        (
            CORE,
            "1e-9 * a.abs().max(b.abs()).max(1.0)",
            "0.0 * a.abs().max(b.abs()).max(1.0)",
        ),
        (REF, "<= 1e-9 * max(", "<= 0.0 * max("),
        (
            REF,
            "> 1e-9 * max(abs(target), abs(after), 1)",
            "> 0.0 * max(abs(target), abs(after), 1)",
        ),
    )
    add(
        "forced day reducing row discarded",
        "test_liquidation_day_honors_reducing_target",
        (
            CORE,
            "if rebalance && let Some",
            "if false && let Some",
        ),
        (
            REF,
            "if rebalance and i in requested_commands:",
            "if False and i in requested_commands:",
        ),
    )
    add(
        "S15 prefix check bypassed",
        "test_prefix_check_when_previous_fees_created_cash_deficit",
        (
            CORE,
            "&& !feasible(projected(account, i, q, st.shares[i], day, m, s), s, day)",
            "&& false && !feasible(projected(account, i, q, st.shares[i], day, m, s), s, day)",
        ),
        (
            REF,
            "if q != 0 and not feasible(project(account, i, q, day), day):",
            "if False and q != 0 and not feasible(project(account, i, q, day), day):",
        ),
    )
    return c


def command(args, name):
    path = OUT / name
    with path.open("w") as stream:
        status = subprocess.run(
            args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        ).returncode
    return status, path.read_text()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    originals = {x: x.read_text() for x in (CORE, TRADE, REF)}
    assert command(["make", "dev", "SANDBOX=1"], "baseline-build.log")[0] == 0
    assert command(["make", "check", "SANDBOX=1"], "baseline-check.log")[0] == 0
    cases = [
        (name, test, [(path, old, new)]) for path, name, old, new, test in old_cases()
    ] + new_cases()
    results = []
    try:
        for index, (name, test, edits) in enumerate(cases, 1):
            for path, old, new in edits:
                data = path.read_text()
                assert old in data, (name, old)
                path.write_text(data.replace(old, new))
            assert (
                command(["make", "dev", "SANDBOX=1"], f"{index}-build.log")[0] == 0
            ), name
            # conftest verifies the imported source hash before collecting tests.
            code, log = command(
                ["uv", "run", "--no-sync", "pytest", "test/" + test, "-q"],
                f"{index}-test.log",
            )
            assert code == 1 and "failed" in log and "extension is stale" not in log, (
                name,
                code,
                log,
            )
            results.append(
                dict(
                    id=index,
                    mutation=name,
                    test=test,
                    scope="dual" if any(p == REF for p, _, _ in edits) else "kernel",
                    exit_code=code,
                )
            )
            for path, data in originals.items():
                path.write_text(data)
            assert (
                command(["make", "dev", "SANDBOX=1"], f"{index}-restore-build.log")[0]
                == 0
            )
            assert (
                command(
                    ["uv", "run", "--no-sync", "pytest", "test/" + test, "-q"],
                    f"{index}-restore-test.log",
                )[0]
                == 0
            )
            print(f"{index}/{len(cases)} killed and restored: {name}", flush=True)
    finally:
        for path, data in originals.items():
            path.write_text(data)
        assert command(["make", "dev", "SANDBOX=1"], "final-build.log")[0] == 0
    for path, data in originals.items():
        snapshot = OUT / (path.parent.parent.name + "-" + path.name)
        snapshot.write_text(data)
        assert (
            command(
                ["git", "diff", "--no-index", "--exit-code", str(snapshot), str(path)],
                snapshot.name + ".diff",
            )[0]
            == 0
        )
    assert command(["make", "check", "SANDBOX=1"], "final-check.log")[0] == 0
    report = dict(
        mutations=results,
        restored_git_diff_exit=0,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in originals
        },
    )
    (ROOT / "bench/results/review-mutations.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
