"""C4 real-source mutations; rebuild Rust and verify worktree imports each time."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORE = "crates/panelbt-core/src/lib.rs"
CRYPTO = "crates/panelbt-core/src/crypto.rs"
WALLET = "crates/panelbt-core/src/wallets.rs"
PAIR = "src/ashare/backtest/crypto_wallets.py"
REFERENCE = "packages/panelbt/python/panelbt/reference_wallets.py"
MODELS = "packages/panelbt/python/panelbt/models.py"
MUTANTS = [
    (
        "spot_decimal_lattice_uses_legacy_rounding",
        CORE,
        "if !enabled || step == 0.0 {",
        "if true {",
    ),
    (
        "dynamic_flow_after_open_check",
        CRYPTO,
        "let opening = value(day, false, m, st, book);",
        "st.cash -= core.flow; let opening = value(day, false, m, st, book); st.cash += core.flow;",
    ),
    ("unrealized_profit_transferable", CORE, "w.unrealized.min(0.0)", "w.unrealized"),
    (
        "im_current_mark_not_previous_settlement",
        CORE,
        "futures::margin(shares[i], f.settlement[mark_row * m.n + i], rule_row, i, m)",
        "futures::margin(shares[i], f.crypto.as_ref().unwrap().open[rule_row * m.n + i], rule_row, i, m)",
    ),
    (
        "im_previous_rule_row",
        CORE,
        "let rule_row = self.day.min(m.t - 1);",
        "let rule_row = self.day.saturating_sub(1).min(m.t - 1);",
    ),
    ("reduction_changes_entry", WALLET, "} else if before * q > 0.0 {", "} else {"),
    (
        "reversal_keeps_old_entry",
        WALLET,
        ["self.entry[i] = 0.0;", "before == 0.0 || before.signum() != after.signum()"],
        [
            "self.entry[i] += 0.0;",
            "before == 0.0 && self.entry[i] == 0.0",
        ],
    ),
    ("omit_deficit_credit", CRYPTO, "w.balance += -st.cash;", "w.balance += 0.0;"),
    (
        "internal_flow_counted_as_return",
        PAIR,
        "equity = self._equity.copy()",
        "equity = self._equity.copy() + np.cumsum(usdm.cash_flow)",
    ),
    (
        "unequal_transfer_amounts",
        PAIR,
        "self._u.step(utarget, flow=accepted)",
        "self._u.step(utarget, flow=accepted * .5)",
    ),
    (
        "bankrupt_accepts_deposit",
        CORE,
        "let actual = if self.core.bankrupt && self.core.out.wallet.is_some() {",
        "let actual = if false {",
    ),
    (
        "follow_target_not_actual_fill",
        PAIR,
        "actual[self._indices[0]] = self._spot_shares * self._units[0][t]",
        "actual[self._indices[0]] = starget * self._units[0][t]",
    ),
    (
        "omit_snapshot_entry",
        WALLET,
        "self.entry.put(w);",
        "vec![0.0; self.entry.len()].put(w);",
    ),
    (
        "reference_wrong_funding_direction",
        REFERENCE,
        "self.money -= paid",
        "self.money += paid",
    ),
]
MUTANTS.extend(
    [
        (
            "review_p1_1_subtraction_one_ulp_only",
            CORE,
            "(quantity / step + 1e-9).floor()",
            "(quantity / step).next_up().floor()",
        ),
        (
            "review_p1_1_capacity_no_quantity_tolerance",
            CORE,
            "q >= position || (step > 0.0 && position - q <= step * 1e-9)",
            "q >= position",
        ),
        (
            "review_p1_2_raw_contract_truncation",
            PAIR,
            "return np.trunc(np.where(np.abs(value - nearest) <= 1e-9, nearest, value))",
            "return np.trunc(value)",
        ),
        (
            "review_p1_3_global_exit_override",
            PAIR,
            "exit_override[self._indices[1]], following, u.shares",
            "True, following, u.shares",
        ),
        (
            "review_p2_4_ignore_fixed_config",
            PAIR,
            '"config": copy.deepcopy(config()) if callable(config) else None,',
            '"config": None,',
        ),
        (
            "review_p2_6_skip_prefunding",
            PAIR,
            "and self.transfer.pre_fund",
            "and False",
        ),
    ]
)
MUTANTS.extend(
    [
        (
            "review_p2_5_reject_legacy_liquidation_snapshot",
            CORE,
            "if wallet_format && self.core.out.wallet.is_none() {",
            "if wallet_format != self.core.out.wallet.is_some() {",
        ),
        (
            "prefund_unused_missing_mark_poison",
            PAIR,
            "prices = np.where(needed, prices, 0.0)",
            "prices = prices.copy()",
        ),
    ]
)
MUTANTS.extend(
    [
        (
            "review2_exit_drops_flat_spot_blocked_future",
            PAIR,
            "affected = self._exit_pending.copy()",
            "affected = self._exit_pending.copy(); "
            "affected[self._indices[0]] &= state.shares != 0",
        ),
        (
            "review2_zero_buffer_skip_actual_credit_check",
            PAIR,
            "while not self._u.opening_feasible(\n"
            "            quantities, prices, flow=min(requested, available)\n"
            "        ):",
            "while False:",
        ),
        (
            "review2_fingerprint_includes_mutable_state",
            PAIR,
            '"config": copy.deepcopy(config()) if callable(config) else None,',
            '"config": copy.deepcopy(config()) if callable(config) else None, '
            '"initial_state": copy.deepcopy(strategy.state()) if protocol else None,',
        ),
        (
            "review2_legacy_display_includes_missing_fields",
            MODELS,
            "if name not in cls._unavailable_fields",
            "if True",
        ),
    ]
)
MUTANTS.extend(
    [
        (
            "review3_preview_omits_opening_fee",
            CRYPTO,
            "fee: fee(quantity),",
            "fee: 0.0,",
        ),
        (
            "review3_missing_exit_schema_defaults_empty",
            PAIR,
            "required_fields = {",
            'data.setdefault("exit_pending", False)\n        required_fields = {',
        ),
    ]
)
TESTS = [
    "test/test_crypto_wallets.py",
    "test/test_crypto_wallets_contract.py",
    "test/test_crypto_wallets_oracle.py",
    "test/test_crypto_wallets_review.py",
    "test/test_crypto_wallets_prefund_boundary.py",
    "test/test_crypto_wallets_review2.py",
    "test/test_crypto_wallets_review3.py",
]


def main():
    from test.build_guard import assert_current_build

    parser = argparse.ArgumentParser()
    parser.add_argument("--select", nargs="+")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    selected = [
        item for item in MUTANTS if args.select is None or item[0] in args.select
    ]
    if args.select:
        assert set(args.select) == {item[0] for item in selected}
    original_hash = assert_current_build()
    paths = {ROOT / item[1] for item in selected}
    originals = {p: p.read_bytes() for p in paths}
    env = dict(
        os.environ,
        CARGO_BUILD_JOBS="2",
        UV_CACHE_DIR=str(ROOT / ".cache/uv"),
        PYTHONPATH=str(ROOT / "packages/panelbt/python") + os.pathsep + str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
    )
    logs = ROOT / "tmp/mutations_crypto_wallets"
    logs.mkdir(parents=True, exist_ok=True)

    def command(args, name, timeout=600):
        p = subprocess.run(
            args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout
        )
        (logs / name).write_text(p.stdout + p.stderr)
        return p

    def build():
        p = command(["nice", "-n", "15", "make", "dev"], "build.txt")
        assert p.returncode == 0, "build error is NOT a killed mutation; see tmp log"

    def probe(path=None):
        p = command(
            [
                sys.executable,
                "-c",
                "import panelbt, ashare.backtest.crypto_wallets as pair, panelbt.reference_wallets as oracle; "
                "import panelbt.models as models; "
                "from test.build_guard import assert_current_build; import json; "
                "print(json.dumps([panelbt.__file__,panelbt._core.__file__,pair.__file__,oracle.__file__,models.__file__,assert_current_build()]))",
            ],
            "probe.txt",
        )
        assert p.returncode == 0, "import/build guard failure is NOT a kill"
        values = json.loads(p.stdout)
        assert all(Path(x).resolve().is_relative_to(ROOT) for x in values[:-1])
        if path is not None and path.suffix == ".py":
            assert str(path) in values[:-1]
        return values[-1]

    assert probe() == original_hash
    baseline = command(
        [
            "nice",
            "-n",
            "15",
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *TESTS,
            "--tb=short",
        ],
        "baseline.txt",
    )
    assert baseline.returncode == 0, baseline.stdout
    report = dict(
        original_source=original_hash,
        total=len(MUTANTS),
        tested_this_run=[item[0] for item in selected],
        mutants=[],
        previous_runs=[],
    )
    report_path = ROOT / "bench/results/mutations_crypto_wallets.json"
    if args.merge:
        previous = json.loads(report_path.read_text())
        # Retain historical evidence with its own source identity; selected
        # mutations alone are rerun against the current worktree/build.
        report["previous_runs"] = previous.get("previous_runs", []) + [
            {
                key: previous[key]
                for key in (
                    "original_source",
                    "restored_source",
                    "total",
                    "killed",
                    "restored_tests_passed",
                )
            }
        ]
        names = {item[0] for item in selected}
        report["mutants"] = [
            item for item in previous["mutants"] if item["name"] not in names
        ]
    try:
        for name, filename, before, after in selected:
            path = ROOT / filename
            old = originals[path].decode()
            edits = (
                zip(before, after, strict=True)
                if isinstance(before, list)
                else [(before, after)]
            )
            for a, b in edits:
                assert old.count(a) == 1, (name, a, old.count(a))
                old = old.replace(a, b)
            path.write_text(old)
            try:
                if path.suffix == ".rs":
                    build()
                source = probe(path)
                assert (source != original_hash) == (path.suffix == ".rs")
                trial = command(
                    [
                        "nice",
                        "-n",
                        "15",
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        *TESTS,
                        "--tb=short",
                    ],
                    name + ".txt",
                )
                failures = [
                    x for x in trial.stdout.splitlines() if x.startswith("FAILED test/")
                ]
                assert (
                    trial.returncode == 1
                    and failures
                    and "ERROR collecting" not in trial.stdout
                ), (name, trial.stdout)
                assert (
                    "ImportError" not in trial.stdout
                    and "SyntaxError" not in trial.stdout
                )
                report["mutants"].append(
                    dict(
                        name=name,
                        file=filename,
                        killed=True,
                        failures=failures,
                        source=source,
                        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                print(name, "KILLED", len(failures), flush=True)
            finally:
                path.write_bytes(originals[path])
                if path.suffix == ".rs":
                    build()
    finally:
        for p, data in originals.items():
            p.write_bytes(data)
        if any(p.suffix == ".rs" for p in originals):
            build()
        assert probe() == original_hash
        final = command(
            [
                "nice",
                "-n",
                "15",
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *TESTS,
                "--tb=short",
            ],
            "restored.txt",
        )
        report["restored_source"] = probe()
        report["restored_tests_passed"] = final.returncode == 0
        order = {item[0]: i for i, item in enumerate(MUTANTS)}
        report["mutants"].sort(key=lambda item: order[item["name"]])
        report["killed"] = len(report["mutants"])
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        assert final.returncode == 0, final.stdout


if __name__ == "__main__":
    main()
