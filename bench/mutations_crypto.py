"""Real Rust/Python mutations imported from this worktree for every trial.

Restores original sources and extension even on failure. A compiler failure,
import failure or stale-build guard is NOT a killed mutant. Run in this
worktree's venv; each subprocess has a five-minute timeout, cargo uses -j 2.
Use --group original then --group review to bound total command duration;
the second completed group merges both reports for the same source build.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CRYPTO = ROOT / "crates/panelbt-core/src/crypto.rs"
FUTURES = ROOT / "crates/panelbt-core/src/futures.rs"
MODELS = ROOT / "packages/panelbt/python/panelbt/models.py"
REFERENCE = ROOT / "packages/panelbt/python/panelbt/reference_crypto.py"

MUTANTS = [
    (
        "funding_direction",
        FUTURES,
        "let paid = shares[i] * per;",
        "let paid = -shares[i] * per;",
    ),
    (
        "funding_right_boundary",
        FUTURES,
        "let per = c.funding[base + i];",
        "let per = c.funding[((day + 1) % m.t) * m.n + i];",
    ),
    (
        "funding_on_liquidated_contracts",
        CRYPTO,
        "let q = -st.shares[i];",
        "let q = -st.shares[i];\n        if !c.funding.is_empty() { let paid = st.shares[i] * c.funding[day*m.n+i]; st.cash -= paid; core.out.daily[30*m.t+day] += paid; core.out.contribution[i] -= paid; }",
    ),
    (
        "lambda_frozen_maintenance",
        CRYPTO,
        "slope += (q - r * q.abs()) * (end - c.open[k]);",
        "slope += (q - 0.0 * r * q.abs()) * (end - c.open[k]);",
    ),
    (
        "omit_post_trade_check",
        CRYPTO,
        "liquidate(core, day, 42, 0.0, m, s, st, book)?;",
        "if false { liquidate(core, day, 42, 0.0, m, s, st, book)?; }",
    ),
    (
        "uncapped_clearance_fee",
        CRYPTO,
        "let fee = due.min(st.cash.max(0.0));",
        "let fee = due;",
    ),
    (
        "delivery_execution_instead_of_settlement",
        FUTURES,
        "let settlement = f.settlement[base + i];",
        "let settlement = if day as i64 == f.expiry[i] { m.price[base+i] } else { f.settlement[base+i] };",
    ),
    (
        "opening_wrongly_exempt_min_notional",
        CRYPTO,
        "if !c.notional.is_empty()\n        && q * f.multiplier[i] * c.open[day * m.n + i] < c.notional[day * m.n + i]",
        "if false && !c.notional.is_empty()\n        && q * f.multiplier[i] * c.open[day * m.n + i] < c.notional[day * m.n + i]",
    ),
    (
        "maintenance_used_for_initial",
        CRYPTO,
        "rate(f.margin_rate, day, i, m)",
        "rate(if c.maint.is_empty() { f.margin_rate } else { c.maint }, day, i, m)",
    ),
    (
        "omit_funding_close_risk",
        CRYPTO,
        "liquidate(core, day, 44, 0.0, m, s, st, book)?;",
        "if false { liquidate(core, day, 44, 0.0, m, s, st, book)?; }",
    ),
    ("tier_uses_low", CRYPTO, "1 => c.high[k],", "1 => c.low[k],"),
    (
        "retain_pending_after_liquidation",
        CRYPTO,
        "book.held.clear();\n    core.pending.clear();",
        "book.held.clear();",
    ),
    (
        "opening_ignores_execution_mark_loss",
        CRYPTO,
        "equity += q * f.multiplier[i] * (mark - x.price) - x.fees.commission;",
        "equity += -x.fees.commission;",
    ),
    (
        "intraday_before_endpoint_only",
        CRYPTO,
        "v.0 - v.1 + slope <= v.2 * 1e-12",
        "v.0 - v.1 + slope < -v.2 * 1e-12",
    ),
    (
        "quantity_filter_wrongly_carried",
        CRYPTO,
        "if !allowed || reducing.abs() > cap {",
        "if q != reducing {",
    ),
    (
        "zero_equity_candidate_profit",
        CRYPTO,
        ["v.0 <= 0.0 && !core.orders.is_empty()", "equity > 0.0\n                &&"],
        ["false", "true\n                &&"],
    ),
    (
        "invalid_reduction_quote_drops_target",
        CRYPTO,
        "blocked[usize::from(reducing < 0.0)] = true;\n                    core.out.counts[28]",
        "core.out.counts[28]",
    ),
    (
        "invalid_opening_quote_drops_target",
        CRYPTO,
        "blocked[usize::from(desired < 0.0)] = true;\n                        core.out.counts[28]",
        "core.out.counts[28]",
    ),
    (
        "weight_intent_missing_marks",
        CRYPTO,
        "if st.shares[i] == 0.0\n            && w != 0.0",
        "if shares && st.shares[i] == 0.0\n            && w != 0.0",
    ),
    (
        "volume_block_lost_after_quantity_filter",
        CRYPTO,
        "if !allowed || reducing.abs() > cap {",
        "if !allowed {",
    ),
    (
        "old_npz_enables_crypto",
        MODELS,
        "values.setdefault(key, None)",
        'values.setdefault(key, np.zeros_like(values["nav"]))',
    ),
    (
        "reference_quantity_filter_wrongly_carried",
        REFERENCE,
        'if obstacles:\n                    blocked.add("buy" if reduction > 0 else "sell")',
        'if qty != reduction:\n                    blocked.add("buy" if reduction > 0 else "sell")',
    ),
    (
        "reference_zero_equity_candidate_profit",
        REFERENCE,
        "if any(qty for _, qty in orders) and self.balance()[0] <= 0:",
        "if False:",
    ),
    (
        "reference_invalid_quote_drops_target",
        REFERENCE,
        [
            'else:\n                        blocked.add("buy" if reduction > 0 else "sell")',
            'else:\n                        blocked.add("buy" if desired > 0 else "sell")',
        ],
        ["else:\n                        pass", "else:\n                        pass"],
    ),
    (
        "reference_weight_intent_missing_marks",
        REFERENCE,
        "self.validate_intent_marks(i, weight)",
        "pass  # mutant: skip intent validation",
    ),
    (
        "inactive_first_row_weight_dropped",
        CRYPTO,
        "f.settlement[base + i]\n        };\n        let raw =",
        "0.0\n        };\n        let raw =",
    ),
    (
        "reference_inactive_first_row_weight_dropped",
        REFERENCE,
        "mark = float(self.f.settlement[max(d - 1, 0), i])",
        "mark = float(self.f.settlement[d - 1, i]) if d else 0.0",
    ),
]


def command(argv, log, env):
    with log.open("w") as stream:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    return result.returncode


def combine_groups(source):
    paths = [ROOT / f".cache/mutations_crypto_{g}.json" for g in ("original", "review")]
    if not all(p.exists() for p in paths):
        return
    groups = [json.loads(p.read_text()) for p in paths]
    if not all(g["original_source"] == source and g.get("restored") for g in groups):
        return
    mutants = [m for g in groups for m in g["mutants"]]
    if len(mutants) != len(MUTANTS) or {m["name"] for m in mutants} != {
        m[0] for m in MUTANTS
    }:
        return
    assert all(m["killed"] and m["failures"] for m in mutants)
    out = ROOT / "bench/results/mutations_crypto.json"
    previous = json.loads(out.read_text()) if out.exists() else None
    if previous and previous["original_source"] == source:
        previous = previous.get("previous_report")
    report = dict(
        groups[0],
        mutants=mutants,
        killed=len(mutants),
        total=len(mutants),
        restored_source=source,
        rust_mutants=sum(m["file"].endswith(".rs") for m in mutants),
        python_mutants=sum(m["file"].endswith(".py") for m in mutants),
        commands=[
            "bench/mutations_crypto.py --group original",
            "bench/mutations_crypto.py --group review",
        ],
        previous_report=previous,
    )
    out.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    parser.add_argument("--group", choices=["original", "review"])
    args = parser.parse_args()
    import panelbt as bt

    from test.build_guard import assert_current_build

    original_hash = assert_current_build()
    print("panelbt.__file__ =", bt.__file__, flush=True)
    print("panelbt._core.__file__ =", bt._core.__file__, flush=True)
    assert Path(bt.__file__).resolve().is_relative_to(ROOT)
    assert Path(bt._core.__file__).resolve().is_relative_to(ROOT)
    originals = {p: p.read_text() for p in (CRYPTO, FUTURES, MODELS, REFERENCE)}
    env = dict(
        os.environ,
        UV_CACHE_DIR=str(ROOT / ".cache/uv"),
        CARGO_BUILD_JOBS="2",
        VIRTUAL_ENV=str(ROOT / ".venv"),
        PYO3_PYTHON=sys.executable,
        PYTHONPATH=str(ROOT / "packages/panelbt/python"),
    )
    logs = ROOT / ".cache/mutations_crypto"
    logs.mkdir(parents=True, exist_ok=True)
    build = [
        "nice",
        "-n",
        "15",
        str(ROOT / ".venv/bin/maturin"),
        "develop",
        "--release",
        "-j",
        "2",
        "--manifest-path",
        "packages/panelbt/Cargo.toml",
    ]
    report = dict(
        original_source=original_hash,
        package=str(Path(bt.__file__).relative_to(ROOT)),
        extension=str(Path(bt._core.__file__).relative_to(ROOT)),
        mutants=[],
    )
    out = ROOT / (
        "bench/results/mutations_crypto.json"
        if not args.group
        else f".cache/mutations_crypto_{args.group}.json"
    )
    try:
        for index, (name, path, before, after) in enumerate(MUTANTS):
            if args.only and name != args.only:
                continue
            if (
                args.group == "original"
                and index >= 14
                or args.group == "review"
                and index < 14
            ):
                continue
            for p, source in originals.items():
                if p.read_text() != source:
                    p.write_text(source)
            edits = (
                zip(before, after, strict=True)
                if isinstance(before, list)
                else [(before, after)]
            )
            source = originals[path]
            occurrences = []
            for old, new in edits:
                assert old in source, name
                occurrences.append(source.count(old))
                source = source.replace(old, new)
            path.write_text(source)
            assert command(build, logs / f"{name}-build.log", env) == 0, (
                f"{name}: build failed, not a kill"
            )
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import panelbt, panelbt.models, panelbt.reference_crypto, hashlib, sys; "
                    "from pathlib import Path; from test.build_guard import assert_current_build; "
                    "print(panelbt.__file__); print(panelbt._core.__file__); "
                    "print(panelbt.models.__file__); print(panelbt.reference_crypto.__file__); "
                    "print(assert_current_build()); print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())",
                    str(path),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            assert probe.returncode == 0, probe.stderr
            loaded = probe.stdout.strip().splitlines()
            assert all(Path(p).resolve().is_relative_to(ROOT) for p in loaded[:4])
            source, file_hash = loaded[-2:]
            assert file_hash == hashlib.sha256(path.read_bytes()).hexdigest()
            assert file_hash != hashlib.sha256(originals[path].encode()).hexdigest()
            if path.suffix == ".rs":
                assert source != original_hash, (
                    "mutant did not reach the imported extension"
                )
            else:
                assert source == original_hash
                assert str(path) in loaded[:4], "Python mutation was not imported"
            log = logs / f"{name}-pytest.log"
            code = command(
                [
                    "nice",
                    "-n",
                    "15",
                    sys.executable,
                    "-m",
                    "pytest",
                    "test/test_crypto.py",
                    "test/test_crypto_review.py",
                    "test/test_crypto_review_intent.py",
                    "-q",
                    "--tb=short",
                ],
                log,
                env,
            )
            failures = [
                line
                for line in log.read_text().splitlines()
                if line.startswith(
                    (
                        "FAILED test/test_crypto.py::",
                        "FAILED test/test_crypto_review.py::",
                        "FAILED test/test_crypto_review_intent.py::",
                    )
                )
            ]
            killed = code == 1 and bool(failures)
            report["mutants"].append(
                dict(
                    name=name,
                    occurrences=occurrences,
                    source=source,
                    file=str(path.relative_to(ROOT)),
                    file_sha256=file_hash,
                    killed=killed,
                    failures=failures,
                )
            )
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(name, "KILLED" if killed else "SURVIVED/ERROR", flush=True)
            assert killed, name
    finally:
        for p, source in originals.items():
            if p.read_text() != source:
                p.write_text(source)
        assert command(build, logs / "restore-build.log", env) == 0, (
            "original extension restore failed"
        )
    report["killed"] = sum(x["killed"] for x in report["mutants"])
    report["total"] = len(report["mutants"])
    restored_source = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from test.build_guard import assert_current_build; print(assert_current_build())",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        timeout=30,
    ).strip()
    assert restored_source == original_hash
    report["restored"] = True
    report["restored_source"] = restored_source
    out.write_text(json.dumps(report, indent=2) + "\n")
    if args.group and not args.only:
        combine_groups(restored_source)


if __name__ == "__main__":
    main()
