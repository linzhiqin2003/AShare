"""Mutation checks for the step-1c rework, run in a disposable copy.

Usage: python bench/mutations_1c_rework.py --root COPY [--only NAME ...]

COPY is a separate checkout of this tree with its own `.venv` (create it with
`rsync -a --exclude /target --exclude /.venv --exclude /.cache --exclude /data
--exclude /.git ./ COPY && make -C COPY dev`). For each mutation: apply exact
edits, rebuild when Rust changed, assert the imported extension is the rebuilt
one (source digest matches, `.so` bytes differ from the clean build), run the
listed tests, restore the bytes. A control run without edits must pass first
and the clean `.so` must be reproduced at the end. Results go to
COPY/.cache/mutations.json.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LIB = "crates/panelbt-core/src/lib.rs"
VAL = "crates/panelbt-core/src/validate.rs"
BIND = "packages/panelbt/src/lib.rs"
API = "packages/panelbt/python/panelbt/api.py"
HELP = "packages/panelbt/python/panelbt/futures.py"
REF = "packages/panelbt/python/panelbt/reference_futures.py"
SPOT_REF = "packages/panelbt/python/panelbt/reference.py"
SO = "packages/panelbt/python/panelbt/_core.abi3.so"

FUT_TESTS = [
    "test/test_futures.py",
    "test/test_futures_invariance.py",
    "test/test_reference_futures.py",
]
VAL_TESTS = ["test/test_validation.py"]
LEGACY = ["test/test_extended.py", "test/test_cleanup.py", "test/test_inputs.py"]


def e(path, old, new, count=1):
    return (path, old, new, count)


MUTATIONS = [
    # ---- Part 1: review fixes
    (
        "P0 legacy cash check on margin-account spot sales with futures",
        [
            e(
                LIB,
                "&& (!CLASSIC\n                                || if st.cash",
                "&& (!(CLASSIC || FUT)\n                                || if st.cash",
            )
        ],
        FUT_TESTS,
    ),
    (
        "P1-3 futures disable the cash specialization",
        [
            e(
                API,
                "    classic = (\n        not margin.allow_short",
                "    classic = (\n        market.futures is None\n        and not margin.allow_short",
            )
        ],
        FUT_TESTS,
    ),
    (
        "P1-2 futures reduction skipped when own cash would turn negative",
        [
            e(
                LIB,
                "let x = futures::quote(q, i, day, m, s);\n                        if x.price > 0.0 && x.price.is_finite() {",
                "let x = futures::quote(q, i, day, m, s);\n                        let probe = futures::project(account, fm, i, q, st.shares[i], book.basis[i], day, m, s);\n                        if x.price > 0.0 && x.price.is_finite() && probe.0.own_cash(s.flags[4]) - probe.1 >= 0.0 {",
            )
        ],
        FUT_TESTS,
    ),
    (
        "P1-1 expired targets not counted on liquidation rows",
        [
            e(
                LIB,
                "if f.kind[i] != 0 && day as i64 > f.expiry[i] && w != 0.0 {",
                "if false && f.kind[i] != 0 && day as i64 > f.expiry[i] && w != 0.0 {",
            )
        ],
        FUT_TESTS,
    ),
    (
        "P1-4 weight to contracts rounds to nearest",
        [
            e(
                LIB,
                "let contracts = raw.signum() * raw.abs().floor();",
                "let contracts = raw.signum() * raw.abs().round();",
            )
        ],
        FUT_TESTS,
    ),
    (
        "P1-4 margin basket dust filter removed",
        [
            e(
                LIB,
                "        if q.abs() > quantity_tolerance(st.shares[i] + desired, st.shares[i]) {",
                "        if q.abs() > 0.0 {",
            )
        ],
        FUT_TESTS,
    ),
    (
        "P1-4 cash basket dust filter removed",
        [
            e(
                LIB,
                "if q > quantity_tolerance(st.shares[i] + desired, st.shares[i]) {",
                "if q > 0.0 {",
            )
        ],
        FUT_TESTS,
    ),
    (
        "P1-4 expiry row not tradable",
        [
            e(
                LIB,
                "if future && day as i64 > m.futures.as_ref().unwrap().expiry[i] {",
                "if future && day as i64 >= m.futures.as_ref().unwrap().expiry[i] {",
            )
        ],
        FUT_TESTS,
    ),
    (
        "P1-4 hedge_contracts rounds to nearest",
        [
            e(
                HELP,
                "contracts = np.floor(unsigned)",
                "contracts = np.floor(unsigned + 0.5)",
            )
        ],
        ["test/test_futures.py"],
    ),
    (
        "P2 roll_targets may roll back",
        [
            e(
                HELP,
                "            active = max(active, previous)",
                "            active = active",
            )
        ],
        ["test/test_futures.py"],
    ),
    (
        "A unheld contract without quote sized at zero",
        [
            e(
                LIB,
                "} else if day > 0 {\n                    m.futures.as_ref().unwrap().settlement[base - m.n + i]",
                "} else if day > 1000000 {\n                    m.futures.as_ref().unwrap().settlement[base - m.n + i]",
            )
        ],
        FUT_TESTS,
    ),
    (
        "rounded-to-zero counter removed",
        [e(LIB, "out.counts[38] += 1;", "out.counts[38] += 0;")],
        FUT_TESTS,
    ),
    (
        "cash account spot buys ignore futures collateral",
        [
            e(
                LIB,
                "let free = |cash: f64| if FUT { cash - fm } else { cash };",
                "let free = |cash: f64| cash;",
            )
        ],
        FUT_TESTS,
    ),
    (
        "cash account monitored without ever holding a contract",
        [
            e(
                LIB,
                "let monitored = !CLASSIC || (FUT && book.seen);",
                "let monitored = !CLASSIC || FUT;",
            )
        ],
        FUT_TESTS,
    ),
    # ---- New reference implementation: key points
    (
        "reference: variation scaled",
        [
            e(
                REF,
                "variation = q * self.book.multiplier[i] * (price - self.book.basis[i])",
                "variation = q * self.book.multiplier[i] * (price - self.book.basis[i]) * 0.999",
            )
        ],
        FUT_TESTS,
    ),
    (
        "reference: collateral without multiplier",
        [
            e(
                REF,
                "return abs(contracts) * self.multiplier[i] * mark",
                "return abs(contracts) * mark",
            )
        ],
        FUT_TESTS,
    ),
    (
        "reference: contracts round to nearest",
        [
            e(
                REF,
                "target = math.copysign(math.floor(abs(x)), x)",
                "target = math.copysign(round(abs(x)), x)",
            )
        ],
        FUT_TESTS,
    ),
    (
        "reference: liquidation-row expired targets not counted",
        [e(REF, "+= self.count_expired(d, d.commands)", "+= 0")],
        FUT_TESTS,
    ),
    # ---- Part 2: kernel content checks
    (
        "check: price domain accepts nonpositive",
        [
            e(
                VAL,
                "Domain::PositiveOrNan => x.is_infinite() || x <= 0.0,",
                "Domain::PositiveOrNan => x.is_infinite(),",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: volume domain accepts infinity",
        [
            e(
                VAL,
                "Domain::NonnegativeOrNan => x.is_infinite() || x < 0.0,",
                "Domain::NonnegativeOrNan => x < 0.0,",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: split domain accepts NaN",
        [
            e(
                VAL,
                "Domain::Positive => !x.is_finite() || x <= 0.0,",
                "Domain::Positive => x.is_infinite() || x <= 0.0,",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: dividend domain accepts negatives",
        [
            e(
                VAL,
                "Domain::Nonnegative => !x.is_finite() || x < 0.0,",
                "Domain::Nonnegative => !x.is_finite(),",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: weights accept infinity",
        [e(VAL, "Domain::Finite => !x.is_finite(),", "Domain::Finite => x.is_nan(),")],
        VAL_TESTS,
    ),
    (
        "check: mask byte 2 accepted",
        [
            e(
                VAL,
                "any_chunk(a, |c| c.iter().any(|&x| x > 1))",
                "any_chunk(a, |c| c.iter().any(|&x| x > 2))",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: settlement of futures columns",
        [e(BIND, "|x: f64| !x.is_finite() || x <= 0.0)", "|x: f64| x <= 0.0)")],
        VAL_TESTS,
    ),
    (
        "check: split on futures columns",
        [e(BIND, "|x: f64| x != 1.0)", "|x: f64| x > 1.0e300)")],
        VAL_TESTS,
    ),
    (
        "check: dividend on futures columns",
        [e(BIND, "|x: f64| x != 0.0)", "|x: f64| x > 1.0e300)")],
        VAL_TESTS,
    ),
    (
        "check: delist on futures columns",
        [e(BIND, "|x: u8| x != 0)", "|x: u8| x > 1)")],
        VAL_TESTS,
    ),
    (
        "check: futures rates",
        [
            e(
                BIND,
                "if validate::any_bad(a, Domain::Nonnegative) {",
                "if validate::any_bad(a, Domain::Finite) {",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: multiplier",
        [
            e(
                BIND,
                ".any(|&i| !multiplier[i].is_finite() || multiplier[i] <= 0.0)",
                ".any(|&i| multiplier[i] < 0.0)",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: expiry",
        [
            e(
                BIND,
                ".any(|&i| expiry[i] < 0 || expiry[i] > t as i64)",
                ".any(|&i| expiry[i] < 0)",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: day order",
        [
            e(
                VAL,
                "days.windows(2).any(|w| w[1] <= w[0])",
                "days.windows(2).any(|w| w[1] < w[0])",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: duplicate columns",
        [
            e(
                VAL,
                "let sorted = cols[a..b].windows(2).all(|x| x[0] < x[1]);",
                "let sorted = cols[a..b].windows(2).all(|x| x[0] <= x[1]);",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: negative without short",
        [
            e(
                VAL,
                ".any(|(&x, &w)| w < 0.0 && !future(x))",
                ".any(|(&x, &w)| w < -1.0 && !future(x))",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: integer contracts",
        [
            e(
                VAL,
                ".any(|(&x, &w)| future(x) && w != w.trunc())",
                ".any(|(&x, &w)| future(x) && w < 0.0 && w != w.trunc())",
            )
        ],
        VAL_TESTS,
    ),
    (
        "check: cash row sum",
        [e(VAL, "if sum > 1.0 + 1e-9 {", "if sum > 1.5 {")],
        VAL_TESTS,
    ),
    (
        "boundary: chunk scan skips each chunk's last element",
        [
            e(
                VAL,
                "        a.par_chunks(CHUNK).any(test)",
                "        a.par_chunks(CHUNK).any(|c| test(&c[..c.len() - 1]))",
            )
        ],
        VAL_TESTS,
    ),
    (
        "boundary: chunk scan skips each chunk's first element",
        [
            e(
                VAL,
                "        a.par_chunks(CHUNK).any(test)",
                "        a.par_chunks(CHUNK).any(|c| test(&c[1..]))",
            )
        ],
        VAL_TESTS,
    ),
    (
        "boundary: column scan skips each block's last row",
        [
            e(
                VAL,
                "        a.par_chunks(rows * n).any(scan)",
                "        a.par_chunks(rows * n).any(|b| scan(&b[..b.len() - n]))",
            )
        ],
        VAL_TESTS,
    ),
    (
        "prepare: first delist reduction keeps the later row",
        [e(VAL, "*x = (*x).min(y);", "*x = (*x).max(y);")],
        LEGACY + VAL_TESTS,
    ),
    # ---- Second review: spot sales in a cash account with futures debt
    (
        "R2 kernel only: legacy check also under debt",
        [
            e(
                LIB,
                "} else {\n                                    x.amount >= x.fees",
                "} else {\n                                    st.cash + x.amount >= x.fees",
            )
        ],
        FUT_TESTS,
    ),
    (
        "R2 reference only: legacy check also under debt",
        [e(SPOT_REF, "else a >= c + tax + fee", "else self.cash + a >= c + tax + fee")],
        FUT_TESTS,
    ),
    (
        "R2 both: new debt branch deleted",
        [
            e(
                LIB,
                "} else {\n                                    x.amount >= x.fees",
                "} else {\n                                    st.cash + x.amount >= x.fees",
            ),
            e(
                SPOT_REF,
                "else a >= c + tax + fee",
                "else self.cash + a >= c + tax + fee",
            ),
        ],
        FUT_TESTS,
    ),
    (
        "R2 kernel only: debt condition inverted",
        [e(LIB, "|| if st.cash >= 0.0 {", "|| if st.cash < 0.0 {")],
        FUT_TESTS,
    ),
    (
        "R2 both: debt condition inverted",
        [
            e(LIB, "|| if st.cash >= 0.0 {", "|| if st.cash < 0.0 {"),
            e(SPOT_REF, "if self.cash >= 0\n", "if self.cash < 0\n"),
        ],
        FUT_TESTS + LEGACY,
    ),
]


def sh(cmd, root, **kw):
    return subprocess.run(cmd, cwd=root, text=True, capture_output=True, **kw)


def so_digest(root):
    return hashlib.sha256((root / SO).read_bytes()).hexdigest()


def build(root):
    r = sh(["make", "dev"], root)
    if r.returncode:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    probe = sh(
        [
            ".venv/bin/python",
            "-c",
            "import sys; sys.path.insert(0, '.'); from test.build_guard import "
            "assert_current_build as a; import panelbt._core as c; a(); print(c.__file__)",
        ],
        root,
    )
    if probe.returncode:
        raise RuntimeError("stale or foreign extension: " + probe.stderr[-1000:])
    assert Path(probe.stdout.strip()).resolve() == (root / SO).resolve(), probe.stdout
    return so_digest(root)


def tests(root, selection):
    r = sh(
        [
            ".venv/bin/python",
            "-m",
            "pytest",
            "-x",
            "-q",
            "-p",
            "no:cacheprovider",
            *selection,
        ],
        root,
    )
    tail = [x for x in r.stdout.splitlines() if x.strip()][-3:]
    return r.returncode, tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    root = args.root.resolve()
    clean = build(root)
    code, tail = tests(root, sorted({t for _, _, ts in MUTATIONS for t in ts}))
    assert code == 0, ("control run must pass", tail)
    results = [{"name": "control (no edit)", "caught": False, "tail": tail}]
    print("control passes; clean .so", clean[:16], flush=True)
    for name, edits, selection in MUTATIONS:
        if args.only and not any(x in name for x in args.only):
            continue
        originals = {}
        try:
            for path, old, new, count in edits:
                p = root / path
                originals.setdefault(path, p.read_bytes())
                current = p.read_text()
                assert current.count(old) == count, (name, path, old)
                p.write_text(current.replace(old, new))
            rust = any(p.endswith(".rs") for p, *_ in edits)
            digest = build(root) if rust else clean
            if rust:
                assert digest != clean, (name, "extension was not rebuilt")
            code, tail = tests(root, selection)
            row = {
                "name": name,
                "caught": code != 0,
                "rebuilt": rust,
                "so": digest[:16],
                "tail": tail,
            }
        except (AssertionError, RuntimeError) as exc:
            row = {"name": name, "caught": None, "error": str(exc)[-800:]}
        finally:
            for path, data in originals.items():
                (root / path).write_bytes(data)
        results.append(row)
        print(
            (
                "CAUGHT "
                if row["caught"]
                else "SURVIVED "
                if row["caught"] is False
                else "ERROR "
            )
            + name,
            flush=True,
        )
    final = build(root)
    results.append({"name": "restored", "so_equals_clean": final == clean})
    (root / ".cache").mkdir(exist_ok=True)
    (root / ".cache/mutations.json").write_text(json.dumps(results, indent=1))
    print("restored .so equals clean:", final == clean)
    sys.exit(0 if final == clean else 1)


if __name__ == "__main__":
    main()
