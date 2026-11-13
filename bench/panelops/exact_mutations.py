"""Mutation check for the rolling-sum exactness work (degenerate windows, short and exact paths).

Each mutant edits crates/panelops-core/src/time.rs in place, rebuilds the extension,
runs the panelops tests and restores the file. Every mutant must make a test fail.
Run from the repository root: python bench/panelops/exact_mutations.py [name substring ...]
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crates/panelops-core/src/time.rs"
PY = ROOT / ".venv/bin/python"

MUTANTS = [
    (
        "incremental: constant windows not detected",
        "            let constant = (O != COUNT) & (counts[k] > 0) & (run[k] >= counts[k]);",
        "            let constant = (O != COUNT) & (counts[k] > 0) & (run[k] >= counts[k]) & false;",
    ),
    (
        "incremental: constant window keeps the main sum (residue carried on)",
        "            main[k] = choose(product, main[k], constant);",
        "            main[k] = choose(product, main[k], false);",
    ),
    (
        "incremental: reset drops the product's rounding error",
        "            correction[k] = choose(residual, correction[k], constant);",
        "            correction[k] = choose(0., correction[k], constant);",
    ),
    (
        "incremental: rebuild row does not advance the equal run",
        "            // The equal run spans all history, so a rebuild only extends it by today.\n            if O != COUNT {",
        "            // The equal run spans all history, so a rebuild only extends it by today.\n            if O != COUNT && false {",
    ),
    (
        "mean of a constant window divides the sum",
        "        choose(last + 0., sum / count as f64, constant)",
        "        choose(last + 0., sum / count as f64, false)",
    ),
    (
        "zero results keep their sign",
        "    } else {\n        sum + 0.\n    }",
        "    } else {\n        sum\n    }",
    ),
    (
        "short windows go incremental",
        "const SHORT_SUM_WINDOW: usize = 3;",
        "const SHORT_SUM_WINDOW: usize = 0;",
    ),
    (
        "short path: inexact error sum not sent to the exact fallback",
        "            redo[k] = !constant & (!value.is_finite() | (lost[k] != 0.));",
        "            redo[k] = !constant & !value.is_finite();",
    ),
    (
        "short path: two-sum errors dropped",
        "                    total[k] = sum + acc;",
        "                    total[k] = sum + 0. * acc;",
    ),
    (
        "short path: middle row of a three-row window dropped",
        "two_sum(a[k], b[k]);",
        "two_sum(a[k], 0. * b[k]);",
    ),
    (
        "short path: leaving row not uncounted",
        "                *c -= usize::from(leaving.get::<O, D>(k).1);",
        "                *c -= usize::from(leaving.get::<O, D>(k).1 & false);",
    ),
    (
        "exact path: expansion drops low parts",
        "        if low != 0. {\n            partials[length] = low;\n            length += 1;\n        }\n        x = high;\n    }\n    partials.truncate(length);\n    if x != 0.",
        "        if low != 0. && false {\n            partials[length] = low;\n            length += 1;\n        }\n        x = high;\n    }\n    partials.truncate(length);\n    if x != 0.",
    ),
    (
        "exact path: no half-way correction",
        "        if y == x - high {\n            high = x;\n        }",
        "        if y == x - high && false {\n            high = x;\n        }",
    ),
    (
        "exact path: constant windows not detected",
        "            let constant = counts[k] > 0 && run[k] >= counts[k];\n            *dest",
        "            let constant = counts[k] > 0 && run[k] >= counts[k] && false;\n            *dest",
    ),
    (
        "exact path: expansion not rebuilt after an overflow",
        "                exact[k].clear();\n                for r in start..=t {",
        "                exact[k].clear();\n                for r in start..start {",
    ),
    (
        "overflow: plain sum instead of the fixed-point sum",
        "        fixed_point_sum(contributions)",
        "        contributions.sum::<f64>()",
    ),
    (
        "fixed-point sum: ties ignore the sticky bits",
        "        if half && (sticky || mantissa & 1 == 1) {",
        "        if half && mantissa & 1 == 1 {",
    ),
    (
        "lanes ignore the mask",
        "            valid: if pan.mask.is_empty() {",
        "            valid: if true {",
    ),
]


def build():
    subprocess.run(
        [
            str(PY.parent / "maturin"),
            "develop",
            "--release",
            "-m",
            "packages/panelops/Cargo.toml",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=1800,
    )


def tests():
    probe = subprocess.run(
        [str(PY), "-c", "import panelops; print(panelops.__file__)"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(probe.stdout.strip()).resolve().is_relative_to(ROOT), probe.stdout
    run = subprocess.run(
        [
            str(PY),
            "-m",
            "pytest",
            "test/panelops",
            "-x",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    failed = [ln for ln in run.stdout.splitlines() if ln.startswith("FAILED")]
    return run.returncode, failed[:1]


def main():
    original = SOURCE.read_text()
    results = []
    only = sys.argv[1:]  # optional substrings: run only the mutants whose name matches
    try:
        for name, old, new in MUTANTS:
            if only and not any(o in name for o in only):
                continue
            assert original.count(old) == 1, name
            SOURCE.write_text(original.replace(old, new))
            build()
            code, failed = tests()
            results.append((name, code != 0, failed))
            print(f"{'caught' if code else 'SURVIVED'}: {name} {failed}", flush=True)
    finally:
        SOURCE.write_text(original)
        build()
    sys.exit(0 if all(caught for _, caught, _ in results) else 1)


if __name__ == "__main__":
    main()
