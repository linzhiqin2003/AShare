"""Regenerate aggregate coverage counts without storing market-data fixtures."""

import json
import subprocess
import sys
from pathlib import Path

import paperbroker

ROOT = Path(__file__).resolve().parents[2]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)


def main():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "packages/paperbroker/tests/test_engine.py::test_random_reference_and_properties",
            "packages/paperbroker/tests/test_wide.py::test_wide_random_differential",
            "test/paper/test_end_to_end.py",
            "-q",
            "-s",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(process.stdout)
    if process.returncode:
        raise SystemExit(process.returncode)
    # Persist counters only. The end-to-end test output includes real fixture prices.
    counts = {}
    for line in process.stdout.splitlines():
        for key in ("PAPER_COVERAGE", "PAPER_WIDE_COVERAGE"):
            marker = key + "="
            if marker in line:
                counts[key] = json.loads(line.split(marker, 1)[1])
    assert set(counts) == {"PAPER_COVERAGE", "PAPER_WIDE_COVERAGE"}
    (ROOT / "bench/paper/coverage-results.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
