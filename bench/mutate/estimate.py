"""Count mutants per shard and estimate the wall time of a full run.

Python counts come from mutmut's own generator (no tests run); Rust counts from
``cargo-mutants --list``. Seconds per mutant are taken from the latest saved run of the
shard when there is one, otherwise from the measured defaults below. Writes
``.cache/testing/mutation_estimate.json`` and prints a table.
"""

from __future__ import annotations

import json
import subprocess

from bench.mutate import rust as rs
from bench.mutate.common import (
    ERROR,
    ROOT,
    SKIPPED,
    TESTING,
    load_shard_results,
    load_shards,
    log,
    now,
)

# Measured on this machine (15 cores shared, 2 parallel mutants), see docs/TESTING_MUTATION.md.
DEFAULT_SECONDS = {"python": 2.0, "rust": 45.0}


def python_count(shard) -> int:
    from mutmut.mutation.file_mutation import mutate_file_contents

    root = ROOT / shard.root
    total = 0
    for path in sorted(root.rglob("*.py")):
        inner = path.relative_to(root).as_posix()
        if "__pycache__" in inner or not shard.matches(inner):
            continue
        try:
            total += len(mutate_file_contents(str(path), path.read_text()).mutant_names)
        except Exception as exc:  # unparsable by libcst: mutmut skips it too
            log(f"skip {path.relative_to(ROOT)}: {type(exc).__name__}")
    return total


def rust_count(shard) -> int:
    cmd = [rs.cargo_mutants(), "mutants", "--list", "-p", shard.package]
    for f in shard.files:
        cmd += ["-f", f]
    out = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=300
    ).stdout
    return sum(1 for line in out.splitlines() if line.strip())


def observed_seconds():
    """Wall seconds per mutant from saved runs: run time x parallelism / mutants."""
    out = {}
    for data in load_shard_results("full"):
        n = sum(m.status not in (SKIPPED, ERROR) for m in data["mutants"])
        secs = data["meta"].get("seconds")
        if n and secs:
            name = data["shard"].split("@")[0]
            out[name] = secs / n  # already wall time at the run's parallelism
    return out


def main():
    shards = load_shards()
    seen = observed_seconds()
    rows = []
    for shard in shards.values():
        n = python_count(shard) if shard.language == "python" else rust_count(shard)
        per = seen.get(shard.name)
        source = "measured" if per else "default"
        if per is None:
            per = DEFAULT_SECONDS[shard.language] / (
                4 if shard.language == "python" else 2
            )
        rows.append(
            {
                "shard": shard.name,
                "language": shard.language,
                "mutants": n,
                "wall_seconds_per_mutant": round(per, 2),
                "basis": source,
                "hours": round(n * per / 3600, 2),
            }
        )
    for r in rows:
        log(
            f"{r['shard']:28s} {r['language']:6s} {r['mutants']:6d} mutants "
            f"{r['wall_seconds_per_mutant']:6.2f}s/mutant ({r['basis']}) ~{r['hours']:.1f} h"
        )
    total = {
        lang: sum(r["hours"] for r in rows if r["language"] == lang)
        for lang in ("python", "rust")
    }
    log(
        f"total: python ~{total['python']:.1f} h, rust ~{total['rust']:.1f} h (4 parallel)"
    )
    TESTING.mkdir(parents=True, exist_ok=True)
    path = TESTING / "mutation_estimate.json"
    path.write_text(
        json.dumps({"generated": now(), "shards": rows, "hours": total}, indent=1)
        + "\n"
    )


if __name__ == "__main__":
    main()
