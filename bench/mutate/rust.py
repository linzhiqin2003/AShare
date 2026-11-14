"""Rust shards: cargo-mutants with bench/mutate/cargo-shim standing in for cargo.

cargo-mutants copies the tree to a scratch directory per job, applies one mutant, and runs
``cargo test --no-run`` (build) then ``cargo test`` (test). Through the CARGO variable both go
to the shim, which also rebuilds the PyO3 extensions from the scratch tree, proves they are the
ones imported, and runs the shard's pytest selection there (see cargo_shim.py).

cargo-mutants must be started as the ``cargo-mutants`` binary, not ``cargo mutants``: cargo sets
CARGO for its subcommands and would override ours.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from bench.mutate.cargo_shim import INFRA_SENTINEL
from bench.mutate.common import (
    ERROR,
    KILLED,
    PYTHON,
    ROOT,
    SURVIVED,
    TIMEOUT,
    UNVIABLE,
    WORK,
    Mutant,
    Shard,
    child_env,
    log,
)
from bench.mutate.python import stop

SHIM = ROOT / "bench/mutate/cargo-shim"
SUMMARY = {
    "CaughtMutant": KILLED,
    "MissedMutant": SURVIVED,
    "Timeout": TIMEOUT,
    "Unviable": UNVIABLE,
}


def cargo_mutants() -> str:
    path = shutil.which("cargo-mutants") or str(
        Path.home() / ".cargo/bin/cargo-mutants"
    )
    if not Path(path).exists():
        raise SystemExit(
            "cargo-mutants not found: cargo install --locked cargo-mutants"
        )
    return path


def outdir(shard: Shard, scope: str) -> Path:
    return WORK / "rs" / (shard.name if scope == "full" else f"{shard.name}-{scope}")


def run_shard(
    shard: Shard,
    scope: str = "full",
    narrow: str | None = None,
    in_diff: Path | None = None,
    jobs: int = 4,
    timeout: float = 3600,
    regex: str | None = None,
    part: str | None = None,
    xdist: int = 1,
):
    files = [narrow] if narrow and narrow.endswith(".rs") else shard.files
    out = outdir(shard, scope)
    if part:
        out = out.with_name(out.name + "-part" + part.replace("/", "of"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tests = shard.test_args()
    if xdist > 1:
        tests += ["-n", str(xdist)]
    workers = max(1, jobs // max(1, xdist))
    cmd = [
        cargo_mutants(),
        "mutants",
        "-p",
        shard.package,
        "--output",
        str(out),
        "-j",
        str(workers),
        "--no-shuffle",
        "--minimum-test-timeout",
        "120",
        "--timeout-multiplier",
        "3",
        "--build-timeout",
        "900",
        "--cap-lints",
        "true",
    ]
    for f in files:
        cmd += ["-f", f]
    if in_diff is not None:
        cmd += ["--in-diff", str(in_diff)]
    if regex:
        cmd += ["--re", regex]
    if part:  # "K/N": cargo-mutants numbers shards from 0
        cmd += ["--shard", part]
    real = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
    env = child_env(
        CARGO=str(SHIM),
        MUTATE_REAL_CARGO=real,
        MUTATE_PYTHON=str(PYTHON),
        MUTATE_PYTEST=json.dumps(tests),
        PYO3_PYTHON=str(PYTHON),
        # Mutants rebuild the extensions once each: trade fat LTO for link time.
        # Floating-point semantics do not depend on LTO or codegen units.
        CARGO_PROFILE_RELEASE_LTO="off",
        CARGO_PROFILE_RELEASE_CODEGEN_UNITS="16",
        CARGO_PROFILE_RELEASE_INCREMENTAL="true",
        CARGO_BUILD_JOBS=str(max(1, jobs // workers)),
    )
    logfile = out / "cargo-mutants.log"
    log(
        f"{shard.name}: cargo-mutants on {', '.join(files)} x{workers}; pytest {' '.join(tests)}"
    )
    started = time.monotonic()
    with logfile.open("w") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop(proc)
            code = "timeout"
    seconds = time.monotonic() - started
    meta = {
        "tool": "cargo-mutants",
        "exit": code,
        "seconds": round(seconds, 1),
        "files": files,
        "tests": tests,
        "log": str(logfile.relative_to(ROOT)),
    }
    mutants, baseline = parse(shard, out / "mutants.out")
    meta["baseline"] = baseline
    # cargo-mutants exit codes: 0 all caught, 2 some missed, 3 timeouts, 4 baseline failed.
    if code not in (0, 2, 3):
        log(f"{shard.name}: cargo-mutants exited {code}; see {meta['log']}")
        log("\n".join(logfile.read_text(errors="replace").splitlines()[-12:]))
    log(f"{shard.name}: {len(mutants)} mutants in {seconds:.0f}s")
    return mutants, meta


def parse(shard: Shard, out: Path):
    path = out / "outcomes.json"
    if not path.exists():
        return [], "missing"
    data = json.loads(path.read_text())
    mutants = []
    baseline = "not run"
    for o in data["outcomes"]:
        scenario = o["scenario"]
        if scenario == "Baseline" or "Baseline" in scenario:
            baseline = o["summary"]
            continue
        m = scenario["Mutant"]
        status = SUMMARY.get(o["summary"], ERROR)
        log_text = (
            (out / o["log_path"]).read_text(errors="replace")
            if o.get("log_path")
            else ""
        )
        note = ""
        if status == KILLED and INFRA_SENTINEL in log_text:
            status, note = ERROR, "scratch extension not loaded (shim proof failed)"
        diff = (
            (out / o["diff_path"]).read_text(errors="replace")
            if o.get("diff_path")
            else ""
        )
        seconds = sum(p.get("duration", 0) for p in o.get("phase_results", []))
        name = m["name"]
        mutants.append(
            Mutant(
                id=name,
                shard=shard.name,
                language="rust",
                file=m["file"],
                line=m["span"]["start"]["line"],
                function=(m.get("function") or {}).get("function_name", ""),
                change=name.split(": ", 1)[1] if ": " in name else name,
                status=status,
                diff=diff,
                seconds=round(seconds, 1),
                note=note,
            )
        )
    return mutants, baseline
