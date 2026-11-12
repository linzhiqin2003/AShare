"""Run real mutations, reject invalid builds, and always restore exact source bytes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mutation_cases import CASES

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / ".cache/panelops/mutations"


def patch(data, edit):
    text = data.decode()
    start = text.index(edit["scope"]) if edit.get("scope") else 0
    # The scope extends through the remainder of the file; patterns include kernel-specific state.
    prefix, part = text[:start], text[start:]
    count = part.count(edit["old"])
    if not count:
        raise ValueError(f"missing mutation pattern: {edit['path']}: {edit['old']!r}")
    return (prefix + part.replace(edit["old"], edit["new"])).encode()


def run(command):
    env = os.environ.copy()
    env["CARGO_HOME"] = str(ROOT / ".cache/cargo-home")
    env["PYO3_PYTHON"] = sys.executable
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)


def build():
    return run(
        [
            "uv",
            "sync",
            "--all-packages",
            "--group",
            "panelops-bench",
            "--group",
            "panelops-crosscheck",
        ]
    )


def identity():
    result = run([sys.executable, "-c", "import panelops; print(panelops.build_id)"])
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun selected records while retaining other results",
    )
    args = parser.parse_args()
    cases = [c for c in CASES if not args.only or c["name"].split()[0] in args.only]
    paths = {e["path"] for c in CASES for e in c["edits"]}
    originals = {p: (ROOT / p).read_bytes() for p in paths}
    errors = []
    for c in cases:
        changed = originals.copy()
        try:
            for e in c["edits"]:
                changed[e["path"]] = patch(changed[e["path"]], e)
        except ValueError as error:
            errors.append((c["name"], str(error)))
    if errors:
        for name, error in errors:
            print(name, error)
        raise SystemExit(1)
    if args.list:
        print(f"{len(cases)} valid source mutation patches")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = (
        json.loads((OUTPUT / "results.json").read_text())["mutations"]
        if args.resume and (OUTPUT / "results.json").exists()
        else []
    )
    completed = {r["name"] for r in records if r["status"] == "KILLED"}
    control = run(["make", "check"])
    (OUTPUT / "control.log").write_text(control.stdout + control.stderr)
    assert control.returncode == 0, (control.stdout + control.stderr)[-3000:]
    baseline = identity()
    report = {
        "control_exit": control.returncode,
        "original_build_id": baseline,
        "mutations": records,
    }
    try:
        pending = [c for c in cases if args.rerun or c["name"] not in completed]
        for index, c in enumerate(pending):
            changed = originals.copy()
            for e in c["edits"]:
                changed[e["path"]] = patch(changed[e["path"]], e)
            touched = {e["path"] for e in c["edits"]}
            for p in touched:
                (ROOT / p).write_bytes(changed[p])
            start = time.perf_counter()
            try:
                # The build fingerprint includes Python wrapper sources as well as Rust.
                native = any(p.startswith(("crates/", "packages/")) for p in touched)
                result = build() if native else None
                status = None
                changed_id = (
                    identity() if native and result.returncode == 0 else baseline
                )
                if result is not None and result.returncode != 0:
                    status = "INVALID_BUILD"
                elif native and changed_id == baseline:
                    status = "NOT_REBUILT"
                if status is None:
                    result = run(
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            *c["tests"],
                            "-q",
                            "-x",
                            "-p",
                            "no:cacheprovider",
                        ]
                    )
                    status = (
                        "KILLED"
                        if result.returncode == 1
                        and "FAILED " in result.stdout
                        and "ERROR collecting" not in result.stdout
                        else "SURVIVED"
                        if result.returncode == 0
                        else "INVALID_TEST"
                    )
                output = result.stdout + result.stderr
                ident = c["name"].split()[0]
                (OUTPUT / (ident + ".log")).write_text(output)
                record = dict(
                    name=c["name"],
                    source_build_id=baseline,
                    status=status,
                    seconds=round(time.perf_counter() - start, 2),
                    tests=c["tests"],
                    failed=[
                        line
                        for line in output.splitlines()
                        if line.startswith("FAILED ")
                    ][:2],
                    build_changed=changed_id != baseline if native else None,
                )
                records = [r for r in records if r["name"] != c["name"]] + [record]
                report["mutations"] = records
                (OUTPUT / "results.json").write_text(json.dumps(report, indent=2))
                print(c["name"], status, record["failed"][:1], flush=True)
            finally:
                for p in touched:
                    (ROOT / p).write_bytes(originals[p])
                # A later pure-Python mutant must never execute a preceding native mutant.
                next_native = index + 1 < len(pending) and any(
                    e["path"].startswith(("crates/", "packages/"))
                    for e in pending[index + 1]["edits"]
                )
                if (
                    any(p.startswith(("crates/", "packages/")) for p in touched)
                    and not next_native
                ):
                    restored = build()
                    assert restored.returncode == 0, restored.stderr[-2000:]
    finally:
        for p, data in originals.items():
            (ROOT / p).write_bytes(data)
        restored = build()
        assert restored.returncode == 0, restored.stderr[-2000:]
        check = run(["make", "check"])
        (OUTPUT / "restored.log").write_text(check.stdout + check.stderr)
        report["restored_check_exit"] = check.returncode
        report["restored_build_id"] = identity()
        report["exact_source_restore"] = all(
            (ROOT / p).read_bytes() == data for p, data in originals.items()
        )
        report["source_hashes"] = {
            p: hashlib.sha256(data).hexdigest() for p, data in originals.items()
        }
        (OUTPUT / "results.json").write_text(json.dumps(report, indent=2))
        assert (
            check.returncode == 0
            and report["exact_source_restore"]
            and report["restored_build_id"] == baseline
        )
    assert all(r["status"] == "KILLED" for r in records), (
        "Review survivors/invalid mutations in ignored results directory"
    )


if __name__ == "__main__":
    main()
