"""Verify a clean exported tree plus the current uncommitted allowed changes."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".cache/rework/clean"


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    archive = ROOT / ".cache/rework/clean-head.tar"
    with archive.open("wb") as out:
        subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, stdout=out, check=True)
    # Exporting the committed skeleton is read-only. No source in excluded
    # concurrent-task paths is inspected or overlaid from the live workspace.
    subprocess.run(["tar", "-xf", str(archive), "-C", str(DEST)], check=True)
    archive.unlink()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "-z"], cwd=ROOT, text=True
    ).split("\0")
    new = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT, text=True
    ).split("\0")
    forbidden = (
        ".worktrees/",
        "data/",
        "packages/paperbroker/",
        "crates/paperbroker-core/",
        "src/ashare/",
        "test/paper/",
        "docs/PAPER_NOTES.md",
    )
    copied = []
    for name in sorted(set(changed + new)):
        if not name or name.startswith(forbidden):
            continue
        src, dest = ROOT / name, DEST / name
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(name)
        elif dest.exists():
            dest.unlink()
    assert not any((DEST / x).exists() for x in (".cache", "target", ".venv"))
    env = os.environ.copy()
    logs = ROOT / ".cache/rework"
    for target in ["dev", "check"]:
        with (logs / f"clean-make-{target}.txt").open("w") as stream:
            result = subprocess.run(
                [
                    "make",
                    target,
                    "SANDBOX=1",
                    f"CARGO_HOME={ROOT / '.cache/cargo-home'}",
                ],
                cwd=DEST,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        assert result.returncode == 0, (target, result.returncode)
    assert not (DEST / ".cache").exists(), "tests wrote a repository .cache directory"
    # Keep the unmodified result tail, excluding pytest's machine-local rootdir.
    tail = (logs / "clean-make-check.txt").read_text().splitlines()[-14:]
    report = dict(
        command="git archive HEAD plus allowed git diff/untracked overlay; make dev SANDBOX=1 && make check SANDBOX=1",
        cache_target_venv_absent_before=True,
        overlaid_files=len(copied),
        source_sha256=hashlib.sha256(
            (DEST / "crates/panelbt-core/src/lib.rs").read_bytes()
        ).hexdigest(),
        check_tail=tail,
    )
    (ROOT / "bench/results/review-clean.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("\n".join(tail))
    shutil.rmtree(DEST)


if __name__ == "__main__":
    main()
