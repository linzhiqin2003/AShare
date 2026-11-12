"""Rebuild and regenerate the reviewed legacy fixture, including causal control.

This is an explicit maintenance command, never called by pytest or make check.
Run review_audit.py afterward before accepting changed corrected digests.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/rework"
OLD = CACHE / "round1"


def command(args, cwd, log):
    with log.open("w") as output:
        subprocess.run(
            args, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True
        )


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    if OLD.exists():
        shutil.rmtree(OLD)
    OLD.mkdir()
    archive = CACHE / "round1.tar"
    with archive.open("wb") as stream:
        subprocess.run(
            ["git", "archive", "faec31e"], cwd=ROOT, stdout=stream, check=True
        )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(OLD)], check=True)
    archive.unlink()
    source = OLD / "crates/panelbt-core/src/lib.rs"
    original = source.read_text()
    try:
        command(["uv", "sync", "--all-packages"], OLD, CACHE / "legacy-build.log")
        command(
            [
                str(OLD / ".venv/bin/python"),
                str(ROOT / "bench/review_legacy.py"),
                "dump",
                str(CACHE / "legacy-original.pkl.gz"),
            ],
            OLD,
            CACHE / "legacy-dump.log",
        )
        adjusted = original.replace(
            "cash += shares[i] * div;",
            "cash += shares[i] * div; if last[i] > 0.0 { last[i] = (last[i] - div).max(0.0); }",
        ).replace("shares[i] *= factor;", "shares[i] *= factor; last[i] /= factor;")
        before = """        if p.is_finite() {
            return p;
        }
    }
    0.0
}"""
        after = """        if p.is_finite() {
            let mut mark = p;
            for d in row + 1..=day {
                if let Some(dividend) = m.dividend { mark = (mark - dividend[d * m.n + col]).max(0.0); }
                if let Some(split) = m.split { mark /= split[d * m.n + col]; }
            }
            return mark;
        }
    }
    0.0
}"""
        assert before in adjusted
        source.write_text(adjusted.replace(before, after))
        command(["uv", "sync", "--all-packages"], OLD, CACHE / "legacy-mark-build.log")
        command(
            [
                str(OLD / ".venv/bin/python"),
                str(ROOT / "bench/review_legacy.py"),
                "dump",
                str(CACHE / "legacy-mark-only.pkl.gz"),
            ],
            OLD,
            CACHE / "legacy-mark-dump.log",
        )
        command(
            [
                sys.executable,
                "bench/review_legacy.py",
                "dump",
                str(CACHE / "legacy-current.pkl.gz"),
            ],
            ROOT,
            CACHE / "legacy-current-dump.log",
        )
        command(
            [
                sys.executable,
                "bench/review_legacy.py",
                "compare",
                str(CACHE / "legacy-original.pkl.gz"),
                str(CACHE / "legacy-current.pkl.gz"),
                "test/fixtures/legacy_sha256.npz",
            ],
            ROOT,
            CACHE / "legacy-compare.log",
        )
        command(
            [sys.executable, "bench/review_causality.py"],
            ROOT,
            CACHE / "legacy-causality.log",
        )
    finally:
        source.write_text(original)
        shutil.rmtree(OLD)
    print(
        "20000 original/corrected digests and corporate-only control regenerated; run review_audit.py"
    )


if __name__ == "__main__":
    main()
