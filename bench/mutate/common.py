"""Shared pieces of the mutation-testing entry point: shards, statuses, results, reports."""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "bench/mutate"
WORK = ROOT / ".cache/mutate"
RESULTS = WORK / "results"
TESTING = ROOT / ".cache/testing"
OUTPUT = TESTING / "mutation.json"
DETAILS = TESTING / "mutation_details.json"
SURVIVORS = TESTING / "mutation_survivors.md"
PYTHON = ROOT / ".venv/bin/python"

# Unified statuses. Score = (killed + timeout) / (killed + timeout + survived).
KILLED, SURVIVED, TIMEOUT, EQUIVALENT = "killed", "survived", "timeout", "equivalent"
NO_TESTS = "no_tests"  # nothing selected exercised it: counted as survived
UNVIABLE = "unviable"  # does not compile (Rust): not a mutant at all
SKIPPED = "skipped"  # not checked (interrupted, filtered): left out
ERROR = "error"  # the harness itself failed: left out, listed in the details
COUNTED_SURVIVED = {SURVIVED, NO_TESTS}


def be_nice():
    """Lower this process's priority; every child (mutmut, cargo, pytest) inherits it.

    Mutation runs share the machine with interactive work: they must never make it sluggish.
    """
    try:
        os.nice(15)
    except OSError:
        pass


def log(*parts):
    print("[mutate]", *parts, flush=True)


def rel(path) -> str:
    """Repository-relative POSIX path (outputs never carry absolute local paths)."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass
class Shard:
    name: str
    language: str  # python | rust
    tests: list[str]
    root: str = ""  # python: mutated source root
    mutate: list[str] = field(default_factory=list)  # python: globs relative to root
    exclude: list[str] = field(default_factory=list)
    package: str = ""  # rust
    files: list[str] = field(default_factory=list)  # rust

    def owns(self, path: str) -> bool:
        """Whether a repository-relative file or directory belongs to this shard."""
        path = path.rstrip("/")
        if self.language == "rust":
            return any(f == path or f.startswith(path + "/") for f in self.files)
        prefix = self.root + "/"
        if not (path + "/").startswith(prefix) and not prefix.startswith(path + "/"):
            return False
        if prefix.startswith(path + "/"):  # a parent directory of the root
            return True
        inner = path[len(prefix) :]
        if inner.endswith(".py"):
            return self.matches(inner)
        # a directory: owned if some glob reaches into it
        return any(
            g.startswith(inner + "/") or fnmatch.fnmatch(inner + "/x.py", g)
            for g in self.mutate
        )

    def matches(self, inner: str) -> bool:
        return any(fnmatch.fnmatch(inner, g) for g in self.mutate) and not any(
            fnmatch.fnmatch(inner, g) for g in self.exclude
        )

    def test_args(self) -> list[str]:
        """Pytest selection with globs expanded (pytest does not expand them)."""
        out = []
        for t in self.tests:
            if any(c in t for c in "*?["):
                out += sorted(rel(p) for p in glob.glob(str(ROOT / t)))
            else:
                out.append(t)
        return out


def load_shards() -> dict[str, Shard]:
    data = tomllib.loads((HERE / "shards.toml").read_text())
    shards = {}
    for language in ("python", "rust"):
        for name, spec in data.get(language, {}).items():
            shards[name] = Shard(name=name, language=language, **spec)
    return shards


def select(shards: dict[str, Shard], module: str | None, language: str | None):
    """Resolve MODULE (a shard name or a path) to [(shard, narrowed path or None)]."""
    chosen = [s for s in shards.values() if not language or s.language == language]
    if not module:
        return [(s, None) for s in chosen]
    out = []
    for item in module.split(","):
        item = item.strip()
        if item in shards:
            out.append((shards[item], None))
            continue
        path = item.rstrip("/")
        if not (ROOT / path).exists():
            raise SystemExit(f"MODULE {item!r} is neither a shard nor an existing path")
        owners = [s for s in chosen if s.owns(path)]
        if not owners:
            raise SystemExit(f"no shard covers {item!r}; add one to shards.toml")
        out += [(s, path) for s in owners]
    return out


@dataclass
class Mutant:
    id: str  # tool's own name (mutmut key / cargo-mutants name)
    shard: str
    language: str
    file: str  # repository-relative
    line: int
    function: str
    change: str  # one-line description of the mutation
    status: str
    diff: str = ""
    seconds: float | None = None
    note: str = ""


def save_shard_result(shard: Shard, scope: str, mutants: list[Mutant], meta: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{scope}-{shard.name}.json"
    payload = {
        "shard": shard.name,
        "language": shard.language,
        "scope": scope,
        "generated": now(),
        "meta": meta,
        "mutants": [asdict(m) for m in mutants],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    return path


def load_shard_results(scope: str, names=None):
    out = []
    for path in sorted(RESULTS.glob(f"{scope}-*.json")):
        data = json.loads(path.read_text())
        if names is None or data["shard"] in names:
            data["mutants"] = [Mutant(**m) for m in data["mutants"]]
            out.append(data)
    return out


def now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def git(*args, check=True) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, capture_output=True, text=True
    ).stdout


def child_env(**extra) -> dict:
    """Environment for every child run: no secrets, bounded threads.

    Real data stays out of reach because ashare.data refuses any data root under a
    ``data/`` directory while pytest is loaded; the tests bring their own fixtures.
    """
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("TUSHARE") or key in ("VIRTUAL_ENV", "PYTHONPATH"):
            env.pop(key)
    env.setdefault("RAYON_NUM_THREADS", "1")
    env.setdefault("POLARS_MAX_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env
