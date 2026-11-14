"""Unified mutation-testing entry point (make mutate-all / mutate / mutate-diff / mutate-report).

    python -m bench.mutate.run full   [--module M] [--language python|rust] [--jobs 2]
    python -m bench.mutate.run diff   --base origin/main [--jobs 2]
    python -m bench.mutate.run report [--scope full|diff]
    python -m bench.mutate.run estimate
    python -m bench.mutate.run controls

Run from the repository root with the project venv (``make dev`` first). See
docs/TESTING_MUTATION.md for the scoring rules and the layout of the outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time

from bench.mutate import python as py
from bench.mutate import report
from bench.mutate import rust as rs
from bench.mutate.common import (
    RESULTS,
    ROOT,
    SKIPPED,
    WORK,
    be_nice,
    git,
    load_shard_results,
    load_shards,
    log,
    save_shard_result,
    select,
)


def slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")


def run_one(shard, scope, narrow, args, **kw):
    if shard.language == "python":
        mutants, meta = py.run_shard(
            shard,
            scope=scope,
            narrow=narrow,
            jobs=args.jobs,
            timeout=args.timeout,
            resume=getattr(args, "resume", False),
            **kw,
        )
    else:
        mutants, meta = rs.run_shard(
            shard,
            scope=scope,
            narrow=narrow,
            jobs=args.jobs,
            timeout=args.timeout,
            regex=args.re,
            part=args.part,
            xdist=shard_xdist(shard, args),
            **kw,
        )
    if narrow:
        meta["narrowed_to"] = narrow
    suffix = slug(narrow) if narrow else ""
    if getattr(args, "part", None):
        meta["part"] = args.part
        suffix += "part" + args.part.replace("/", "of")
    if getattr(args, "re", None):
        meta["re"] = args.re
        suffix += "re" + hashlib.sha256(args.re.encode()).hexdigest()[:8]
    tag = _Tagged(shard, f"{shard.name}@{suffix}") if suffix else shard
    save_shard_result(tag, scope, mutants, meta)
    return mutants


class _Tagged:
    """Shard view whose saved result file name carries the narrowed path."""

    def __init__(self, shard, name):
        self._shard, self.name = shard, name
        self.language = shard.language

    def __getattr__(self, item):
        return getattr(self._shard, item)


def shard_xdist(shard, args):
    return max(1, min(args.xdist, args.jobs))


def merged(scope):
    """Saved results of a scope, one entry per shard; newer files win per mutant id."""
    by_shard = {}
    for data in sorted(load_shard_results(scope), key=lambda d: d["generated"]):
        name = data["shard"].split("@")[0]
        entry = by_shard.setdefault(
            name,
            {
                "shard": name,
                "language": data["language"],
                "generated": data["generated"],
                "meta": {},
                "by_id": {},
            },
        )
        entry["generated"] = data["generated"]
        entry["meta"][data["shard"]] = data["meta"]
        entry.setdefault("narrowed", set()).add(data["meta"].get("narrowed_to"))
        entry.setdefault("parts", set()).add(data["meta"].get("part"))
        for m in data["mutants"]:
            entry["by_id"][m.id] = m
    out = []
    for entry in by_shard.values():
        entry["mutants"] = list(entry.pop("by_id").values())
        narrowed = entry.pop("narrowed")
        if None not in narrowed:  # only parts of the shard were run: say which
            entry["shard"] += ":" + ",".join(sorted(narrowed))
        parts = entry.pop("parts")
        if None not in parts:
            total = int(next(iter(parts)).split("/")[1])
            if len(parts) < total:
                done = ",".join(sorted(p.split("/")[0] for p in parts))
                entry["shard"] += f" (parts {done} of {total})"
        unchecked = sum(m.status == SKIPPED for m in entry["mutants"])
        if unchecked:  # interrupted run: continue with --resume
            total = len(entry["mutants"])
            entry["shard"] += f" (checked {total - unchecked} of {total})"
        out.append(entry)
    return sorted(out, key=lambda e: (e["language"], e["shard"]))


def cmd_full(args):
    shards = load_shards()
    chosen = select(shards, args.module, args.language)
    started = time.monotonic()
    for shard, narrow in chosen:
        run_one(shard, "full", narrow, args)
    log(f"full run of {len(chosen)} shard(s) took {time.monotonic() - started:.0f}s")
    report.write(merged("full"), "full")


def changed_lines(base: str, path: str):
    """Changed line ranges of a file relative to the merge base (working tree included)."""
    out = git("diff", "--merge-base", base, "-U0", "--", path)
    ranges = []
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", out, re.M):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        if count:
            ranges.append((start, start + count - 1))
    return ranges


def cmd_diff(args):
    shards = load_shards()
    files = [
        f
        for f in git(
            "diff", "--merge-base", args.base, "--name-only", "--diff-filter=AMR"
        ).split()
        if (ROOT / f).exists()
    ]
    log(f"{len(files)} changed file(s) against {args.base}")
    for old in RESULTS.glob("diff-*.json"):
        old.unlink()
    ran = 0
    for shard in shards.values():
        if shard.language == "python":
            mine = [f for f in files if f.endswith(".py") and shard.owns(f)]
            if not mine:
                continue
            names = []
            for f in mine:
                names += py.changed_function_globs(
                    shard, f, changed_lines(args.base, f)
                )
            if not names:
                log(f"{shard.name}: changes touch no function bodies, skipped")
                continue
            run_one(shard, "diff", None, args, files=mine, names=names)
        else:
            mine = [f for f in files if f in shard.files]
            if not mine:
                continue
            patch = WORK / "rs" / f"{shard.name}.diff"
            patch.parent.mkdir(parents=True, exist_ok=True)
            patch.write_text(git("diff", "--merge-base", args.base, "--", *mine))
            run_one(shard, "diff", None, args, in_diff=patch)
        ran += 1
    if not ran:
        log("no mutable code changed")
    report.write(merged("diff"), "diff")


def cmd_report(args):
    report.write(merged(args.scope), args.scope)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("full", "diff"):
        s = sub.add_parser(name)
        s.add_argument(
            "--jobs",
            type=int,
            default=2,
            help="parallel mutants (all tools); keep at 2 on a shared machine",
        )
        s.add_argument(
            "--timeout", type=float, default=3600, help="wall limit per shard, seconds"
        )
        s.add_argument(
            "--xdist",
            type=int,
            default=1,
            help="pytest-xdist workers per Rust mutant (taken out of --jobs)",
        )
        s.add_argument("--re", help="Rust only: cargo-mutants --re filter")
        s.add_argument(
            "--part",
            help="Rust only: K/N, run the K-th of N slices (K from 0) to keep runs short",
        )
        if name == "full":
            s.add_argument("--module", help="shard name(s) or path(s), comma separated")
            s.add_argument("--language", choices=["python", "rust"])
            s.add_argument(
                "--resume",
                action="store_true",
                help="Python: continue an interrupted shard (keeps verdicts already reached)",
            )
        else:
            s.add_argument("--base", default="origin/main")
    s = sub.add_parser("report")
    s.add_argument("--scope", default="full", choices=["full", "diff"])
    sub.add_parser("estimate")
    c = sub.add_parser("controls")
    c.add_argument("--language", choices=["python", "rust"])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    be_nice()
    if args.command in ("full", "diff") and not 1 <= args.jobs <= 4:
        sys.exit("--jobs must be between 1 and 4")
    if args.command == "full":
        cmd_full(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "estimate":
        from bench.mutate.estimate import main as estimate

        estimate()
    elif args.command == "controls":
        from bench.mutate.controls import main as controls

        controls(args.language)


if __name__ == "__main__":
    main()
