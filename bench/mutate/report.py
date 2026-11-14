"""Equivalent-mutant registry, score aggregation and the two outputs.

``.cache/testing/mutation.json`` (fixed format, read by ``make test-report``)::

    {"generated": ISO, "scope": "full|diff",
     "modules": [{"module", "language", "killed", "survived", "timeout", "equivalent", "score"}],
     "total": {same counters + score}}

``killed`` excludes timeouts; ``score = (killed + timeout) / (killed + timeout + survived)``.
``survived`` includes mutants that no selected test reached. Equivalent mutants (registered
in bench/mutate/equivalent.toml) and unviable / skipped / errored ones are outside the
denominator; the last three are counted in ``mutation_details.json``. A module with nothing
in the denominator scores 1.0 (nothing escaped).
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections import Counter

from bench.mutate.common import (
    COUNTED_SURVIVED,
    DETAILS,
    EQUIVALENT,
    HERE,
    KILLED,
    OUTPUT,
    SURVIVORS,
    TESTING,
    TIMEOUT,
    Mutant,
    log,
    now,
)

REGISTRY = HERE / "equivalent.toml"


def change_key(m: Mutant) -> str:
    """Identity of a mutation that survives renumbering and line shifts."""
    text = f"{m.file}|{m.function}|{m.change}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_registry():
    if not REGISTRY.exists():
        return []
    return tomllib.loads(REGISTRY.read_text()).get("mutant", [])


def apply_equivalents(mutants: list[Mutant]):
    """Mark registered equivalents. Returns entries that matched nothing (stale or not run)."""
    entries = load_registry()
    by_key = {}
    for e in entries:
        by_key.setdefault((e["file"], e["function"], e["change"]), e)
    used = set()
    for m in mutants:
        e = by_key.get((m.file, m.function, m.change))
        if e is None:
            continue
        used.add((e["file"], e["function"], e["change"]))
        if m.status in COUNTED_SURVIVED:
            m.status = EQUIVALENT
            m.note = e.get("reason", "")
        else:  # registered as equivalent but now killed: the registration is wrong or stale
            m.note = "registered as equivalent but " + m.status
    files = {m.file for m in mutants}
    return [
        e
        for e in entries
        if e["file"] in files and (e["file"], e["function"], e["change"]) not in used
    ]


def counts(mutants):
    c = Counter(m.status for m in mutants)
    killed = c[KILLED]
    timeout = c[TIMEOUT]
    survived = sum(c[s] for s in COUNTED_SURVIVED)
    denom = killed + timeout + survived
    return {
        "killed": killed,
        "survived": survived,
        "timeout": timeout,
        "equivalent": c[EQUIVALENT],
        "score": round((killed + timeout) / denom, 4) if denom else 1.0,
    }


def write(results, scope: str):
    """results: [{"shard", "language", "generated", "meta", "mutants": [Mutant]}]."""
    stale = []
    modules = []
    every = []
    details = []
    for r in results:
        stale += apply_equivalents(r["mutants"])
        every += r["mutants"]
        modules.append(
            {"module": r["shard"], "language": r["language"], **counts(r["mutants"])}
        )
        c = Counter(m.status for m in r["mutants"])
        details.append(
            {
                "module": r["shard"],
                "generated": r["generated"],
                "statuses": dict(sorted(c.items())),
                "meta": r["meta"],
            }
        )
    TESTING.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": now(),
        "scope": scope,
        "modules": modules,
        "total": counts(every),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    DETAILS.write_text(
        json.dumps(
            {
                "generated": payload["generated"],
                "scope": scope,
                "modules": details,
                "stale_equivalents": stale,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )
    write_survivors(every, scope, stale)
    log(f"wrote {OUTPUT.relative_to(OUTPUT.parents[2])}: total {payload['total']}")
    return payload


def write_survivors(mutants, scope, stale):
    alive = [m for m in mutants if m.status in COUNTED_SURVIVED]
    alive.sort(key=lambda m: (m.shard, m.file, m.line, m.change))
    lines = [
        f"# 存活的变异（{scope}，{now()}）",
        "",
        f"共 {len(alive)} 个。逐个看：补测试，或确属等价就登记到 `bench/mutate/equivalent.toml`。",
        "`no tests` 表示所选测试没有一个经过这段代码。",
        "",
    ]
    if stale:
        lines += ["## 登记了但这次没对上的等价变异", ""]
        lines += [f"- `{e['file']}` `{e['function']}`: {e['change']}" for e in stale]
        lines.append("")
    shard = None
    for m in alive:
        if m.shard != shard:
            shard = m.shard
            if lines[-1]:
                lines.append("")
            lines += [
                f"## {shard}",
                "",
                "| 文件:行 | 函数 | 变异 | 状态 | id |",
                "|---|---|---|---|---|",
            ]
        change = m.change.replace("|", "\\|").replace("\n", " ")
        status = "no tests" if m.status != "survived" else "survived"
        lines.append(
            f"| `{m.file}:{m.line}` | `{m.function}` | `{change}` | {status} | `{m.id}` |"
        )
    SURVIVORS.write_text("\n".join(lines) + "\n")
