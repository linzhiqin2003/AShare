"""阶段 2：首批因子与代表性表达式批量求值，拆出算子内核、字段准备、调度与缓存。"""

import cProfile
import io
import pstats
import shutil
import time

import numpy as np
import panelops
from common import Phases, digest, measure
from expressions import expressions

import ashare.data
import ashare.factor.engine as engine
from ashare.factor import Evaluator, default_registry, field

CACHE_SUBSET = 10


def open_source(ctx, fields=None):
    data = ashare.data.open(ctx["data"])
    return data.panel(list(fields or data._meta["fields"]))


def instrument(phases):
    """包住求值器调用的 panelops 入口，按算子类别累计内核时间与输出面板数。"""
    original = {}
    for name in panelops.__all__:
        fn = getattr(panelops, name)
        if callable(fn) and (name.startswith(("ts_", "cs_")) or name == "elementwise"):
            original[name] = fn
            kind = (
                "kernel_ts"
                if name.startswith("ts_")
                else ("kernel_cs" if name.startswith("cs_") else "kernel_elementwise")
            )
            setattr(panelops, name, phases.wrap(kind, fn))
    return lambda: [setattr(panelops, k, v) for k, v in original.items()]


def distinct_nodes(exprs):
    """每个表达式内部去重后的节点数，与跨表达式去重后的节点数。"""
    per, union = 0, set()
    for expression in exprs:
        seen, stack = set(), [expression]
        while stack:
            node = stack.pop()
            if node.digest in seen:
                continue
            seen.add(node.digest)
            stack.extend(node.args)
        per += len(seen)
        union |= seen
    return per, len(union)


def run(ctx):
    cases = []
    panelops.set_num_threads(ctx["threads"])
    source = open_source(ctx)
    evaluator = Evaluator()

    library = default_registry()
    stats, values = measure(
        lambda: {
            f.name: digest(evaluator.evaluate(f.expression, source)) for f in library
        },
        ctx["repeat"](2),
    )
    cases.append(
        {
            "case": "library_all",
            "scale": {"factors": len(library), "shape": source.shape},
            **stats,
            "summary": values,
        }
    )

    batch = expressions()
    per_node, union = distinct_nodes(batch.values())
    phases = Phases()
    restore = instrument(phases)
    timings = {}

    def evaluate_batch():
        out = {}
        for name, expression in batch.items():
            start = time.perf_counter()
            value = evaluator.evaluate(expression, source)
            timings[name] = min(timings.get(name, np.inf), time.perf_counter() - start)
            out[name] = digest(value)
        return out

    stats, values = measure(evaluate_batch, ctx["repeat"](2))
    restore()
    breakdown = {
        k: {
            "seconds": v["seconds"] / stats["repeat"],
            "calls": v["calls"] // stats["repeat"],
        }
        for k, v in phases.report().items()
    }
    cases.append(
        {
            "case": "expressions_batch",
            "scale": {
                "expressions": len(batch),
                "nodes_per_expression_sum": per_node,
                "nodes_distinct_across_batch": union,
                "shape": source.shape,
            },
            **stats,
            "summary": values,
            "detail": {
                "kernel_breakdown": breakdown,
                "slowest": dict(sorted(timings.items(), key=lambda kv: -kv[1])[:15]),
                "per_expression": {k: round(v, 4) for k, v in timings.items()},
            },
        }
    )

    leaves = sorted({name for e in batch.values() for name in e.fields})

    def prepare_fields():
        for name in leaves:
            dtype = "int64" if name == "industry" else "float64"
            evaluator.evaluate(field(name, dtype=dtype), source)

    stats, _ = measure(prepare_fields, ctx["repeat"](2))
    cases.append(
        {"case": "field_leaf_prepare", "scale": {"fields": len(leaves)}, **stats}
    )

    subset = dict(list(batch.items())[:CACHE_SUBSET])
    cache_dir = ctx["work"] / "factor-cache"
    cached = Evaluator(cache_dir)

    def clear():
        shutil.rmtree(cache_dir, ignore_errors=True)

    def through_cache():
        hits, out = 0, {}
        for name, expression in subset.items():
            out[name] = digest(cached.evaluate(expression, source))
            hits += cached.cache_hit
        return out, hits

    nocache, _ = measure(
        lambda: [evaluator.evaluate(e, source) for e in subset.values()],
        ctx["repeat"](1),
    )
    miss, (miss_values, miss_hits) = measure(
        through_cache, ctx["repeat"](1), setup=clear
    )
    hit, (hit_values, hits) = measure(through_cache, ctx["repeat"](2))
    sample = next(cache_dir.glob("*.npy"))
    checksum, _ = measure(lambda: engine._checksum(sample), ctx["repeat"](3))
    for name, stats, summary in (
        ("cache_none_subset", nocache, None),
        ("cache_miss_write_subset", miss, {"hits": miss_hits, **miss_values}),
        ("cache_hit_subset", hit, {"hits": hits, **hit_values}),
        ("cache_checksum_one_file", checksum, None),
    ):
        cases.append(
            {
                "case": name,
                "scale": {
                    "expressions": len(subset),
                    "file_bytes": sample.stat().st_size,
                },
                **stats,
                "summary": summary,
            }
        )
    clear()

    profiler = cProfile.Profile()
    profiler.enable()
    for expression in subset.values():
        evaluator.evaluate(expression, source)
    profiler.disable()
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).sort_stats("tottime").print_stats(12)
    (ctx["work"] / "factors_cprofile.txt").write_text(text.getvalue())
    top = pstats.Stats(profiler).sort_stats("tottime")
    rows = sorted(top.stats.items(), key=lambda kv: -kv[1][2])[:12]
    cases.append(
        {
            "case": "cprofile_subset",
            "scale": {"expressions": len(subset)},
            "repeat": 1,
            "detail": {
                f"{k[0].rsplit('/', 1)[-1]}:{k[2]}": round(v[2], 4) for k, v in rows
            },
        }
    )
    return cases
