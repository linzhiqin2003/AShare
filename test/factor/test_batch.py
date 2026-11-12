# ruff: noqa: E402
"""批量求值（P-009）：与逐个 evaluate 逐位相同、分批、缓存、释放。"""

import sys
from pathlib import Path

import panelops

import ashare

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(panelops.__file__).resolve().is_relative_to(ROOT)

import numpy as np
import pytest

import ashare.factor as af
import ashare.factor.engine as engine

sys.path.insert(0, str(ROOT / "bench/e2e"))
from expressions import expressions as bench_expressions  # noqa: E402

FLOATS = (
    "close adj_factor open high low vwap volume amount turnover total_market_cap "
    "pe_ttm pb ps_ttm dividend_yield_ttm"
).split()


class Source:
    """合成面板：缺失、正负无穷、零、整数并列、行业 -1 都有。"""

    def __init__(self, t=160, n=23, seed=7):
        rng = np.random.default_rng(seed)
        self.dates = [f"d{i}" for i in range(t)]
        self.symbols = [f"s{j}" for j in range(n)]
        self.version = "synthetic"
        self.reads = 0
        self.fields = {}
        for k, name in enumerate(FLOATS):
            x = np.exp(rng.normal(0, 0.3, (t, n)).cumsum(0) * 0.05) * (1 + k)
            if k % 3 == 0:
                x = np.round(x)  # 整数并列
            x[rng.random((t, n)) < 0.08] = np.nan
            x[rng.random((t, n)) < 0.01] = np.inf
            x[rng.random((t, n)) < 0.01] = -np.inf
            x[rng.random((t, n)) < 0.02] = 0.0
            self.fields[name] = x
        industry = rng.integers(0, 5, (t, n))
        industry[rng.random((t, n)) < 0.05] = -1
        self.fields["industry"] = industry

    def __getitem__(self, name):
        self.reads += 1
        return self.fields[name]


def batch():
    out = {f.name: f.expression for f in af.default_registry()}
    out.update(bench_expressions())
    out["dup_kmid"] = out["a158_kmid"]
    out["const"] = af.constant(2.5)
    out["leaf"] = af.field("close")
    return out


def same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("threads", [1, 0])
def test_batch_bitwise_matches_single(masked, threads):
    source = Source()
    mask = (
        np.random.default_rng(1).random((len(source.dates), len(source.symbols))) > 0.1
        if masked
        else None
    )
    exprs = batch()
    previous = panelops.get_num_threads()
    panelops.set_num_threads(threads)
    try:
        single = {
            k: af.Evaluator().evaluate(e, source, mask=mask) for k, e in exprs.items()
        }
        ev = af.Evaluator()
        got = list(ev.evaluate_many(exprs, source, mask=mask))
    finally:
        panelops.set_num_threads(previous)
    assert [k for k, _ in got] == list(exprs)
    for name, value in got:
        same(value, single[name])
        assert not value.flags.writeable
    assert ev.last_batch == {"expressions": len(exprs), "cache_hits": 0, "batches": 1}
    # 跨表达式去重：节点数等于并集
    union = set()
    for e in exprs.values():
        stack = [e]
        while stack:
            node = stack.pop()
            if node.digest not in union:
                union.add(node.digest)
                stack.extend(node.args)
    assert len(ev.last_counts) == len(union)


def test_field_leaves_read_once_and_sequence_input():
    source = Source()
    exprs = list(batch().values())
    got = list(af.Evaluator().evaluate_many(exprs, source))
    assert [k for k, _ in got] == list(range(len(exprs)))
    fields = set().union(*(e.fields for e in exprs))
    assert source.reads == len(fields)


@pytest.mark.parametrize("panels", [1, 3, 6, 12])
def test_memory_limit_splits_and_bounds_peak(panels):
    source = Source(t=60, n=7)
    exprs = batch()
    limit = panels * 8 * 60 * 7
    single = {k: af.Evaluator().evaluate(e, source) for k, e in exprs.items()}
    ev = af.Evaluator()
    got = dict(ev.evaluate_many(exprs, source, memory_limit=limit))
    for name in exprs:
        same(got[name], single[name])
    assert ev.last_batch["batches"] > 1
    # 每批模拟峰值不超过上限，单个超限的表达式自成一批
    for group in engine._split(list(exprs.values()), panels):
        assert len(group) == 1 or engine._peak(group) <= panels


def test_peak_model_counts_real_peak():
    """模拟峰值不小于求值器实际同时持有的数组数。"""
    source = Source(t=40, n=5)
    exprs = list(batch().values())
    ev = af.Evaluator()
    for _ in ev.evaluate_many(exprs, source):
        pass
    assert engine._peak(exprs) >= ev.last_peak_arrays


def test_results_not_retained_by_evaluator():
    import gc
    import weakref

    source = Source(t=40, n=5)
    exprs = batch()
    refs = []
    ev = af.Evaluator()
    for name, value in ev.evaluate_many(exprs, source):
        base = value.base if value.base is not None else value
        if name in ("a158_klen", "gtja_014"):
            refs.append(weakref.ref(base))
        del value, base
    gc.collect()
    assert refs and all(r() is None for r in refs)


def test_cache_hits_interleave_in_order(tmp_path):
    source = Source(t=50, n=6)
    exprs = batch()
    names = list(exprs)
    warm = {k: exprs[k] for k in names[::3]}
    ev = af.Evaluator(tmp_path)
    first = dict(ev.evaluate_many(warm, source))
    assert ev.last_batch["cache_hits"] == 0
    got = list(ev.evaluate_many(exprs, source))
    assert [k for k, _ in got] == names
    assert ev.last_batch["cache_hits"] == len(warm)
    for name, value in got:
        same(value, af.Evaluator().evaluate(exprs[name], source))
        if name in first:
            same(value, first[name])
    # 批量写的缓存能被单个 evaluate 命中
    for name in names[1::3]:
        ev.evaluate(exprs[name], source)
        assert ev.cache_hit


def test_validation_is_eager():
    source = Source(t=20, n=4)
    ev = af.Evaluator()
    with pytest.raises(TypeError):
        ev.evaluate_many([af.field("close"), "x"], source)
    with pytest.raises(ValueError):
        ev.evaluate_many([af.field("close")], source, memory_limit=0)
    with pytest.raises(ValueError):
        ev.evaluate_many([af.field("close")], source, universe_id="")
    with pytest.raises(ValueError):
        ev.evaluate_many([af.field("close")], source, mask=np.ones((3, 3), dtype=bool))
    assert list(ev.evaluate_many([], source)) == []


def test_peak_model_exact_small_cases():
    x, y = af.field("close"), af.field("open")
    # x 存活时写 x+1：2 个面板，再加调用方手里的 1 个
    assert engine._peak([x + 1]) == 3
    # x、y 存活时写 x*y
    assert engine._peak([x * y]) == 4
    # x 被第二个根复用，保留到写 x-1 时：x、x-1、调用方手里的 x+1
    assert engine._peak([x + 1, x - 1]) == 3
    assert engine._peak([af.constant(1.0)]) == 1


def test_masked_constant_root_and_dedup_reads():
    source = Source(t=30, n=4)
    mask = np.ones((30, 4), dtype=bool)
    mask[::3, 1] = False
    ((_, value),) = af.Evaluator().evaluate_many([af.constant(2.0)], source, mask=mask)
    assert np.isnan(value[~mask]).all() and (value[mask] == 2.0).all()
    source.reads = 0
    x = af.field("close")
    list(af.Evaluator().evaluate_many([x + 1, x * 2, x - 3], source))
    assert source.reads == 1
