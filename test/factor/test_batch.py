# ruff: noqa: E402
"""批量求值（P-009）：与逐个 evaluate 逐位相同、分批、缓存、释放。"""

import os
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
        market = rng.normal(0, 1, t)
        market[::7] = np.nan
        market[3] = np.inf
        self.fields["mkt"] = market
        self.fields["flag"] = rng.random((t, n)) < 0.5

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


@pytest.mark.parametrize("masked", [False, True])
def test_unnormalized_leaves_match_normalized(masked, monkeypatch):
    """P-010：作为算子输入的字段不再预先改写非有限值，结果与改写过的逐位相同。"""
    source = Source(t=120, n=17, seed=11)
    mask = np.random.default_rng(5).random((120, 17)) > 0.15 if masked else None
    exprs = batch()
    # 再加几个专门让 ±inf 进入各类算子的表达式
    x, y = af.field("close"), af.field("open")
    exprs |= {
        "fill": af.fill_nan(x, y),
        "isnan": af.is_nan(x),
        "where": af.where(af.gt(x, 1.0), y, x),
        "delay": af.call("ts_delay", x, periods=1),
        "ffill": af.call("ts_ffill", x, limit=3),
        "corr": af.call("ts_corr", x, y, window=5),
        "sumif": af.call("ts_sumif", x, af.gt(y, 2.0), window=5),
        "rank": af.call("cs_rank", x),
        "neutral": af.neutralize(
            x, y, groups=af.field("industry", dtype="int64"), weights=y
        ),
        # 暴露与权重用不同字段，暴露里的 ±inf 不会被权重的有限性检查顺带挡掉
        "neutral_split": af.neutralize(
            x,
            af.field("high"),
            groups=af.field("industry", dtype="int64"),
            weights=af.field("volume"),
        ),
        "neutral_plain": af.neutralize(x, af.field("low")),
        "zscore": af.call("cs_zscore", x),
        "demean": af.call("cs_demean", x),
        "scale": af.call("cs_scale", x),
        "market": af.field("mkt", broadcast=True) * x,
        "market_ts": af.call("ts_mean", af.field("mkt", broadcast=True), window=3),
        "flag": af.call("ts_sum", af.field("flag"), window=4) - x,
    }
    plain = dict(af.Evaluator().evaluate_many(exprs, source, mask=mask))
    single = {
        k: af.Evaluator().evaluate(e, source, mask=mask) for k, e in exprs.items()
    }
    monkeypatch.setitem(panelops.api.BACKENDS, "_force_normalize", "rust")
    normalized = dict(af.Evaluator().evaluate_many(exprs, source, mask=mask))
    for name in exprs:
        same(plain[name], normalized[name])
        same(single[name], normalized[name])
    # 字段直接作为结果时仍改写成 NaN
    assert np.isnan(plain["leaf"][~np.isfinite(source.fields["close"])]).all()


def _greedy_split(roots, limit):
    """拆批的定义：逐个加入，第一次超限就另起一批（P-009 最初的实现）。"""
    batches, current = [], []
    for root in roots:
        if current and engine._peak([*current, root]) > limit:
            batches.append(current)
            current = []
        current.append(root)
    if current:
        batches.append(current)
    return batches


def test_split_matches_greedy_definition():
    exprs = list(batch().values())
    rng = np.random.default_rng(0)
    for trial in range(60):
        order = [exprs[i] for i in rng.integers(0, len(exprs), rng.integers(1, 60))]
        limit = int(rng.integers(1, 40)) if trial % 4 else 10**6
        got = engine._split(order, limit)
        expected = _greedy_split(order, limit)
        assert [[r.digest for r in b] for b in got] == [
            [r.digest for r in b] for b in expected
        ]


def test_split_simulations_grow_logarithmically(monkeypatch):
    exprs = list(batch().values()) * 20
    calls = []
    real = engine._peak
    monkeypatch.setattr(engine, "_peak", lambda roots: calls.append(1) or real(roots))
    assert len(engine._split(exprs, 10**6)) == 1
    assert len(calls) <= 2 * int(np.log2(len(exprs))) + 2


def test_default_memory_limit_is_twelfth_of_ram():
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    assert engine.default_memory_limit() == total // 12


def test_cache_hit_flag_and_duplicate_writes(tmp_path, monkeypatch):
    source = Source(t=30, n=5)
    x = af.field("close")
    a, b = af.call("ts_mean", x, window=3), x * 2
    ev = af.Evaluator(tmp_path)
    writes = []
    real = af.Evaluator._store
    monkeypatch.setattr(
        af.Evaluator,
        "_store",
        staticmethod(lambda p, r: writes.append(p) or real(p, r)),
    )
    got = list(ev.evaluate_many([a, b, a, a], source))
    assert [k for k, _ in got] == [0, 1, 2, 3] and len(writes) == 2
    same(got[0][1], got[2][1])
    assert not ev.cache_hit
    list(ev.evaluate_many([a, b, a], source))
    assert ev.cache_hit and ev.last_batch["cache_hits"] == 3 and len(writes) == 2
    list(ev.evaluate_many([a, x + 5], source))
    assert not ev.cache_hit
    list(ev.evaluate_many([], source))
    assert not ev.cache_hit
