"""搜索空间：量纲规则、规范形与去重键、三种生成方式的确定性与合法性。"""

import itertools

import numpy as np
import pytest

from ashare.factor import call, field
from ashare.factor.mining import (
    Enumerate,
    Evolution,
    Random,
    Space,
    Unit,
    canonical,
    dedup_key,
)
from ashare.factor.mining import strategies as st
from ashare.factor.mining.space import Invalid

close, open_, volume, amount = (
    field("close"),
    field("open"),
    field("volume"),
    field("amount"),
)


def test_units():
    sp = Space.default()
    assert sp.unit(close / open_) == Unit()
    assert sp.unit(call("ts_corr", close, volume, window=10)) == Unit()
    with pytest.raises(Invalid, match="量纲不同"):
        sp.unit(call("cs_rank", canonical(close + volume)))
    # 价格 × 成交量 = 成交额量纲，与成交额相除无量纲
    assert sp.unit(amount / canonical(close * volume)) == Unit()
    # 同底对数相减无量纲；对数相加被拒
    assert sp.unit(call("log", close) - call("log", open_)) == Unit()
    with pytest.raises(Invalid):
        sp.unit(canonical(call("log", close) + call("log", open_)))
    assert sp.unit(call("ts_delta", call("log", close), periods=5)) == Unit()
    # 根必须无量纲：价格水平本身不是候选
    with pytest.raises(Invalid, match="无量纲"):
        sp.unit(call("ts_mean", close, window=5))
    assert Space.default(require_dimensionless=False).unit(
        call("ts_mean", close, window=5)
    ) == Unit(1, 0)


def test_space_constraints():
    sp = Space.default(max_depth=2, max_nodes=4)
    deep = call("cs_rank", call("ts_mean", close / open_, window=5))
    with pytest.raises(Invalid, match="深度"):
        sp.unit(deep)
    with pytest.raises(Invalid, match="参数"):
        sp.unit(call("ts_mean", close / open_, window=7))
    with pytest.raises(Invalid, match="冗余"):
        sp.unit(call("cs_rank", call("cs_rank", close / open_)))
    with pytest.raises(Invalid, match="相同"):
        sp.unit(close - close)
    root = Space.default(cs_positions="root")
    assert root.valid(call("cs_rank", close / open_))
    assert not root.valid(call("ts_mean", call("cs_rank", close / open_), window=5))
    with pytest.raises(ValueError, match="CATALOG"):
        Space(ops={"ts_kurt": {"window": (5,)}})
    with pytest.raises(ValueError, match="参数"):
        Space(ops={"ts_mean": {}})


def test_canonical_and_dedup_key():
    a = close / open_
    b = call("ts_mean", volume, window=5) / call("ts_mean", volume, window=20)
    assert canonical(a + b).digest == canonical(b + a).digest
    assert canonical(a * b) == canonical(b * a)
    assert dedup_key(call("cs_rank", a)) == dedup_key(a)
    assert dedup_key(call("cs_zscore", a)) == dedup_key(a)
    assert dedup_key(call("negate", a)) != dedup_key(
        a
    )  # 只做多：方向不同就是不同的候选
    assert dedup_key(call("ts_mean", call("cs_rank", a), window=5)) != dedup_key(a)


def _small():
    return Space(
        fields={"close": "price", "open": "price", "volume": "volume"},
        ops={
            "divide": {},
            "multiply": {},
            "ts_mean": {"window": (5, 10)},
            "cs_rank": {},
            "negate": {},
        },
        max_depth=2,
        max_nodes=5,
    )


def test_enumerate_is_complete_and_valid():
    sp = _small()
    out = list(st.enumerate_space(sp))
    keys = [dedup_key(e) for e in out]
    assert len(keys) == len(set(keys))
    assert all(sp.valid(e) for e in out)
    texts = {str(e) for e in out}
    assert "(close / open)" in texts or "(open / close)" in texts
    assert out == list(st.enumerate_space(sp))
    # 穷举覆盖：随机采到的每个候选都在穷举里（去重键意义下）
    rng = np.random.default_rng(0)
    for _ in range(200):
        e = st.random_candidate(sp, rng)
        assert dedup_key(e) in set(keys)


def _collect(strategy, space):
    seen = []

    def evaluate(batch):
        seen.extend(str(e) for e in batch)
        return [float(len(str(e)) % 7) for e in batch]

    strategy.run(space, evaluate)
    return seen


def test_strategies_deterministic():
    sp = Space.default()
    for s in (Random(n=40, seed=3, batch=7), Evolution(population=12, generations=3, seed=5),
              Enumerate(limit=50, batch=9)):  # fmt: skip
        a, b = _collect(s, sp), _collect(s, sp)
        assert a == b and len(a) > 0
    assert _collect(Random(n=40, seed=3), sp) != _collect(Random(n=40, seed=4), sp)
    assert len(_collect(Random(n=40, seed=3), sp)) == 40


def test_evolution_uses_fitness_and_stays_valid():
    sp = Space.default()
    got = []

    def evaluate(batch):
        got.append(list(batch))
        # 偏好含 volume 的表达式
        return [1.0 if "volume" in str(e) else 0.0 for e in batch]

    Evolution(population=20, generations=4, seed=1).run(sp, evaluate)
    assert len(got) >= 3
    assert all(sp.valid(e) for b in got for e in b)
    share = [np.mean(["volume" in str(e) for e in b]) for b in got]
    assert share[-1] > share[0]
    keys = [dedup_key(e) for b in got for e in b]
    assert len(keys) == len(set(keys))  # 同一候选不重复提出


def test_mutation_and_crossover_valid():
    sp = Space.default()
    rng = np.random.default_rng(9)
    pool = [st.random_candidate(sp, rng) for _ in range(30)]
    for a, b in itertools.pairwise(pool):
        m = st.mutate(sp, rng, a)
        c = st.crossover(sp, rng, a, b)
        for x in (m, c):
            assert x is None or sp.valid(x)
