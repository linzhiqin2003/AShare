"""整代 GPU 粗筛（`Screener`）：共享子树、分批、确定性、交给 CPU 的判定、t 值误差门禁。

来源：docs/GPU_PROBE.md 第六节「精确性」「性能」与本模块的设计（字段只上传一次、共享子树、
按显存分批、算不了的整棵表达式交给 CPU）。没装 MLX 或没有 GPU 时整个文件 skip。
"""

import math

import gpu_cases as gc
import numpy as np
import pytest

from ashare.factor.evaluate import FactorEvaluator
from ashare.factor.expression import call, constant, field
from ashare.factor.mining import Truncated
from ashare.factor.mining.gpu import GPU_SCREEN_T_MARGIN, available
from ashare.factor.mining.strategies import random_candidate

OK, WHY = available()
pytestmark = pytest.mark.skipif(not OK, reason=f"GPU 不可用：{WHY}")


@pytest.fixture(scope="module")
def setup():
    from ashare.factor.mining.gpu import Screener

    panel = gc.small_panel()
    a, v, h = gc.split_of(panel).rows(panel.dates)
    view = Truncated(panel, h)
    fe = FactorEvaluator(view, segments={"train": (a, v), "valid": (v, h)})
    sh = fe.screen_horizon
    sc = Screener(view, fe.universe, fe.returns[sh])
    return view, fe, sc


def population(n=120, seed=11):
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        e = random_candidate(gc.SPACE, rng)
        if e.digest not in seen:
            seen.add(e.digest)
            out.append(e)
    return out


def _t(fe, ic):
    t = fe.screen_metrics(np.asarray(ic, np.float64))["train"]["ic"]["t"]
    return t if t is not None and math.isfinite(t) else None


def test_population_is_fully_supported(setup):
    """测试种群（覆盖默认搜索空间全部算子）全部能在 GPU 上算，误差测试才有意义。"""
    _, _, sc = setup
    pop = population()
    assert all(sc.unsupported(e) is None for e in pop)
    ops = {n.op for e in pop for n in _nodes(e)}
    assert {"ts_corr", "ts_rank", "ts_zscore", "ts_std", "cs_rank", "divide"} <= ops


def _nodes(e):
    out, stack = [], [e]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.args)
    return out


def test_daily_ic_matches_cpu_and_t_error_below_margin(setup):
    """边际门禁：GPU t 值与 CPU float64 的最大误差不到边际的 1/1.2（docs/GPU_PROBE.md 第四节）。
    日 IC 的缺失位置逐日一致（t 不是有限数的两边都不是，GPU 这边会送复算）。"""
    view, fe, sc = setup
    from ashare.factor.engine import Evaluator

    pop = population(300)
    outs = sc.daily_ic(pop)
    ev = Evaluator()
    worst, compared, routed = 0.0, 0, 0
    for e, oc in zip(pop, outs, strict=True):
        if (
            oc.ic is None
        ):  # 成交额 × 成交额 × 价格 ≈ 1e18、ts_corr(low, -low) 这类：交给 CPU
            assert oc.reason in ("float32 range", "rounding ties"), oc.reason
            routed += 1
            continue
        _, ic = fe.screen(ev.evaluate(e, view))
        tc, tg = _t(fe, ic), _t(fe, oc.ic)
        if tc is None or tg is None:
            assert tc is None and tg is None, (str(e), tc, tg)
            continue
        assert np.array_equal(np.isfinite(ic), np.isfinite(oc.ic)), str(e)
        # 日值本身不设逐日界：float32 下相邻两只的因子值可能对调或变成并列（时序排名、
        # 极值），小截面上一次对调就能让当天的 Spearman 变 6·2·(n−1)/(n(n²−1))（n = 41 时
        # 约 7e-3）。算子级 Rank IC（输入相同）的 1e-3 见差分测试；这里守的是 t 值误差。
        worst = max(worst, abs(tc - tg))
        compared += 1
    assert compared >= 200 and routed <= 20
    assert worst * 1.2 <= GPU_SCREEN_T_MARGIN, worst


def test_rounding_ties_go_to_cpu(setup):
    """数学上恒为常数（相关恒为 ±1）的表达式，排序由舍入决定，float32 的 t 值代表不了 float64
    的（小面板上差 1.24），整棵交给 CPU。"""
    _, _, sc = setup
    low, vol = field("low"), field("volume")
    exprs = [call("ts_corr", low, call("negate", low), window=10),
             call("ts_corr", call("abs", vol), call("negate", vol), window=10)]  # fmt: skip
    for oc in sc.daily_ic(exprs):
        assert oc.ic is None and oc.reason == "rounding ties", oc
    (ok,) = sc.daily_ic([call("ts_zscore", low, window=10)])
    assert ok.ic is not None and ok.noise < 0.01


def test_shared_batch_equals_single_and_split_batches(setup):
    """共享子树、按显存分批都不改变结果：整批、逐个、强制每批一个根，逐位相同。"""
    view, fe, _ = setup
    from ashare.factor.mining.gpu import Screener

    pop = population(40, seed=3)
    pop = pop + pop[:5]  # 重复的根
    whole = Screener(view, fe.universe, fe.returns[fe.screen_horizon])
    batched = whole.daily_ic(pop)
    assert whole.counts["batches"] == 1
    tiny = Screener(view, fe.universe, fe.returns[fe.screen_horizon], memory_limit=1)
    split = tiny.daily_ic(pop)
    assert tiny.counts["batches"] == len(pop)
    single = [whole.daily_ic([e])[0] for e in pop]
    for a, b, c in zip(batched, split, single, strict=True):
        assert np.array_equal(a.ic, b.ic, equal_nan=True)
        assert np.array_equal(a.ic, c.ic, equal_nan=True)
    assert whole.counts["fields"] <= len(gc.SPACE.fields)  # 每个字段只上传一次


def test_repeat_is_bitwise_identical(setup):
    """确定性：同一输入在同一设备上重复跑（新建 Screener、重新编译内核）逐位相同。"""
    view, fe, sc = setup
    from ashare.factor.mining.gpu import Screener

    pop = population(60, seed=5)
    first = sc.daily_ic(pop)
    for _ in range(2):
        again = Screener(view, fe.universe, fe.returns[fe.screen_horizon]).daily_ic(pop)
        for a, b in zip(first, again, strict=True):
            assert a.reason == b.reason and a.noise == b.noise
            if a.ic is not None:
                assert a.ic.tobytes() == b.ic.tobytes()


def test_unsupported_and_unsafe_go_to_cpu(setup):
    _, _, sc = setup
    close, amount = field("close"), field("amount")
    cases = {
        "median": call("ts_median", close, window=5),
        "min_periods": call("ts_mean", close, window=5, min_periods=3),
        "missing field": call("ts_mean", field("no_such_field"), window=5),
        "int field": call("cs_rank", field("industry", dtype="int64")),
        "constant root": constant(1.0),
        "constant operands": call("negate", constant(2.0)),
    }
    for name, e in cases.items():
        (oc,) = sc.daily_ic([e])
        assert oc.ic is None and oc.reason.startswith("unsupported"), name
    # 成交额的四次方远超 float32 的安全范围：整棵交给 CPU
    big = call("cs_rank", amount * amount * amount * amount / (close * close))
    (oc,) = sc.daily_ic([big])
    assert oc.ic is None and oc.reason == "float32 range"
    # 同一批里能算的照常给出结果
    ok = call("ts_mean", close, window=5) / close
    outs = sc.daily_ic([big, ok])
    assert outs[0].ic is None and outs[1].ic is not None


def test_rows_prefix_matches_full(setup):
    """rows=k 只在前 k 行上算（mine 里取验证段起点），前 k 行的日 IC 与整段逐位相同，其后为 NaN。"""
    view, fe, sc = setup
    from ashare.factor.mining.gpu import Screener

    pop = population(30, seed=9)
    k = 250
    part = Screener(view, fe.universe, fe.returns[fe.screen_horizon], rows=k).daily_ic(
        pop
    )
    full = sc.daily_ic(pop)
    for a, b in zip(part, full, strict=True):
        if b.ic is None:
            continue
        assert a.ic is not None and len(a.ic) == len(b.ic)
        assert a.ic[:k].tobytes() == b.ic[:k].tobytes()
        assert np.isnan(a.ic[k:]).all()
