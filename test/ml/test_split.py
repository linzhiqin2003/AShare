"""切分：间隔由标签推出、不能减小；任意参数下训练样本的标签结束日 < 后一段首日。"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("lightgbm", reason="需要可选依赖组 ml")

import ashare.ml as ml  # noqa: E402

from .synthetic import dates  # noqa: E402

D = dates(2000, start=2010)


def _naive_ok(sched, span):
    """直白检查：逐个训练/验证日，标签结束日严格早于下一段第一天。"""
    for f in sched.folds:
        segs = [f.train] + ([f.valid] if f.valid else []) + [f.test]
        for (a0, b0), (a1, _b1) in zip(segs, segs[1:], strict=False):
            for s in range(a0, b0):
                if s + span >= a1:
                    return False
            if b0 > a1:
                return False
    return True


@settings(max_examples=60, deadline=None)
@given(
    h=st.integers(1, 25),
    lag=st.integers(1, 3),
    extra=st.integers(0, 10),
    retrain=st.sampled_from(["month", "quarter", "halfyear", "year", 17, 63]),
    window=st.sampled_from(["rolling", "expanding"]),
    valid=st.sampled_from([0, 30, 120]),
    start=st.integers(600, 1500),
)
def test_gaps_property(h, lag, extra, retrain, window, valid, start):
    span = h + lag
    s = ml.schedule(
        D,
        span,
        start=start,
        retrain=retrain,
        window=window,
        train_days=400,
        valid_days=valid,
        gap=span + extra,
        min_train_days=100,
    )
    assert len(s) > 0
    assert _naive_ok(s, span)
    assert s.gap == span + extra
    tests = [f.test for f in s.folds]
    for (_a, b), (c, _d) in zip(tests, tests[1:], strict=False):
        assert b == c  # 预测期首尾相接，不重不漏
    assert tests[0][0] == start and tests[-1][1] == len(D)
    if window == "rolling":
        assert all(f.train[1] - f.train[0] == 400 for f in s.folds)
    else:
        assert all(f.train[0] == 0 for f in s.folds)


def test_gap_cannot_shrink():
    with pytest.raises(ValueError, match="不能小于标签跨度"):
        ml.schedule(D, 6, start=800, gap=5)
    s = ml.schedule(D, 6, start=800)
    assert s.gap == 6
    for f in s.folds:
        assert f.valid[0] - (f.train[1] - 1) == 7
        assert f.test[0] - (f.valid[1] - 1) == 7


def test_check_schedule_catches_overlap():
    s = ml.schedule(D, 6, start=800)
    assert ml.check_schedule(s)
    with pytest.raises(ValueError):
        ml.check_schedule(s, span=7)  # 按更长的标签看，这个间隔不够


def test_quarter_boundaries_and_years():
    s = ml.schedule(D, 6, start=800, retrain="quarter", train_years=3, valid_days=60)
    for f in s.folds:
        first = str(D[f.test[0]])
        assert (
            first[4:6] in ("01", "04", "07", "10") and first[6:] == "01" or f.index == 0
        )
        train_first, train_last = str(D[f.train[0]]), str(D[f.train[1] - 1])
        assert int(train_last[:4]) - int(train_first[:4]) in (2, 3)


def test_describe_lists_folds_and_skipped():
    s = ml.schedule(D, 6, start=300, retrain="year", train_days=400)
    text = s.describe()
    assert "间隔 6 个交易日" in text and "跳过" in text
    assert text.count("\n") >= len(s) + 1
    assert len(s.table()) == len(s)
