# ruff: noqa: E402
"""rank_ic_many（P-011）：远期收益只排一次，结果与逐个 rank_ic 逐位相同。"""

from pathlib import Path

import panelops

ROOT = Path(__file__).resolve().parents[2]
assert Path(panelops.__file__).resolve().is_relative_to(ROOT)

import numpy as np
import panelops as po
import pytest


def panels(kind, t=90, n=33, seed=0):
    r = np.random.default_rng(seed)
    if kind == "ints":
        x = r.integers(-2, 3, (t, n)).astype(float)
    elif kind == "zeros":
        x = np.where(r.random((t, n)) < 0.5, 0.0, -0.0) + r.integers(0, 2, (t, n))
    else:
        x = r.normal(size=(t, n))
    x[r.random((t, n)) < 0.1] = np.nan
    x[r.random((t, n)) < 0.02] = np.inf
    x[r.random((t, n)) < 0.02] = -np.inf
    x[3] = np.nan  # 整行缺失
    x[4, 1:] = np.nan  # 只剩一个
    x[5] = 1.0  # 常数行
    return np.ascontiguousarray(x)


@pytest.mark.parametrize("threads", [1, 0])
@pytest.mark.parametrize("kind", ["normal", "ints", "zeros"])
@pytest.mark.parametrize("masked", [False, True])
def test_matches_rank_ic_bitwise(threads, kind, masked):
    y = panels(kind, seed=1)
    xs = [panels(k, seed=s) for s, k in enumerate(["normal", "ints", "zeros", kind])]
    mask = np.random.default_rng(7).random(y.shape) > 0.2 if masked else None
    previous = po.get_num_threads()
    po.set_num_threads(threads)
    try:
        got = po.rank_ic_many((x for x in xs), y, mask=mask)
        expected = [po.rank_ic(x, y, mask=mask) for x in xs]
    finally:
        po.set_num_threads(previous)
    assert len(got) == len(xs)
    for g, e in zip(got, expected, strict=True):
        assert g.dtype == e.dtype and g.shape == e.shape
        assert g.tobytes() == e.tobytes()
    assert np.isnan(got[0][3]) and np.isnan(got[0][4])


def test_numpy_backend_and_edges():
    y = panels("normal", seed=2)
    xs = [panels("ints", seed=3), panels("normal", seed=4)]
    got = po.rank_ic_many(xs, y, backend="numpy")
    for g, x in zip(got, xs, strict=True):
        np.testing.assert_array_equal(g, po.rank_ic(x, y, backend="numpy"))
    assert po.rank_ic_many([], y) == []
    empty = np.empty((4, 0))
    (out,) = po.rank_ic_many([empty], empty)
    assert out.shape == (4,) and np.isnan(out).all()
    with pytest.raises(ValueError):
        po.rank_ic_many([y[:-1].copy()], y)
    with pytest.raises(TypeError):
        po.rank_ic_many([y.astype(np.float32)], y)
    with pytest.raises(ValueError):
        po.rank_ic_many([y], y, mask=np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError):
        po.rank_ic_many([y], y, backend="bogus")


def test_days_with_two_or_three_common_samples():
    """恰好 2 个公共样本的日子相关系数是 ±1，也必须照常计算。"""
    y = np.full((6, 5), np.nan)
    x = np.full((6, 5), np.nan)
    y[0, :2], x[0, :2] = [1.0, 2.0], [3.0, 4.0]  # +1
    y[1, 1:3], x[1, 1:3] = [1.0, 2.0], [4.0, 3.0]  # -1
    y[2, :3], x[2, :3] = [1.0, 2.0, 3.0], [1.0, 3.0, 2.0]
    y[3, :2], x[3, 2:4] = [1.0, 2.0], [1.0, 2.0]  # 没有公共样本
    y[4, 0], x[4, 0] = 1.0, 2.0  # 只有 1 个
    y[5], x[5] = [1.0, 1.0, 2.0, 2.0, 3.0], [5.0, 4.0, 3.0, 2.0, 1.0]
    (got,) = po.rank_ic_many([x], y)
    expected = po.rank_ic(x, y)
    assert got.tobytes() == expected.tobytes()
    assert np.isclose(got[0], 1.0) and np.isclose(got[1], -1.0)
    assert np.isnan(got[3]) and np.isnan(got[4])


def test_shape_error_message_matches_rank_ic():
    y = panels("normal", seed=2)
    bad = np.ascontiguousarray(y[:-1])
    with pytest.raises(ValueError) as single:
        po.rank_ic(bad, y)
    with pytest.raises(ValueError) as many:
        po.rank_ic_many([bad], y)
    assert str(many.value) == str(single.value)
    with pytest.raises(TypeError, match="iterable of panels"):
        po.rank_ic_many(y, y)


def test_numpy_dispatch_keeps_mask_and_follows_backends(monkeypatch):
    y = panels("ints", seed=5)
    xs = [panels("normal", seed=6), panels("zeros", seed=7)]
    mask = np.random.default_rng(3).random(y.shape) > 0.3
    got = po.rank_ic_many(xs, y, mask=mask, backend="numpy")
    for g, x in zip(got, xs, strict=True):
        expected = po.rank_ic(x, y, mask=mask, backend="numpy")
        assert g.tobytes() == expected.tobytes()

    def refuse(*args):
        raise AssertionError("native path used despite BACKENDS")

    monkeypatch.setitem(po.api.BACKENDS, "rank_ic", "numpy")
    monkeypatch.setattr(po.api._core, "return_order", refuse)
    got = po.rank_ic_many(xs, y, mask=mask)
    for g, x in zip(got, xs, strict=True):
        assert g.tobytes() == po.rank_ic(x, y, mask=mask).tobytes()
