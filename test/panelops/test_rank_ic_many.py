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
