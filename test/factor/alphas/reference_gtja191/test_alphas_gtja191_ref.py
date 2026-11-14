"""国君 191 参考实现整体：186 条都能算、形状对、批量与逐条一致，几条用 pandas 直接抽查。"""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare.factor.alphas.reference import gtja191_ref as g
from ashare.factor.alphas.reference import gtja191_ref_ops as op

EXCLUDED = {30, 143, 165, 166, 183}
SOURCES = Path(__file__).resolve().parents[4] / "docs/factors/sources/gtja191.jsonl"


def frames(t=320, n=12, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.Index([f"d{i:04d}" for i in range(t)])
    symbols = pd.Index([f"S{j:02d}" for j in range(n)])
    steps = rng.normal(0, 0.02, (t, n))
    steps[:, 3] = 0.0  # 常数价格
    close = np.round(10 * np.exp(np.cumsum(steps, axis=0)), 2)
    high = np.round(close * (1 + rng.uniform(0, 0.03, (t, n))), 2)
    low = np.round(close * (1 - rng.uniform(0, 0.03, (t, n))), 2)
    open_ = np.round(low + (high - low) * rng.uniform(0, 1, (t, n)), 2)
    volume = np.round(rng.lognormal(12, 1, (t, n)), -2)
    quote = np.ones((t, n), bool)
    quote[:40, 1] = False  # 中途上市
    quote[100:104, 2] = False  # 停牌
    quote[:, 4] = False  # 从未上市
    data = {}
    for name, v in [
        ("open", open_),
        ("high", high),
        ("low", low),
        ("close", close),
        ("volume", volume),
        ("amount", volume * (high + low) / 2),
        ("vwap", (high + low) / 2),
    ]:
        data[name] = pd.DataFrame(
            np.where(quote, v, np.nan), index=dates, columns=symbols
        )
    prev = data["close"].shift(1)
    data["returns"] = data["close"] / prev - 1
    bench = 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, t)))
    data["benchmark_close"] = pd.Series(bench, index=dates)
    data["benchmark_open"] = pd.Series(
        bench * (1 + rng.normal(0, 0.003, t)), index=dates
    )
    return data


@pytest.fixture(scope="module")
def data():
    return frames()


@pytest.fixture(scope="module")
def results(data):
    return g.compute_many(data, sorted(g.REFERENCES))


def test_registry_names():
    expected = {f"gtja191_{n:03d}" for n in range(1, 192) if n not in EXCLUDED}
    assert set(g.REFERENCES) == expected
    assert len(g.REFERENCES) == 186


def test_formula_comments_verbatim():
    formulas = {}
    for line in SOURCES.read_text().splitlines():
        d = json.loads(line)
        formulas[f"gtja191_{d['n']:03d}"] = d["formula"]
    source = Path(g.__file__).read_text()
    for name in g.REFERENCES:
        assert f"# {formulas[name]}\ndef {name}(" in source, name


def test_independent_of_engine():
    for module in (g, op):
        imports = [
            line
            for line in inspect.getsource(module).splitlines()
            if line.startswith(("import ", "from "))
        ]
        for line in imports:
            assert "panelops" not in line and "ashare" not in line, line


def test_all_run_shape_and_finite(data, results):
    shape = data["close"].shape
    for name, values in results.items():
        assert values.shape == shape, name
        assert not np.isinf(values).any(), name
        if name != "gtja191_064":
            # 064：12 只股票的小截面上 RANK(MEAN(VOLUME,60)) 的 4 日窗口约四分之一是常数，
            # CORR 为 NaN，再经 TSMAX(·,13)、DECAYLINEAR(·,14) 的全窗口口径传播成全 NaN。
            assert np.isfinite(values).any(), name
        # 从未上市的一列全 NaN
        assert np.isnan(values[:, 4]).all(), name


def test_single_matches_batch(data, results):
    for name in ("gtja191_001", "gtja191_069", "gtja191_149", "gtja191_190"):
        single = g.REFERENCES[name](data)
        assert list(single.index) == list(data["close"].index)
        assert list(single.columns) == list(data["close"].columns)
        np.testing.assert_array_equal(single.to_numpy(), results[name])


def _check(results, name, expected):
    np.testing.assert_allclose(
        results[name],
        np.asarray(expected, float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_spot_checks_with_plain_pandas(data, results):
    c, o, h, low, v = (data[k] for k in ("close", "open", "high", "low", "volume"))
    _check(results, "gtja191_014", c - c.shift(5))
    _check(results, "gtja191_018", c / c.shift(5))
    _check(results, "gtja191_126", (c + h + low) / 3)
    _check(results, "gtja191_132", data["amount"].rolling(20).sum() / 20)
    up = (c > c.shift(1)).astype(float).where(c.notna() & c.shift(1).notna())
    _check(results, "gtja191_053", up.rolling(12).sum() / 12 * 100)
    _check(results, "gtja191_065", (c.rolling(6).sum() / 6) / c)
    _check(results, "gtja191_185", (-1 * (1 - o / c) ** 2).rank(axis=1, pct=True))
    ratio = v / (v.rolling(20).sum() / 20)
    _check(results, "gtja191_168", -1 * v / (v.rolling(20).sum() / 20))
    assert np.isfinite(ratio.to_numpy()).any()


def test_sma_based_factor_by_loop(data, results):
    # 024：SMA(CLOSE-DELAY(CLOSE,5),5,1)，逐列逐日递推
    x = (data["close"] - data["close"].shift(5)).to_numpy()
    expected = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        state, seen = None, 0
        for i in range(x.shape[0]):
            if np.isnan(x[i, j]):
                continue
            state = x[i, j] if state is None else (1 * x[i, j] + 4 * state) / 5
            seen += 1
            if seen >= 5:
                expected[i, j] = state
    _check(results, "gtja191_024", expected)


def test_149_suspension_nan_and_partial_window(data, results):
    """R-G-02：当天收益为 NaN（停牌、复牌首日）时为 NaN；开头不足 252 行用已有行，配对 ≥ 20 出数。"""
    c = data["close"].to_numpy()
    bc = data["benchmark_close"].to_numpy()
    r = c[1:] / c[:-1] - 1
    r = np.vstack([np.full((1, c.shape[1]), np.nan), r])
    xb = np.concatenate([[np.nan], bc[1:] / bc[:-1] - 1])
    down = np.concatenate([[False], bc[1:] < bc[:-1]])
    expected = np.full(c.shape, np.nan)
    for j in range(c.shape[1]):
        for i in range(c.shape[0]):
            if np.isnan(r[i, j]):
                continue
            lo = max(0, i - 251)
            ok = [k for k in range(lo, i + 1) if down[k] and not np.isnan(r[k, j])]
            if len(ok) < 20:
                continue
            xs, ys = xb[ok], r[ok, j]
            dx, dy = xs - xs.mean(), ys - ys.mean()
            expected[i, j] = (dx * dy).sum() / (dx * dx).sum()
    _check(results, "gtja191_149", expected)
    got = results["gtja191_149"]
    assert np.isnan(got[100:105, 2]).all()  # 停牌 100–103，104 复牌首日收益为 NaN
    assert np.isfinite(got[105, 2]) and np.isfinite(got[99, 2])
    assert np.isfinite(got[:251, 0]).any()  # 开头不足 252 行也出数
