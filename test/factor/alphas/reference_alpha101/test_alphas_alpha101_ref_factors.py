"""Alpha101 参考实现的因子层测试：原文注释逐字、101 条都能跑、D-13 两种读法、几条手算核对。"""

import ast
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from ashare.factor.alphas.reference import alpha101_ref as ref
from ashare.factor.alphas.reference import alpha101_ref_ops as op

ROOT = pathlib.Path(__file__).resolve().parents[4]
SOURCE = ROOT / "docs/factors/sources/alpha101.jsonl"
MODULE = ROOT / "src/ashare/factor/alphas/reference/alpha101_ref.py"


def test_formula_comments_verbatim():
    formulas = {}
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        formulas[row["id"]] = row["formula"]
    lines = MODULE.read_text(encoding="utf-8").splitlines()
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    defs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("alpha101_")
    ]
    assert [d.name for d in defs] == list(formulas)
    for d in defs:
        comment = lines[d.lineno - 2]
        assert comment == "# 原文：" + formulas[d.name], d.name


def test_registry_covers_101():
    assert len(ref.REFERENCES) == 101
    assert list(ref.REFERENCES) == [f"alpha101_{i:03d}" for i in range(1, 102)]


def synthetic_frames(t=300, n=16, seed=3):
    """``reference_frames`` 形状的合成输入：上市前空行、停牌、并列价格、常数量、行业缺失。

    自带，不依赖 ``inputs``（因子库正在迁移出本仓库）。``t`` 取 300，够 250 日窗口出数。
    """
    rng = np.random.default_rng(seed)
    dates = pd.Index([f"d{i:04d}" for i in range(t)])
    symbols = pd.Index([f"{600000 + j}.SH" for j in range(n)])
    quote = np.ones((t, n), bool)
    quote[:40, 1] = False  # 中途上市
    quote[200:, 2] = False  # 退市
    for j in range(n):
        for start in rng.integers(0, t, 2):
            quote[start : start + rng.integers(1, 5), j] = False
    steps = rng.normal(0, 0.02, (t, n))
    steps[:, 5] = 0.0
    close = np.round(10 * np.exp(np.cumsum(steps, axis=0)), 2)
    high = np.round(close * (1 + rng.uniform(0, 0.03, (t, n))), 2)
    low = np.round(close * (1 - rng.uniform(0, 0.03, (t, n))), 2)
    high[::17, 6] = low[::17, 6] = close[::17, 6]  # 一字板：high == low
    open_ = np.round(low + (high - low) * rng.uniform(0, 1, (t, n)), 2)
    volume = np.round(rng.lognormal(12, 1, (t, n)), -2)
    volume[:, 8] = 1e6
    amount = volume * (low + high) / 2
    previous = np.vstack([np.full((1, n), np.nan), close[:-1]])
    industry = rng.choice([801010.0, 801030.0, 801040.0], (t, n))
    industry[:, 9] = np.nan

    def frame(values):
        return pd.DataFrame(
            np.where(quote, values, np.nan), index=dates, columns=symbols
        )

    out = {
        "open": frame(open_),
        "high": frame(high),
        "low": frame(low),
        "close": frame(close),
        "vwap": frame(amount / volume),
        "volume": frame(volume),
        "amount": frame(amount),
        "returns": frame(close / previous - 1),
        "cap": frame(close * 1e8),
    }
    out["listed"] = pd.DataFrame(quote, index=dates, columns=symbols)
    out["industry"] = pd.DataFrame(industry, index=dates, columns=symbols)
    return out


@pytest.fixture(scope="module")
def frames():
    return synthetic_frames()


def test_all_run_shape_and_finite(frames):
    out = ref.compute_many(frames, list(ref.NAMES))
    shape = frames["close"].shape
    for name, values in out.items():
        assert values.shape == shape, name
        assert not np.isinf(values).any(), name
    # 每条单独调用（REFERENCES 接口）与批量结果逐位相同
    for name in ("alpha101_001", "alpha101_048", "alpha101_100"):
        single = ref.REFERENCES[name](frames)
        assert isinstance(single, pd.DataFrame)
        np.testing.assert_array_equal(single.to_numpy(), out[name])


def test_argmax_readings_switch(frames):
    a = ref.compute_many(frames, list(ref.ARGMAX_DEPENDENT), argmax="days_ago")
    b = ref.compute_many(frames, list(ref.ARGMAX_DEPENDENT), argmax="position")
    for name in ref.ARGMAX_DEPENDENT:
        both = np.isfinite(a[name]) & np.isfinite(b[name])
        if both.any():
            assert not np.array_equal(a[name][both], b[name][both]), name
    # 不受影响的条目不接受这个参数也不变
    plain = ref.references("position")["alpha101_002"]
    np.testing.assert_array_equal(
        np.asarray(plain(frames)), np.asarray(ref.REFERENCES["alpha101_002"](frames))
    )


def test_alpha101_057_by_hand(frames):
    """057 = −(close − vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)，按两种读法逐格手算。"""
    close = frames["close"].to_numpy()
    vwap = frames["vwap"].to_numpy()
    for how in ("days_ago", "position"):
        am = np.full(close.shape, np.nan)
        for j in range(close.shape[1]):
            for i in range(29, close.shape[0]):
                w = close[i - 29 : i + 1, j]
                if np.isnan(w).any():
                    continue
                hits = np.flatnonzero(w == w.max())
                am[i, j] = 29 - hits[-1] if how == "days_ago" else hits[0] + 1
        r = pd.DataFrame(am).rank(axis=1, pct=True).to_numpy()
        dl = np.full(r.shape, np.nan)
        dl[1:] = (r[:-1] * 1 + r[1:] * 2) / 3
        with np.errstate(divide="ignore", invalid="ignore"):
            expect = 0 - (1 * ((close - vwap) / dl))
        expect[~np.isfinite(expect)] = np.nan
        got = ref.alpha101_057(frames, argmax=how).to_numpy()
        np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_alpha101_101_and_012_by_hand(frames):
    c, o, h, low = (frames[k].to_numpy() for k in ("close", "open", "high", "low"))
    np.testing.assert_allclose(
        ref.alpha101_101(frames).to_numpy(),
        (c - o) / ((h - low) + 0.001),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    v = frames["volume"].to_numpy()
    dv = np.full(v.shape, np.nan)
    dv[1:] = v[1:] - v[:-1]
    dc = np.full(c.shape, np.nan)
    dc[1:] = c[1:] - c[:-1]
    np.testing.assert_array_equal(
        ref.alpha101_012(frames).to_numpy(), np.sign(dv) * (-1 * dc)
    )


def test_ternary_nan_on_suspension(frames):
    """007：volume 缺失（停牌）时条件为 NaN，结果为 NaN，不落到常数分支 −1。"""
    out = ref.alpha101_007(frames).to_numpy()
    vol = frames["volume"].to_numpy()
    assert np.isnan(out[np.isnan(vol)]).all()


def test_indneutralize_uses_industry(frames):
    """048 的分子只在有行业的格子有值。"""
    out = ref.alpha101_048(frames).to_numpy()
    assert np.isnan(out[np.isnan(frames["industry"].to_numpy())]).all()


def test_argmax_reading_rejected():
    with pytest.raises(ValueError):
        op.ts_argmax(np.zeros((3, 1)), 2, how="bogus")
