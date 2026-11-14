"""Qlib 转换程序：158 条全部能转、确定、生成文件与原文一致；拒绝没见过的写法。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare.factor import call, field
from ashare.factor.alphas import (
    ALPHA158_FORMULAS,
    alpha101_registry,
    alpha158_registry,
    build_alpha158,
    gtja191_registry,
)
from ashare.factor.alphas.reference import qlib_alpha158 as ref
from ashare.factor.formula import ops, qlib

ROOT = Path(__file__).resolve().parents[3]


def test_formulas_module_is_generated_from_sources():
    rows = [
        json.loads(line)
        for line in (ROOT / "docs/factors/sources/alpha158.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [(r["n"], r["name"], r["formula"]) for r in rows] == list(ALPHA158_FORMULAS)
    assert build_alpha158.main(["--check"]) == 0


def test_all_158_convert_deterministically():
    first = [qlib.to_expr(f).digest for _, _, f in ALPHA158_FORMULAS]
    second = [qlib.to_expr(f).digest for _, _, f in ALPHA158_FORMULAS]
    assert first == second and len(set(first)) == 158
    registry = alpha158_registry()
    assert len(registry) == 158
    assert registry["alpha158_kmid"].description == "($close-$open)/$open"
    assert len(alpha101_registry()) <= 101 and len(gtja191_registry()) <= 186


def test_conversion_structure_by_hand():
    close = field("close") * field("adj_factor")
    domain = ops.listed_domain()
    assert (
        qlib.to_expr("Ref($close, 5)/$close")
        == call("ts_delay", close, periods=5) / close * domain
    )
    assert (
        qlib.to_expr("Mean($close>Ref($close, 1), 5)")
        == call(
            "ts_mean",
            ops.qlib_compare("gt", close, call("ts_delay", close, periods=1)),
            window=5,
            min_periods=1,
        )
        * domain
    )
    assert qlib.to_expr("Quantile($close, 5, 0.8)/$close") == (
        call("ts_quantile", close, window=5, min_periods=1, q=0.8) / close * domain
    )
    assert qlib.to_expr("Greater($close-Ref($close, 1), 0)") == (
        call("maximum", close - call("ts_delay", close, periods=1), 0) * domain
    )
    # 优先级与 Python 相同：乘除先于加减，比较最后。
    assert qlib.parse("2*$close-$high-$low") == (
        "bin",
        "-",
        ("bin", "-", ("bin", "*", ("num", 2), ("field", "close")), ("field", "high")),
        ("field", "low"),
    )


@pytest.mark.parametrize(
    "text",
    [
        "Foo($close, 5)",
        "$amount/$close",
        "Mean($close, $open)",
        "Ref($close, -1)",
        "Mean($close, 5",
        "$close > $open > $low",
        "1+2",
        "Quantile($close, 5)",
        "$close ? 1 : 0",
    ],
)
def test_rejects_unknown_syntax(text):
    with pytest.raises((SyntaxError, ValueError)):
        qlib.to_expr(text)


def test_cython_port_matches_direct_regression():
    """参考实现里移植的 rolling.pyx：完整窗口与 numpy 最小二乘一致（含缺失与初始空窗）。"""
    rng = np.random.default_rng(0)
    values = 50 + np.cumsum(rng.normal(size=120))
    values[[10, 11, 40]] = np.nan
    w = 6
    slope = ref._cython_rolling(values, w, "slope")
    resi = ref._cython_rolling(values, w, "resi")
    rsq = ref._cython_rolling(values, w, "rsquare")
    for t in range(len(values)):
        pos = np.arange(max(0, t - w + 1), t + 1)
        y = values[pos]
        ok = np.isfinite(y)
        if ok.sum() < 2:
            assert np.isnan(slope[t])
            continue
        b, a = np.polyfit(pos[ok], y[ok], 1)
        np.testing.assert_allclose(slope[t], b, rtol=1e-7, atol=1e-9)
        r = np.corrcoef(pos[ok], y[ok])[0, 1]
        np.testing.assert_allclose(rsq[t], r * r, rtol=1e-7, atol=1e-9)
        if np.isfinite(values[t]):
            np.testing.assert_allclose(
                resi[t], values[t] - (a + b * t), rtol=1e-7, atol=1e-7
            )
        else:
            assert np.isnan(resi[t])


def test_reference_compiles_like_qlib():
    tree = ref.compile_formula("Mean($close>Ref($close, 1), 5)-2*$open")
    assert str(tree) == "Sub(Mean(Gt($close,Ref($close,1)),5),Mul($open,2))"
    with pytest.raises(NameError):
        ref.compile_formula("Foo($close, 5)")
    data = {
        "close": pd.Series([1.0, 2.0, np.nan, 2.0]),
        "open": pd.Series([1.0, 1.0, 1.0, 1.0]),
    }
    out = ref.compile_formula("$close>Ref($close, 1)").load(data, {})
    assert out.tolist() == [False, True, False, False]  # NaN 比较为 False
