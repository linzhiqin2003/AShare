"""Qlib 表达式转换：结构、优先级、确定性；拒绝没见过的写法。表达式都是自编的。"""

import numpy as np
import pytest

from ashare.factor import Evaluator, call, field
from ashare.factor.formula import ops, qlib
from ashare.factor.formula.testing import synthetic_source

OPEN = field("open") * field("adj_factor")
VWAP = field("vwap") * field("adj_factor")
HIGH = field("high") * field("adj_factor")
LOW = field("low") * field("adj_factor")
DOMAIN = ops.listed_domain()


def test_conversion_structure_by_hand():
    assert (
        qlib.to_expr("Ref($vwap, 4)*$open")
        == call("ts_delay", VWAP, periods=4) * OPEN * DOMAIN
    )
    assert (
        qlib.to_expr("Mean($high<$low, 6)")
        == call("ts_mean", ops.qlib_compare("lt", HIGH, LOW), window=6, min_periods=1)
        * DOMAIN
    )
    assert qlib.to_expr("Quantile($open, 9, 0.3)") == (
        call("ts_quantile", OPEN, window=9, min_periods=1, q=0.3) * DOMAIN
    )
    assert qlib.to_expr("Less($vwap-$open, 2)") == (
        call("minimum", VWAP - OPEN, 2) * DOMAIN
    )
    # 优先级与 Python 相同：乘除先于加减，比较最后。
    assert qlib.parse("3*$open-$vwap/$high") == (
        "bin",
        "-",
        ("bin", "*", ("num", 3), ("field", "open")),
        ("bin", "/", ("field", "vwap"), ("field", "high")),
    )


def test_conversion_is_deterministic_and_every_function_converts():
    texts = [
        "Mean($open, 6)-Sum($vwap, 4)*Std($low, 9)",
        "Greater($open, 1)+Less($vwap, 2)+Abs($low-$vwap)",
        "Max($open, 9)-Min($vwap, 4)",
        "Rank($open, 4)+Slope($vwap, 6)-Resi($open, 4)*Rsquare($low, 9)",
        "IdxMax($open, 6)+IdxMin($vwap, 4)+Corr($high, $low, 9)",
        "Quantile($open, 9, 0.3)+Ref($vwap, 4)+Log($open)",
    ]
    first = [qlib.to_expr(t).digest for t in texts]
    assert first == [qlib.to_expr(t).digest for t in texts]
    assert len(set(first)) == len(texts)
    used = {name for t in texts for name in qlib.FUNCTIONS if f"{name}(" in t}
    assert used == set(qlib.FUNCTIONS)
    source = synthetic_source(seed=3)
    for t in texts:
        values = Evaluator().evaluate(qlib.to_expr(t), source)
        assert values.shape == source.shape and np.isfinite(values).any()
        # 上市区间外一律为 NaN。
        assert np.isnan(values[~np.asarray(source["is_listed"], bool)]).all()


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
