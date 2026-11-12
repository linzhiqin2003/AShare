"""Qlib 表达式转换：结构、优先级、确定性；拒绝没见过的写法。表达式都是自编的。"""

import numpy as np
import pytest

from ashare.factor import Evaluator, call, field
from ashare.factor.formula import ops, qlib
from ashare.factor.formula.testing import synthetic_source

CLOSE = field("close") * field("adj_factor")
DOMAIN = ops.listed_domain()


def test_conversion_structure_by_hand():
    assert (
        qlib.to_expr("Ref($close, 7)/$close")
        == call("ts_delay", CLOSE, periods=7) / CLOSE * DOMAIN
    )
    assert (
        qlib.to_expr("Mean($close>Ref($close, 2), 7)")
        == call(
            "ts_mean",
            ops.qlib_compare("gt", CLOSE, call("ts_delay", CLOSE, periods=2)),
            window=7,
            min_periods=1,
        )
        * DOMAIN
    )
    assert qlib.to_expr("Quantile($close, 13, 0.7)/$close") == (
        call("ts_quantile", CLOSE, window=13, min_periods=1, q=0.7) / CLOSE * DOMAIN
    )
    assert qlib.to_expr("Less($close-Ref($close, 2), 0)") == (
        call("minimum", CLOSE - call("ts_delay", CLOSE, periods=2), 0) * DOMAIN
    )
    # 优先级与 Python 相同：乘除先于加减，比较最后。
    assert qlib.parse("3*$close-$high-$low") == (
        "bin",
        "-",
        ("bin", "-", ("bin", "*", ("num", 3), ("field", "close")), ("field", "high")),
        ("field", "low"),
    )


def test_conversion_is_deterministic_and_every_function_converts():
    texts = [
        "Mean($high, 7)/Std($close, 13)",
        "Sum(Greater($close-$open, 0), 7)/(Sum(Abs($close-$open), 7)+1e-12)",
        "Max($high, 13)/Min($low, 13)",
        "Rank($volume, 7)*Slope($close, 13)+Resi($close, 7)-Rsquare($vwap, 13)",
        "IdxMax($high, 7)-IdxMin($low, 7)+Corr($close, Log($volume+1), 13)",
        "Quantile($close, 7, 0.2)/Ref($close, 3)-Less($low, Ref($low, 1))",
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
