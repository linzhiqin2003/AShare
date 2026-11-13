"""补充算子定义表与可组合构造器的结构检查。

这里只守住清单覆盖、``ts_range`` 的形状，以及未实现算子必须显式报错而不是静默给出
错误结果；数值对拍在 ``test_ops_extra_numeric.py``。
"""

import pytest

from ashare.factor import Expr, field
from ashare.factor.formula import ops_extra

X = field("x")
Y = field("y")


def test_catalog_is_complete():
    assert len(ops_extra.FORMULAS) == 69  # 补充清单 68 个 + 新增 ts_range
    assert "ts_range" in ops_extra.FORMULAS
    assert set(ops_extra.BUILDERS) | set(ops_extra.UNSUPPORTED) == set(
        ops_extra.FORMULAS
    )
    assert not set(ops_extra.BUILDERS) & set(ops_extra.UNSUPPORTED)


def test_ts_range_is_max_minus_min():
    expression = ops_extra.build("ts_range", X, window=5)
    assert isinstance(expression, Expr)
    assert expression.op == "subtract"
    assert {child.op for child in expression.args} == {"ts_max", "ts_min"}


def test_normalized_rank_uses_kernel_normalize():
    expression = ops_extra.build("ts_rank", X, window=5)
    assert "normalize=True" in str(expression)


def test_conflicts_are_documented():
    assert {"ts_rank", "ts_quantile", "ts_decay_linear", "scale", "ts_backfill"} <= set(
        ops_extra.CONFLICTS
    )


def test_unsupported_operators_raise():
    with pytest.raises(NotImplementedError):
        ops_extra.build("bucket", X, 10)
    with pytest.raises(NotImplementedError):
        ops_extra.build("ts_backfill", X, lookback=20, k=2)
    with pytest.raises(NotImplementedError):
        ops_extra.build("scale", X, longscale=1.0, shortscale=1.0)
    with pytest.raises(KeyError):
        ops_extra.build("not_an_operator", X)


def test_group_neutralize_goes_through_cs_neutralize():
    expression = ops_extra.build("group_neutralize", X, Y)
    assert expression.op == "cs_neutralize"
    assert dict(expression.params)["has_groups"] is True


def test_coverage_matches_tables():
    covered, total = ops_extra.coverage()
    assert total == len(ops_extra.FORMULAS)
    assert covered == len(ops_extra.BUILDERS)
    assert set(ops_extra.missing()) == set(ops_extra.UNSUPPORTED)
