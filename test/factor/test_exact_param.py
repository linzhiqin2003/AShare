# ruff: noqa: E402
"""滑动求和的 exact 参数在表达式层只收 bool，默认值不进规范形式。"""

from pathlib import Path

import ashare

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)

import numpy as np
import pytest

import ashare.factor as af


def test_exact_false_keeps_existing_digest_and_text():
    x = af.field("close")
    base = af.call("ts_sum", x, window=5)
    assert str(base) == "ts_sum(close, min_periods=5, window=5)"
    for false in (False, np.bool_(False)):
        e = af.call("ts_sum", x, window=5, exact=false)
        assert e.digest == base.digest and str(e) == str(base)


def test_exact_true_is_a_distinct_expression():
    x = af.field("close")
    a = af.call("ts_mean", x, window=5, exact=True)
    b = af.call("ts_mean", x, window=5, exact=np.bool_(True))
    assert a.digest == b.digest
    assert str(a) == "ts_mean(close, exact=True, min_periods=5, window=5)"
    assert a.digest != af.call("ts_mean", x, window=5).digest


@pytest.mark.parametrize("value", (0, 1, None, "no", 1.0))
def test_exact_rejects_non_bool(value):
    with pytest.raises(TypeError, match="exact"):
        af.call("ts_sum", af.field("close"), window=5, exact=value)
