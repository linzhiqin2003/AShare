"""机制测试的合成数据（见 ``ashare.factor.formula.testing``）。"""

import pytest

from ashare.factor.formula.testing import synthetic_source

__all__ = ["synthetic_source"]


@pytest.fixture
def source():
    return synthetic_source()
