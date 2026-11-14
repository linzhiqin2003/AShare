"""集成测试共用的合成数据目录：整个会话（含 xdist 各进程）只建一次，只读共享。"""

import pytest

from test.fixtures.synthetic_datadir import cached_datadir


@pytest.fixture(scope="session")
def datadir():
    return cached_datadir()
