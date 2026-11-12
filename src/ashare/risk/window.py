"""特质方差的预计算结果：可代替残差窗口传给 `specific_variance`。"""

from typing import NamedTuple

import numpy as np


class WindowVariances(NamedTuple):
    base: np.ndarray
    corr0: np.ndarray
    corrnw: np.ndarray
    count: np.ndarray
