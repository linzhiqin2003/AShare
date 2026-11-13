"""配置用的小规模协方差估计：复用 `ashare.risk.statistical.shrinkage_components`。

大类资产只有几个到几十个，这里把风险模块的低秩表示 ``U Uᵀ + diag(d)`` 展开成 n×n 稠密矩阵
（风险模块本身不展开，因为股票有几千只）。单位是日收益的方差。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ashare.risk.statistical import shrinkage_components


@dataclass(frozen=True)
class Covariance:
    """协方差估计方法。

    half_life：指数加权的半衰期（交易日），``None`` 表示窗口内等权。
    shrinkage：``"auto"`` 用 Ledoit–Wolf 公式求收缩强度；``0`` 不收缩（样本协方差）；
        0 到 1 之间的数为固定强度。
    target：收缩目标，``"identity"``（等方差）或 ``"constant_correlation"``（常相关）。

    缺失收益按 `ashare.risk` 的约定：逐列有效权重归一化，不当作 0。
    """

    half_life: float | None = 63.0
    shrinkage: str | float = "auto"
    target: str = "constant_correlation"

    def __post_init__(self):
        if self.half_life is not None and not (
            np.isfinite(self.half_life) and self.half_life > 0
        ):
            raise ValueError("half_life: 必须为正数或 None")
        if self.shrinkage != "auto" and not (
            isinstance(self.shrinkage, (int, float)) and 0 <= self.shrinkage <= 1
        ):
            raise ValueError("shrinkage: 只能是 'auto' 或 [0,1] 内的数")
        if self.target not in ("identity", "constant_correlation"):
            raise ValueError("target: identity 或 constant_correlation")

    def estimate(self, returns):
        """returns：L×n 日收益（NaN 为缺失）→ (n×n 协方差, 说明)。"""
        x = np.asarray(returns, dtype=float)
        if x.ndim != 2 or x.shape[1] == 0 or x.shape[0] < 2:
            raise ValueError("returns: 需要至少两行、一列的二维数组")
        half = np.inf if self.half_life is None else float(self.half_life)
        strength = None if self.shrinkage == "auto" else float(self.shrinkage)
        u, d, meta = shrinkage_components(x, half, self.target, strength)
        sigma = u @ u.T + np.diag(d)
        sigma = (sigma + sigma.T) * 0.5
        return sigma, {
            "shrinkage": float(meta["shrinkage"]),
            "average_correlation": float(meta["average_correlation"]),
        }


def sample_volatility(returns):
    """窗口内每列的样本标准差（ddof=1，忽略 NaN；有效值少于 2 个为 NaN），用来识别零波动资产。"""
    x = np.asarray(returns, dtype=float)
    ok = np.isfinite(x)
    count = ok.sum(axis=0)
    with np.errstate(all="ignore"):
        mean = np.where(ok, x, 0.0).sum(axis=0) / count
        squares = np.where(ok, (x - mean) ** 2, 0.0).sum(axis=0)
        out = np.sqrt(squares / (count - 1))
    return np.where(count >= 2, out, np.nan)
