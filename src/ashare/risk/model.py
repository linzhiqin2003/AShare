"""统一的日频风险快照；正式路径仅使用因子结构。"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def immutable(value, ndim, name, dtype=np.float64):
    a = np.array(value, dtype=dtype, order="C", copy=True)
    if a.ndim != ndim or (dtype == np.float64 and not np.isfinite(a).all()):
        raise ValueError(f"{name}: 维度错误或包含非有限数")
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class RiskModel:
    symbols: tuple[str, ...]
    B: np.ndarray
    factors: tuple[str, ...]
    F: np.ndarray
    D: np.ndarray
    date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name, dim in (("B", 2), ("F", 2), ("D", 1)):
            object.__setattr__(self, name, immutable(getattr(self, name), dim, name))
        object.__setattr__(self, "symbols", tuple(map(str, self.symbols)))
        object.__setattr__(self, "factors", tuple(self.factors))
        n, k = self.B.shape
        if n == 0 or len(self.symbols) != n or len(set(self.symbols)) != n:
            raise ValueError("symbols: 需要唯一、非空且与 B 一致")
        if self.F.shape != (k, k) or self.D.shape != (n,) or len(self.factors) != k:
            raise ValueError("B/F/D/factors: 形状不一致")
        if len(set(self.factors)) != k or (self.D < 0).any():
            raise ValueError("因子名重复或特质方差为负")
        scale = max(float(np.max(np.abs(self.F), initial=0)), 1e-15)
        if not np.allclose(self.F, self.F.T, atol=scale * 1e-12, rtol=0):
            raise ValueError("F: 必须对称")
        object.__setattr__(self, "F", immutable((self.F + self.F.T) * 0.5, 2, "F"))
        diagonal = np.diag(self.F)
        eigen_min = (
            diagonal.min(initial=0)
            if np.count_nonzero(self.F) == np.count_nonzero(diagonal)
            else np.linalg.eigvalsh(self.F).min(initial=0)
        )
        if k and eigen_min < -scale * 1e-10:
            raise ValueError("F: 必须半正定")

    def _weights(self, weights, benchmark=None):
        w = np.asarray(weights, dtype=float)
        if w.shape != self.D.shape or not np.isfinite(w).all():
            raise ValueError("weights: 需要长度 N 的有限向量")
        if benchmark is not None:
            w = w - self._weights(benchmark)
        return w

    def matvec(self, weights):
        w = self._weights(weights)
        return self.B @ (self.F @ (self.B.T @ w)) + self.D * w

    def variance(self, weights, benchmark=None):
        w = self._weights(weights, benchmark)
        y = self.B.T @ w
        return float(y @ self.F @ y + np.dot(self.D, w * w))

    def volatility(self, weights, benchmark=None):
        return float(np.sqrt(max(0.0, self.variance(weights, benchmark))))

    def decomposition(self, weights, benchmark=None):
        w = self._weights(weights, benchmark)
        y = self.B.T @ w
        f = y * (self.F @ y)
        d = self.D * w * w
        return {"factor": f, "specific": d, "total": float(f.sum() + d.sum())}

    def marginal_risk(self, weights, benchmark=None):
        w = self._weights(weights, benchmark)
        vol = self.volatility(w)
        return self.matvec(w) / vol if vol else np.zeros_like(w)

    def contributions(self, weights, benchmark=None):
        w = self._weights(weights, benchmark)
        return w * self.marginal_risk(w)

    def dense(self):
        """只供小规模测试；上限不能由调用方绕过。"""
        if len(self.D) > 256:
            raise ValueError("禁止为 N > 256 展开稠密协方差")
        return (self.B @ self.F) @ self.B.T + np.diag(self.D)

    def subset(self, indices):
        i = np.asarray(indices, dtype=int)
        return RiskModel(
            tuple(self.symbols[j] for j in i),
            self.B[i],
            self.factors,
            self.F,
            self.D[i],
            self.date,
            self.metadata.copy(),
        )
