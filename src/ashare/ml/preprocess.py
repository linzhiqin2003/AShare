"""截面预处理：每一步只用当日截面，训练和预测走同一份结果（存在特征库里）。

顺序：去极值 → 中性化 → 标准化（或排名化）→ 缺失值处理。全部用 `panelops` 的截面
算子，按日在股票池掩码内做，不用未来数据。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import panelops

NEUTRALIZE = ("industry", "size")


@dataclass(frozen=True)
class Preprocess:
    """截面预处理的定义；进特征库的缓存键。

    winsor：`"quantile"`（按 lower/upper 分位截断）、`"mad"`（中位数 ± mad_scale × MAD）
    或 None。neutralize：`("industry",)`、`("size",)` 或两者，对申万行业哑变量和对数
    总市值回归取残差。standardize：`"zscore"`、`"rank"`（排名映射到 [−0.5, 0.5]）或 None。
    fill：`"zero"` 把股票池内的缺失填 0（标准化后即截面均值/中位）；`"none"` 保留 NaN
    （LightGBM 能用缺失信息，线性模型在取数时补 0）。
    """

    winsor: str | None = "quantile"
    lower: float = 0.01
    upper: float = 0.99
    mad_scale: float = 5.0
    neutralize: tuple = ()
    standardize: str | None = "zscore"
    fill: str = "zero"

    def __post_init__(self):
        if self.winsor not in ("quantile", "mad", None):
            raise ValueError("winsor: 只能是 quantile、mad 或 None")
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("lower/upper: 需要 0 ≤ lower < upper ≤ 1")
        if not self.mad_scale > 0:
            raise ValueError("mad_scale: 必须为正")
        object.__setattr__(self, "neutralize", tuple(self.neutralize))
        if set(self.neutralize) - set(NEUTRALIZE) or len(set(self.neutralize)) != len(
            self.neutralize
        ):
            raise ValueError(f"neutralize: 只能取 {NEUTRALIZE} 的子集")
        if self.standardize not in ("zscore", "rank", None):
            raise ValueError("standardize: 只能是 zscore、rank 或 None")
        if self.fill not in ("zero", "none"):
            raise ValueError("fill: 只能是 zero 或 none")

    def fields(self):
        out = []
        if "industry" in self.neutralize:
            out.append("industry")
        if "size" in self.neutralize:
            out.append("total_market_cap")
        return tuple(out)

    def describe(self):
        return asdict(self)

    @property
    def digest(self):
        text = json.dumps(["preprocess", 1, self.describe()], sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()

    def context(self, panel):
        """中性化用到的暴露（行业编码、对数市值）；建库时算一次，各特征共用。"""
        ctx = {}
        if "industry" in self.neutralize:
            ctx["groups"] = np.ascontiguousarray(panel["industry"], dtype=np.int64)
        if "size" in self.neutralize:
            cap = np.asarray(panel["total_market_cap"], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                ctx["size"] = np.ascontiguousarray(
                    np.where(cap > 0, np.log(cap), np.nan)
                )
        return ctx

    def apply(self, x, mask, ctx=None):
        """对一张 T×N 特征面板做截面预处理，返回 float64 T×N（掩码外为 NaN）。"""
        ctx = ctx or {}
        x = np.ascontiguousarray(x, dtype=np.float64)
        mask = np.ascontiguousarray(mask, dtype=bool)
        if self.winsor == "quantile":
            x = panelops.cs_winsor_quantile(
                x, lower=self.lower, upper=self.upper, mask=mask
            )
        elif self.winsor == "mad":
            x = panelops.cs_winsor_mad(x, scale=self.mad_scale, mask=mask)
        if self.neutralize:
            exposures = [ctx["size"]] if "size" in self.neutralize else []
            x = panelops.cs_neutralize(
                x, exposures=exposures, groups=ctx.get("groups"), mask=mask
            )
        if self.standardize == "zscore":
            x = panelops.cs_zscore(x, mask=mask)
        elif self.standardize == "rank":
            x = panelops.cs_rank(x, normalize=True, mask=mask) - 0.5
        if self.fill == "zero":
            x = np.where(mask & ~np.isfinite(x), 0.0, x)
        x[~mask] = np.nan
        return x
