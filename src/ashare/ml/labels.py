"""标签：按回测的成交约定实际拿得到的未来收益。

时间轴（行号就是交易日，与 `ashare.backtest.timing` 相同）：

    t            t+lag                      t+lag+h
    │ 收盘后算特征 │ 按成交价买入（can_buy）    │ 按成交价卖出
    ├────────────┼──────────────────────────┤
    信号日         成交日 = execution_days(t)   标签结束日 end_day(t)

label[t] = P[t+lag+h] / P[t+lag] − 1，P = 成交价 × 复权因子。复权因子把分红再投资、
送转都算进去，与 `panelbt` 按面板公司行为记账的差别只有红利税和零股，见 ML_NOTES。

- 买入日不可买（停牌、涨停、一字板，口径同 `market_from_panel`）：默认
  `blocked="exclude"`，样本不进训练集，但仍然给预测值；`"keep"` 照样进训练集。
- 卖出日不可卖：默认 `sell_blocked="price"`，按当日成交价算（偏乐观，见文档）；
  `"drop"` 把标签置缺失。成交价缺失（停牌、退市）时标签缺失，不进训练集。
- 末尾 lag+h 行没有标签。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import numpy as np
import panelops
from panelbt.markets.ashare import tradability

from ..backtest.market import EXEC_PRICES, MASKS, RULES

RELATIVE = ("none", "market", "industry")


@dataclass(frozen=True)
class Label:
    """标签定义。horizon：持有交易日数；lag：信号日到买入日的行数（同回测的 lag）。

    relative：`"none"` 原始收益，`"market"` 扣除股票池等权均值，`"industry"` 扣除
    同行业均值（行业取信号日）。rank：训练目标改为当日截面排名（居中到 ±0.5）。
    评估（Rank IC、分组收益）始终用原始收益。
    """

    horizon: int = 5
    lag: int = 1
    exec: str = "open"
    relative: str = "none"
    rank: bool = False
    blocked: str = "exclude"
    sell_blocked: str = "price"

    def __post_init__(self):
        for name in ("horizon", "lag"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}: 必须是正整数")
        if self.exec not in EXEC_PRICES:
            raise ValueError(f"exec: 只能是 {tuple(EXEC_PRICES)}")
        if self.relative not in RELATIVE:
            raise ValueError(f"relative: 只能是 {RELATIVE}")
        if self.blocked not in ("exclude", "keep"):
            raise ValueError("blocked: 只能是 exclude 或 keep")
        if self.sell_blocked not in ("price", "drop"):
            raise ValueError("sell_blocked: 只能是 price 或 drop")
        if not isinstance(self.rank, bool):
            raise ValueError("rank: 必须是 bool")

    @property
    def span(self):
        """信号日到标签结束日的行数（lag + horizon）。切分的最小间隔由它推出。"""
        return self.lag + self.horizon

    def end_day(self, t):
        """信号日 t 的标签结束日（行号）。"""
        return np.asarray(t) + self.span

    def fields(self, panel=None):
        """计算标签需要的面板字段。"""
        names = [EXEC_PRICES[self.exec], "adj_factor"]
        if self.exec == "open" and (panel is None or _has_all(panel, MASKS)):
            names += list(MASKS)
        else:
            names += list(RULES)
        if self.relative == "industry":
            names.append("industry")
        return tuple(dict.fromkeys(names))

    def describe(self):
        return asdict(self)

    @property
    def digest(self):
        text = json.dumps(["label", 1, self.describe()], sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


PRESETS = {
    "ret1": Label(horizon=1),
    "ret5": Label(horizon=5),
    "ret10": Label(horizon=10),
    "ret20": Label(horizon=20),
}


def label(spec="ret5", **overrides):
    """取预设标签并按需改参数：`label("ret5", relative="industry", rank=True)`。"""
    if isinstance(spec, Label):
        return replace(spec, **overrides) if overrides else spec
    if spec not in PRESETS:
        raise ValueError(f"未知标签预设 {spec!r}；可用：{sorted(PRESETS)}")
    return replace(PRESETS[spec], **overrides)


def _has_all(panel, names):
    for name in names:
        try:
            panel[name]
        except (KeyError, FileNotFoundError):
            return False
    return True


def tradable(panel, exec="open"):
    """(can_buy, can_sell)，与 `ashare.backtest.market_inputs` 同一口径。"""
    price = EXEC_PRICES[exec]
    if exec == "open" and _has_all(panel, MASKS):
        return tuple(np.asarray(panel[m], dtype=bool) for m in MASKS)
    p = {name: np.asarray(panel[name]) for name in RULES}
    return tradability(
        np.asarray(panel[price]),
        p["high"],
        p["low"],
        p["upper_limit"],
        p["lower_limit"],
        p["is_suspended"] | ~p["has_quote"],
    )


@dataclass(frozen=True)
class LabelValues:
    """raw：原始未来收益（评估用）；target：训练目标；buyable：买入日可买（行 = 信号日）。"""

    raw: np.ndarray
    target: np.ndarray
    buyable: np.ndarray
    definition: Label

    def trainable(self, universe):
        """可进训练集的样本：在股票池内、目标有限，且（默认）买入日可买。"""
        ok = universe & np.isfinite(self.target)
        if self.definition.blocked == "exclude":
            ok &= self.buyable
        return ok


def compute(panel, definition, universe):
    """按定义算标签面板（T×N，行 = 信号日）。universe：T×N 布尔股票池。"""
    definition = label(definition)
    t, n = len(panel.dates), len(panel.symbols)
    universe = np.asarray(universe, dtype=bool)
    if universe.shape != (t, n):
        raise ValueError("universe: 形状必须与面板一致")
    price = np.asarray(panel[EXEC_PRICES[definition.exec]], dtype=np.float64)
    adjusted = price * np.asarray(panel["adj_factor"], dtype=np.float64)
    adjusted = np.where(np.isfinite(adjusted) & (adjusted > 0), adjusted, np.nan)
    can_buy, can_sell = tradable(panel, definition.exec)
    lag, span = definition.lag, definition.span
    raw = np.full((t, n), np.nan)
    buyable = np.zeros((t, n), dtype=bool)
    if t > span:
        raw[: t - span] = adjusted[span:] / adjusted[lag : t - span + lag] - 1
        buyable[: t - span] = can_buy[lag : t - span + lag]
        if definition.sell_blocked == "drop":
            raw[: t - span][~can_sell[span:]] = np.nan
    raw[~universe] = np.nan
    target = raw
    if definition.relative == "market":
        target = panelops.cs_demean(raw, mask=universe)
    elif definition.relative == "industry":
        groups = np.ascontiguousarray(panel["industry"], dtype=np.int64)
        target = panelops.cs_group_demean(raw, groups=groups, mask=universe)
    if definition.rank:
        target = panelops.cs_rank(target, normalize=True, mask=universe) - 0.5
    return LabelValues(raw, np.ascontiguousarray(target), buyable, definition)
