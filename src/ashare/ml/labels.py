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
  `"drop"` 把标签置缺失；`"carry"` 顺延到之后第一个可卖且有价的日子卖出，最多顺延
  `max_carry` 天，仍卖不出则标签缺失。成交价缺失（停牌、退市）时：price、drop 标签缺失；
  carry 同样顺延。
- carry 时标签的最晚结束日是 t+lag+h+max_carry，所以 `span` 也加上 max_carry，切分的最小
  间隔随之变大；每个样本的实际卖出日记在 `LabelValues.end`，防前视检查按它核对。
- 末尾 span 行没有标签。
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
SELL_BLOCKED = ("price", "drop", "carry")


@dataclass(frozen=True)
class Label:
    """标签定义。horizon：持有交易日数；lag：信号日到买入日的行数（同回测的 lag）。

    relative：`"none"` 原始收益，`"market"` 扣除股票池等权均值，`"industry"` 扣除
    同行业均值（行业取信号日）。训练目标的截面变换（在 relative 之后做，二选一）：
    rank=True 改为当日截面排名（居中到 ±0.5），zscore=True 改为当日截面标准分。
    默认都不做，训练目标就是原始收益。评估（Rank IC、分组收益）始终用原始收益。
    sell_blocked、max_carry 见模块说明。

    新增的选项取默认值时不进 `describe()`，所以不改变已有定义的身份（试验编号）。
    """

    horizon: int = 5
    lag: int = 1
    exec: str = "open"
    relative: str = "none"
    rank: bool = False
    blocked: str = "exclude"
    sell_blocked: str = "price"
    zscore: bool = False
    max_carry: int = 10

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
        if self.sell_blocked not in SELL_BLOCKED:
            raise ValueError(f"sell_blocked: 只能是 {SELL_BLOCKED}")
        for name in ("rank", "zscore"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name}: 必须是 bool")
        if self.rank and self.zscore:
            raise ValueError("rank 与 zscore 只能选一个")
        k = self.max_carry
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("max_carry: 必须是正整数")

    @property
    def carry(self):
        """卖不出时最多顺延的天数（只有 sell_blocked="carry" 时非 0）。"""
        return self.max_carry if self.sell_blocked == "carry" else 0

    @property
    def span(self):
        """信号日到标签最晚结束日的行数（lag + horizon，carry 时再加 max_carry）。
        切分的最小间隔由它推出。"""
        return self.lag + self.horizon + self.carry

    def end_day(self, t):
        """信号日 t 的标签最晚结束日（行号）。实际卖出日见 `LabelValues.end`。"""
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
        d = asdict(self)
        # 后加的选项取默认值时不进定义，已有试验的编号不变。
        if not d["zscore"]:
            del d["zscore"]
        if self.sell_blocked != "carry":
            del d["max_carry"]
        return d

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
    """raw：原始未来收益（评估用）；target：训练目标；buyable：买入日可买（行 = 信号日）；
    end：实际卖出日的行号（int32，没有标签的格子为 -1）。"""

    raw: np.ndarray
    target: np.ndarray
    buyable: np.ndarray
    definition: Label
    end: np.ndarray | None = None

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
    lag, hold = definition.lag, definition.lag + definition.horizon
    span = definition.span
    raw = np.full((t, n), np.nan)
    end = np.full((t, n), -1, dtype=np.int32)
    buyable = np.zeros((t, n), dtype=bool)
    if t > span:
        m = t - span  # 有完整标签窗口的信号日数
        buy = adjusted[lag : m + lag]
        buyable[:m] = can_buy[lag : m + lag]
        if definition.sell_blocked == "carry":
            # 从 t+lag+h 起逐日找第一个可卖且有价的日子，最多到 t+lag+h+max_carry。
            sell = np.full((m, n), np.nan)
            when = np.full((m, n), -1, dtype=np.int32)
            for d in range(definition.carry + 1):
                rows = slice(hold + d, hold + d + m)
                ok = np.isnan(sell) & can_sell[rows] & np.isfinite(adjusted[rows])
                sell[ok] = adjusted[rows][ok]
                day = np.arange(hold + d, hold + d + m)[:, None]
                when[ok] = np.broadcast_to(day, (m, n))[ok]
            raw[:m] = sell / buy - 1
            end[:m] = when
        else:
            raw[:m] = adjusted[hold : hold + m] / buy - 1
            if definition.sell_blocked == "drop":
                raw[:m][~can_sell[hold : hold + m]] = np.nan
            end[:m] = np.arange(hold, hold + m)[:, None]
    raw[~universe] = np.nan
    end[~np.isfinite(raw)] = -1
    target = raw
    if definition.relative == "market":
        target = panelops.cs_demean(raw, mask=universe)
    elif definition.relative == "industry":
        groups = np.ascontiguousarray(panel["industry"], dtype=np.int64)
        target = panelops.cs_group_demean(raw, groups=groups, mask=universe)
    if definition.rank:
        target = panelops.cs_rank(target, normalize=True, mask=universe) - 0.5
    elif definition.zscore:
        target = panelops.cs_zscore(target, mask=universe)
    return LabelValues(raw, np.ascontiguousarray(target), buyable, definition, end)
