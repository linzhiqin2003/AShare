"""把数据层面板、`panelbt.Result` 和风险模型的输入整理成归因要用的数组。

时间约定（全包统一，防前视）：
- `weights[t]` 是第 t 日**收盘后**的持仓权重，它赚的是第 t+1 日的收益。
- 内部一律换成「第 t 日持有的权重」`held[t] = weights[t-1]`，与 `returns[t]`
  （t-1 收盘到 t 收盘的总收益）相乘；`held[0] = 0`。
- 风险模型的暴露 `B[t]` 用第 t 日收盘的信息，解释第 t+1 日的收益，与 `held[t+1]` 同一时点。
"""

from __future__ import annotations

import numpy as np

from .adapters import INDEX_FIELDS


class LazyPanel:
    """按需读取字段的面板视图：`dates`、`symbols`、`panel[field]`。

    ```python
    import ashare.data, ashare.analysis as an
    panel = an.LazyPanel(ashare.data.open(), start="2016-01-04")
    panel["close"]          # T×N，第一次访问时才读
    ```
    """

    def __init__(self, data, start=None, end=None, symbols=None):
        self._data, self._args = data, (start, end, symbols)
        probe = data.panel(["close"], start=start, end=end, symbols=symbols)
        self.dates = np.asarray(probe.dates)
        self.symbols = tuple(map(str, probe.symbols))
        self.version = getattr(data, "version", "")
        self._cache = {"close": probe["close"]}

    def __getitem__(self, name):
        if name not in self._cache:
            start, end, symbols = self._args
            self._cache[name] = self._data.panel(
                [name], start=start, end=end, symbols=symbols
            )[name]
        return self._cache[name]

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)


def field(panel, name, dtype=float):
    try:
        value = panel[name]
    except KeyError:
        raise KeyError(
            f"面板缺少字段 {name!r}；请用 ashare.analysis.LazyPanel(ashare.data.open()) "
            "或在 data.panel([...]) 里带上它"
        ) from None
    return np.asarray(value, dtype=dtype)


def carry_forward(values):
    """沿时间轴用最近一个有效值填 NaN（panelbt 估值价的口径）。"""
    v = np.array(values, dtype=float, copy=True)
    valid = np.isfinite(v)
    index = np.where(valid, np.arange(len(v))[:, None], 0)
    np.maximum.accumulate(index, axis=0, out=index)
    out = v[index, np.arange(v.shape[1])[None, :]]
    seen = np.maximum.accumulate(valid, axis=0)
    out[~seen] = np.nan
    return out


def total_returns(panel):
    """t-1 收盘到 t 收盘的总收益（含分红送转，按复权因子），T×N。

    停牌或缺价沿用最近收盘价（收益为 0，与 panelbt 用最近有效收盘价估值一致）；
    上市前为 NaN；第 0 行为 NaN。
    """
    price = carry_forward(field(panel, "close")) * carry_forward(
        field(panel, "adj_factor")
    )
    r = np.full_like(price, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = price[1:] / price[:-1] - 1
    r[1:][~(price[:-1] > 0)] = np.nan
    return r


def held_from_weights(weights):
    """收盘后权重 `weights[t]` → 第 t 日持有的权重 `held[t] = weights[t-1]`。"""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or not np.isfinite(w).all():
        raise ValueError("weights: 需要有限的 T×N 权重")
    held = np.zeros_like(w)
    held[1:] = w[:-1]
    return held


def result_weights(result, panel):
    """`panelbt.Result`（须 record_trades=True）→ 收盘后权重 T×N 与各项日序列。

    权重 = 持仓股数 × 收盘价（缺价沿用最近有效值）÷ 当日权益。返回 (weights, info)，
    info 里是组合日收益、成本率和权益（成本率以收益分母 E[t-1]+资金流 为分母）。
    """
    if getattr(result, "positions", None) is None:
        raise ValueError("Result 没有持仓：回测时请设 Config(record_trades=True)")
    q = np.asarray(result.positions, dtype=float)
    if q.ndim != 2:
        raise ValueError("只支持单策略 Result；批量结果请逐个取出")
    t, n = q.shape
    if (t, n) != (len(panel.dates), len(panel.symbols)):
        raise ValueError(f"Result 形状 {q.shape} 与面板 {(t, n)} 不一致")
    close = carry_forward(field(panel, "close"))
    equity = np.asarray(result.equity, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = np.where(q != 0, q * close, 0.0) / equity[:, None]
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    flow = np.asarray(result.cash_flow, dtype=float)
    denom = np.empty(t)
    denom[1:] = equity[:-1] + flow[1:]
    # E[-1] 是初始资金：由第 0 日收益反推（E0 / (1 + r0)），不必另传 Config。
    r0 = float(result.returns[0])
    denom[0] = equity[0] / (1 + r0) if r0 != -1 else np.nan
    cost = (
        np.asarray(result.commission)
        + np.asarray(result.tax)
        + np.asarray(result.other_fee)
        + np.asarray(result.slippage_cost)
        + np.asarray(result.impact_cost)
        + np.asarray(result.borrow_cost)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        cost_rate = np.where(denom != 0, cost / denom, 0.0)
    info = {
        "returns": np.asarray(result.returns, dtype=float),
        "cost_rate": cost_rate,
        "equity": equity,
        "denominator": denom,
    }
    return weights, info


def benchmark_held(panel, benchmark, returns):
    """第 t 日持有的基准权重 T×N。

    `benchmark` 可以是指数代码（"000300.SH"）、面板权重字段名或收盘后权重数组。
    数据层的指数权重是月末快照、从下一交易日起可用并沿用到下一份快照；这里在两份
    快照之间按成分股的收益漂移（不漂移就等于每天按快照再平衡，不是指数的真实持仓）。
    第 t 行出现新快照时，它就是 t 日持有的权重（快照描述的是此前月末收盘的权重）。
    """
    if benchmark is None:
        return None
    if isinstance(benchmark, str):
        name = INDEX_FIELDS.get(benchmark, benchmark)
        raw = np.nan_to_num(field(panel, name))
        if raw.shape != returns.shape:
            raise ValueError("基准权重字段与面板形状不一致")
        r = np.nan_to_num(returns)
        held = np.zeros_like(raw)
        change = np.ones(len(raw), bool)
        change[1:] = np.any(raw[1:] != raw[:-1], axis=1)
        for day in range(len(raw)):
            total = raw[day].sum()
            if change[day] or day == 0:
                held[day] = raw[day] / total if total > 0 else 0.0
            else:
                grown = held[day - 1] * (1 + r[day - 1])
                s = grown.sum()
                held[day] = grown / s if s > 0 else 0.0
        return held
    return held_from_weights(benchmark)


def industry_codes(panel, rows=None):
    """面板的申万一级行业编码 → (逐日整数类别 T×N，类别名)。0 / 缺失记为「未知」。"""
    raw = np.nan_to_num(field(panel, "industry")).astype(np.int64)
    codes = np.unique(raw[raw > 0])
    names = tuple(str(c) for c in codes) + ("unknown",)
    lookup = {int(c): i for i, c in enumerate(codes)}
    out = np.full(raw.shape, len(codes), dtype=np.int64)
    for code, i in lookup.items():
        out[raw == code] = i
    return out, names
