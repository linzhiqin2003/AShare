"""常见量价特征：`default_registry` 里的示例因子，加二十多个常见写法（动量、反转、
波动、换手、量价相关、K 线位置等）。全部只用信号日及以前的数据；用作机器学习的
起步特征集和真实数据例子，不代表这些特征已经单独检验过。"""

from __future__ import annotations

import math

from ..factor import adjusted_close, call, default_registry, field, returns


def _roll(op, *args, **params):
    *panels, window = args
    return call(
        op, *panels, window=window, min_periods=math.ceil(0.8 * window), **params
    )


def extra_features():
    """自己补的二十多个量价特征 {名字: Expr}。"""
    price = adjusted_close()
    ret = returns()
    delay = lambda x, n: call("ts_delay", x, periods=n)  # noqa: E731
    volume, amount, turnover = field("volume"), field("amount"), field("turnover")
    high, low, close, open_ = field("high"), field("low"), field("close"), field("open")
    f = {}
    for w in (60, 120, 250):
        f[f"mom_{w}"] = price / delay(price, w) - 1
    f["mom_60_5"] = delay(price, 5) / delay(price, 60) - 1
    f["reversal_1"] = -ret
    f["vol_60"] = _roll("ts_std", ret, 60)
    f["vol_ratio_20_60"] = _roll("ts_std", ret, 20) / _roll("ts_std", ret, 60)
    f["downside_vol_20"] = _roll("ts_std", call("minimum", ret, 0.0), 20)
    f["skew_20"] = _roll("ts_skew", ret, 20)
    f["kurt_20"] = _roll("ts_kurt", ret, 20)
    f["turnover_5"] = _roll("ts_mean", turnover, 5)
    f["turnover_ratio_5_60"] = _roll("ts_mean", turnover, 5) / _roll(
        "ts_mean", turnover, 60
    )
    f["turnover_std_20"] = _roll("ts_std", turnover, 20)
    f["corr_price_volume_20"] = _roll("ts_corr", price, volume, 20)
    f["corr_ret_volume_20"] = _roll("ts_corr", ret, call("log", volume + 1.0), 20)
    f["volume_ratio_5_60"] = _roll("ts_mean", volume, 5) / _roll("ts_mean", volume, 60)
    f["amount_ratio_5_60"] = _roll("ts_mean", amount, 5) / _roll("ts_mean", amount, 60)
    f["range_20"] = _roll("ts_mean", (high - low) / close, 20)
    lo20, hi20 = (
        _roll("ts_min", low * field("adj_factor"), 20),
        _roll("ts_max", high * field("adj_factor"), 20),
    )
    f["position_20"] = (price - lo20) / (hi20 - lo20 + 1e-12)
    f["ma_gap_20"] = price / _roll("ts_mean", price, 20) - 1
    f["ma_gap_60"] = price / _roll("ts_mean", price, 60) - 1
    f["drawdown_60"] = price / _roll("ts_max", price, 60) - 1
    f["intraday_20"] = _roll("ts_mean", close / open_ - 1, 20)
    f["overnight_20"] = _roll("ts_mean", open_ / field("previous_close") - 1, 20)
    f["vwap_gap_5"] = _roll("ts_mean", close / field("vwap") - 1, 5)
    f["ts_rank_price_20"] = _roll("ts_rank", price, 20, normalize=True)
    f["log_price"] = call("log", close)
    f["free_float_ratio"] = field("free_shares") / field("total_shares")
    return f


def basic_features():
    """`default_registry()` 的因子 + `extra_features()`，{名字: Expr}，顺序固定。"""
    out = {f.name: f.expression for f in default_registry()}
    out.update(extra_features())
    return out
