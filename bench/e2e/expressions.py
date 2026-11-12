"""剖析用的代表性因子表达式：仿 Alpha101、Alpha158、国泰君安 191 的常见写法。

只用于测量求值耗时的分布，不代表这些因子已经复现或检验过；覆盖时序、截面、
嵌套和条件四类算子，名字前缀标出来源风格。
"""

from ashare.factor import adjusted_close, call, clip, field, neutralize, returns


def _ts(op, *args, **params):
    *panels, window = args
    return call(op, *panels, window=window, **params)


def _cs(op, x, **params):
    return call(op, x, **params)


def expressions():
    """返回 {名字: Expr}，顺序固定。"""
    close = adjusted_close()
    adj = field("adj_factor")
    open_ = field("open") * adj
    high = field("high") * adj
    low = field("low") * adj
    vwap = field("vwap") * adj
    volume = field("volume")
    amount = field("amount")
    turnover = field("turnover")
    cap = field("total_market_cap")
    industry = field("industry", dtype="int64")
    ret = returns()
    delay = lambda x, n: call("ts_delay", x, periods=n)  # noqa: E731
    delta = lambda x, n: call("ts_delta", x, periods=n)  # noqa: E731
    rank = lambda x: _cs("cs_rank", x, normalize=True)  # noqa: E731
    up = call("gt", ret, 0.0)
    e = {}

    # Alpha158 风格：K 线形态与滚动统计
    e["a158_kmid"] = (close - open_) / open_
    e["a158_klen"] = (high - low) / open_
    e["a158_kup"] = (high - call("maximum", open_, close)) / open_
    e["a158_ksft"] = (2 * close - high - low) / open_
    for w in (5, 20, 60):
        e[f"a158_roc{w}"] = delay(close, w) / close
        e[f"a158_ma{w}"] = _ts("ts_mean", close, w) / close
        e[f"a158_std{w}"] = _ts("ts_std", close, w) / close
    e["a158_beta20"] = _ts("ts_slope", close, 20) / close
    e["a158_rsqr20"] = _ts("ts_rsquare", close, 20)
    e["a158_resi20"] = _ts("ts_residual", close, 20) / close
    e["a158_max20"] = _ts("ts_max", high, 20) / close
    e["a158_min20"] = _ts("ts_min", low, 20) / close
    e["a158_qtlu20"] = _ts("ts_quantile", close, 20, q=0.8) / close
    e["a158_rank20"] = _ts("ts_rank", close, 20, normalize=True)
    e["a158_rsv20"] = (close - _ts("ts_min", low, 20)) / (
        _ts("ts_max", high, 20) - _ts("ts_min", low, 20) + 1e-12
    )
    e["a158_imax20"] = _ts("ts_argmax", high, 20) / 20.0
    e["a158_imxd20"] = (_ts("ts_argmax", high, 20) - _ts("ts_argmin", low, 20)) / 20.0
    e["a158_corr20"] = _ts("ts_corr", close, call("log", volume + 1.0), 20)
    e["a158_cord20"] = _ts(
        "ts_corr", close / delay(close, 1), call("log", volume / delay(volume, 1)), 20
    )
    e["a158_cntp20"] = _ts("ts_mean", up, 20)
    e["a158_sump20"] = _ts("ts_sumif", call("abs", ret), up, 20) / (
        _ts("ts_sum", call("abs", ret), 20) + 1e-12
    )
    e["a158_vma20"] = _ts("ts_mean", volume, 20) / (volume + 1e-12)
    e["a158_vstd20"] = _ts("ts_std", volume, 20) / (volume + 1e-12)
    e["a158_wvma20"] = _ts("ts_std", call("abs", ret) * volume, 20) / (
        _ts("ts_mean", call("abs", ret) * volume, 20) + 1e-12
    )
    e["a158_skew60"] = _ts("ts_skew", ret, 60)
    e["a158_kurt60"] = _ts("ts_kurt", ret, 60)
    e["a158_med20"] = _ts("ts_median", ret, 20)
    e["a158_mad20"] = _ts("ts_mad", ret, 20)

    # Alpha101 风格：截面秩、时序相关与嵌套
    e["a101_001"] = (
        rank(
            _ts(
                "ts_argmax",
                call(
                    "signed_power",
                    call("where", call("lt", ret, 0.0), _ts("ts_std", ret, 20), close),
                    2.0,
                ),
                5,
            )
        )
        - 0.5
    )
    e["a101_002"] = -_ts(
        "ts_corr",
        rank(delta(call("log", volume + 1.0), 2)),
        rank((close - open_) / open_),
        6,
    )
    e["a101_003"] = -_ts("ts_corr", rank(open_), rank(volume), 10)
    e["a101_004"] = -_ts("ts_rank", rank(low), 9, normalize=True)
    e["a101_006"] = -_ts("ts_corr", open_, volume, 10)
    e["a101_009"] = call(
        "where",
        call("lt", 0.0, _ts("ts_min", delta(close, 1), 5)),
        delta(close, 1),
        call(
            "where",
            call("lt", _ts("ts_max", delta(close, 1), 5), 0.0),
            delta(close, 1),
            -delta(close, 1),
        ),
    )
    e["a101_012"] = call("sign", delta(volume, 1)) * -delta(close, 1)
    e["a101_013"] = -rank(_ts("ts_cov", rank(close), rank(volume), 5))
    e["a101_014"] = -rank(delta(ret, 3)) * _ts("ts_corr", open_, volume, 10)
    e["a101_016"] = -rank(_ts("ts_cov", rank(high), rank(volume), 5))
    e["a101_018"] = -rank(
        _ts("ts_std", call("abs", close - open_), 5)
        + (close - open_)
        + _ts("ts_corr", close, open_, 10)
    )
    e["a101_020"] = (
        -rank(open_ - delay(high, 1))
        * rank(open_ - delay(close, 1))
        * rank(open_ - delay(low, 1))
    )
    e["a101_026"] = -_ts(
        "ts_max",
        _ts(
            "ts_corr",
            _ts("ts_rank", volume, 5, normalize=True),
            _ts("ts_rank", high, 5, normalize=True),
            5,
        ),
        3,
    )
    e["a101_033"] = rank(-(1.0 - open_ / close))
    e["a101_034"] = rank(
        (1.0 - rank(_ts("ts_std", ret, 2) / (_ts("ts_std", ret, 5) + 1e-12)))
        + (1.0 - rank(delta(close, 1)))
    )
    e["a101_041"] = call("power", high * low, 0.5) - vwap
    e["a101_042"] = rank(vwap - close) / rank(vwap + close)
    e["a101_044"] = -_ts("ts_corr", high, rank(volume), 5)
    e["a101_053"] = -delta(((close - low) - (high - close)) / (close - low + 1e-12), 9)
    e["a101_054"] = -((low - close) * call("power", open_, 5.0)) / (
        (low - high - 1e-12) * call("power", close, 5.0)
    )
    e["a101_101"] = (close - open_) / (high - low + 0.001)
    e["a101_decay"] = _ts("ts_decay_linear", rank(delta(vwap, 1)), 10)

    # 国泰君安 191 风格：条件累计、EMA、回归
    e["gtja_014"] = close - delay(close, 5)
    e["gtja_018"] = close / delay(close, 5)
    e["gtja_031"] = (close - _ts("ts_mean", close, 12)) / _ts("ts_mean", close, 12)
    e["gtja_034"] = _ts("ts_mean", close, 12) / close
    e["gtja_046"] = (
        _ts("ts_mean", close, 3)
        + _ts("ts_mean", close, 6)
        + _ts("ts_mean", close, 12)
        + _ts("ts_mean", close, 24)
    ) / (4.0 * close)
    e["gtja_053"] = _ts("ts_sum", up, 12)
    e["gtja_063"] = _ts("ts_ema", call("maximum", close - delay(close, 1), 0.0), 6) / (
        _ts("ts_ema", call("abs", close - delay(close, 1)), 6) + 1e-12
    )
    e["gtja_079"] = _ts("ts_ema", call("maximum", close - delay(close, 1), 0.0), 12) / (
        _ts("ts_ema", call("abs", close - delay(close, 1)), 12) + 1e-12
    )
    e["gtja_088"] = (close - delay(close, 20)) / delay(close, 20)
    e["gtja_096"] = _ts(
        "ts_ema",
        _ts(
            "ts_ema",
            (close - _ts("ts_min", low, 9))
            / (_ts("ts_max", high, 9) - _ts("ts_min", low, 9) + 1e-12),
            3,
        ),
        3,
    )
    e["gtja_106"] = close - delay(close, 20)
    e["gtja_150"] = (close + high + low) / 3.0 * volume
    e["gtja_beta60"] = _ts(
        "ts_regression_slope",
        ret,
        _cs("cs_demean", ret),
        60,
    )
    e["gtja_resid60"] = _ts("ts_regression_residual", ret, _cs("cs_demean", ret), 60)

    # 截面处理与中性化
    log_cap = call("log", cap)
    e["cs_turnover_z"] = _cs(
        "cs_zscore", _cs("cs_winsor_mad", _ts("ts_mean", turnover, 20))
    )
    e["cs_mom_ind"] = call(
        "cs_group_demean", rank(delay(close, 21) / delay(close, 252)), industry
    )
    e["cs_rev_neutral"] = neutralize(
        _cs("cs_winsor_quantile", -returns(5)), log_cap, groups=industry
    )
    e["cs_vol_neutral"] = neutralize(
        _cs("cs_zscore", _ts("ts_std", ret, 60)), log_cap, groups=industry
    )
    e["cs_ep_scaled"] = _cs("cs_scale", 1 / field("pe_ttm"))
    e["cs_liquidity"] = clip(
        _cs("cs_zscore", call("log", _ts("ts_mean", amount, 20) + 1.0)), -3.0, 3.0
    )
    e["ts_ffill_bp"] = call("ts_ffill", 1 / field("pb"), limit=20)
    e["ts_zscore_vol"] = _ts("ts_zscore", volume, 20)
    e["ts_idxmax_ret"] = _ts("ts_idxmax", ret, 20)
    e["ts_product_ret"] = _ts("ts_product", 1.0 + ret, 10)
    e["ts_var_ret"] = _ts("ts_var", ret, 120)
    return e
