"""补充算子的定义表与可组合实现。

本模块在 ``ops`` 之外补充一批算子，按项目「公式层不写新内核、只用现有算子精确组合」
的约定做三件事：

1. ``FORMULAS``：每个算子一句话或公式（按公式重写）；
2. ``BUILDERS``：能用现有 ``panelops`` 算子精确拼出来的，给出 ``Expr`` 构造器；
3. ``UNSUPPORTED``：需要新内核的，注明原因，``build`` 时明确报错。

与内核同名算子的语义差异记在 ``CONFLICTS``，且**只在本模块内生效**：内核里
``ts_rank`` / ``ts_quantile`` / ``ts_decay_linear`` / ``scale`` 的原语义不改，避免
破坏现有测试、缓存键和基准。``ts_range`` 是本模块新增的算子（``ts_max - ts_min``）。

本文件未在本地构建验证（开发机没有 Rust 工具链，``make dev`` 不可用）；提交前请在
项目环境跑 ``make check``。
"""

from ..expression import Expr, call, field, neutralize

# name -> 一句话或公式（按公式重写）
FORMULAS = {
    # 算术
    "exp": "e^x",
    "sign": "+1 / 0 / -1，按符号；缺失仍缺失",
    "log": "ln(x)",
    "max": "max(x, y, ...)",
    "abs": "|x|",
    "sigmoid": "1 / (1 + e^(-x))",
    "divide": "x / y",
    "min": "min(x, y, ...)",
    "signed_power": "sign(x) * |x|^y",
    "inverse": "1 / x",
    "round": "四舍五入到整数",
    "sqrt": "sqrt(x)，x < 0 时为 NaN",
    "s_log_1p": "sign(x) * ln(1 + |x|)",
    "reverse": "-x",
    "log_diff": "ln(x_t) - ln(x_{t-1})",
    "power": "x^y",
    # 逻辑
    "if_else": "cond ? a : b",
    # 时序
    "ts_corr": "窗口 Pearson 相关",
    "ts_zscore": "(x - mean_d) / std_d",
    "ts_returns": "(x_t - x_{t-d}) / x_{t-d}；mode=2 时分母取两端均值",
    "ts_std_dev": "窗口标准差",
    "ts_backfill": "窗口 lookback 内最近的有效值；k 表示第 k 近",
    "last_diff_value": "窗口内最近一个与当日不同的值",
    "ts_scale": "(x - min_d) / (max_d - min_d) + c",
    "ts_entropy": "窗口直方图分布的信息熵",
    "ts_vector_neut": "x - proj_y(x)，窗口内投影",
    "ts_sum": "窗口求和",
    "ts_decay_exp_window": "sum(f^i * x_{t-i}) / sum(f^i)，i = 0..d-1",
    "ts_av_diff": "x - mean_d",
    "ts_kurtosis": "窗口四阶标准化中心矩 - 3",
    "ts_mean": "窗口均值",
    "ts_min_max_diff": "x - f * (min_d + max_d)",
    "ts_arg_max": "窗口最大值距今天数",
    "ts_max": "窗口最大值",
    "ts_min_max_cps": "(min_d + max_d) - f * x",
    "ts_rank": "窗口内秩，归一化到 [0, 1] 再加常数 c",
    "ts_ir": "mean_d / std_d",
    "ts_delay": "x_{t-d}",
    "ts_weighted_decay": "k * x_t + (1 - k) * x_{t-1}",
    "ts_quantile": "窗口秩经指定分布的逆分布函数映射",
    "ts_min": "窗口最小值",
    "ts_count_nans": "窗口内缺失值个数",
    "ts_min_diff": "x - min_d",
    "ts_decay_linear": "sum(i * x_{t-d+i}) / sum(i)，i = 1..d；dense=false 时缺失按 0",
    "ts_arg_min": "窗口最小值距今天数",
    "ts_regression": "窗口 OLS 的 beta / alpha / 残差等（rettype 选）",
    "ts_skewness": "窗口三阶标准化中心矩",
    "ts_max_diff": "x - max_d",
    "ts_median": "窗口中位数",
    "ts_delta": "x_t - x_{t-d}",
    # 截面
    "winsorize": "截面裁剪到 mean ± k * std",
    "rank": "截面秩，映射到 [0, 1]",
    "vector_neut": "x - proj_y(x)，截面投影",
    "regression_neut": "y - (a + b * x)，截面 OLS 残差",
    "zscore": "(x - mean) / std，截面",
    "scale": "x / sum|x|，可分别指定长、短账本",
    "rank_by_side": "正负两边分别截面排名后按账本缩放",
    "normalize": "截面去均值；可选除 std、再截断到 ±limit",
    "quantile": "截面秩经指定分布的逆分布函数映射",
    # 变换
    "bucket": "截面秩分段成组号",
    "trade_when": "exit ? NaN : (trigger ? y : 前值)",
    # 分组
    "group_mean": "组内均值（调和均值口径）",
    "group_median": "组内中位数",
    "group_rank": "组内秩，映射到 [0, 1]",
    "group_normalize": "组内 x / sum|x|",
    "group_zscore": "组内 (x - mean) / std",
    "group_sum": "组内求和",
    "group_neutralize": "组内去均值",
    # 新增
    "ts_range": "max_d - min_d",
}


def _shift(x, periods):
    return call("ts_delay", x, periods=periods)


# ---- 直接对应内核算子 -------------------------------------------------------


def exp(x):
    return call("exp", x)


def sign(x):
    return call("sign", x)


def log(x):
    return call("log", x)


def abs_(x):
    return call("abs", x)


def sqrt(x):
    return call("sqrt", x)


def power(x, y):
    return call("power", x, y)


def signed_power(x, y):
    return call("signed_power", x, y)


def divide(x, y):
    return call("divide", x, y)


def reverse(x):
    return call("negate", x)


def if_else(cond, a, b):
    return call("where", cond, a, b)


def _fold(op, args):
    if len(args) < 2:
        raise ValueError(f"{op} 至少需要 2 个输入")
    out = args[0]
    for item in args[1:]:
        out = call(op, out, item)
    return out


def max_(*args):
    return _fold("maximum", args)


def min_(*args):
    return _fold("minimum", args)


def ts_sum(x, window=20):
    return call("ts_sum", x, window=window)


def ts_mean(x, window=20):
    return call("ts_mean", x, window=window)


def ts_std_dev(x, window=20):
    return call("ts_std", x, window=window)


def ts_min(x, window=20):
    return call("ts_min", x, window=window)


def ts_max(x, window=20):
    return call("ts_max", x, window=window)


def ts_median(x, window=20):
    return call("ts_median", x, window=window)


def ts_delta(x, window=1):
    return call("ts_delta", x, periods=window)


def ts_delay(x, window=1):
    return call("ts_delay", x, periods=window)


def ts_corr(x, y, window=20):
    return call("ts_corr", x, y, window=window)


def ts_zscore(x, window=20):
    return call("ts_zscore", x, window=window)


def ts_skewness(x, window=20):
    return call("ts_skew", x, window=window)


def ts_kurtosis(x, window=20):
    return call("ts_kurt", x, window=window)


def ts_arg_max(x, window=20):
    return call("ts_argmax", x, window=window)


def ts_arg_min(x, window=20):
    return call("ts_argmin", x, window=window)


# ---- 用现有算子组合 ---------------------------------------------------------


def ts_range(x, window=20):
    """max_d - min_d。"""
    return call("ts_max", x, window=window) - call("ts_min", x, window=window)


def ts_ir(x, window=20):
    """mean_d / std_d。"""
    return call("ts_mean", x, window=window) / call("ts_std", x, window=window)


def ts_av_diff(x, window=20):
    """x - mean_d。"""
    return x - call("ts_mean", x, window=window)


def ts_min_diff(x, window=20):
    """x - min_d。"""
    return x - call("ts_min", x, window=window)


def ts_max_diff(x, window=20):
    """x - max_d。"""
    return x - call("ts_max", x, window=window)


def ts_min_max_diff(x, window=20, f=0.5):
    """x - f * (min_d + max_d)。"""
    return x - f * (call("ts_min", x, window=window) + call("ts_max", x, window=window))


def ts_min_max_cps(x, window=20, f=2.0):
    """(min_d + max_d) - f * x。"""
    return (call("ts_min", x, window=window) + call("ts_max", x, window=window)) - f * x


def log_diff(x):
    """ln(x_t) - ln(x_{t-1})。"""
    return call("log", x) - call("log", _shift(x, 1))


def s_log_1p(x):
    """sign(x) * ln(1 + |x|)。"""
    return call("sign", x) * call("log", 1 + call("abs", x))


def inverse(x):
    """1 / x。"""
    return 1 / x


def ts_returns(x, window=1, mode=1):
    """(x_t - x_{t-d}) / x_{t-d}；mode=2 时分母取两端均值。"""
    past = _shift(x, window)
    if mode == 1:
        return (x - past) / past
    if mode == 2:
        return (x - past) / ((x + past) / 2)
    raise ValueError("mode 只能是 1 或 2")


def ts_weighted_decay(x, k=0.5):
    """k * x_t + (1 - k) * x_{t-1}。"""
    return k * x + (1 - k) * _shift(x, 1)


def ts_scale(x, window=20, constant=0.0):
    """(x - min_d) / (max_d - min_d) + c。"""
    low = call("ts_min", x, window=window)
    high = call("ts_max", x, window=window)
    return (x - low) / (high - low) + constant


def ts_count_nans(x, window=20):
    """窗口内缺失值个数。"""
    return call("ts_sum", call("is_nan", x), window=window, min_periods=1)


def ts_decay_linear(x, window=20, dense=False):
    """线性衰减均值；dense=False 时缺失按 0（先填 0 再算）。"""
    filled = x if dense else call("fill_nan", x, 0.0)
    return call("ts_decay_linear", filled, window=window, min_periods=window)


def ts_decay_exp_window(x, window=20, factor=0.5):
    """sum(f^i * x_{t-i}) / sum(f^i)。"""
    if not 0.0 < factor < 1.0:
        raise ValueError("factor 须在 (0, 1)")
    total = None
    weight = 0.0
    for i in range(window):
        term = x if i == 0 else _shift(x, i)
        w = factor**i
        total = w * term if total is None else total + w * term
        weight += w
    return total / weight


def ts_rank(x, window=20, constant=0.0):
    """窗口秩归一化到 [0, 1] 后加常数。"""
    return call("ts_rank", x, window=window, normalize=True) + constant


def ts_backfill(x, lookback=252, k=1):
    """最近有效值；只实现 k=1（等价 ts_ffill(limit)）。"""
    if k != 1:
        raise NotImplementedError("k != 1 需要状态型内核，暂未实现")
    return call("ts_ffill", x, limit=lookback)


def normalize(x, use_std=False, limit=0.0):
    """截面去均值；use_std 时除 std；limit 非零时截断到 ±limit。"""
    out = call("cs_zscore", x) if use_std else call("cs_demean", x)
    if limit:
        out = call("clip", out, lower=-limit, upper=limit)
    return out


def vector_neut(x, y):
    """x 对 y 的截面投影取残差。"""
    return neutralize(x, y)


def regression_neut(y, x):
    """y 对 x 的截面 OLS 残差。"""
    return neutralize(y, x)


def group_neutralize(x, group):
    """按组去均值。"""
    return neutralize(x, groups=group)


def ts_vector_neut(x, y, window=20):
    """x 减去在窗口内对 y 的投影（用斜率乘 y 近似）。"""
    slope = call("ts_regression_slope", x, y, window=window)
    return x - slope * y


# ---- 需要新内核，明确报错 ---------------------------------------------------

UNSUPPORTED = {
    "sigmoid": "需要新增逐元素算子",
    "round": "需要新增逐元素算子",
    "ts_entropy": "需要窗口直方图/熵内核",
    "last_diff_value": "需要状态型扫描内核",
    "ts_quantile": "需要秩到逆分布的映射内核",
    "quantile": "需要截面秩到逆分布的映射内核",
    "winsorize": "需要截面均值/标准差口径的裁剪内核",
    "scale": "需要截面绝对值和的归约内核",
    "rank_by_side": "需要分边排名与账本缩放内核",
    "bucket": "需要分组建模内核",
    "trade_when": "需要状态保持型内核",
    "ts_regression": "需要多 rettype 回归内核（现有 slope/residual 只覆盖部分情形）",
    "group_mean": "需要分组归约内核",
    "group_median": "需要分组归约内核",
    "group_rank": "需要组内排序内核",
    "group_normalize": "需要分组归约内核",
    "group_zscore": "需要组内均值/标准差内核",
    "group_sum": "需要分组归约内核",
}

# 与内核同名算子的语义差异（仅在本模块内按本表口径实现）
CONFLICTS = {
    "ts_rank": "本表口径为归一化秩 + constant；内核默认返回整数平均秩",
    "ts_quantile": "本表口径为秩到逆分布映射；与内核的窗口分位插值不同，暂未实现",
    "ts_decay_linear": "本表口径缺失按 0；内核忽略缺失后重归一",
    "scale": "本表口径支持长/短账本分别缩放；内核只有一个账本，暂未实现",
    "ts_backfill": "本表口径带 k；本模块只支持 k=1（等价 ts_ffill(limit)）",
}

BUILDERS = {
    "exp": exp,
    "sign": sign,
    "log": log,
    "max": max_,
    "abs": abs_,
    "divide": divide,
    "min": min_,
    "signed_power": signed_power,
    "inverse": inverse,
    "sqrt": sqrt,
    "s_log_1p": s_log_1p,
    "reverse": reverse,
    "log_diff": log_diff,
    "power": power,
    "if_else": if_else,
    "ts_corr": ts_corr,
    "ts_zscore": ts_zscore,
    "ts_returns": ts_returns,
    "ts_std_dev": ts_std_dev,
    "ts_backfill": ts_backfill,
    "ts_scale": ts_scale,
    "ts_vector_neut": ts_vector_neut,
    "ts_sum": ts_sum,
    "ts_decay_exp_window": ts_decay_exp_window,
    "ts_av_diff": ts_av_diff,
    "ts_kurtosis": ts_kurtosis,
    "ts_mean": ts_mean,
    "ts_min_max_diff": ts_min_max_diff,
    "ts_arg_max": ts_arg_max,
    "ts_max": ts_max,
    "ts_min_max_cps": ts_min_max_cps,
    "ts_rank": ts_rank,
    "ts_ir": ts_ir,
    "ts_delay": ts_delay,
    "ts_weighted_decay": ts_weighted_decay,
    "ts_min": ts_min,
    "ts_count_nans": ts_count_nans,
    "ts_min_diff": ts_min_diff,
    "ts_decay_linear": ts_decay_linear,
    "ts_arg_min": ts_arg_min,
    "ts_skewness": ts_skewness,
    "ts_max_diff": ts_max_diff,
    "ts_median": ts_median,
    "ts_delta": ts_delta,
    "rank": lambda x: call("cs_rank", x, normalize=True),
    "vector_neut": vector_neut,
    "regression_neut": regression_neut,
    "zscore": lambda x: call("cs_zscore", x),
    "normalize": normalize,
    "group_neutralize": group_neutralize,
    "ts_range": ts_range,
}


def build(name, *args, **params):
    """按本模块口径构造表达式；未实现的算子明确报 NotImplementError。"""
    if name not in FORMULAS:
        raise KeyError(f"未收录的算子：{name}")
    if name in UNSUPPORTED:
        raise NotImplementedError(f"{name}: {UNSUPPORTED[name]}")
    if name not in BUILDERS:
        raise NotImplementedError(f"{name}: 缺少构造器")
    return BUILDERS[name](*args, **params)


def supported():
    """已能构造的算子名（按 FORMULAS 顺序）。"""
    return [name for name in FORMULAS if name in BUILDERS]


def missing():
    """尚未实现的算子名与原因。"""
    return {name: UNSUPPORTED[name] for name in FORMULAS if name in UNSUPPORTED}


def coverage():
    """(已实现数, 总数)。"""
    return len(supported()), len(FORMULAS)


__all__ = [
    "CONFLICTS",
    "FORMULAS",
    "UNSUPPORTED",
    "Expr",
    "build",
    "coverage",
    "field",
    "missing",
    "supported",
]
