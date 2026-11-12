"""用现有算子精确组合出来的算子（不写新内核）。

每个函数只拼 ``Expr``；精确性靠两点：
比较运算的结果恰好是 0.0 或 1.0，以及 IEEE 除以 1.0 不改变任何位。
"""

from ..expression import Expr, call, field

QLIB_STD_EPS = 2e-5  # Qlib Corr/Rsquare 里 np.isclose(std, 0, atol=2e-05) 的界


def keep_where(keep, x):
    """keep 为 1 处保留 x，为 0 或 NaN 处给 NaN。

    keep 必须是比较、逻辑运算的 0/1 结果：x / 1.0 与 x 逐位相同，x / 0.0 按 SPEC 为 NaN。
    """
    return x / keep


def valid(x):
    """有效（有限）为 1，缺失为 0。"""
    return call("not_", call("is_nan", x))


def ts_valid_count(x, window):
    """窗口内有效值的个数（按行计，含当日）；窗口全空时为 0。"""
    return call("ts_sum", valid(x), window=window, min_periods=1)


def cs_valid_count(x):
    """当天截面上有效值的个数；全空的行为 NaN。

    全部相等的值平均秩是 (n+1)/2，所以 n = 2·rank(1) − 1，整数和半整数在双精度里都精确。
    """
    ones = x * 0 + 1
    return 2 * call("cs_rank", ones) - 1


def cs_pct_rank(x):
    """截面百分位：平均秩 / 有效个数，取值 (0, 1]，NaN 不参与（pandas rank(pct=True)）。"""
    return call("cs_rank", x) / cs_valid_count(x)


def ts_pct_rank(x, window, min_periods):
    """时序百分位：当天值在窗口内的平均秩 / 窗口内有效个数；当天缺失为 NaN。"""
    rank = call("ts_rank", x, window=window, min_periods=min_periods)
    return rank / ts_valid_count(x, window)


def ewm(x, n, m, min_periods):
    """国君 SMA(A, n, m)：α = m/n 的递推指数平均，首个有效值作初值，缺失日保持状态。

    输出在当天输入缺失时为 NaN（约定 D-10），状态不受影响。
    """
    state = call("ts_ema", x, window=n, min_periods=min_periods, alpha=m / n)
    return keep_where(valid(x), state)


def decayed_mean(x, n, base=0.9):
    """国君 WMA(A, n)：权重 base**i（i 为距今天数，今天 0），除以权重和；任一天缺失为 NaN。"""
    weights = [base**i for i in range(n)]
    total = None
    for i, w in enumerate(weights):
        term = (x if i == 0 else call("ts_delay", x, periods=i)) * w
        total = term if total is None else total + term
    return total / sum(weights)


def filter_out(x, condition):
    """国君 FILTER(A, cond)：条件为真保留，为假或缺失给 NaN。"""
    return keep_where(call("ne", condition, 0), x)


# ---- Qlib 表达式专用 -----------------------------------------------------------


def listed_domain():
    """上市区间内为 1，之外为 NaN（0/0）：Qlib 的股票序列只有这些行。"""
    listed = field("is_listed")
    return listed / listed


def qlib_compare(op, a, b):
    """Qlib 的 Gt/Lt：NaN 参与比较得 False（0），上市区间外仍为 NaN。"""
    return call("fill_nan", call(op, a, b), 0) * listed_domain()


def qlib_rank(x, window):
    """Qlib main 的 Rank：rolling(N, min_periods=1).rank(pct=True)。"""
    return ts_pct_rank(x, window, 1)


def _small_std(x, window):
    """窗口标准差 ≤ 2e-5 为 1，否则（含 NaN 标准差）为 0。"""
    std = call("ts_std", x, window=window, min_periods=1)
    return call("fill_nan", call("le", std, QLIB_STD_EPS), 0)


def qlib_corr(a, b, window):
    """Qlib Corr：成对滚动相关；任一序列自己的窗口标准差 ≤ 2e-5 时置 NaN。"""
    corr = call("ts_corr", a, b, window=window, min_periods=1)
    guard = call("or_", _small_std(a, window), _small_std(b, window))
    return keep_where(call("eq", guard, 0), corr)


def qlib_rsquare(x, window):
    """Qlib Rsquare：对窗口位置回归的 R²；窗口标准差 ≤ 2e-5 时置 NaN。"""
    rsq = call("ts_rsquare", x, window=window, min_periods=1)
    return keep_where(call("eq", _small_std(x, window), 0), rsq)


def qlib_idx(x, window, op, *, nan_first=True):
    """Qlib IdxMax/IdxMin：rolling(N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)。

    逐条复现 np.argmax 的行为：并列取最早；窗口里有 NaN 时返回第一个 NaN 的位置
    （``nan_first``，约定 D-16）；上市不满 N 行时位置从上市第一行数起；窗口内没有有效值、
    或当天不在上市区间时为 NaN。
    """
    if op not in ("ts_idxmax", "ts_idxmin"):
        raise ValueError("op 只能是 ts_idxmax 或 ts_idxmin")
    domain = listed_domain()
    rows = call("ts_sum", domain, window=window, min_periods=1)
    position = call(op, x, window=window, min_periods=1, ties="earliest")
    if nan_first:
        missing = call("is_nan", x) * domain
        has_missing = call(
            "gt", call("ts_sum", missing, window=window, min_periods=1), 0
        )
        first_missing = call(
            "ts_idxmax", missing, window=window, min_periods=1, ties="earliest"
        )
        position = call("where", has_missing, first_missing, position)
    found = call("gt", ts_valid_count(x, window), 0)
    return keep_where(found, position - (window - rows)) * domain


__all__ = [
    "Expr",
    "keep_where",
    "valid",
    "ts_valid_count",
    "cs_valid_count",
    "cs_pct_rank",
    "ts_pct_rank",
    "ewm",
    "decayed_mean",
    "filter_out",
    "listed_domain",
    "qlib_compare",
    "qlib_rank",
    "qlib_corr",
    "qlib_rsquare",
    "qlib_idx",
]
