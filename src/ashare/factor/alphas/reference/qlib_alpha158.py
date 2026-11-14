"""Alpha158 的独立参考实现：照 Qlib 源码的 pandas 写法逐只股票计算。

不经过转换程序（``qlib.py``）、``Expr`` 和 ``panelops``：表达式字符串按 Qlib 自己的办法
（正则把 ``$x`` 换成 ``Feature("x")``、把 ``Name(`` 换成算子类，再 ``eval``）变成算子树，
算子的实现逐行对照 ``docs/factors/sources/alpha158_defs.md`` 里的 Qlib 原文：

- 每只股票一条 ``pandas.Series``，只含上市区间的行（Qlib 的股票序列从第一条数据开始，
  停牌日是 NaN 行）；
- 滚动算子用 ``series.rolling(N, min_periods=1)``；``Rank`` 取 main 分支的
  ``rolling.rank(pct=True)``；``IdxMax/IdxMin`` 用 ``rolling.apply(lambda x: x.argmax() + 1,
  raw=True)``；``Corr``、``Rsquare`` 带 ``np.isclose(std, 0, atol=2e-05)`` 判定；
- ``Slope/Resi/Rsquare`` 把 rolling.pyx 的 Cython ``update`` 逐行移植成 Python（窗口初始为
  N 个 NaN、``na_count = N``，见 CONVENTIONS Q-Q-05）；
- 比较 ``>``、``<`` 用 ``np.greater/np.less``，NaN 比较为 False。

输入是 ``reference_frames`` 给出的 DataFrame 字典（见 ``docs/REFERENCE_INPUTS.md``）。
"""

import math
import re
from collections import deque

import numpy as np
import pandas as pd

FIELD_NAMES = ("open", "high", "low", "close", "vwap", "volume")


class Expression:
    """Qlib ``Expression`` 的最小子集：运算符重载与按股票缓存的 ``load``。"""

    def load(self, data, cache):
        key = str(self)
        if key not in cache:
            cache[key] = self._load(data, cache)
        return cache[key]

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)  # base.py 原文：__rmul__ 返回 Mul(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __gt__(self, other):
        return Gt(self, other)

    def __lt__(self, other):
        return Lt(self, other)


def _value(x, data, cache):
    return x.load(data, cache) if isinstance(x, Expression) else x


class Feature(Expression):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "$" + self.name

    def _load(self, data, cache):
        return data[self.name]


class NpPairOperator(Expression):
    def __init__(self, left, right, func):
        self.left, self.right, self.func = left, right, func

    def __str__(self):
        return f"{type(self).__name__}({self.left},{self.right})"

    def _load(self, data, cache):
        left = _value(self.left, data, cache)
        right = _value(self.right, data, cache)
        return getattr(np, self.func)(left, right)


class Add(NpPairOperator):
    def __init__(self, left, right):
        super().__init__(left, right, "add")


class Sub(NpPairOperator):
    def __init__(self, left, right):
        super().__init__(left, right, "subtract")


class Mul(NpPairOperator):
    def __init__(self, left, right):
        super().__init__(left, right, "multiply")


class Div(NpPairOperator):
    def __init__(self, left, right):
        super().__init__(left, right, "divide")


class Greater(NpPairOperator):
    def __init__(self, left, right):
        super().__init__(left, right, "maximum")


class Less(NpPairOperator):
    def __init__(self, left, right):
        super().__init__(left, right, "minimum")


class Gt(NpPairOperator):
    def __init__(self, left, right):
        super().__init__(left, right, "greater")


class Lt(NpPairOperator):
    def __init__(self, left, right):
        super().__init__(left, right, "less")


class NpElemOperator(Expression):
    def __init__(self, feature, func):
        self.feature, self.func = feature, func

    def __str__(self):
        return f"{type(self).__name__}({self.feature})"

    def _load(self, data, cache):
        return getattr(np, self.func)(self.feature.load(data, cache))


class Abs(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "abs")


class Log(NpElemOperator):
    def __init__(self, feature):
        super().__init__(feature, "log")


class Rolling(Expression):
    func = None

    def __init__(self, feature, N):
        self.feature, self.N = feature, N

    def __str__(self):
        return f"{type(self).__name__}({self.feature},{self.N})"

    def _load(self, data, cache):
        series = self.feature.load(data, cache)
        return getattr(series.rolling(self.N, min_periods=1), self.func)()


class Ref(Rolling):
    def _load(self, data, cache):
        series = self.feature.load(data, cache)
        if series.empty:
            return series
        return series.shift(self.N)


class Mean(Rolling):
    func = "mean"


class Sum(Rolling):
    func = "sum"


class Std(Rolling):
    func = "std"


class Max(Rolling):
    func = "max"


class Min(Rolling):
    func = "min"


class IdxMax(Rolling):
    def _load(self, data, cache):
        series = self.feature.load(data, cache)
        return series.rolling(self.N, min_periods=1).apply(
            lambda x: x.argmax() + 1, raw=True
        )


class IdxMin(Rolling):
    def _load(self, data, cache):
        series = self.feature.load(data, cache)
        return series.rolling(self.N, min_periods=1).apply(
            lambda x: x.argmin() + 1, raw=True
        )


class Quantile(Rolling):
    def __init__(self, feature, N, qscore):
        super().__init__(feature, N)
        self.qscore = qscore

    def __str__(self):
        return f"Quantile({self.feature},{self.N},{self.qscore})"

    def _load(self, data, cache):
        series = self.feature.load(data, cache)
        return series.rolling(self.N, min_periods=1).quantile(self.qscore)


class Rank(Rolling):
    def _load(self, data, cache):
        series = self.feature.load(data, cache)
        return series.rolling(self.N, min_periods=1).rank(pct=True)


def _cython_rolling(values, window, kind):
    """rolling.pyx 的 Slope/Resi/Rsquare.update 逐行移植。"""
    barv = deque([math.nan] * window)
    na_count = window
    i_sum = x_sum = x2_sum = y_sum = y2_sum = xy_sum = 0.0
    out = np.empty(len(values))
    for k, val in enumerate(values):
        val = float(val)
        barv.append(val)
        xy_sum = xy_sum - y_sum
        x2_sum = x2_sum + i_sum - 2 * x_sum
        x_sum = x_sum - i_sum
        old = barv[0]
        if not math.isnan(old):
            i_sum -= 1
            y_sum -= old
            y2_sum -= old * old
        else:
            na_count -= 1
        barv.popleft()
        if math.isnan(val):
            na_count += 1
        else:
            i_sum += 1
            x_sum += window
            x2_sum += window * window
            y_sum += val
            y2_sum += val * val
            xy_sum += window * val
        n = window - na_count
        with np.errstate(divide="ignore", invalid="ignore"):
            num = np.float64(n * xy_sum - x_sum * y_sum)
            if kind == "rsquare":
                den = np.sqrt(
                    np.float64(
                        (n * x2_sum - x_sum * x_sum) * (n * y2_sum - y_sum * y_sum)
                    )
                )
                r = num / den
                out[k] = r * r
                continue
            slope = num / np.float64(n * x2_sum - x_sum * x_sum)
            if kind == "slope":
                out[k] = slope
                continue
            x_mean = np.float64(x_sum) / n
            y_mean = np.float64(y_sum) / n
            interp = y_mean - slope * x_mean
            out[k] = val - (slope * window + interp)
    return out


class Slope(Rolling):
    def _load(self, data, cache):
        series = self.feature.load(data, cache)
        return pd.Series(
            _cython_rolling(series.values, self.N, "slope"), index=series.index
        )


class Resi(Rolling):
    def _load(self, data, cache):
        series = self.feature.load(data, cache)
        return pd.Series(
            _cython_rolling(series.values, self.N, "resi"), index=series.index
        )


class Rsquare(Rolling):
    def _load(self, data, cache):
        source = self.feature.load(data, cache)
        series = pd.Series(
            _cython_rolling(source.values, self.N, "rsquare"), index=source.index
        )
        series.loc[
            np.isclose(source.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
        ] = np.nan
        return series


class Corr(Expression):
    def __init__(self, left, right, N):
        self.left, self.right, self.N = left, right, N

    def __str__(self):
        return f"Corr({self.left},{self.right},{self.N})"

    def _load(self, data, cache):
        left = self.left.load(data, cache)
        right = self.right.load(data, cache)
        res = left.rolling(self.N, min_periods=1).corr(right)
        res.loc[
            np.isclose(left.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(right.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
        ] = np.nan
        return res


OPERATORS = {
    cls.__name__: cls
    for cls in (
        Abs,
        Log,
        Greater,
        Less,
        Gt,
        Lt,
        Ref,
        Mean,
        Sum,
        Std,
        Max,
        Min,
        IdxMax,
        IdxMin,
        Quantile,
        Rank,
        Slope,
        Resi,
        Rsquare,
        Corr,
    )
}


def parse_field(text):
    """Qlib ``parse_field`` 的写法：``$x`` → ``Feature("x")``，``Name(`` → 算子类调用。"""
    text = re.sub(r"\$(\w+)", r'Feature("\1")', text)
    return re.sub(r"(\w+\s*)\(", r"OPS.\1(", text)


def compile_formula(text):
    """表达式字符串 → 算子树（只认本模块定义的算子）。"""
    namespace = {"OPS": _Namespace(), "__builtins__": {}}
    tree = eval(parse_field(text), namespace)  # noqa: S307 - 与 Qlib 相同的求值方式，输入只来自原文文件
    if not isinstance(tree, Expression):
        raise ValueError(f"不是表达式：{text}")
    return tree


class _Namespace:
    def __getattr__(self, name):
        name = name.strip()
        if name == "Feature":
            return Feature
        if name not in OPERATORS:
            raise NameError(f"参考实现没有算子 {name}")
        return OPERATORS[name]


def _domain(listed):
    """一只股票的上市区间（行号首尾）；没有上市行时为 None。区间不连续时报错。"""
    rows = np.flatnonzero(listed)
    if len(rows) == 0:
        return None
    if rows[-1] - rows[0] + 1 != len(rows):
        raise ValueError("上市区间不连续，参考实现不支持")
    return rows[0], rows[-1] + 1


def compute_columns(frames, formulas, columns):
    """对指定列（证券下标）逐只股票计算一组表达式；返回 ``{名字: ndarray[T, len(columns)]}``。"""
    trees = {name: compile_formula(text) for name, text in formulas.items()}
    listed = frames["listed"].to_numpy()
    values = {name: frames[name].to_numpy() for name in FIELD_NAMES}
    index = frames["close"].index
    out = {name: np.full((len(index), len(columns)), np.nan) for name in trees}
    for k, j in enumerate(columns):
        span = _domain(listed[:, j])
        if span is None:
            continue
        a, b = span
        data = {
            name: pd.Series(values[name][a:b, j], index=index[a:b])
            for name in FIELD_NAMES
        }
        cache = {}
        for name, tree in trees.items():
            result = tree.load(data, cache)
            out[name][a:b, k] = np.asarray(result, dtype=float)
    return out


def compute(data, *, formula):
    """REFERENCE_INPUTS 约定的单因子接口：返回与输入同形状的 DataFrame。"""
    columns = range(data["close"].shape[1])
    values = compute_columns(data, {"x": formula}, columns)["x"]
    return pd.DataFrame(
        values, index=data["close"].index, columns=data["close"].columns
    )
