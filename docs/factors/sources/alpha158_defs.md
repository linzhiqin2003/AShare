# Qlib Alpha158 表达式算子定义（原文抄录）

出处（均为 GitHub 原始文件，curl 直接取回，未经任何模型转述）：
- 算子：https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py
- Slope/Rsquare/Resi 的滚动实现：https://github.com/microsoft/qlib/blob/main/qlib/data/_libs/rolling.pyx
- 运算符重载（`>`、`<`、`+`、`-`、`*`、`/` 映射到哪个算子）：https://github.com/microsoft/qlib/blob/main/qlib/data/base.py
- 对照旧版本：https://github.com/microsoft/qlib/blob/v0.8.6/qlib/data/ops.py
- Alpha158 默认配置：https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py 中 `Alpha158.get_feature_config`（conf 见下）

## 要点（直接读自下方原文，不是推断）

- `Ref($x, d)`：docstring 原文 "N = 0, retrieve the first data; N > 0, retrieve data of N periods ago; N < 0, future data"；实现为 `series.shift(self.N)`。Alpha158 只用到 d=1 和 d=5/10/20/30/60（正数，即 d 期以前）。
- 滚动窗口是否含当日：`Rolling._load_internal` 用 pandas `series.rolling(self.N, min_periods=1)`，pandas 的 rolling 窗口以当前行结尾（含当日），且 `min_periods=1`（窗口不满也出值）。Slope/Rsquare/Resi 的 Cython 实现同样把当前值 push 进长度为 N 的 deque 后计算（含当日）。
- `Rank($x, N)`：时序分位，不是截面排名。main 版 docstring 为 "Rolling Rank (Percentile)"，实现为 `rolling(...).rank(pct=True)`（pandas 有该方法时），否则 `percentileofscore(x1, x1[-1]) / 100`。**v0.8.6 版本**的同一函数除以 `len(x1)` 而不是 100（见文末原文），两个版本数值口径不同。
- `Slope/Rsquare/Resi`：自变量是窗口内的位置序号。Slope 的 docstring 原文："This operator calculate the slope between `idx` and `feature`. (e.g. [<feature_t1>, <feature_t2>, <feature_t3>] and [1, 2, 3])"。rolling.pyx 中自变量 x 用 1..window 的位置（最新值取 x = window）；`Resi` 返回 `val - (slope*self.window + interp)`，即**当日**值相对拟合直线的残差；`Rsquare` 返回 rvalue 的平方，ops.py 中窗口内标准差接近 0 时置为 NaN。
- `IdxMax/IdxMin`：实现为 `rolling(N, min_periods=1).apply(lambda x: x.argmax() + 1)`，即窗口内最大（小）值的位置序号，从窗口中最早一天数起为 1。
- `Corr`：pandas `rolling(N, min_periods=1).corr(...)`，任一侧窗口内标准差接近 0（atol=2e-05）时置为 NaN。
- `Greater/Less`：`np.maximum` / `np.minimum`，逐元素。`>` 和 `<` 由 base.py 重载为 `Gt`/`Lt`，返回布尔序列；`Mean($close>Ref($close, 1), d)` 是对布尔序列求滚动均值。
- `Std`：pandas `rolling.std()`（默认 ddof=1，pandas 默认值，非 qlib 源码中写明）。

## Alpha158 默认配置（handler.py 原文）

```python
    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)
```

## ops.py（main）各算子原文

```python
class Abs(NpElemOperator):
    """Feature Absolute Value

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with absolute output
    """

    def __init__(self, feature):
        super(Abs, self).__init__(feature, "abs")
```

```python
class Log(NpElemOperator):
    """Feature Log

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with log
    """

    def __init__(self, feature):
        super(Log, self).__init__(feature, "log")
```

```python
class Greater(NpPairOperator):
    """Greater Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        greater elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(Greater, self).__init__(feature_left, feature_right, "maximum")
```

```python
class Less(NpPairOperator):
    """Less Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        smaller elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(Less, self).__init__(feature_left, feature_right, "minimum")
```

```python
class Gt(NpPairOperator):
    """Greater Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left > right`
    """

    def __init__(self, feature_left, feature_right):
        super(Gt, self).__init__(feature_left, feature_right, "greater")
```

```python
class Lt(NpPairOperator):
    """Less Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left < right`
    """

    def __init__(self, feature_left, feature_right):
        super(Lt, self).__init__(feature_left, feature_right, "less")
```

```python
class Rolling(ExpressionOps):
    """Rolling Operator
    The meaning of rolling and expanding is the same in pandas.
    When the window is set to 0, the behaviour of the operator should follow `expanding`
    Otherwise, it follows `rolling`

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size
    func : str
        rolling method

    Returns
    ----------
    Expression
        rolling outputs
    """

    def __init__(self, feature, N, func):
        self.feature = feature
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # NOTE: remove all null check,
        # now it's user's responsibility to decide whether use features in null days
        # isnull = series.isnull() # NOTE: isnull = NaN, inf is not null
        if isinstance(self.N, int) and self.N == 0:
            series = getattr(series.expanding(min_periods=1), self.func)()
        elif isinstance(self.N, float) and 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = getattr(series.rolling(self.N, min_periods=1), self.func)()
            # series.iloc[:self.N-1] = np.nan
        # series[isnull] = np.nan
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if 0 < self.N < 1:
            return int(np.log(1e-6) / np.log(1 - self.N))  # (1 - N)**window == 1e-6
        return self.feature.get_longest_back_rolling() + self.N - 1

    def get_extended_window_size(self):
        if self.N == 0:
            # FIXME: How to make this accurate and efficiently? Or  should we
            # remove such support for N == 0?
            get_module_logger(self.__class__.__name__).warning("The Rolling(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        elif 0 < self.N < 1:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            size = int(np.log(1e-6) / np.log(1 - self.N))
            lft_etd = max(lft_etd + size - 1, lft_etd)
            return lft_etd, rght_etd
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N - 1, lft_etd)
            return lft_etd, rght_etd
```

```python
class Ref(Rolling):
    """Feature Reference

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        N = 0, retrieve the first data; N > 0, retrieve data of N periods ago; N < 0, future data

    Returns
    ----------
    Expression
        a feature instance with target reference
    """

    def __init__(self, feature, N):
        super(Ref, self).__init__(feature, N, "ref")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # N = 0, return first day
        if series.empty:
            return series  # Pandas bug, see: https://github.com/pandas-dev/pandas/issues/21049
        elif self.N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(self.N)  # copy
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        return self.feature.get_longest_back_rolling() + self.N

    def get_extended_window_size(self):
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning("The Ref(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N, lft_etd)
            rght_etd = max(rght_etd - self.N, rght_etd)
            return lft_etd, rght_etd
```

```python
class Mean(Rolling):
    """Rolling Mean (MA)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling average
    """

    def __init__(self, feature, N):
        super(Mean, self).__init__(feature, N, "mean")
```

```python
class Sum(Rolling):
    """Rolling Sum

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling sum
    """

    def __init__(self, feature, N):
        super(Sum, self).__init__(feature, N, "sum")
```

```python
class Std(Rolling):
    """Rolling Std

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling std
    """

    def __init__(self, feature, N):
        super(Std, self).__init__(feature, N, "std")
```

```python
class Max(Rolling):
    """Rolling Max

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max
    """

    def __init__(self, feature, N):
        super(Max, self).__init__(feature, N, "max")
```

```python
class Min(Rolling):
    """Rolling Min

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min
    """

    def __init__(self, feature, N):
        super(Min, self).__init__(feature, N, "min")
```

```python
class IdxMax(Rolling):
    """Rolling Max Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max index
    """

    def __init__(self, feature, N):
        super(IdxMax, self).__init__(feature, N, "idxmax")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series
```

```python
class IdxMin(Rolling):
    """Rolling Min Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min index
    """

    def __init__(self, feature, N):
        super(IdxMin, self).__init__(feature, N, "idxmin")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series
```

```python
class Quantile(Rolling):
    """Rolling Quantile

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling quantile
    """

    def __init__(self, feature, N, qscore):
        super(Quantile, self).__init__(feature, N, "quantile")
        self.qscore = qscore

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature, self.N, self.qscore)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).quantile(self.qscore)
        else:
            series = series.rolling(self.N, min_periods=1).quantile(self.qscore)
        return series
```

```python
class Rank(Rolling):
    """Rolling Rank (Percentile)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling rank
    """

    def __init__(self, feature, N):
        super(Rank, self).__init__(feature, N, "rank")

    # for compatiblity of python 3.7, which doesn't support pandas 1.4.0+ which implements Rolling.rank
    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        rolling_or_expending = series.expanding(min_periods=1) if self.N == 0 else series.rolling(self.N, min_periods=1)
        if hasattr(rolling_or_expending, "rank"):
            return rolling_or_expending.rank(pct=True)

        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / 100

        return rolling_or_expending.apply(rank, raw=True)
```

```python
class Slope(Rolling):
    """Rolling Slope
    This operator calculate the slope between `idx` and `feature`.
    (e.g. [<feature_t1>, <feature_t2>, <feature_t3>] and [1, 2, 3])

    Usage Example:
    - "Slope($close, %d)/$close"

    # TODO:
    # Some users may want pair-wise rolling like `Slope(A, B, N)`

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression slope of given window
    """

    def __init__(self, feature, N):
        super(Slope, self).__init__(feature, N, "slope")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_slope(series.values), index=series.index)
        else:
            series = pd.Series(rolling_slope(series.values, self.N), index=series.index)
        return series
```

```python
class Rsquare(Rolling):
    """Rolling R-value Square

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(Rsquare, self).__init__(feature, N, "rsquare")

    def _load_internal(self, instrument, start_index, end_index, *args):
        _series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_rsquare(_series.values), index=_series.index)
        else:
            series = pd.Series(rolling_rsquare(_series.values, self.N), index=_series.index)
            series.loc[np.isclose(_series.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return series
```

```python
class Resi(Rolling):
    """Rolling Regression Residuals

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression residuals of given window
    """

    def __init__(self, feature, N):
        super(Resi, self).__init__(feature, N, "resi")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_resi(series.values), index=series.index)
        else:
            series = pd.Series(rolling_resi(series.values, self.N), index=series.index)
        return series
```

```python
class PairRolling(ExpressionOps):
    """Pair Rolling Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling output of two input features
    """

    def __init__(self, feature_left, feature_right, N, func):
        # TODO: in what case will a const be passed into `__init__` as `feature_left` or `feature_right`
        self.feature_left = feature_left
        self.feature_right = feature_right
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature_left, self.feature_right, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, Expression), self.feature_right, Expression]
        ), "at least one of two inputs is Expression instance"

        if isinstance(self.feature_left, Expression):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, Expression):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right

        if self.N == 0:
            series = getattr(series_left.expanding(min_periods=1), self.func)(series_right)
        else:
            series = getattr(series_left.rolling(self.N, min_periods=1), self.func)(series_right)
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if isinstance(self.feature_left, Expression):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, Expression):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, Expression):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0
        if isinstance(self.feature_right, Expression):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning(
                "The PairRolling(ATTR, 0) will not be accurately calculated"
            )
            return -np.inf, max(lr, rr)
        else:
            return max(ll, rl) + self.N - 1, max(lr, rr)
```

```python
class Corr(PairRolling):
    """Rolling Correlation

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling correlation of two input features
    """

    def __init__(self, feature_left, feature_right, N):
        super(Corr, self).__init__(feature_left, feature_right, N, "corr")

    def _load_internal(self, instrument, start_index, end_index, *args):
        res: pd.Series = super(Corr, self)._load_internal(instrument, start_index, end_index, *args)

        # NOTE: Load uses MemCache, so calling load again will not cause performance degradation
        series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        res.loc[
            np.isclose(series_left.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(series_right.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
        ] = np.nan
        return res
```

## rolling.pyx（main）Slope / Resi / Rsquare 原文

```cython
cdef class Slope(Rolling):
    """1-D array rolling slope"""
    cdef double i_sum # can be used as i2_sum
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double xy_sum
    def __init__(self, int window):
        super(Slope, self).__init__(window)
        self.i_sum  = 0
        self.x_sum  = 0
        self.x2_sum = 0
        self.y_sum  = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2*self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        cdef double _val
        _val = self.barv.front()
        if not isnan(_val):
            self.i_sum -= 1
            self.y_sum -= _val
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
            # return NAN
        else:
            self.i_sum  += 1
            self.x_sum  += self.window
            self.x2_sum += self.window * self.window
            self.y_sum  += val
            self.xy_sum += self.window * val
        cdef int N = self.window - self.na_count
        return (N*self.xy_sum - self.x_sum*self.y_sum) / \
            (N*self.x2_sum - self.x_sum*self.x_sum)
```

```cython
cdef class Resi(Rolling):
    """1-D array rolling residuals"""
    cdef double i_sum # can be used as i2_sum
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double xy_sum
    def __init__(self, int window):
        super(Resi, self).__init__(window)
        self.i_sum  = 0
        self.x_sum  = 0
        self.x2_sum = 0
        self.y_sum  = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2*self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        cdef double _val
        _val = self.barv.front()
        if not isnan(_val):
            self.i_sum -= 1
            self.y_sum -= _val
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
            # return NAN
        else:
            self.i_sum  += 1
            self.x_sum  += self.window
            self.x2_sum += self.window * self.window
            self.y_sum  += val
            self.xy_sum += self.window * val
        cdef int N = self.window - self.na_count
        slope = (N*self.xy_sum - self.x_sum*self.y_sum) / \
                (N*self.x2_sum - self.x_sum*self.x_sum)
        x_mean = self.x_sum / N
        y_mean = self.y_sum / N
        interp = y_mean - slope*x_mean
        return val - (slope*self.window + interp)
```

```cython
cdef class Rsquare(Rolling):
    """1-D array rolling rsquare"""
    cdef double i_sum
    cdef double x_sum
    cdef double x2_sum
    cdef double y_sum
    cdef double y2_sum
    cdef double xy_sum
    def __init__(self, int window):
        super(Rsquare, self).__init__(window)
        self.i_sum  = 0
        self.x_sum  = 0
        self.x2_sum = 0
        self.y_sum  = 0
        self.y2_sum = 0
        self.xy_sum = 0

    cdef double update(self, double val):
        self.barv.push_back(val)
        self.xy_sum = self.xy_sum - self.y_sum
        self.x2_sum = self.x2_sum + self.i_sum - 2*self.x_sum
        self.x_sum = self.x_sum - self.i_sum
        cdef double _val
        _val = self.barv.front()
        if not isnan(_val):
            self.i_sum  -= 1
            self.y_sum  -= _val
            self.y2_sum -= _val * _val
        else:
            self.na_count -= 1
        self.barv.pop_front()
        if isnan(val):
            self.na_count += 1
            # return NAN
        else:
            self.i_sum  += 1
            self.x_sum  += self.window
            self.x2_sum += self.window * self.window
            self.y_sum  += val
            self.y2_sum += val * val
            self.xy_sum += self.window * val
        cdef int N = self.window - self.na_count
        cdef double rvalue
        rvalue = (N*self.xy_sum - self.x_sum*self.y_sum) / \
            sqrt((N*self.x2_sum - self.x_sum*self.x_sum) * (N*self.y2_sum - self.y_sum*self.y_sum))
        return rvalue * rvalue
```

## base.py（main）运算符重载原文

```python
    def __gt__(self, other):
        from .ops import Gt  # pylint: disable=C0415

        return Gt(self, other)
    def __lt__(self, other):
        from .ops import Lt  # pylint: disable=C0415

        return Lt(self, other)
    def __add__(self, other):
        from .ops import Add  # pylint: disable=C0415

        return Add(self, other)
    def __radd__(self, other):
        from .ops import Add  # pylint: disable=C0415

        return Add(other, self)
    def __sub__(self, other):
        from .ops import Sub  # pylint: disable=C0415

        return Sub(self, other)
    def __rsub__(self, other):
        from .ops import Sub  # pylint: disable=C0415

        return Sub(other, self)
    def __mul__(self, other):
        from .ops import Mul  # pylint: disable=C0415

        return Mul(self, other)
    def __rmul__(self, other):
        from .ops import Mul  # pylint: disable=C0415

        return Mul(self, other)
    def __truediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(self, other)
    def __rtruediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(other, self)
```

## v0.8.6 ops.py 中 Rank 的原文（与 main 不同）

```python
class Rank(Rolling):
    """Rolling Rank (Percentile)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling rank
    """

    def __init__(self, feature, N):
        super(Rank, self).__init__(feature, N, "rank")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO: implement in Cython

        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / len(x1)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(rank, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(rank, raw=True)
        return series
```

