"""Qlib 表达式到本仓库表达式的确定性转换。

只认这些写法：字段 ``$open $high $low $close $vwap $volume``，数字，``+ - * /``，
比较 ``> <``，一元负号，以及下面 ``FUNCTIONS`` 里的函数。遇到别的写法直接报错，不猜。
语义照 Qlib 源码（``qlib/data/ops.py`` 与 ``rolling.pyx``，main 分支的 ``Rank``）。

    >>> to_expr("Mean($high, 7)/Std($close, 13)")   # doctest: +SKIP

转换分两步：``parse`` 得到语法树（元组），``build`` 把语法树换成 ``Expr``。同一个字符串
永远得到同一个表达式（摘要相同）。
"""

import re

from ..expression import call, field
from . import ops

_factor = field("adj_factor")
FIELDS = {
    "open": field("open") * _factor,
    "high": field("high") * _factor,
    "low": field("low") * _factor,
    "close": field("close") * _factor,
    "vwap": field("vwap") * _factor,
    "volume": field("volume") / _factor,
}


def _rolling(op, **extra):
    def build(x, n):
        return call(op, x, window=_int(n), min_periods=1, **extra)

    return build


def _int(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"窗口必须是正整数：{n!r}")
    return n


def _ref(x, n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Ref 只支持正整数（向过去取值）")
    return call("ts_delay", x, periods=n)


def _quantile(x, n, q):
    return call("ts_quantile", x, window=_int(n), min_periods=1, q=float(q))


# 函数名 → (参数种类, 构造函数)：e 序列，n 数字，v 序列或数字。
FUNCTIONS = {
    "Ref": ("en", _ref),
    "Mean": ("en", _rolling("ts_mean")),
    "Sum": ("en", _rolling("ts_sum")),
    "Std": ("en", _rolling("ts_std", ddof=1)),
    "Max": ("en", _rolling("ts_max")),
    "Min": ("en", _rolling("ts_min")),
    "Quantile": ("enn", _quantile),
    "Rank": ("en", lambda x, n: ops.qlib_rank(x, _int(n))),
    "Slope": ("en", _rolling("ts_slope")),
    "Resi": ("en", _rolling("ts_residual")),
    "Rsquare": ("en", lambda x, n: ops.qlib_rsquare(x, _int(n))),
    "IdxMax": ("en", lambda x, n: ops.qlib_idx(x, _int(n), "ts_idxmax")),
    "IdxMin": ("en", lambda x, n: ops.qlib_idx(x, _int(n), "ts_idxmin")),
    "Corr": ("een", lambda a, b, n: ops.qlib_corr(a, b, _int(n))),
    "Greater": ("ev", lambda a, b: call("maximum", a, b)),
    "Less": ("ev", lambda a, b: call("minimum", a, b)),
    "Abs": ("e", lambda x: call("abs", x)),
    "Log": ("e", lambda x: call("log", x)),
}

_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)|(?P<field>\$[a-z]+)"
    r"|(?P<name>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>[-+*/(),<>]))"
)


def tokenize(text):
    tokens, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise SyntaxError(f"无法识别的字符：{text[pos:]!r}")
        kind = m.lastgroup
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


class _Parser:
    """递归下降，优先级与 Python 相同（Qlib 用 eval 求值）：比较 < 加减 < 乘除 < 一元负号。"""

    def __init__(self, text):
        self.tokens, self.i = tokenize(text), 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def take(self, value=None):
        token = self.peek()
        if token[0] is None or (value is not None and token[1] != value):
            raise SyntaxError(f"期望 {value!r}，得到 {token[1]!r}")
        self.i += 1
        return token

    def parse(self):
        tree = self.comparison()
        if self.peek()[0] is not None:
            raise SyntaxError(f"多余的内容：{self.peek()[1]!r}")
        return tree

    def comparison(self):
        left = self.additive()
        if self.peek() in (("op", ">"), ("op", "<")):
            op = self.take()[1]
            right = self.additive()
            if self.peek() in (("op", ">"), ("op", "<")):
                raise SyntaxError("不支持连续比较")
            left = ("cmp", op, left, right)
        return left

    def additive(self):
        left = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            left = ("bin", op, left, self.term())
        return left

    def term(self):
        left = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            left = ("bin", op, left, self.unary())
        return left

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        return self.primary()

    def primary(self):
        kind, value = self.peek()
        if kind == "num":
            self.take()
            number = float(value)
            return ("num", int(number) if re.fullmatch(r"\d+", value) else number)
        if kind == "field":
            self.take()
            if value[1:] not in FIELDS:
                raise SyntaxError(f"未知字段：{value}")
            return ("field", value[1:])
        if kind == "name":
            self.take()
            if value not in FUNCTIONS:
                raise SyntaxError(f"未知函数：{value}")
            self.take("(")
            args = [self.comparison()]
            while self.peek() == ("op", ","):
                self.take()
                args.append(self.comparison())
            self.take(")")
            return ("call", value, tuple(args))
        if (kind, value) == ("op", "("):
            self.take()
            inner = self.comparison()
            self.take(")")
            return inner
        raise SyntaxError(f"意外的记号：{value!r}")


def parse(text):
    """Qlib 表达式 → 语法树（嵌套元组）。"""
    return _Parser(text).parse()


def build(tree):
    """语法树 → ``Expr``（或数字）。"""
    kind = tree[0]
    if kind == "num":
        return tree[1]
    if kind == "field":
        return FIELDS[tree[1]]
    if kind == "neg":
        return -_series(build(tree[1]))
    if kind == "bin":
        left, right = build(tree[2]), build(tree[3])
        if not hasattr(left, "op") and not hasattr(right, "op"):
            raise SyntaxError("两个数字之间的运算应在原文里写成一个数")
        op = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}[tree[1]]
        return call(op, left, right)
    if kind == "cmp":
        op = {">": "gt", "<": "lt"}[tree[1]]
        return ops.qlib_compare(op, build(tree[2]), build(tree[3]))
    if kind == "call":
        spec, builder = FUNCTIONS[tree[1]]
        args = tree[2]
        if len(args) != len(spec):
            raise SyntaxError(f"{tree[1]} 需要 {len(spec)} 个参数，得到 {len(args)}")
        values = []
        for want, arg in zip(spec, args, strict=True):
            if want == "n":
                if arg[0] != "num":
                    raise SyntaxError(f"{tree[1]} 的这个参数必须是数字")
                values.append(arg[1])
            elif want == "v":
                values.append(build(arg))
            else:
                values.append(_series(build(arg)))
        return builder(*values)
    raise SyntaxError(f"未知语法节点：{kind}")


def _series(value):
    if not hasattr(value, "op"):
        raise SyntaxError("这里需要序列，得到数字")
    return value


def to_expr(text):
    """Qlib 表达式字符串 → ``Expr``。

    最外层乘上市区间（1 或 NaN）：Qlib 的股票序列只有上市期间的行，``min_periods=1`` 的
    滚动统计在退市后的行上不会有输出；面板上这些行存在，要显式置 NaN。
    """
    return _series(build(parse(text))) * ops.listed_domain()
