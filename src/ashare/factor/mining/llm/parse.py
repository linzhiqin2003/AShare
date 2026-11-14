"""模型写的表达式文本 → `Expr`，以及 `Expr` → 同一种文本。

只构造节点，不执行任何 Python（没有 eval / exec）。白名单就是搜索空间本身：算子名必须在
``space.grids`` 里，字段名必须在 ``space.fields`` 里，参数只能是该算子在空间里声明过的
取值；最后交给 `Space.unit` 检查量纲、深度、节点数、位置和冗余，任何一条不满足都抛
`ParseError`，消息说明哪里不对（会原样反馈给模型）。

语法（与 `render` 的输出相同）::

    表达式 := 项 (("+" | "-") 项)*
    项     := 因子 (("*" | "/") 因子)*
    因子   := "-" 因子 | 字段 | 算子 "(" 实参 ("," 实参)* ")" | "(" 表达式 ")"
    实参   := 表达式 | 参数名 "=" 整数 | 整数（按算子声明的参数顺序）

四则运算对应 add、subtract、multiply、divide，一元负号对应 negate，都要求这些算子在空间里。
没有数值常量：数字只能作为算子的参数出现。
"""

from __future__ import annotations

import re

from ..space import CATALOG, Invalid, canonical

MAX_TEXT = 600  # 一条表达式的最大字符数
MAX_NESTING = 24  # 括号与调用的最大嵌套层数（远大于任何空间的 max_depth）

_INFIX = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/(),=]))"
)


class ParseError(ValueError):
    """表达式文本不合语法，或不满足搜索空间的约束。"""


def _tokens(text):
    out, pos = [], 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ParseError(f"无法识别的字符 {text[pos:].lstrip()[:1]!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


def _number(text):
    if re.fullmatch(r"\d+", text):
        return int(text)
    value = float(text)
    return int(value) if value.is_integer() else value


class _Parser:
    def __init__(self, text, space):
        self.tokens, self.i, self.space, self.depth = _tokens(text), 0, space, 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def take(self, value=None):
        kind, got = self.peek()
        if kind is None:
            raise ParseError(
                "表达式不完整" if value is None else f"缺少 {value!r}（表达式提前结束）"
            )
        if value is not None and got != value:
            raise ParseError(f"期望 {value!r}，得到 {got!r}")
        self.i += 1
        return kind, got

    def nest(self):
        self.depth += 1
        if self.depth > MAX_NESTING:
            raise ParseError(f"嵌套超过 {MAX_NESTING} 层")

    def parse(self):
        if not self.tokens:
            raise ParseError("表达式为空")
        tree = self.additive()
        if self.peek()[0] is not None:
            raise ParseError(f"多余的内容 {self.peek()[1]!r}")
        return tree

    def additive(self):
        left = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = _INFIX[self.take()[1]]
            left = self.node(op, [left, self.term()], {})
        return left

    def term(self):
        left = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = _INFIX[self.take()[1]]
            left = self.node(op, [left, self.unary()], {})
        return left

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            self.nest()
            if self.peek()[0] == "num":
                raise ParseError("不支持数值常量：数字只能作为算子参数")
            out = self.node("negate", [self.unary()], {})
            self.depth -= 1
            return out
        return self.primary()

    def primary(self):
        kind, value = self.peek()
        if kind == "num":
            raise ParseError(
                f"不支持数值常量 {value}：数字只能作为算子参数（如 window=20）"
            )
        if (kind, value) == ("op", "("):
            self.take()
            self.nest()
            inner = self.additive()
            self.take(")")
            self.depth -= 1
            return inner
        if kind == "name":
            self.take()
            if self.peek() == ("op", "("):
                return self.call(value)
            if value in self.space.fields:
                return self.space.leaf(value)
            if value in CATALOG:
                raise ParseError(f"算子 {value} 后面要跟括号和参数")
            raise ParseError(
                f"未知字段 {value}（可用字段：{', '.join(self.space.fields)}）"
            )
        if kind is None:
            raise ParseError("表达式不完整")
        raise ParseError(f"意外的记号 {value!r}")

    def call(self, op):
        if op not in self.space.grids:
            if op in self.space.fields:
                raise ParseError(f"{op} 是字段，不能当函数调用")
            raise ParseError(
                f"未知算子 {op}（可用算子：{', '.join(self.space.grids)}）"
            )
        rule = CATALOG[op]
        self.take("(")
        self.nest()
        args, positional, named = [], [], {}
        while True:
            kind, value = self.peek()
            if kind == "name" and self.tokens[self.i + 1 : self.i + 2] == [("op", "=")]:
                self.take()
                self.take("=")
                kind2, number = self.take()
                if kind2 != "num":
                    raise ParseError(f"{op}: 参数 {value} 必须是数字")
                if value not in rule.params:
                    raise ParseError(
                        f"{op}: 没有参数 {value}"
                        + (
                            f"（参数是 {', '.join(rule.params)}）"
                            if rule.params
                            else "（不接受参数）"
                        )
                    )
                if value in named:
                    raise ParseError(f"{op}: 参数 {value} 重复")
                named[value] = _number(number)
            elif kind == "num":
                self.take()
                if named:
                    raise ParseError(f"{op}: 位置参数不能写在关键字参数后面")
                positional.append(_number(value))
            else:
                if positional or named:
                    raise ParseError(f"{op}: 操作数要写在参数前面")
                args.append(self.additive())
            if self.peek() == ("op", ","):
                self.take()
                continue
            self.take(")")
            break
        self.depth -= 1
        if len(args) != rule.arity:
            raise ParseError(f"{op}: 需要 {rule.arity} 个操作数，给了 {len(args)} 个")
        if len(positional) > len(rule.params):
            raise ParseError(
                f"{op}: 参数过多（参数是 {', '.join(rule.params) or '无'}）"
            )
        for name, number in zip(rule.params, positional, strict=False):
            if name in named:
                raise ParseError(f"{op}: 参数 {name} 重复")
            named[name] = number
        return self.node(op, args, named)

    def node(self, op, args, params):
        if op not in self.space.grids:
            symbol = {v: k for k, v in _INFIX.items()}.get(op)
            how = f"（{symbol!r} 对应 {op}）" if symbol else ""
            raise ParseError(f"算子 {op} 不在空间里{how}")
        rule = CATALOG[op]
        missing = [p for p in rule.params if p not in params]
        if missing:
            raise ParseError(f"{op}: 缺少参数 {', '.join(missing)}")
        if params not in self.space.grids[op]:
            allowed = _allowed(self.space.grids[op])
            raise ParseError(f"{op}: 参数 {_fmt(params)} 不在允许的取值里（{allowed}）")
        return self.space.build(op, args, params)


def _fmt(params):
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def _allowed(grid):
    keys = sorted({k for g in grid for k in g})
    return "; ".join(
        f"{k} ∈ {{{', '.join(str(v) for v in sorted({g[k] for g in grid}))}}}"
        for k in keys
    )


def parse(text, space):
    """表达式文本 → 规范形 `Expr`（满足 space 的全部约束），否则抛 `ParseError`。"""
    if not isinstance(text, str):
        raise ParseError("表达式必须是字符串")
    text = text.strip().strip("`").strip()
    if len(text) > MAX_TEXT:
        raise ParseError(f"表达式超过 {MAX_TEXT} 个字符")
    expr = canonical(_Parser(text, space).parse())
    try:
        space.unit(expr)
    except Invalid as exc:
        raise ParseError(str(exc)) from None
    return expr


_PREC = {"add": 1, "subtract": 1, "multiply": 2, "divide": 2}


def render(expr, _context=0, _right=False):
    """`Expr` → `parse` 能读回同一棵树的文本：四则运算写中缀（只加必要的括号：优先级更低、
    或同级时作为右操作数），其余写函数形式，参数只写算子声明的那几个（关键字形式）。
    只用于空间内的小表达式（深度受 ``max_depth`` 限制）。"""
    params = dict(expr.params)
    if expr.op == "field":
        return params["value"]
    if expr.op in _PREC:
        prec = _PREC[expr.op]
        symbol = {v: k for k, v in _INFIX.items()}[expr.op]
        text = (
            f"{render(expr.args[0], prec)} {symbol} {render(expr.args[1], prec, True)}"
        )
        wrap = prec < _context or (_right and prec == _context)
        return f"({text})" if wrap else text
    rule = CATALOG.get(expr.op)
    names = rule.params if rule is not None else ()
    pieces = [render(a) for a in expr.args]
    pieces += [f"{k}={params[k]!r}" for k in names]
    return f"{expr.op}({', '.join(pieces)})"
