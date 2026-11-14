"""翻译里的比较方向与原文逐条机械对照（两套翻译的每个因子）。

从文档串（原文公式）里抽出所有比较运算 ``<``、``>``、``<=``、``>=``、``==``、``=``（国君的相等）、
``!=``，记下运算符和「哪一侧是数字常数、常数是多少」；从翻译的源码里抽出所有
``d.lt/gt/le/ge/eq/ne`` 调用，记下同样的信息。两边按出现顺序逐个比对，顺序对不上时再按多重集
比对（比较嵌在另一个比较的操作数里时，原文的中序与调用的前序可能不同）。

只认「紧挨着比较运算符、且被更低优先级的符号或括号隔开」的数字为常数操作数：
``(1 < (x))`` 的左侧是常数 1；``a + 1 < b`` 的左侧是 ``a + 1``，不算常数。
"""

import ast
import inspect
import re
import textwrap
from collections import Counter

import pytest

from ashare.factor.alphas import alpha101, gtja191

OPS = {"<": "lt", ">": "gt", "<=": "le", ">=": "ge", "==": "eq", "=": "eq", "!=": "ne"}
TOKEN = re.compile(
    r"\s*(<=|>=|==|!=|&&|\|\||[<>=?:&|(),+\-*/^？：]|\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+|[A-Za-z_][A-Za-z_0-9.]*)"
)
# 比较运算的操作数在这些符号处结束（比比较优先级低，或是分组边界）。
LEFT_STOP = {"(", ",", "?", ":", "？", "：", "&", "&&", "|", "||", "OR", None}
RIGHT_STOP = {")", ",", "?", ":", "？", "：", "&", "&&", "|", "||", "OR", None}


def _tokens(text):
    out, pos = [], 0
    text = text.rstrip().rstrip(";")
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            pos += 1  # 原文里的少数怪字符（en dash 等）不影响比较的抽取
            continue
        out.append(m.group(1))
        pos = m.end()
    return out


def _number(tok):
    try:
        return float(tok)
    except (TypeError, ValueError):
        return None


def formula_comparisons(text):
    toks = _tokens(text)
    out = []
    for i, t in enumerate(toks):
        if t not in OPS:
            continue
        left = right = None
        # 左侧：数字（可带一元负号），再往左是边界
        j = i - 1
        if j >= 0 and _number(toks[j]) is not None:
            value, k = _number(toks[j]), j - 1
            if k >= 0 and toks[k] == "-" and (k - 1 < 0 or toks[k - 1] in LEFT_STOP):
                value, k = -value, k - 1
            if (toks[k] if k >= 0 else None) in LEFT_STOP:
                left = value
        # 右侧：数字（可带一元负号），再往右是边界
        j = i + 1
        sign = 1.0
        if j < len(toks) and toks[j] == "-":
            sign, j = -1.0, j + 1
        if j < len(toks) and _number(toks[j]) is not None:
            nxt = toks[j + 1] if j + 1 < len(toks) else None
            if nxt in RIGHT_STOP:
                right = sign * _number(toks[j])
        out.append((OPS[t], left, right))
    return out


def _const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
    ):
        return -float(node.operand.value)
    return None


def translation_comparisons(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "d"
            and node.func.attr in set(OPS.values())
        ):
            a, b = node.args
            calls.append(
                ((node.lineno, node.col_offset), (node.func.attr, _const(a), _const(b)))
            )
    return [c for _, c in sorted(calls)]


def _functions():
    for module, prefix in ((alpha101, "alpha101"), (gtja191, "gtja191")):
        for name in sorted(vars(module)):
            if re.fullmatch(rf"{prefix}_\d{{3}}", name):
                yield name, getattr(module, name)


def mismatch(function):
    want = formula_comparisons(function.__doc__)
    got = translation_comparisons(function)
    if want == got or Counter(want) == Counter(got):
        return None
    return {"原文": want, "翻译": got}


def test_extractor_on_known_shapes():
    assert formula_comparisons("((1 < (v / m)) || ((v / m) == 1)) ? 1 : 0") == [
        ("lt", 1.0, None),
        ("eq", None, 1.0),
    ]
    assert formula_comparisons("(0.25 < (a - b)) ? x : ((a + 1 < b) ? 1 : y)") == [
        ("lt", 0.25, None),
        ("lt", None, None),
    ]
    assert formula_comparisons("(x < -1) ? 1 : 0") == [("lt", None, -1.0)]
    assert formula_comparisons("COUNT(CLOSE=DELAY(CLOSE,1),5)") == [("eq", None, None)]


def test_detects_flipped_constant_comparison():
    def flipped():
        """(1 < x) ? 1 : 0"""
        return d.where(d.gt(1, d.x), 1, 0)  # noqa: F821  只解析源码，不执行

    assert mismatch(flipped) is not None


@pytest.mark.parametrize(
    "name,function", list(_functions()), ids=lambda v: v if isinstance(v, str) else ""
)
def test_comparison_direction_matches_formula(name, function):
    assert mismatch(function) is None, (name, mismatch(function))
