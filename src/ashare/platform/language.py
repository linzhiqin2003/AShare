"""有资源上限的表达式解释器：只构造 DSL 节点，不执行 Python 文本。"""

import ast
import copy
import hashlib
import inspect
import itertools
import json
import math
import operator
from dataclasses import dataclass
from functools import lru_cache

from ashare.factor import Expr, call, constant, field
from ashare.factor.formula import dsl
from ashare.factor.formula.plugins import discover
from ashare.factor.library import FIELDS, default_registry

MAX_SOURCE = 16000
MAX_NODES = 512
MAX_DEPTH = 32
MAX_VARIANTS = 8
MAX_NUMBER = 1e9
MAX_WINDOW = 2520

# 名称、DSL 构造器、面向编辑器的签名、中文说明；这张表也是调用白名单。
_SPECS = [
    ("ts_std", "stddev", "x, window", "滚动标准差，分母 n−1"),
    ("ts_mean", "ts_mean", "x, window", "滚动均值，窗口必须满"),
    ("ts_sum", "ts_sum", "x, window", "滚动求和，窗口必须满"),
    ("ts_min", "ts_min", "x, window", "滚动最小值"),
    ("ts_max", "ts_max", "x, window", "滚动最大值"),
    ("ts_rank", "ts_rank", "x, window", "当天值在窗口内的百分位"),
    ("ts_delay", "delay", "x, window", "向过去移位指定交易日"),
    ("ts_delta", "delta", "x, window", "与指定交易日前的差值"),
    ("ts_corr", "correlation", "x, y, window", "滚动 Pearson 相关系数"),
    ("ts_cov", "covariance", "x, y, window", "滚动样本协方差"),
    ("ts_decay_linear", "decay_linear", "x, window", "线性衰减加权平均"),
    ("ts_argmax", "ts_argmax", "x, window", "窗口最大值距今天数"),
    ("ts_argmin", "ts_argmin", "x, window", "窗口最小值距今天数"),
    ("ts_product", "ts_product", "x, window", "滚动乘积"),
    ("ts_count", "ts_count", "condition, window", "条件成立的天数"),
    ("ts_sumif", "ts_sumif", "x, condition, window", "条件成立时滚动求和"),
    ("rank", "rank", "x", "截面百分位，平均秩除以有效数"),
    ("scale", "scale", "x, a=1", "按截面绝对值之和归一化"),
    ("log", "log", "x", "自然对数，非正数为空"),
    ("abs", "abs", "x", "绝对值"),
    ("sign", "sign", "x", "数值符号"),
    ("sqrt", "sqrt", "x", "平方根"),
    ("power", "power", "x, y", "实数幂"),
    ("signed_power", "signed_power", "x, a", "保留符号的幂"),
    ("max", "maximum", "x, y", "逐元素取大"),
    ("min", "minimum", "x, y", "逐元素取小"),
    ("where", "where", "condition, if_true, if_false", "按条件选择数值"),
    ("adv", "adv", "window", "滚动平均成交额，单位元"),
]
OPERATORS = {
    name: (getattr(dsl, target), f"{name}({signature})", description)
    for name, target, signature, description in _SPECS
}
ALIASES = {name: target for name, target, _, _ in _SPECS}
FIELD_HELP = {
    "open": "复权开盘价",
    "high": "复权最高价",
    "low": "复权最低价",
    "close": "复权收盘价",
    "volume": "复权成交量",
    "vwap": "复权均价",
    "returns": "日收益，上市首日为空",
    "cap": "总市值，元",
    "amount": "成交额，元",
    "turnover": "换手率，比例",
}
FIELD_NODES = {name: field(name) for name in FIELDS}
FIELD_NODES.update(
    {name: getattr(dsl, name) for name in FIELD_HELP if hasattr(dsl, name)}
)


def factor_name(name):
    """字段名保留原义；重名登记因子使用明确的前缀。"""
    return f"factor_{name}" if name in FIELD_NODES or name in OPERATORS else name


class ExpressionError(ValueError):
    def __init__(self, reason, node=None, *, line=1, column=1):
        self.line = getattr(node, "lineno", line)
        self.column = getattr(node, "col_offset", column - 1) + 1
        self.reason = reason
        super().__init__(f"表达式第 {self.line} 行：{reason}")

    def detail(self):
        return {"message": str(self), "line": self.line, "column": self.column}


@lru_cache(maxsize=1)
def registry():
    result = {f.name: f for f in default_registry()}
    for group in discover().values():
        for factor in group.registry():
            if factor.name in result:
                raise ValueError("因子插件名称重复")
            result[factor.name] = factor
    return result


@dataclass(frozen=True)
class Variant:
    source: str
    expression: Expr
    parameters: tuple


_ALLOWED = (
    ast.Module,
    ast.Assign,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UnaryOp,
    ast.USub,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.Call,
    ast.List,
)
_BINARY = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide"}
_SCALAR_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_COMPARE = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Eq: "eq",
    ast.NotEq: "ne",
}


def _tree(source):
    if not isinstance(source, str) or len(source.encode()) > MAX_SOURCE:
        raise ExpressionError("输入过长，最多 16000 字节")
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError, ValueError) as error:
        raise ExpressionError(
            "语法不正确",
            line=getattr(error, "lineno", 1) or 1,
            column=getattr(error, "offset", 1) or 1,
        ) from None
    if not tree.body:
        raise ExpressionError("请先输入表达式")
    stack, count, lists = [(tree, 0, None)], 0, []
    while stack:
        node, depth, parent = stack.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise ExpressionError("表达式过于复杂或嵌套过深", node)
        if not isinstance(node, _ALLOWED):
            raise ExpressionError("不支持此语法", node)
        if isinstance(node, ast.Name) and ("__" in node.id or len(node.id) > 80):
            raise ExpressionError("不允许此名称", node)
        if isinstance(node, ast.Constant):
            if (
                type(node.value) not in (int, float)
                or abs(node.value) > MAX_NUMBER
                or not math.isfinite(node.value)
            ):
                raise ExpressionError("只接受绝对值不超过十亿的有限数字", node)
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ExpressionError("赋值左侧须是一个名称", node)
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.keywords
        ):
            raise ExpressionError("只支持白名单函数的普通位置参数", node)
        if isinstance(node, ast.List):
            if not isinstance(parent, ast.Call) or node not in parent.args:
                raise ExpressionError("扫描列表只能写在函数参数位置", node)
            if not 1 <= len(node.elts) <= MAX_VARIANTS:
                raise ExpressionError("扫描列表须有 1–8 项", node)
            lists.append(node)
        stack.extend((child, depth + 1, node) for child in ast.iter_child_nodes(node))
    lists.sort(key=lambda n: (n.lineno, n.col_offset))
    if math.prod(len(n.elts) for n in lists) > MAX_VARIANTS:
        raise ExpressionError("一个 tab 最多运行 8 组，请减少参数组合")
    return tree, lists


def _value(node, names):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ExpressionError(f"未知名称 {node.id}", node)
        return names[node.id]
    if isinstance(node, ast.UnaryOp):
        return -_value(node.operand, names)
    if isinstance(node, ast.BinOp):
        left, right = _value(node.left, names), _value(node.right, names)
        if type(left) in (int, float) and type(right) in (int, float):
            try:
                result = _SCALAR_BINARY[type(node.op)](left, right)
            except (ZeroDivisionError, OverflowError):
                raise ExpressionError("数字运算无效", node) from None
            if abs(result) > MAX_NUMBER or not math.isfinite(result):
                raise ExpressionError("数字运算超出范围", node)
            return result
        return call(_BINARY[type(node.op)], left, right)
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ExpressionError("请将连续比较拆成单个比较", node)
        return call(
            _COMPARE[type(node.ops[0])],
            _value(node.left, names),
            _value(node.comparators[0], names),
        )
    if isinstance(node, ast.Call):
        name = node.func.id
        if name not in OPERATORS:
            raise ExpressionError(f"未知函数 {name}", node)
        fn = OPERATORS[name][0]
        args = [_value(a, names) for a in node.args]
        if name.startswith("ts_") or name == "adv":
            if (
                not args
                or type(args[-1]) not in (int, float)
                or not 0 <= args[-1] <= MAX_WINDOW
            ):
                raise ExpressionError(f"窗口须在 0–{MAX_WINDOW} 之间", node)
        try:
            inspect.signature(fn).bind(*args)
            return fn(*args)
        except (TypeError, ValueError, OverflowError):
            raise ExpressionError(f"函数 {name} 的参数不正确", node) from None
    raise ExpressionError("不支持此表达式", node)


def parse(source, factors=None):
    tree, lists = _tree(source)
    factors = registry() if factors is None else factors
    variants = []
    for picks in itertools.product(*(range(len(n.elts)) for n in lists)):
        replacements = {
            (n.lineno, n.col_offset): n.elts[i]
            for n, i in zip(lists, picks, strict=True)
        }

        class Expand(ast.NodeTransformer):
            def visit_List(self, node, choices=replacements):
                return self.visit(
                    copy.deepcopy(choices[(node.lineno, node.col_offset)])
                )

        resolved = Expand().visit(copy.deepcopy(tree))
        names = {
            factor_name(name): f.expression for name, f in factors.items()
        } | FIELD_NODES
        value = None
        for statement in resolved.body:
            value = _value(statement.value, names)
            if isinstance(statement, ast.Assign):
                name = statement.targets[0].id
                if (
                    name in OPERATORS
                    or name in FIELD_NODES
                    or name in {factor_name(n) for n in factors}
                ):
                    raise ExpressionError(
                        "不能覆盖算子、字段或已登记因子的名称", statement
                    )
                names[name] = value
        expression = value if isinstance(value, Expr) else constant(value)
        if expression.lookback is not None and expression.lookback > 10000:
            raise ExpressionError("累计回看窗口过长")
        variants.append(
            Variant(
                ast.unparse(resolved),
                expression,
                tuple(
                    ast.unparse(n.elts[i]) for n, i in zip(lists, picks, strict=True)
                ),
            )
        )
    return variants


def completions(best=None):
    best = best or {}
    items = [
        {
            "label": name,
            "kind": "operator",
            "signature": sig,
            "detail": desc,
            "hint": sig,
        }
        for name, (_, sig, desc) in OPERATORS.items()
    ]
    items += [
        {
            "label": name,
            "kind": "field",
            "detail": FIELD_HELP.get(name, f"{name}，{FIELDS.get(name, ('',))[0]}"),
            "hint": FIELD_LABELS.get(name, FIELD_HELP.get(name, name)),
        }
        for name in FIELD_NODES
    ]
    items += [
        {
            "label": factor_name(f.name),
            "kind": "factor",
            "detail": f.description,
            "hint": f"{best[f.name]:.2f}" if f.name in best else "",
        }
        for f in registry().values()
    ]
    return sorted(
        items,
        key=lambda item: (
            {"factor": 0, "field": 1, "operator": 2}[item["kind"]],
            item["label"],
        ),
    )


def family_id(expression):
    """保留算子与字段结构，抽去数值旋钮；跨批次排除参数同族。"""

    def shape(node):
        params = [
            (key, "parameter" if type(value) in (int, float) else value)
            for key, value in node.params
        ]
        return [node.op, [shape(arg) for arg in node.args], params]

    return hashlib.sha256(
        json.dumps(shape(expression), separators=(",", ":")).encode()
    ).hexdigest()


def suggested_name(source):
    tree, _ = _tree(source)
    last = tree.body[-1]
    if isinstance(last, ast.Assign):
        return last.targets[0].id
    call_node = next((n for n in ast.walk(last) if isinstance(n, ast.Call)), None)
    return call_node.func.id if call_node else "signal"


FIELD_LABELS = {
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "volume": "成交量",
    "vwap": "成交均价",
    "returns": "日收益率",
    "cap": "总市值",
    "amount": "成交额",
    "turnover": "换手率",
    "adj_factor": "复权因子",
    "previous_close": "昨收价",
    "upper_limit": "涨停价",
    "lower_limit": "跌停价",
    "total_market_cap": "总市值",
    "float_market_cap": "流通市值",
    "total_shares": "总股本",
    "float_shares": "流通股本",
    "free_shares": "自由流通股本",
    "industry": "行业",
    "listing_age": "上市天数",
    "is_listed": "是否上市",
    "is_st": "是否 ST",
    "is_suspended": "是否停牌",
    "has_quote": "是否有行情",
    "tradable": "是否可交易",
    "pe": "市盈率",
    "pe_ttm": "滚动市盈率",
    "pb": "市净率",
    "ps": "市销率",
    "ps_ttm": "滚动市销率",
    "dividend_yield": "股息率",
    "dividend_yield_ttm": "滚动股息率",
    "cash_dividend": "现金分红",
    "split": "拆股比例",
    "delist": "是否退市",
    "action_source": "公司行为来源",
    "industry_source": "行业来源",
    **{f"member_csi{code}": f"中证{code}成分" for code in (300, 500, 1000, 2000)},
    **{f"weight_csi{code}": f"中证{code}权重" for code in (300, 500, 1000, 2000)},
    "member_csi_all": "全指成分",
    "weight_csi_all": "全指权重",
}
