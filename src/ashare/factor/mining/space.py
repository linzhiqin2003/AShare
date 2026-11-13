"""搜索空间：用户声明可用字段、算子与参数取值、深度与节点数上限、类型（量纲）约束。

量纲用「价格、成交量」两个基本量的指数加一个对数标记表示：价格 (1,0)、成交量 (0,1)、
成交额与市值 (1,1)、比例与排名 (0,0)。规则（`CATALOG` 里每个算子一条）：

- 加、减、取大、取小：两边量纲必须相同（价格加成交量在生成时就排除）；
- 乘、除：指数相加、相减（价格除以价格得无量纲）；
- 对数：log(价格) 带对数标记，同底的对数相减（`ts_delta`、减法）得无量纲；
- 排名、标准化、相关、符号：结果无量纲；滑动均值、最大最小、延迟、差分：量纲不变。

另有：截面算子的位置（``cs_positions="any"`` 或 ``"root"``）、显然冗余的嵌套（排名套
排名、取负套取负、两个相同的操作数相减）在生成时排除。截面算子（排名、标准化）的参数
默认必须无量纲：``cs_rank(close)`` 这类对价格水平、成交量水平的截面排名量的是「股价高低、
公司大小」，不是这里要找的东西，根节点无量纲的要求也拦不住它（排名的结果无量纲），
所以单独排除；``allow_level_ranks=True`` 放开。

表达式的规范形在 `Expr` 的摘要之上再做一步：可交换算子（加、乘、取大、取小）的操作数
按摘要排序。去重键在规范形上再做三步（只用于判重，不改变生成和求值）：滑动相关
``ts_corr`` 的两个操作数按摘要排序（相关系数对称）；作用在非负量上的 ``abs`` 去掉
（非负字段见 `NONNEGATIVE_FIELDS`，滑动均值、最大最小、标准差等保持非负）；根上的截面
排名与截面标准化（逐日单调变换，不改变排序）剥掉。

```python
from ashare.factor.mining import Space
space = Space.default()                          # 量价字段 + 常用时序、截面算子
space = Space(fields={"close": "price", "volume": "volume"},
              ops={"ts_mean": {"window": (5, 20)}, "divide": {}, "cs_rank": {}},
              max_depth=3, max_nodes=6)
```
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..expression import Expr, call
from ..expression import field as leaf

# ---- 量纲 -------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    price: int = 0
    volume: int = 0
    log: bool = False

    @property
    def dimensionless(self):
        return self.price == 0 and self.volume == 0 and not self.log

    def __str__(self):
        if self.dimensionless:
            return "1"
        base = "·".join(
            f"{n}^{e}" if e != 1 else n
            for n, e in (("价格", self.price), ("量", self.volume))
            if e
        )
        return f"log({base or '1'})" if self.log else base


ONE = Unit()
UNITS = {
    "price": Unit(1, 0),
    "volume": Unit(0, 1),
    "amount": Unit(1, 1),
    "ratio": ONE,
}


class Invalid(ValueError):
    """表达式不满足空间的约束。"""


def _same(units, op):
    a, b = units
    if a != b:
        raise Invalid(f"{op}: 量纲不同（{a} 与 {b}）")
    if op == "subtract" and a.log:
        return ONE
    if op == "add" and a.log:
        raise Invalid("add: 对数相加没有意义")
    return a


def _mul(units, op):
    a, b = units
    if (a.log and not b.dimensionless) or (b.log and not a.dimensionless):
        raise Invalid(f"{op}: 对数量只能与无量纲量相乘除")
    sign = 1 if op == "multiply" else -1
    if a.log or b.log:
        return a if a.log else (b if op == "multiply" else _raise(op))
    return Unit(a.price + sign * b.price, a.volume + sign * b.volume)


def _raise(op):
    raise Invalid(f"{op}: 无量纲量除以对数量没有意义")


def _keep(units, op):
    return units[0]


def _one(units, op):
    return ONE


def _log(units, op):
    a = units[0]
    if a.log:
        raise Invalid("log: 不能对对数再取对数")
    return ONE if a.dimensionless else Unit(a.price, a.volume, True)


def _delta(units, op):
    a = units[0]
    return ONE if a.log else a


def _scale_free(units, op):
    """标准差这类与平移无关、随尺度伸缩的统计量：对数量的结果无量纲。"""
    a = units[0]
    return ONE if a.log else a


@dataclass(frozen=True)
class OpRule:
    arity: int
    kind: str  # "elem"、"ts"、"cs"
    unit: object
    params: tuple = ()  # 允许的参数名
    commutative: bool = False
    idempotent: bool = False  # 套在自己身上没有意义


CATALOG = {
    "add": OpRule(2, "elem", _same, commutative=True),
    "subtract": OpRule(2, "elem", _same),
    "multiply": OpRule(2, "elem", _mul, commutative=True),
    "divide": OpRule(2, "elem", _mul),
    "maximum": OpRule(2, "elem", _same, commutative=True),
    "minimum": OpRule(2, "elem", _same, commutative=True),
    "negate": OpRule(1, "elem", _keep, idempotent=True),
    "abs": OpRule(1, "elem", _keep, idempotent=True),
    "sign": OpRule(1, "elem", _one, idempotent=True),
    "log": OpRule(1, "elem", _log),
    "ts_delay": OpRule(1, "ts", _keep, ("periods",), idempotent=True),
    "ts_delta": OpRule(1, "ts", _delta, ("periods",)),
    "ts_mean": OpRule(1, "ts", _keep, ("window",)),
    "ts_sum": OpRule(1, "ts", _keep, ("window",)),
    "ts_std": OpRule(1, "ts", _scale_free, ("window",)),
    "ts_max": OpRule(1, "ts", _keep, ("window",)),
    "ts_min": OpRule(1, "ts", _keep, ("window",)),
    "ts_median": OpRule(1, "ts", _keep, ("window",)),
    "ts_decay_linear": OpRule(1, "ts", _keep, ("window",)),
    "ts_ema": OpRule(1, "ts", _keep, ("window",)),
    "ts_rank": OpRule(1, "ts", _one, ("window",)),
    "ts_zscore": OpRule(1, "ts", _one, ("window",)),
    "ts_skew": OpRule(1, "ts", _one, ("window",)),
    "ts_slope": OpRule(1, "ts", _keep, ("window",)),
    "ts_argmax": OpRule(1, "ts", _one, ("window",)),
    "ts_argmin": OpRule(1, "ts", _one, ("window",)),
    "ts_corr": OpRule(2, "ts", _one, ("window",)),
    "cs_rank": OpRule(1, "cs", _one, idempotent=True),
    "cs_zscore": OpRule(1, "cs", _one, idempotent=True),
}
# 根上去掉不改变逐日排序的变换（去重键）
MONOTONE_ROOT = ("cs_rank", "cs_zscore")
# 去重键里按对称处理的二元算子（不在 CATALOG 的可交换里：生成时两种顺序都会出现）
SYMMETRIC_KEY = ("ts_corr",)
# 取值非负的字段（价格、成交量、成交额、换手率、市值）：abs 套在它们上面是恒等
NONNEGATIVE_FIELDS = frozenset(
    {"open", "high", "low", "close", "vwap", "volume", "amount", "turnover",
     "total_market_cap", "float_market_cap", "adj_factor"}
)  # fmt: skip
# 非负输入得非负结果的算子（全部操作数非负时）
_KEEP_NONNEG = frozenset(
    {"add", "multiply", "divide", "maximum", "minimum", "ts_delay", "ts_mean", "ts_sum",
     "ts_max", "ts_min", "ts_median", "ts_decay_linear", "ts_ema"}
)  # fmt: skip
# 结果总是非负的算子
_ALWAYS_NONNEG = frozenset({"abs", "ts_std"})
# 两个截面单调变换互相嵌套同样冗余
_CS_MONOTONE = set(MONOTONE_ROOT)

DEFAULT_FIELDS = {
    "open": "price",
    "high": "price",
    "low": "price",
    "close": "price",
    "vwap": "price",
    "volume": "volume",
    "amount": "amount",
    "turnover": "ratio",
}
DEFAULT_OPS = {
    "add": {},
    "subtract": {},
    "multiply": {},
    "divide": {},
    "negate": {},
    "abs": {},
    "log": {},
    "ts_delay": {"periods": (1, 5)},
    "ts_delta": {"periods": (1, 5, 20)},
    "ts_mean": {"window": (5, 10, 20, 60)},
    "ts_std": {"window": (5, 20, 60)},
    "ts_max": {"window": (5, 20)},
    "ts_min": {"window": (5, 20)},
    "ts_rank": {"window": (5, 20)},
    "ts_zscore": {"window": (20, 60)},
    "ts_decay_linear": {"window": (5, 10)},
    "ts_corr": {"window": (10, 20)},
    "cs_rank": {},
}


def _grid(params):
    """{参数: 取值} → 有序的参数组合列表（按参数名排序展开）。"""
    keys = sorted(params)
    combos = [()]
    for k in keys:
        values = params[k]
        values = (
            (values,) if isinstance(values, (int, float, str, bool)) else tuple(values)
        )
        if not values:
            raise ValueError(f"参数 {k} 没有取值")
        combos = [c + ((k, v),) for c in combos for v in values]
    return [dict(c) for c in combos]


@dataclass(frozen=True)
class Space:
    """搜索空间。fields：{字段名: 量纲}（"price"、"volume"、"amount"、"ratio" 或 `Unit`）；
    ops：{算子名: {参数名: 取值序列}}，算子必须在 `CATALOG` 里；max_depth：最大深度
    （字段叶子深度为 0）；max_nodes：最大节点数（字段也算）；cs_positions：截面算子可
    出现的位置，"any" 或 "root"；require_dimensionless：根节点必须无量纲（价格水平本身、
    成交量本身这类与股价高低、公司大小直接相关的量不作为候选）；allow_level_ranks：
    是否允许截面算子作用在有量纲的量上（默认不允许，见模块说明）。"""

    fields: dict = field(default_factory=lambda: dict(DEFAULT_FIELDS))
    ops: dict = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_OPS.items()}
    )
    max_depth: int = 3
    max_nodes: int = 7
    cs_positions: str = "any"
    require_dimensionless: bool = True
    allow_level_ranks: bool = False

    def __post_init__(self):
        fields = {}
        for name, unit in self.fields.items():
            if isinstance(unit, str):
                if unit not in UNITS:
                    raise ValueError(f"字段 {name}: 量纲只能是 {sorted(UNITS)} 或 Unit")
                unit = UNITS[unit]
            if not isinstance(unit, Unit):
                raise TypeError(f"字段 {name}: 量纲类型无效")
            fields[str(name)] = unit
        if not fields:
            raise ValueError("fields: 至少一个字段")
        ops = {}
        for name, params in self.ops.items():
            if name not in CATALOG:
                raise ValueError(f"算子 {name} 不在 CATALOG 里（没有量纲规则）")
            rule = CATALOG[name]
            params = dict(params or {})
            unknown = set(params) - set(rule.params)
            missing = set(rule.params) - set(params)
            if unknown or missing:
                raise ValueError(
                    f"算子 {name}: 需要参数 {rule.params}，给的是 {sorted(params)}"
                )
            ops[name] = _grid(params)
        if self.cs_positions not in ("any", "root"):
            raise ValueError('cs_positions: 只能是 "any" 或 "root"')
        if self.max_depth < 0 or self.max_nodes < 1:
            raise ValueError("max_depth、max_nodes 无效")
        object.__setattr__(self, "fields", dict(sorted(fields.items())))
        object.__setattr__(self, "_grids", dict(sorted(ops.items())))

    @classmethod
    def default(cls, **overrides):
        """默认空间：量价基础字段 + 常用时序／截面算子，深度 3、最多 7 个节点。"""
        return cls(**overrides)

    def describe(self):
        return {
            "fields": {k: [u.price, u.volume, u.log] for k, u in self.fields.items()},
            "ops": {k: v for k, v in self._grids.items()},
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "cs_positions": self.cs_positions,
            "require_dimensionless": self.require_dimensionless,
            "allow_level_ranks": self.allow_level_ranks,
        }

    @property
    def digest(self):
        text = json.dumps(["space", 1, self.describe()], sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def grids(self):
        return self._grids

    # -- 构造与检查 --

    def leaf(self, name):
        return leaf(name)

    def build(self, op, args, params):
        """按规则建一个节点（可交换算子的操作数排好序）。不检查约束。"""
        args = list(args)
        if CATALOG[op].commutative:
            args.sort(key=lambda a: a.digest)
        return call(op, *args, **params)

    def unit(self, expr):
        """表达式的量纲；不满足空间约束时抛 `Invalid`（同时检查深度、节点数、位置、冗余）。"""
        info = self._check(expr, root=True)
        if self.require_dimensionless and not info[0].dimensionless:
            raise Invalid(f"根节点必须无量纲（现在是 {info[0]}）")
        return info[0]

    def valid(self, expr):
        try:
            self.unit(expr)
        except Invalid:
            return False
        return True

    def _check(self, expr, root=False):
        # 返回 (量纲, 深度, 节点数)；显式栈后序，记忆化
        memo = {}
        stack = [(expr, False, root)]
        while stack:
            node, ready, is_root = stack.pop()
            if node.digest in memo and not ready:
                continue
            if not ready:
                stack.append((node, True, is_root))
                stack.extend((a, False, False) for a in reversed(node.args))
                continue
            memo[node.digest] = self._node(
                node, [memo[a.digest] for a in node.args], is_root
            )
        unit, depth, nodes = memo[expr.digest]
        if depth > self.max_depth:
            raise Invalid(f"深度 {depth} 超过 {self.max_depth}")
        if nodes > self.max_nodes:
            raise Invalid(f"节点数 {nodes} 超过 {self.max_nodes}")
        return unit, depth, nodes

    def _node(self, node, children, is_root):
        params = dict(node.params)
        if node.op == "field":
            name = params["value"]
            if name not in self.fields:
                raise Invalid(f"字段 {name} 不在空间里")
            return self.fields[name], 0, 1
        if node.op not in self._grids:
            raise Invalid(f"算子 {node.op} 不在空间里")
        rule = CATALOG[node.op]
        if len(children) != rule.arity:
            raise Invalid(f"{node.op}: 操作数个数不对")
        declared = {k: params[k] for k in rule.params}
        if declared not in self._grids[node.op]:
            raise Invalid(f"{node.op}: 参数 {declared} 不在空间里")
        if rule.kind == "ts":
            extra = {k: v for k, v in params.items() if k not in rule.params}
            if "min_periods" in extra and extra["min_periods"] != params.get("window"):
                raise Invalid(f"{node.op}: 只允许满窗口")
        if rule.kind == "cs" and self.cs_positions == "root" and not is_root:
            raise Invalid(f"{node.op}: 截面算子只能在根上")
        if (
            rule.kind == "cs"
            and not self.allow_level_ranks
            and not children[0][0].dimensionless
        ):
            raise Invalid(
                f"{node.op}: 截面算子的参数必须无量纲（现在是 {children[0][0]}）"
            )
        child_ops = [a.op for a in node.args]
        if rule.idempotent and node.op in child_ops:
            raise Invalid(f"{node.op} 套在自己上冗余")
        if node.op in _CS_MONOTONE and set(child_ops) & _CS_MONOTONE:
            raise Invalid("截面单调变换嵌套冗余")
        if rule.arity == 2 and node.args[0] == node.args[1]:
            raise Invalid(f"{node.op}: 两个操作数相同")
        if rule.commutative and node.args[0].digest > node.args[1].digest:
            raise Invalid(f"{node.op}: 可交换算子的操作数未按规范顺序排列")
        unit = rule.unit([c[0] for c in children], node.op)
        depth = 1 + max(c[1] for c in children)
        nodes = 1 + sum(c[2] for c in children)
        return unit, depth, nodes


def canonical(expr):
    """把可交换算子的操作数按摘要排序（自底向上重建），其余不变。"""
    memo = {}
    stack = [(expr, False)]
    while stack:
        node, ready = stack.pop()
        if node.digest in memo and not ready:
            continue
        if not ready:
            stack.append((node, True))
            stack.extend((a, False) for a in node.args)
            continue
        args = [memo[a.digest] for a in node.args]
        rule = CATALOG.get(node.op)
        if rule is not None and rule.commutative:
            args.sort(key=lambda a: a.digest)
        memo[node.digest] = (
            node
            if all(a is b for a, b in zip(args, node.args, strict=True))
            else Expr(node.op, tuple(args), node.params)
        )
    return memo[expr.digest]


def key_form(expr):
    """去重键用的形式：规范形 + ``ts_corr`` 操作数排序 + 去掉非负量上的 ``abs``。"""
    memo, nonneg = {}, {}
    stack = [(expr, False)]
    while stack:
        node, ready = stack.pop()
        if node.digest in memo and not ready:
            continue
        if not ready:
            stack.append((node, True))
            stack.extend((a, False) for a in node.args)
            continue
        args = [memo[a.digest] for a in node.args]
        rule = CATALOG.get(node.op)
        if (rule is not None and rule.commutative) or node.op in SYMMETRIC_KEY:
            args.sort(key=lambda a: a.digest)
        if node.op == "abs" and nonneg[node.args[0].digest]:
            out = args[0]
        elif all(a is b for a, b in zip(args, node.args, strict=True)):
            out = node
        else:
            out = Expr(node.op, tuple(args), node.params)
        memo[node.digest] = out
        if node.op == "field":
            nonneg[node.digest] = dict(node.params).get("value") in NONNEGATIVE_FIELDS
        else:
            nonneg[node.digest] = node.op in _ALWAYS_NONNEG or (
                node.op in _KEEP_NONNEG and all(nonneg[a.digest] for a in node.args)
            )
    return memo[expr.digest]


def dedup_key(expr):
    """去重键：`key_form` 再去掉根上的逐日单调变换后的摘要。键相同的候选只评一次。"""
    node = key_form(expr)
    while node.op in MONOTONE_ROOT:
        node = node.args[0]
    return node.digest


def size(expr):
    """节点数（字段也算；公共子式按出现次数计）。"""
    count, stack = 0, [expr]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.args)
    return count


def subtrees(expr):
    """先序列出 (路径, 子树)；路径是从根开始的操作数下标元组。"""
    out, stack = [], [((), expr)]
    while stack:
        path, node = stack.pop()
        out.append((path, node))
        for i in reversed(range(len(node.args))):
            stack.append((path + (i,), node.args[i]))
    return out


def replace(expr, path, new):
    """把 path 处的子树换成 new，返回新表达式（可交换算子重新排序）。"""
    if not path:
        return new
    i = path[0]
    args = list(expr.args)
    args[i] = replace(args[i], path[1:], new)
    rule = CATALOG.get(expr.op)
    if rule is not None and rule.commutative:
        args.sort(key=lambda a: a.digest)
    return Expr(expr.op, tuple(args), expr.params)
