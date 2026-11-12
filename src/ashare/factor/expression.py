"""可哈希的表达式节点；参数在构建时校验，摘要采用缓存的 Merkle 编码。"""

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from functools import cache, cached_property

import panelops

ELEMENTWISE = {
    "add": 2,
    "subtract": 2,
    "multiply": 2,
    "divide": 2,
    "log": 1,
    "abs": 1,
    "negate": 1,
    "gt": 2,
    "ge": 2,
    "lt": 2,
    "le": 2,
    "eq": 2,
    "ne": 2,
    "and_": 2,
    "or_": 2,
    "not_": 1,
    "where": 3,
    "maximum": 2,
    "minimum": 2,
    "sign": 1,
    "power": 2,
    "signed_power": 2,
    "sqrt": 1,
    "exp": 1,
    "clip": 1,
    "is_nan": 1,
    "fill_nan": 2,
}
_OPERATORS = {name for name in panelops.__all__ if name.startswith(("ts_", "cs_"))}


def _parameter(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("表达式参数必须是有限 JSON 标量")


@cache
def _signature(op):
    return inspect.signature(getattr(panelops, op))


def _normalized(op, args, params):
    """统一默认参数，避免省略默认值与显式填写产生两个缓存。"""
    if op in ELEMENTWISE:
        if len(args) != ELEMENTWISE[op]:
            raise TypeError(f"{op} 的操作数数量不符")
        if op == "clip":
            if set(params) != {"lower", "upper"}:
                raise TypeError("clip 需要 lower 和 upper")
            lo, hi = params["lower"], params["upper"]
            if (
                isinstance(lo, bool)
                or isinstance(hi, bool)
                or not isinstance(lo, (int, float))
                or not isinstance(hi, (int, float))
                or lo > hi
            ):
                raise ValueError("clip 边界必须为递增有限数值")
        elif params:
            raise TypeError(f"{op} 不接受参数")
        return params
    if op == "cs_neutralize":
        params = {
            "exposure_count": len(args) - 1,
            "has_groups": False,
            "has_weights": False,
            "rcond": 1e-12,
        } | params
        if set(params) != {"exposure_count", "has_groups", "has_weights", "rcond"}:
            raise TypeError("中性化存在未知参数")
        count = panelops.api.integer(params["exposure_count"], "exposure_count", 0)
        if any(type(params[k]) is not bool for k in ("has_groups", "has_weights")):
            raise TypeError("中性化标志必须为 bool")
        if len(args) != 1 + count + params["has_groups"] + params["has_weights"]:
            raise ValueError("中性化面板数量不符")
        panelops.api.scalar(params["rcond"], "rcond", 1e-300, 1)
        return params
    if op == "cs_group_demean":
        if len(args) != 2:
            raise TypeError("分组去均值需要分数和行业两个节点")
        bound = _signature(op).bind(args[0], groups=args[1], **params)
    else:
        bound = _signature(op).bind(*args, **params)
    if set(params) & {"mask", "backend", "layout", "groups", "exposures", "weights"}:
        raise TypeError("面板与执行配置不能作为标量节点参数")
    bound.apply_defaults()
    normalized = {
        k: v
        for k, v in bound.arguments.items()
        if k
        not in {
            "x",
            "y",
            "condition",
            "mask",
            "backend",
            "groups",
            "exposures",
            "weights",
        }
    }
    for key in ("window", "min_periods", "periods", "ddof", "limit"):
        if key in normalized and normalized[key] is not None:
            panelops.api.integer(
                normalized[key], key, 0 if key in ("periods", "ddof") else 1
            )
    if "window" in normalized:
        window = normalized["window"]
        if normalized.get("min_periods") is None:
            normalized["min_periods"] = window
        if normalized["min_periods"] > window:
            raise ValueError("min_periods 不得大于 window")
    if "alpha" in normalized and normalized["alpha"] is None:
        normalized["alpha"] = 2 / (normalized["window"] + 1)
    for key in ("q", "lower", "upper", "alpha"):
        if key in normalized:
            panelops.api.scalar(
                normalized[key], key, 1e-300 if key == "alpha" else 0, 1
            )
    if "normalize" in normalized and type(normalized["normalize"]) is not bool:
        raise TypeError("normalize 必须为 bool")
    if "scale" in normalized:
        panelops.api.scalar(normalized["scale"], "scale", 0, float("inf"))
    if "lower" in normalized and normalized["lower"] > normalized["upper"]:
        raise ValueError("lower 不得大于 upper")
    return normalized


@dataclass(frozen=True, eq=False)
class Expr:
    """字段、标量常量或算子节点；结构相等与数值比较明确分开。"""

    op: str
    args: tuple["Expr", ...] = ()
    params: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        if self.op not in _OPERATORS | set(ELEMENTWISE) | {"field", "constant"}:
            raise ValueError(f"未知算子：{self.op}")
        args = tuple(self.args)
        if any(not isinstance(a, Expr) for a in args):
            raise TypeError("节点操作数必须是 Expr")
        params = {k: _parameter(v) for k, v in self.params}
        if len(params) != len(self.params):
            raise ValueError("重复参数")
        if self.op == "constant":
            if (
                args
                or set(params) != {"value"}
                or isinstance(params["value"], bool)
                or not isinstance(params["value"], (int, float))
            ):
                raise ValueError("常量必须是有限数值")
        elif self.op == "field":
            if (
                args
                or not isinstance(params.get("value"), str)
                or not params["value"]
                or set(params) - {"value", "dtype", "broadcast"}
            ):
                raise ValueError("无效字段叶子")
            params = {"dtype": "float64", "broadcast": False} | params
            if (
                params["dtype"] not in ("float64", "int64")
                or type(params["broadcast"]) is not bool
            ):
                raise ValueError("字段 dtype 或 broadcast 无效")
            if params["broadcast"] and params["dtype"] != "float64":
                raise ValueError("市场序列必须为 float64")
        else:
            params = _normalized(self.op, args, params)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "params", tuple(sorted(params.items())))
        # 子节点摘要已经缓存，构建长链时每个节点只编码一次。
        _ = self.digest, self.fields, self.lookback

    def canonical(self):
        """嵌套列表形式；后序遍历显式栈，任意深度的链都不触发递归上限。"""
        done = {}
        stack = [self]
        while stack:
            node = stack[-1]
            pending = [a for a in node.args if id(a) not in done]
            if pending:
                stack.extend(reversed(pending))
                continue
            stack.pop()
            done[id(node)] = [
                2,
                node.op,
                [done[id(a)] for a in node.args],
                list(node.params),
            ]
        return done[id(self)]

    @cached_property
    def digest(self):
        value = [2, self.op, [a.digest for a in self.args], list(self.params)]
        return hashlib.sha256(
            json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def __eq__(self, other):
        return isinstance(other, Expr) and self.digest == other.digest

    def __hash__(self):
        return int(self.digest[:15], 16)

    @cached_property
    def fields(self):
        if self.op == "field":
            return frozenset([dict(self.params)["value"]])
        return frozenset().union(*(a.fields for a in self.args))

    @cached_property
    def lookback(self):
        """额外历史行数；EMA 和无限前填为 None。"""
        parents = [a.lookback for a in self.args]
        params = dict(self.params)
        if (
            self.op == "ts_ema"
            or (self.op == "ts_ffill" and params["limit"] is None)
            or None in parents
        ):
            return None
        base = max(parents, default=0)
        if self.op in ("ts_delay", "ts_delta"):
            return base + params["periods"]
        if self.op == "ts_ffill":
            return base + params["limit"]
        if self.op.startswith("ts_"):
            return base + params["window"] - 1
        return base

    def __str__(self):
        """显式栈逐个输出片段再一次拼接，深链不递归、不反复复制子串。"""
        infix = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
        parts = []
        stack = [self]
        while stack:
            item = stack.pop()
            if isinstance(item, str):
                parts.append(item)
                continue
            p = dict(item.params)
            if item.op == "field":
                parts.append(p["value"] + ("[market]" if p["broadcast"] else ""))
            elif item.op == "constant":
                parts.append(repr(p["value"]))
            elif item.op in infix:
                stack.extend(
                    reversed(
                        ["(", item.args[0], f" {infix[item.op]} ", item.args[1], ")"]
                    )
                )
            else:
                pieces = [*item.args, *(f"{k}={v!r}" for k, v in item.params)]
                tokens = [item.op + "("]
                for i, piece in enumerate(pieces):
                    tokens.extend([", "] if i else [])
                    tokens.append(piece)
                tokens.append(")")
                stack.extend(reversed(tokens))
        return "".join(parts)

    def __add__(self, other):
        return call("add", self, other)

    def __radd__(self, other):
        return call("add", other, self)

    def __sub__(self, other):
        return call("subtract", self, other)

    def __rsub__(self, other):
        return call("subtract", other, self)

    def __mul__(self, other):
        return call("multiply", self, other)

    def __rmul__(self, other):
        return call("multiply", other, self)

    def __truediv__(self, other):
        return call("divide", self, other)

    def __rtruediv__(self, other):
        return call("divide", other, self)

    def __neg__(self):
        return call("negate", self)


def field(name, *, dtype="float64", broadcast=False):
    return Expr(
        "field", params=(("value", name), ("dtype", dtype), ("broadcast", broadcast))
    )


def constant(value):
    return Expr("constant", params=(("value", value),))


def call(op, *args, **params):
    return Expr(
        op,
        tuple(a if isinstance(a, Expr) else constant(a) for a in args),
        tuple(params.items()),
    )


def neutralize(expression, *exposures, groups=None, weights=None, rcond=1e-12):
    args = [expression, *exposures]
    if groups is not None:
        args.append(groups)
    if weights is not None:
        args.append(weights)
    return call(
        "cs_neutralize",
        *args,
        exposure_count=len(exposures),
        has_groups=groups is not None,
        has_weights=weights is not None,
        rcond=rcond,
    )


def clip(x, lower, upper):
    return call("clip", x, lower=lower, upper=upper)


def _named(op):
    def construct(*args):
        return call(op, *args)

    construct.__name__ = op
    construct.__doc__ = f"构建 {op} 逐元素表达式，缺失规则见 SPEC。"
    return construct


for _name in ELEMENTWISE:
    if _name != "clip":
        globals()[_name] = _named(_name)
