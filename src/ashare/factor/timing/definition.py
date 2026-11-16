"""因子定义：AST 规范化与隔离名字空间，数据只能通过 view 显式提供。"""

import ast
import builtins
import copy
import inspect
import json
import math
import textwrap
from types import CodeType

import numpy as np
import polars as pl

from ashare.data.crypto.storage import digest, encoded

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "len",
        "range",
        "min",
        "max",
        "sum",
        "abs",
        "float",
        "int",
        "bool",
        "enumerate",
        "zip",
        "tuple",
        "list",
        "dict",
        "slice",
    )
}
SAFE_BUILTINS["__import__"] = (
    builtins.__import__
)  # NumPy 内部懒加载需要；因子 AST 禁止 import 和 __ 名字。
# 这层防常见误用；不是恶意 Python 的操作系统安全沙箱。
FORBIDDEN = {
    "open",
    "exec",
    "eval",
    "compile",
    "getattr",
    "setattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "input",
    "print",
    "breakpoint",
    "load",
    "save",
    "loadtxt",
    "savetxt",
    "fromfile",
    "tofile",
    "memmap",
    "read_csv",
    "read_parquet",
    "scan_parquet",
    "scan_csv",
    "read_ipc",
    "write_csv",
    "write_parquet",
    "write_ipc",
}


def _constant_payload(value):
    if isinstance(value, CodeType):
        return _code_payload(value)
    if isinstance(value, (tuple, frozenset)):
        items = [_constant_payload(v) for v in value]
        return [
            type(value).__name__,
            sorted(items, key=repr) if isinstance(value, frozenset) else items,
        ]
    return [type(value).__name__, repr(value)]


def _code_payload(code):
    # marshal 的对象引用标志不稳定；只取实际执行语义，不含引用计数/调试位置。
    return dict(
        bytecode=code.co_code.hex(),
        constants=[_constant_payload(v) for v in code.co_consts],
        names=code.co_names,
        variables=code.co_varnames,
        free=code.co_freevars,
        cells=code.co_cellvars,
        args=[code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount],
        flags=code.co_flags,
        exceptions=code.co_exceptiontable.hex(),
    )


class Factor:
    __slots__ = ("_params", "_code", "_parameter_names", "family")

    def __setattr__(self, name, value):
        raise AttributeError("Factor 定义只读")

    def __init__(self, function, *, params=None, family="exploration"):
        object.__setattr__(self, "_params", encoded(dict(params or {})))
        object.__setattr__(self, "family", str(family))
        encoded(self.params)
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("因子必须为可读取源码的普通函数")
        fn = tree.body[0]
        if (
            fn.decorator_list
            or fn.args.defaults
            or any(v is not None for v in fn.args.kw_defaults)
            or fn.args.vararg
            or fn.args.kwarg
        ):
            raise ValueError("因子不允许装饰器/默认参数/可变参数；通过 params 冻结参数")
        closure = inspect.getclosurevars(function)
        safe_globals = {"np": np, "pl": pl, "math": math}
        if closure.nonlocals or any(
            k not in safe_globals or v is not safe_globals[k]
            for k, v in closure.globals.items()
        ):
            raise ValueError(
                "因子数据与依赖只能来自 view/params；不允许闭包或外部全局状态"
            )
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                    ast.Global,
                    ast.Nonlocal,
                    ast.ClassDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                raise ValueError("因子不允许导入或外部状态")
            if isinstance(node, ast.Attribute) and (
                node.attr.startswith("_")
                or node.attr in FORBIDDEN
                or node.attr.startswith(("read_", "write_", "scan_"))
            ):
                raise ValueError("因子不允许访问私有属性或外部 I/O")
            if isinstance(node, ast.Name) and (
                node.id.startswith("__") or node.id in FORBIDDEN
            ):
                raise ValueError("因子不允许反射或外部 I/O")
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        if not args or set(a.arg for a in args[1:]) != set(self.params):
            raise ValueError("第一个参数为 view，其余参数必须在 params 中精确给出")
        fn.name = "_factor"
        fn.returns = None
        for a in args:
            a.annotation = None
        if (
            fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)
        ):
            fn.body.pop(0)
        canonical = copy.deepcopy(tree)
        # 名字、注释、排版、局部变量重命名不产生新定义。
        mapping = {a.arg: f"arg{i}" for i, a in enumerate(args)}
        for node in ast.walk(canonical):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id not in mapping
            ):
                mapping[node.id] = f"local{len(mapping)}"
        for node in ast.walk(canonical):
            if isinstance(node, ast.Name):
                node.id = mapping.get(node.id, node.id)
            elif isinstance(node, ast.arg):
                node.arg = mapping.get(node.arg, node.arg)
        # 执行规范化后的同一份代码；重建位置后，排版/重命名不会影响编译结果。
        code = compile(ast.unparse(canonical), "<timing-factor>", "exec")
        object.__setattr__(self, "_code", code)
        object.__setattr__(
            self, "_parameter_names", tuple((k, mapping[k]) for k in self.params)
        )

    @property
    def definition_hash(self):
        # 从实际执行的不可变 code object 与参数计算，不读取外部挂载的身份。
        return digest(
            encoded(
                dict(
                    code=digest(encoded(_code_payload(self._code))),
                    params={
                        target: self.params[name]
                        for name, target in self._parameter_names
                    },
                )
            )
        )

    @property
    def params(self):
        return json.loads(self._params)

    def __call__(self, view):
        namespace = dict(np=np, pl=pl, math=math, __builtins__=SAFE_BUILTINS)
        exec(self._code, namespace)
        return namespace["_factor"](
            view,
            **{target: self.params[name] for name, target in self._parameter_names},
        )


def require_factor(factor):
    """入口先检验精确类型；子类与任何挂载哈希的 callable 都不受信任。"""
    if type(factor) is not Factor:
        raise ValueError("FACTOR_TYPE: 评估只接受未派生的 Factor 定义对象")
    return factor.definition_hash
