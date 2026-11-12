"""从「一个函数一条因子」的翻译模块构造注册表。

翻译模块里的函数名形如 ``<前缀>_NNN``（三位编号），没有参数，返回一个表达式，文档字符串是
原文公式。``notes`` 是「因子名 → ((标记, 编号), ...)」表，构造时按名字挂到 ``Factor.notes``；
``excluded`` 是裁决不收的编号，翻译模块里出现这些函数时报错。
"""

import re

from ..library import Registry


def registry_from_module(module, prefix, count, *, notes=None, excluded=None):
    if not prefix.isidentifier():
        raise ValueError(f"前缀须是标识符：{prefix!r}")
    notes = notes or {}
    excluded = excluded or {}
    registry = Registry()
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{3}})")
    names = sorted(n for n in vars(module) if pattern.fullmatch(n))
    for name in names:
        number = int(pattern.fullmatch(name).group(1))
        if not 1 <= number <= count:
            raise ValueError(f"编号越界：{name}")
        if number in excluded:
            raise ValueError(f"{name} 已裁决不收（{excluded[number]}），不应有翻译")
        function = getattr(module, name)
        registry.register(
            name,
            function(),
            (function.__doc__ or "").strip(),
            notes.get(name, ()),
        )
    stray = set(notes) - set(registry.names())
    unknown = {n for n in stray if not pattern.fullmatch(n)}
    if unknown:
        raise ValueError(f"标记表里有不属于 {prefix} 的名字：{sorted(unknown)}")
    return registry
