"""外部因子包的发现与加载（Python 入口点组 ``ashare.factors``）。

外部包在自己的 ``pyproject.toml`` 里声明::

    [project.entry-points."ashare.factors"]
    mypack = "mypack.plugin:register"

``register(hub)`` 收到一个 ``Hub``，用 ``hub.add(FactorSet(...))`` 挂上一套或几套因子。
没有装任何外部包时只有内置的示例集 ``example``。用法和最小例子见 ``docs/DESIGN.md`` 6.2.1。

加载出错（入口点导入失败、``register`` 抛错、集合重名）一律直接报错并指明是哪个入口点，
不静默跳过：少了一套因子比报错更难发现。
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata

from ..library import Registry

ENTRY_POINT_GROUP = "ashare.factors"


@dataclass(frozen=True)
class FactorSet:
    """一套可以构造注册表、可以被核对工具核对的因子。

    - ``build(**options) -> Registry``：构造注册表；``options`` 只能用 ``options`` 里声明的名字。
    - ``reference``：独立参考实现的模块名（提供 ``compute_many`` 或 ``REFERENCES``），没有就 None。
    - ``chunk``：参考实现能按证券分块计算时给块大小（纯时序的因子集），否则 None（整截面）。
    - ``exceptions``：``{因子名: (KnownException, ...)}``，核对工具的已知例外。
    """

    name: str
    build: Callable[..., Registry]
    reference: str | None = None
    chunk: int | None = None
    exceptions: Mapping = field(default_factory=dict)
    options: tuple = ()
    description: str = ""
    origin: str = "ashare"

    def registry(self, **options):
        unknown = set(options) - set(self.options)
        if unknown:
            raise ValueError(
                f"因子集 {self.name} 不认识选项 {sorted(unknown)}，可选 {list(self.options)}"
            )
        registry = self.build(**options)
        if not isinstance(registry, Registry):
            raise TypeError(f"因子集 {self.name} 的 build 没有返回 Registry")
        return registry


class Hub:
    """``register(hub)`` 收到的对象：收集各个入口点挂上来的因子集。"""

    def __init__(self, origin="ashare"):
        self._sets = {}
        self._origin = origin

    def add(self, factor_set):
        if not isinstance(factor_set, FactorSet):
            raise TypeError("hub.add 需要 FactorSet")
        if not factor_set.name.isidentifier():
            raise ValueError(f"因子集名字须是标识符：{factor_set.name!r}")
        if factor_set.name in self._sets:
            other = self._sets[factor_set.name].origin
            raise ValueError(
                f"因子集重名：{factor_set.name}（{other} 与 {self._origin}）"
            )
        self._sets[factor_set.name] = FactorSet(
            **{**factor_set.__dict__, "origin": self._origin}
        )

    @property
    def sets(self):
        return dict(self._sets)


def _builtin(hub):
    from . import example

    example.register(hub)


_CACHE = {}


def discover(*, refresh=False, entry_points=None):
    """全部可用的因子集 ``{名字: FactorSet}``：内置示例加上各入口点挂上来的。

    ``entry_points`` 只给测试用（替代 ``importlib.metadata`` 的查询结果）。
    """
    if entry_points is None and not refresh and "sets" in _CACHE:
        return dict(_CACHE["sets"])
    hub = Hub()
    _builtin(hub)
    found = (
        metadata.entry_points(group=ENTRY_POINT_GROUP)
        if entry_points is None
        else entry_points
    )
    for ep in sorted(found, key=lambda e: e.name):
        hub._origin = f"{ENTRY_POINT_GROUP}:{ep.name} = {ep.value}"
        try:
            register = ep.load()
        except Exception as error:
            raise ImportError(f"无法加载外部因子包入口点 {hub._origin}") from error
        if not callable(register) or len(inspect.signature(register).parameters) != 1:
            raise TypeError(f"入口点 {hub._origin} 须是一个参数的函数 register(hub)")
        register(hub)
    sets = hub.sets
    if entry_points is None:
        _CACHE["sets"] = sets
    return dict(sets)


def factor_set(name):
    sets = discover()
    if name not in sets:
        raise KeyError(f"没有因子集 {name!r}；可用的有 {sorted(sets)}")
    return sets[name]


def load_registry(name, **options):
    """按名字取一套因子的注册表，例如 ``load_registry("example")``。"""
    return factor_set(name).registry(**options)
