"""公共数据结构及 numpy 输入校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def array(value: ArrayLike, name: str, dtype: Any, ndim: int) -> NDArray:
    try:
        a = np.asarray(value)
        if dtype == np.int64 and (
            a.dtype.kind not in "iu" or (a.size and np.any(a > np.iinfo(np.int64).max))
        ):
            raise ValueError(f"{name}: 期望 int64 整数下标")
        a = np.ascontiguousarray(a, dtype=dtype)
    except (TypeError, OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name}: 期望可转换为 {np.dtype(dtype)} 的数组；{exc}"
        ) from exc
    if a.ndim != ndim:
        raise ValueError(f"{name}: 期望 {ndim} 维数组，实际形状 {a.shape}")
    # 只读视图不改变调用方原数组的可写标志；运行期间调用方不得修改底层数据。
    a = a.view()
    a.flags.writeable = False
    return a


def nonnegative(
    a: NDArray, name: str, *, nan: bool = False, positive: bool = False
) -> None:
    bad = np.isinf(a) | ((a <= 0) if positive else (a < 0))
    if not nan:
        bad |= np.isnan(a)
    if np.any(bad):
        raise ValueError(
            f"{name}: 期望有限{'正' if positive else '非负'}数"
            + (" 或 NaN" if nan else "")
        )


def vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: 期望数值标量或长度 {size} 的数组") from exc
    if a.ndim == 0:
        a = np.full(size, a.item())
    a = array(a, name, np.float64, 1)
    if a.shape != (size,):
        raise ValueError(f"{name}: 期望标量或长度 {size}，实际形状 {a.shape}")
    nonnegative(a, name)
    return a


@dataclass(frozen=True)
class Market:
    """对齐行情；在构造时校验一次，允许多个 run 共享只读面板。"""

    exec_price: ArrayLike
    close: ArrayLike
    can_buy: ArrayLike
    can_sell: ArrayLike
    volume: ArrayLike | None = None
    split: ArrayLike | None = None
    cash_div: ArrayLike | None = None
    delist: ArrayLike | None = None

    def __post_init__(self) -> None:
        shape = None
        for name in (
            "exec_price",
            "close",
            "can_buy",
            "can_sell",
            "volume",
            "split",
            "cash_div",
            "delist",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            dtype = (
                np.bool_ if name in ("can_buy", "can_sell", "delist") else np.float64
            )
            a = array(value, name, dtype, 2)
            if shape is None:
                shape = a.shape
                if 0 in shape:
                    raise ValueError("exec_price: 期望非空 T×N 面板")
            if a.shape != shape:
                raise ValueError(
                    f"{name}: 期望与 exec_price 相同的形状 {shape}，实际 {a.shape}"
                )
            if dtype == np.float64:
                nonnegative(
                    a,
                    name,
                    nan=name in ("exec_price", "close", "volume"),
                    positive=name in ("exec_price", "close", "split"),
                )
            object.__setattr__(self, name, a)
        if any(
            getattr(self, key) is None
            for key in ("exec_price", "close", "can_buy", "can_sell")
        ):
            raise ValueError("Market: exec_price、close、can_buy、can_sell 必须提供")

    @property
    def shape(self) -> tuple[int, int]:
        return self.exec_price.shape  # type: ignore[union-attr]


@dataclass(frozen=True)
class Targets:
    """稀疏调仓计划；空的有效调仓行表示清仓。"""

    days: ArrayLike
    indptr: ArrayLike
    cols: ArrayLike
    weights: ArrayLike

    def __post_init__(self) -> None:
        for name in ("days", "indptr", "cols", "weights"):
            dtype = np.float64 if name == "weights" else np.int64
            value = getattr(self, name)
            # 空列表没有类型信息，按约定创建整数数组。
            if dtype == np.int64 and np.asarray(value).size == 0:
                value = np.asarray(value, dtype=np.int64)
            object.__setattr__(self, name, array(value, name, dtype, 1))
        d, p, c, w = self.days, self.indptr, self.cols, self.weights
        if np.any(d < 0) or np.any(np.diff(d) <= 0):
            raise ValueError("days: 期望非负且严格升序的调仓日下标")
        if (
            len(p) != len(d) + 1
            or p[0] != 0
            or p[-1] != len(c)
            or np.any(np.diff(p) < 0)
        ):
            raise ValueError("indptr: 期望长度 R+1、从 0 单调不减到 nnz")
        if len(c) != len(w) or np.any(c < 0):
            raise ValueError("cols/weights: 期望相同长度且 cols 非负")
        nonnegative(w, "weights")
        rows = np.repeat(np.arange(len(d)), np.diff(p))
        sums = np.bincount(rows, weights=w, minlength=len(d))
        if np.any(sums > 1 + 1e-9):
            raise ValueError("weights: 期望每个调仓行之和 <= 1 + 1e-9")
        order = np.lexsort((c, rows))
        ordered = c[order]
        if len(c) > 1 and np.any((np.diff(rows[order]) == 0) & (np.diff(ordered) == 0)):
            raise ValueError("cols: 同一调仓行不允许重复列号")
        if not np.array_equal(order, np.arange(len(c))):
            object.__setattr__(self, "cols", array(ordered, "cols", np.int64, 1))
            object.__setattr__(
                self, "weights", array(w[order], "weights", np.float64, 1)
            )

    def validate_shape(self, t: int, n: int) -> None:
        if len(self.days) and self.days[-1] >= t:
            raise ValueError(f"days: 期望下标 < T={t}")
        if len(self.cols) and np.max(self.cols) >= n:
            raise ValueError(f"cols: 期望下标 < N={n}")

    @classmethod
    def from_dense(cls, weights: ArrayLike) -> Targets:
        w = array(weights, "weights", np.float64, 2)
        days = np.flatnonzero(~np.all(np.isnan(w), axis=1))
        rows, cols = np.nonzero((w != 0) & ~np.isnan(w))
        counts = np.bincount(rows, minlength=w.shape[0])[days]
        indptr = np.concatenate(([0], np.cumsum(counts)))
        return cls(days, indptr, cols, w[rows, cols])

    def to_dense(self, t: int, n: int) -> FloatArray:
        if not isinstance(t, int) or not isinstance(n, int) or t < 0 or n < 0:
            raise ValueError("T/N: 期望非负整数")
        self.validate_shape(t, n)
        w = np.full((t, n), np.nan)
        w[self.days] = 0
        rows = np.repeat(self.days, np.diff(self.indptr))
        w[rows, self.cols] = self.weights
        return w


@dataclass(frozen=True)
class Costs:
    commission: float = 2.5e-4
    min_commission: float = 5.0
    tax_sell: ArrayLike = 0.0
    tax_buy: ArrayLike = 0.0
    transfer_fee: float = 1e-5
    slippage: float = 5e-4


@dataclass(frozen=True)
class Config:
    initial_cash: float = 1e7
    min_qty: ArrayLike = 100.0
    qty_step: ArrayLike = 100.0
    max_participation: float = 0.0
    carry_blocked: bool = True
    delist_recovery: float = 1.0
    cash_buffer: float = 0.0
    record_trades: bool = False


@dataclass(frozen=True)
class Trade:
    day: int
    col: int
    qty: float
    price: float
    commission: float
    tax: float
    transfer_fee: float
    slippage_cost: float


@dataclass(frozen=True)
class DelistEvent:
    day: int
    col: int
    qty: float
    price: float
    proceeds: float


METRICS = (
    "nav",
    "cash",
    "turnover",
    "commission",
    "tax",
    "transfer_fee",
    "slippage_cost",
    "n_blocked_buy",
    "n_blocked_sell",
)
COUNTERS = (
    "trades",
    "blocked_buy",
    "blocked_sell",
    "cash_scaled",
    "min_commission",
    "delist",
    "dividend",
    "split",
    "carry_trade",
)


@dataclass
class Result:
    nav: FloatArray
    cash: FloatArray
    turnover: FloatArray
    commission: FloatArray
    tax: FloatArray
    transfer_fee: FloatArray
    slippage_cost: FloatArray
    n_blocked_buy: IntArray
    n_blocked_sell: IntArray
    delist_events: list[DelistEvent] | list[list[DelistEvent]]
    trades: list[Trade] | list[list[Trade]] | None = None
    positions: FloatArray | None = None
    diagnostics: dict[str, int | IntArray] = field(default_factory=dict)
