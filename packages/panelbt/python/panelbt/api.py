"""校验、零拷贝只读绑定和 numpy 结果整理。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _core
from .models import (
    COUNTERS,
    METRICS,
    Config,
    Costs,
    DelistEvent,
    Market,
    Result,
    Targets,
    Trade,
    vector,
)


def _prepare(
    market: Market, targets: Sequence[Targets], costs: Costs, config: Config
) -> tuple:
    if not isinstance(market, Market):
        raise ValueError("market: 期望 Market")
    t, n = market.shape
    for target in targets:
        if not isinstance(target, Targets):
            raise ValueError("targets: 期望 Targets 或 Targets 序列")
        target.validate_shape(t, n)
    scalars = [
        costs.commission,
        costs.min_commission,
        costs.transfer_fee,
        costs.slippage,
        config.initial_cash,
        config.max_participation,
        config.delist_recovery,
        config.cash_buffer,
    ]
    names = [
        "commission",
        "min_commission",
        "transfer_fee",
        "slippage",
        "initial_cash",
        "max_participation",
        "delist_recovery",
        "cash_buffer",
    ]
    for name, value in zip(names, scalars, strict=True):
        try:
            valid = np.isscalar(value) and np.isfinite(value) and value >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{name}: 期望有限非负标量")
    if config.initial_cash <= 0:
        raise ValueError("initial_cash: 期望 > 0")
    if costs.slippage >= 1:
        raise ValueError("slippage: 期望 < 1")
    for name in ("max_participation", "delist_recovery", "cash_buffer"):
        if getattr(config, name) > 1:
            raise ValueError(f"{name}: 期望在 [0, 1] 内")
    vectors = [
        vector(costs.tax_sell, t, "tax_sell"),
        vector(costs.tax_buy, t, "tax_buy"),
        vector(config.min_qty, n, "min_qty"),
        vector(config.qty_step, n, "qty_step"),
    ]
    empty_f = np.empty((0, 0), dtype=np.float64)
    empty_b = np.empty((0, 0), dtype=np.bool_)
    floats = [
        getattr(market, name) if getattr(market, name) is not None else empty_f
        for name in ("exec_price", "close", "volume", "split", "cash_div")
    ]
    masks = [
        getattr(market, name) if getattr(market, name) is not None else empty_b
        for name in ("can_buy", "can_sell", "delist")
    ]
    plans = [(x.days, x.indptr, x.cols, x.weights) for x in targets]
    return (
        floats,
        masks,
        plans,
        vectors,
        scalars,
        [config.carry_blocked, config.record_trades],
    )


def _result(raw: dict[str, Any]) -> Result:
    daily = raw["daily"]
    values = {
        name: daily[i].astype(np.int64) if name.startswith("n_blocked") else daily[i]
        for i, name in enumerate(METRICS)
    }
    return Result(
        **values,
        positions=raw["positions"],
        trades=None if raw["trades"] is None else [Trade(*x) for x in raw["trades"]],
        delist_events=[DelistEvent(*x) for x in raw["events"]],
        diagnostics=dict(zip(COUNTERS, raw["counts"], strict=True)),
    )


def run(
    market: Market,
    targets: Targets,
    costs: Costs | None = None,
    config: Config | None = None,
) -> Result:
    """目标在给定行成交，不自动移动信号；nav 从初始资金 1 开始。"""
    prepared = _prepare(market, [targets], costs or Costs(), config or Config())
    return _result(_core.run(*prepared, 1)[0])


def run_batch(
    market: Market,
    targets: Sequence[Targets],
    costs: Costs | None = None,
    config: Config | None = None,
    threads: int = 0,
) -> Result:
    """K 组目标共享市场；threads=0 使用所有核，输出前置 K 维。"""
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 0:
        raise ValueError("threads: 期望非负整数，0 表示全部核")
    config = config or Config()
    prepared = _prepare(market, targets, costs or Costs(), config)
    return _batch_result(
        _core.run(*prepared, threads), *market.shape, config.record_trades
    )


def _batch_result(raw: list[dict[str, Any]], t: int, n: int, record: bool) -> Result:
    results = [_result(x) for x in raw]
    daily = {
        name: np.stack([getattr(x, name) for x in results])
        if results
        else np.empty(
            (0, t), dtype=np.int64 if name.startswith("n_blocked") else np.float64
        )
        for name in METRICS
    }
    positions = None
    if record:
        positions = (
            np.stack([x.positions for x in results]) if results else np.empty((0, t, n))
        )
    return Result(
        **daily,
        positions=positions,
        trades=[x.trades for x in results] if record else None,
        delist_events=[x.delist_events for x in results],
        diagnostics={
            key: np.array([x.diagnostics[key] for x in results], dtype=np.int64)
            for key in COUNTERS
        },
    )
