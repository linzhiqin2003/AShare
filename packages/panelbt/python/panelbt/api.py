"""Validated zero-copy binding and NumPy result assembly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _core
from .models import (
    COUNTERS,
    EXTRA_COUNTERS,
    METRICS,
    RULE_COUNTERS,
    Config,
    Costs,
    DelistEvent,
    ExpiryEvent,
    LiquidationEvent,
    Market,
    Result,
    Rules,
    Targets,
    Trade,
    TradeReason,
    array,
    vector,
)


def _panel(value, t, n, name, *, optional=False):
    if value is None and optional:
        return np.empty((0, 0))
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.shape == (t,):
        a = a.reshape(t, 1)
    if a.shape not in ((1, 1), (t, 1), (t, n)):
        raise ValueError(f"{name}: expected scalar, T or T×N")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f"{name}: expected finite nonnegative values")
    return array(a, name, np.float64, 2)


_EMPTY = np.empty(0)
_EMPTY.flags.writeable = False


def _percent(value, t, n, name):
    """Rule percentage as a flat array of length 0, 1, N or T×N."""
    if value is None:
        return _EMPTY
    a = np.asarray(value, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1)
    elif a.ndim == 1:
        if a.shape != (n,):
            raise ValueError(f"{name}: expected scalar, length N or T×N")
    elif a.shape == (t, n):
        a = a.ravel()
    else:
        raise ValueError(f"{name}: expected scalar, length N or T×N")
    if np.any(np.isinf(a) | (a <= 0)):
        raise ValueError(f"{name}: expected positive fractions or NaN")
    return array(a, name, np.float64, 1)


def _rules(market: Market, rules: Rules | None, any_limits: bool):
    """Kernel rule arguments, or None when neither rules nor limits are used."""
    if rules is None:
        if any_limits:
            raise ValueError("limits: limit targets require Config(rules=Rules(...))")
        return None
    if not isinstance(rules, Rules):
        raise ValueError("rules: expected Rules")
    if market.futures is not None:
        raise ValueError(
            "rules: futures columns are not supported with exit rules or limit targets"
        )
    if market.open is None or market.high is None or market.low is None:
        raise ValueError("rules: Market needs open, high and low panels")
    timing = {"open": 0, "close": 1, "average": 2}.get(rules.fill_time)
    if timing is None:
        raise ValueError("fill_time: expected open, close or average")
    if rules.both_hit not in ("stop", "target"):
        raise ValueError("both_hit: expected stop or target")
    if rules.limit_valid not in ("day", "until_next"):
        raise ValueError("limit_valid: expected day or until_next")
    for name in ("max_hold", "cooldown"):
        value = getattr(rules, name)
        if value is not None and (
            not isinstance(value, int | np.integer)
            or isinstance(value, bool)
            or value < (1 if name == "max_hold" else 0)
        ):
            raise ValueError(
                f"{name}: expected None or an integer >= "
                f"{1 if name == 'max_hold' else 0}"
            )
    for name in ("dividend_adjusts_cost", "sell_lock"):
        if not isinstance(getattr(rules, name), bool):
            raise ValueError(f"{name}: expected bool")
    t, n = market.shape
    arrays = [market.open.ravel(), market.high.ravel(), market.low.ravel()]
    arrays += [
        _percent(getattr(rules, k), t, n, k)
        for k in ("stop_loss", "take_profit", "trailing_stop")
    ]
    terms = (
        int(rules.max_hold or 0),
        timing,
        rules.both_hit == "target",
        -1 if rules.cooldown is None else int(rules.cooldown),
        rules.dividend_adjusts_cost,
        rules.sell_lock,
        rules.limit_valid == "until_next",
    )
    return arrays, terms


def _prepare(
    market: Market, targets: Sequence[Targets], costs: Costs, config: Config
) -> tuple:
    if not isinstance(market, Market):
        raise ValueError("market: expected Market")
    t, n = market.shape
    margin = config.margin
    for target in targets:
        if not isinstance(target, Targets):
            raise ValueError("targets: expected Targets")
    # Bounds, sign, integer-contract and row-sum checks for all plans run in
    # the kernel, in parallel, before any simulation starts.
    names = (
        "commission_rate",
        "commission_min",
        "commission_per_share",
        "commission_max_rate",
        "impact_exponent",
    )
    scalars = [getattr(costs, k) for k in names]
    scalars[1] = 0.0 if scalars[1] is None else scalars[1]
    scalars[3] = -1.0 if scalars[3] is None else scalars[3]
    cnames = (
        "initial_cash",
        "max_participation",
        "delist_recovery",
        "cash_buffer",
        "trade_band",
    )
    scalars += [getattr(config, k) for k in cnames]
    for i, (name, value) in enumerate(zip((*names, *cnames), scalars, strict=True)):
        if i == 3 and costs.commission_max_rate is None:
            continue
        try:
            valid = np.isscalar(value) and np.isfinite(value) and value >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{name}: expected finite nonnegative scalar")
    if config.initial_cash <= 0 or costs.impact_exponent <= 0:
        raise ValueError("initial_cash/impact_exponent: expected > 0")
    for name in ("max_participation", "delist_recovery", "cash_buffer"):
        if getattr(config, name) > 1:
            raise ValueError(f"{name}: expected [0, 1]")
    if costs.impact_k is not None and market.volume is None:
        raise ValueError("impact_k requires volume")
    if margin.margin_call not in ("liquidate", "record"):
        raise ValueError("margin_call: expected liquidate or record")
    vectors = [
        vector(getattr(costs, k), t, k)
        for k in (
            "tax_sell",
            "tax_buy",
            "fee_rate_sell",
            "fee_rate_buy",
            "fee_per_share_sell",
            "fee_per_share_buy",
        )
    ]
    vectors += [
        vector(config.min_qty, n, "min_qty"),
        vector(config.qty_step, n, "qty_step"),
    ]
    vectors += [
        vector(getattr(config, k), t, k) for k in ("dividend_tax_rate", "cash_rate")
    ]
    if (vectors[8] > 1).any():
        raise ValueError("dividend_tax_rate: expected [0, 1]")
    flow = np.asarray(config.cash_flow, dtype=float)
    if flow.ndim == 0:
        flow = np.full(t, flow)
    flow = array(flow, "cash_flow", np.float64, 1)
    if flow.shape != (t,) or not np.isfinite(flow).all():
        raise ValueError("cash_flow: expected finite length T")
    vectors += [flow]
    vectors += [
        vector(getattr(margin, k), t, k)
        for k in (
            "long_initial",
            "short_initial",
            "long_maintenance",
            "short_maintenance",
            "financing_rate",
        )
    ]
    ef, eb = np.empty((0, 0)), np.empty((0, 0), bool)
    floats = [
        getattr(market, k) if getattr(market, k) is not None else ef
        for k in ("exec_price", "close", "volume", "split", "cash_div")
    ]
    slip = _panel(costs.slippage, t, n, "slippage")
    if (slip >= 1).any():
        raise ValueError("slippage: expected < 1")
    # Slippage and impact require scalar or full panels; borrow additionally accepts T.
    if np.asarray(costs.slippage).ndim == 1:
        raise ValueError("slippage: expected scalar or T×N")
    impact = _panel(costs.impact_k, t, n, "impact_k", optional=True)
    if costs.impact_k is not None and np.asarray(costs.impact_k).shape != (t, n):
        raise ValueError("impact_k: expected T×N")
    floats += [slip, impact, _panel(margin.borrow_rate, t, n, "borrow_rate")]
    eb = eb.view(np.uint8)
    masks = [
        getattr(market, k).view(np.uint8) if getattr(market, k) is not None else eb
        for k in ("can_buy", "can_sell", "delist", "can_short")
    ]
    plans = [
        (
            x.days,
            x.indptr,
            x.cols,
            x.weights,
            x.kind == "shares",
            _EMPTY if x.limits is None else x.limits,
        )
        for x in targets
    ]
    # The cash specialization is chosen by the account terms alone; futures
    # columns do not change how the spot side of a cash account is executed.
    classic = (
        not margin.allow_short
        and not margin.allow_financing
        and np.all(vectors[11] == 1)
        and np.all(vectors[13] == 0)
        and np.all(vectors[14] == 0)
    )
    flags = [
        config.carry_blocked,
        config.record_trades,
        margin.allow_short,
        margin.allow_financing,
        margin.short_proceeds_usable,
        margin.margin_call == "liquidate",
        classic,
    ]
    if market.futures is None:
        futures = (
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            [],
        )
    else:
        f = market.futures
        futures = (
            f.is_future.view(np.uint8),
            f.multiplier,
            f.expiry,
            [
                f.settlement,
                f.margin_rate,
                f.fee_rate,
                f.fee_per_contract,
                f.delivery_fee_rate,
                f.delivery_fee_per_contract,
            ],
        )
    return (
        floats,
        masks,
        plans,
        vectors,
        scalars,
        flags,
        market.first_delist,
        (market.action_indptr, market.action_cols),
        futures,
    )


def _checks(targets: Sequence[Targets], config: Config) -> tuple:
    """Per-plan content flags and the account context for kernel checks."""
    margin = config.margin
    return (
        [x.validated for x in targets],
        (margin.allow_short, not margin.allow_financing and not margin.allow_short),
    )


def _rule_args(market: Market, targets: Sequence[Targets], config: Config):
    return _rules(market, config.rules, any(x.limits is not None for x in targets))


def _validated(market: Market, targets: Sequence[Targets]) -> bool:
    return market.validated and all(x.validated for x in targets)


def _trades(raw: dict[str, Any]):
    if raw["trades"] is None:
        return None
    if not raw["reasons"]:
        return [Trade(*x) for x in raw["trades"]]
    reasons = dict(raw["reasons"])
    return [
        Trade(*x, TradeReason(reasons[k])) if k in reasons else Trade(*x)
        for k, x in enumerate(raw["trades"])
    ]


def _diagnostics(raw: dict[str, Any]) -> dict:
    out = dict(zip(COUNTERS, raw["counts"][: len(COUNTERS)], strict=True))
    if raw["rule_counts"]:
        out.update(zip(RULE_COUNTERS, raw["rule_counts"], strict=True))
    return out


def _result(raw: dict[str, Any], validated: bool = True) -> Result:
    values = {
        k: raw["daily"][i].astype(np.int64)
        if k.startswith("n_")
        else raw["daily"][i].astype(bool)
        if k == "margin_call"
        else raw["daily"][i]
        for i, k in enumerate(METRICS)
    }
    return Result(
        **values,
        positions=raw["positions"],
        trades=_trades(raw),
        delist_events=[DelistEvent(*x) for x in raw["events"]],
        diagnostics=_diagnostics(raw),
        coverage=dict(zip(EXTRA_COUNTERS, raw["counts"][len(COUNTERS) :], strict=True)),
        contribution=raw["contribution"],
        liquidation_events=[LiquidationEvent(*x) for x in raw["liquidations"]],
        bankruptcy_events=raw["bankruptcies"],
        expiry_events=[ExpiryEvent(*x) for x in raw["expiries"]],
        inputs_validated=validated,
    )


def run(
    market: Market,
    targets: Targets,
    costs: Costs | None = None,
    config: Config | None = None,
) -> Result:
    """Execute targets on their given rows without shifting signals."""
    config = config or Config()
    prepared = _prepare(market, [targets], costs or Costs(), config)
    raw = _core.run(
        *prepared,
        *_checks([targets], config),
        1,
        _rule_args(market, [targets], config),
    )[0]
    return _result(raw, _validated(market, [targets]))


def run_batch(
    market: Market,
    targets: Sequence[Targets],
    costs: Costs | None = None,
    config: Config | None = None,
    threads: int = 0,
) -> Result:
    """Share one market across K strategies; threads=0 uses available CPUs."""
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 0:
        raise ValueError("threads: expected nonnegative integer")
    config = config or Config()
    return _batch_result(
        _core.run(
            *_prepare(market, targets, costs or Costs(), config),
            *_checks(targets, config),
            threads,
            _rule_args(market, targets, config),
        ),
        *market.shape,
        config.record_trades,
        _validated(market, targets),
    )


def _batch_result(raw, t, n, record, validated=True):
    if not record:
        # Assemble one contiguous daily tensor without K intermediate Results.
        tensor = (
            np.stack([x["daily"] for x in raw])
            if raw
            else np.empty((0, len(METRICS), t))
        )
        counts = np.asarray([x["counts"] for x in raw], dtype=np.int64).reshape(
            len(raw), len(COUNTERS) + len(EXTRA_COUNTERS)
        )
        rule_counts = (
            np.asarray([x["rule_counts"] for x in raw], dtype=np.int64)
            if raw and raw[0]["rule_counts"]
            else None
        )
        daily = {
            key: tensor[:, i].astype(np.int64)
            if key.startswith("n_")
            else tensor[:, i].astype(bool)
            if key == "margin_call"
            else tensor[:, i]
            for i, key in enumerate(METRICS)
        }
        return Result(
            **daily,
            positions=None,
            trades=None,
            diagnostics={key: counts[:, i] for i, key in enumerate(COUNTERS)}
            | (
                {}
                if rule_counts is None
                else {k: rule_counts[:, i] for i, k in enumerate(RULE_COUNTERS)}
            ),
            coverage={
                key: counts[:, len(COUNTERS) + i]
                for i, key in enumerate(EXTRA_COUNTERS)
            },
            delist_events=[[DelistEvent(*e) for e in x["events"]] for x in raw],
            liquidation_events=[
                [LiquidationEvent(*e) for e in x["liquidations"]] for x in raw
            ],
            bankruptcy_events=[x["bankruptcies"] for x in raw],
            expiry_events=[[ExpiryEvent(*e) for e in x["expiries"]] for x in raw],
            contribution=np.stack([x["contribution"] for x in raw])
            if raw
            else np.empty((0, n)),
            inputs_validated=validated,
        )
    results = [_result(x, validated) for x in raw]
    daily = {
        k: np.stack([getattr(x, k) for x in results])
        if results
        else np.empty(
            (0, t),
            dtype=bool
            if k == "margin_call"
            else np.int64
            if k.startswith("n_")
            else float,
        )
        for k in METRICS
    }
    return Result(
        **daily,
        positions=(
            np.stack([x.positions for x in results]) if results else np.empty((0, t, n))
        )
        if record
        else None,
        trades=[x.trades for x in results] if record else None,
        delist_events=[x.delist_events for x in results],
        diagnostics={
            k: np.array([x.diagnostics[k] for x in results], dtype=np.int64)
            for k in (results[0].diagnostics if results else COUNTERS)
        },
        coverage={
            k: np.array([x.coverage[k] for x in results], dtype=np.int64)
            for k in EXTRA_COUNTERS
        },
        contribution=np.stack([x.contribution for x in results])
        if results
        else np.empty((0, n)),
        liquidation_events=[x.liquidation_events for x in results],
        bankruptcy_events=[x.bankruptcy_events for x in results],
        expiry_events=[x.expiry_events for x in results],
        inputs_validated=validated,
    )
