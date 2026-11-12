"""Explicit input-conversion boundaries for the two timing scopes."""

import importlib
from dataclasses import fields

import panelbt as panel
from panelbt import _core
from panelbt.api import _prepare, _result
from panelbt.reference import run as reference


def clone_inputs(market, targets):
    m = panel.Market(
        **{x.name: getattr(market, x.name) for x in fields(market) if x.init}
    )
    plans = [
        panel.Targets(x.days, x.indptr, x.cols, x.weights, x.kind) for x in targets
    ]
    return m, plans


def prepare(engine, market, target, costs, config, task):
    if engine == "panelbt":
        args = _prepare(market, [target], costs, config)
        return lambda: _result(_core.run(*args, 1)[0]).nav
    if engine == "reference":
        return lambda: reference(market, target, costs, config).nav
    if engine in ("numpy_full", "numpy_sparse"):
        module = importlib.import_module("numpy_daily")
        return lambda: module.run(market, target, task, sparse=engine == "numpy_sparse")
    module = importlib.import_module(engine)
    args = (
        module.prepare(market, target, task)
        if engine in ("bt_strategy", "backtrader_strategy", "vectorbt_strategy")
        else module.prepare(market, target)
    )
    return lambda: module.execute(args)


JOBS = []
WARM_JIT = False


def initialize_block(engine, t, n, interval, task, variants):
    from common import inputs, plan

    global JOBS, WARM_JIT
    WARM_JIT = engine == "vectorbt_strategy"
    m, _, c, g = inputs(t, n, interval, task=task)
    JOBS = [
        prepare(engine, m, plan(t, n, interval, i, task), c, g, task) for i in variants
    ]


def execute_block():
    return [fn() for fn in JOBS]


def ready():
    if WARM_JIT and JOBS:
        JOBS[0]()
    return len(JOBS)
