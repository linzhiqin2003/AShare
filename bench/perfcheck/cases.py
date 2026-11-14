"""固定基准：全部是合成数据，不读 data/ 下的真实行情，不联网。

每个基准是一个 `setup(workdir) -> call` 函数：setup 只做一次（造数据、预备输入），
返回的无参 `call` 被反复计时。名字一经存进基线就不要改，改了算新基准。
"""

import queue
import shutil
from pathlib import Path

import numpy as np

CASES = {}
# 由 __main__ 按 --threads 设定：平台任务的同时变体数与算子线程数一致。
THREADS = 4


def case(name, repeat=9):
    def register(setup):
        CASES[name] = (setup, repeat)
        return setup

    return register


_PANELS = {}


def _panel(n):
    from bench.platform.synthetic import panel

    if n not in _PANELS:
        _PANELS[n] = panel(n=n, full=True)
    return _PANELS[n]


def _alpha(source):
    from ashare.factor import Evaluator
    from ashare.platform.language import parse

    expression = parse("-ts_std(returns, 20)", factors={})[0].expression
    return Evaluator().evaluate(expression, source)


@case("groupbt.single")
def _groupbt_single(workdir):
    from ashare.factor import groupbt

    source = _panel(300)
    alpha = _alpha(source)
    return lambda: groupbt.run(alpha, source, "ALL")


@case("groupbt.run_many8")
def _groupbt_many(workdir):
    from ashare.factor import groupbt

    source = _panel(300)
    alpha = _alpha(source)
    configurations = [
        dict(universe="ALL", group_neutralize=g, rebalance=r)
        for g in (None, "industry", "cap", "cap+industry")
        for r in (1, 5)
    ]
    return lambda: groupbt.run_many(
        [alpha] * len(configurations), source, configurations, workers=THREADS
    )


@case("factor.evaluate")
def _evaluate(workdir):
    from ashare.factor import Evaluator
    from ashare.platform.language import parse

    source = _panel(1000)
    expression = parse(
        "rank(ts_corr(close, volume, 20)) - ts_rank(returns, 60)", factors={}
    )[0].expression
    return lambda: Evaluator().evaluate(expression, source)


@case("factor.evaluate_many4")
def _evaluate_many(workdir):
    from ashare.factor import Evaluator
    from ashare.platform.language import parse

    source = _panel(1000)
    expressions = {
        str(i): v.expression
        for i, v in enumerate(parse("-ts_std(returns, [5, 10, 20, 60])", factors={}))
    }
    return lambda: list(Evaluator().evaluate_many(expressions, source))


@case("panelbt.weights")
def _weights(workdir):
    import panelbt as bt

    rng = np.random.default_rng(3)
    weights = rng.random((2400, 2000))
    weights /= weights.sum(axis=1, keepdims=True)
    returns = rng.normal(0, 0.01, (2400, 2000))
    return lambda: bt.run_weights(weights, returns)


@case("panelbt.trades")
def _trades(workdir):
    import panelbt as bt

    from bench.compare.common import inputs

    market, target, costs, config = inputs(2400, 2000, interval=1)
    return lambda: bt.run(market, target, costs, config)


def _operator(op, paired=False):
    def setup(workdir):
        import panelops as po

        rng = np.random.default_rng(7)
        x = rng.normal(size=(2400, 2000))
        x[rng.random(x.shape) < 0.03] = np.nan
        y = rng.normal(size=x.shape)
        fn = getattr(po, "ts_" + op)
        if paired:
            return lambda: fn(x, y, 20)
        return lambda: fn(x, 20)

    return setup


for _op, _paired in (("std", False), ("mean", False), ("rank", False), ("corr", True)):
    case(f"panelops.ts_{_op}")(_operator(_op, _paired))


_EIGHT = (
    "-ts_std(returns, [10, 20, 40, 60])",
    {"group_neutralize": ["industry", "cap+industry"]},
)
_FOUR = (
    "-ts_std(returns, [20, 60])",
    {"group_neutralize": ["industry", "cap+industry"]},
)


def _platform_job(workdir, source, settings):
    from ashare.platform.config import expand_settings
    from ashare.platform.language import parse
    from bench.platform.synthetic import write_panel

    data = Path(workdir) / "platform-data"
    if not data.exists():
        write_panel(data, _panel(120))
    variants = parse(source, factors={})
    pairs = [(v, c) for c in expand_settings(settings, len(variants)) for v in variants]
    return data, pairs


def _compute(data, pairs, results):
    from ashare.platform import runner
    from ashare.platform.storage import Store

    shutil.rmtree(results, ignore_errors=True)
    store = Store(results)
    job, runs = store.begin("tab", "perfcheck", pairs, {})
    cache = runner.PanelCache(data)
    runs = runner.prepare_signals(
        runs, cache, results / "work" / job, lambda i, p: None
    )
    out = queue.Queue()
    runner.calculate_many(runs, cache, job, out, THREADS)
    payloads = []
    while not out.empty():
        _, kind, payload = out.get_nowait()
        if kind == "result":
            payloads.append(payload)
    return store, job, payloads


def _platform(source, settings):
    """平台工作进程里的一次任务（同进程）：读面板、求值、并发回测与统计，到结果全部算出为止。

    不含 spawn、HTTP 和落盘；落盘受磁盘争用影响大，单独见 `platform.save8groups`、
    `platform.holdings_archive`，免得磁盘抖动淹没计算上的回归。"""

    def setup(workdir):
        data, pairs = _platform_job(workdir, source, settings)
        results = Path(workdir) / "platform-results"

        def call():
            store, _, _ = _compute(data, pairs, results)
            return store.close

        return call

    return setup


case("platform.4groups", repeat=9)(_platform(*_FOUR))
case("platform.8groups", repeat=9)(_platform(*_EIGHT))


@case("platform.save8groups", repeat=7)
def _platform_save(workdir):
    """8 个变体的结果落盘（序列 NPZ、SQLite 元数据、检验）并结束任务；后台持仓档案不计时。"""
    import copy

    from ashare.platform.storage import Store

    data, pairs = _platform_job(workdir, *_EIGHT)
    template = Path(workdir) / "platform-template"
    store, _, payloads = _compute(data, pairs, template)
    store.close()
    results = Path(workdir) / "platform-save"

    def call():
        shutil.rmtree(results, ignore_errors=True)
        store = Store(results)
        job, runs = store.begin("tab", "perfcheck", pairs, {})
        for run, payload in zip(runs, payloads, strict=True):
            payload = dict(
                payload,
                run=run["id"],
                summary=copy.deepcopy(payload["summary"]),
                arrays=dict(payload["arrays"]),
            )
            store.save(**payload)
        store.finish(job, "complete")
        return store.close

    return call


@case("platform.holdings_archive")
def _holdings_archive(workdir):
    """一个变体的持仓档案（日调仓，约 2600 个快照）压缩写盘，平台在后台线程里做。"""
    from ashare.platform.analysis import run_variant
    from ashare.platform.config import DEFAULTS
    from ashare.platform.holdings import archive_arrays, write_archive
    from ashare.platform.language import parse

    expression = parse("-ts_std(returns, 20)", factors={})[0].expression
    _, arrays = run_variant(
        expression, _panel(120), dict(DEFAULTS, group_neutralize="industry")
    )
    archive = archive_arrays(arrays)
    path = Path(workdir) / "holdings.npz"
    return lambda: write_archive(path, archive)
