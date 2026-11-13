"""搜索：留出段隔离、确定性、去重计数、断点续跑、留出段一次性解锁、折算用的试验数。"""

import json
import math

import numpy as np
import panelops
import pytest

pytest.importorskip("lightgbm")  # 标签与台账复用 ashare.ml（可选依赖组 ml）

from ashare.factor import (  # noqa: E402
    call,
    field,
    stats,  # noqa: E402
)
from ashare.factor.mining import (  # noqa: E402
    Evolution,
    Random,
    Space,
    Split,
    mine,
    open_ledger,
    record_trial,
)
from ashare.factor.mining.synthetic import market, planted_expression  # noqa: E402

SPACE = Space(
    fields={"close": "price", "open": "price", "high": "price", "volume": "volume"},
    ops={
        "divide": {},
        "negate": {},
        "ts_delay": {"periods": (1, 5)},
        "ts_mean": {"window": (5, 10)},
        "ts_rank": {"window": (5,)},
        "cs_rank": {},
    },
    max_depth=3,
    max_nodes=6,
)


@pytest.fixture(scope="module")
def panel():
    return market(t=420, n=60, seed=3, strength=0.003)


def _split(p):
    return Split(valid_start=str(p.dates[200]), holdout_start=str(p.dates[320]))


def _strip(result):
    rows = [{k: v for k, v in r.items() if k not in ("trial_id", "cached")}
            for r in result.candidates]  # fmt: skip
    sig = dict(result.summary["significant"])
    return json.dumps(
        {"rows": rows, "selected": [r["key"] for r in result.selected], "sig": sig},
        sort_keys=True, default=str,
    )  # fmt: skip


def _run(p, tmp, name, strategy=None, **kw):
    kw.setdefault("screen_t", 0.5)
    return mine(p, split=_split(p), space=SPACE, root=tmp / name,
                strategy=strategy or Random(n=30, seed=2, batch=8), **kw)  # fmt: skip


def test_split_is_required(panel, tmp_path):
    with pytest.raises(TypeError):
        mine(panel, space=SPACE, root=tmp_path)
    with pytest.raises(TypeError, match="Split"):
        mine(panel, split=None, space=SPACE, root=tmp_path)
    dates = market(t=800, n=5).dates
    s = Split.recent(dates, holdout_years=1, valid_years=1)
    a, v, h = s.rows(dates)
    assert 0 == a < v < h < len(dates)
    last = dates[-1]
    assert (
        dates[h - 1] <= f"{int(last[:4]) - 1}{last[4:]}" < dates[h]
    )  # 留出段正好是最后一年
    with pytest.raises(ValueError, match="为空"):
        Split.recent(panel.dates, holdout_years=1, valid_years=1).rows(panel.dates)


def test_holdout_perturbation_changes_nothing(panel, tmp_path):
    base = _run(panel, tmp_path, "a")
    h = _split(panel).rows(panel.dates)[2]
    rng = np.random.default_rng(99)
    changed = {}
    for name in panel.fields():
        x = np.array(panel[name], copy=True)
        if x.dtype == bool:
            x[h:] = rng.random(x[h:].shape) < 0.5
        elif x.dtype.kind in "iu":
            x[h:] = rng.integers(0, 2000, x[h:].shape)
        else:
            x[h:] = rng.random(x[h:].shape) * 100 + 1
        changed[name] = x
    other = panel.replace(**changed)
    assert other.version != panel.version
    moved = _run(other, tmp_path, "b")
    assert _strip(moved) == _strip(base)
    assert base.summary["backtested"] > 0 and base.summary["screened_out"] > 0
    # 粗筛掉的同样记一次试验
    assert open_ledger(root=tmp_path / "a").trials() == base.summary["unique"]
    # 对照：扰动留出段之前的一行，结果就会变（证明比较是敏感的）
    x = np.array(panel["open"], copy=True)
    x[h - 30 :] *= rng.random(x[h - 30 :].shape) + 0.5
    early = _run(panel.replace(open=x), tmp_path, "c")
    assert _strip(early) != _strip(base)


def test_deterministic_across_threads(panel, tmp_path):
    before = panelops.get_num_threads()
    try:
        panelops.set_num_threads(1)
        a = _run(panel, tmp_path, "t1", strategy=Evolution(population=10, generations=2, seed=4),
                 threads=1)  # fmt: skip
        panelops.set_num_threads(4)
        b = _run(panel, tmp_path, "t4", strategy=Evolution(population=10, generations=2, seed=4),
                 threads=4)  # fmt: skip
    finally:
        panelops.set_num_threads(before)
    assert _strip(a) == _strip(b)
    c = _run(
        panel, tmp_path, "t5", strategy=Evolution(population=10, generations=2, seed=5)
    )
    assert _strip(c) != _strip(a)


class Fixed:
    """自定义策略：一次性提出给定的表达式（同时验证策略协议对外开放）。"""

    def __init__(self, exprs):
        self.exprs = list(exprs)

    def describe(self):
        return {"strategy": "fixed", "n": len(self.exprs)}

    def run(self, space, evaluate):
        evaluate(self.exprs)


def test_duplicates_still_count_as_trials(panel, tmp_path):
    close = field("close")
    lagged = call("ts_delay", close, periods=5)
    planted = planted_expression()
    # 与 planted 逐日排序完全相同但写法不同（去重键不同）：数值重复
    twin = lagged / close
    other = call("negate", field("open") / call("ts_delay", field("open"), periods=5))
    existing = {"planted": call("cs_rank", planted)}
    res = _run(panel, tmp_path, "d", strategy=Fixed([planted, twin, other, planted]),
               existing=existing, screen_t=-10)  # fmt: skip
    s = res.summary
    assert s["proposed"] == 4 and s["unique"] == 3  # 规范形相同的只评一次
    dup = {r["expr"]: r["duplicate_of"] for r in res.candidates}
    assert dup[str(planted)] == "existing:planted"
    assert dup[str(twin)] == "existing:planted"
    assert s["duplicates"] >= 2
    led = open_ledger(root=tmp_path / "d")
    assert led.trials() == s["unique"] == s["trials_total"]
    assert s["distinct"] == s["backtested"] - s["duplicates"]
    # 手工试验同样累计
    record_trial(led, call("negate", field("close")), None, data_version="x")
    assert led.trials() == s["unique"] + 1


def test_dsr_uses_cumulative_trials(panel, tmp_path):
    root = tmp_path / "n"
    first = _run(panel, tmp_path, "n", strategy=Random(n=20, seed=7))
    second = _run(panel, tmp_path, "n", strategy=Random(n=20, seed=8))
    total = open_ledger(root=root).trials()
    assert second.summary["trials_total"] == total > first.summary["trials_total"]
    row = next(r for r in second.candidates if r["dsr_train"] is not None)
    lo = row["train"]["long"]
    expect = stats.deflated_sharpe(
        lo["sr_period"], lo["days"], lo["skew"], lo["kurt"],
        n_trials=total, sr_var=second.summary["sr_var_train"],
    )  # fmt: skip
    assert row["dsr_train"] == expect
    smaller = stats.deflated_sharpe(lo["sr_period"], lo["days"], lo["skew"], lo["kurt"],
                                    n_trials=20, sr_var=second.summary["sr_var_train"])  # fmt: skip
    assert smaller >= expect


class Killed(Exception):
    pass


def test_resume_matches_single_run(panel, tmp_path):
    strategy = Evolution(population=10, generations=3, seed=11)
    whole = _run(panel, tmp_path, "whole", strategy=strategy)
    count = {"n": 0}

    def kill_after(record):
        count["n"] += 1
        if count["n"] == 17:
            raise Killed

    with pytest.raises(Killed):
        _run(panel, tmp_path, "part", strategy=strategy, progress=kill_after)
    resumed = _run(panel, tmp_path, "part", strategy=strategy)
    assert resumed.summary["cache_hits"] == 17
    assert _strip(resumed) == _strip(whole)
    assert resumed.summary["trials_total"] == whole.summary["trials_total"]


def test_unlock_holdout_once(panel, tmp_path):
    res = _run(panel, tmp_path, "u", select=3)
    assert "未解锁" in res.text()
    out = res.unlock_holdout(panel)
    assert len(out) == min(3, res.summary["distinct"])
    assert all(o["holdout"]["start"] == _split(panel).holdout_start for o in out)
    with pytest.raises(RuntimeError, match="解锁过"):
        res.unlock_holdout(panel)
    led = open_ledger(root=tmp_path / "u")
    assert sum(r["event"] == "holdout_unlock" for r in led.records()) == 1
    with pytest.raises(ValueError, match="版本"):
        res.unlock_holdout(panel.replace(open=np.asarray(panel["open"]) * 1.01))


def test_text_without_expressions(panel, tmp_path):
    res = _run(panel, tmp_path, "x")
    hidden = res.text(show_expressions=False)
    assert all(r["expr"] not in hidden for r in res.candidates)
    json.dumps(res.to_dict())
    assert math.isfinite(res.summary["per_second"])


TINY = Space(
    fields={"close": "price", "open": "price", "volume": "volume"},
    ops={"divide": {}, "negate": {}, "ts_delay": {"periods": (1, 5)},
         "ts_mean": {"window": (5, 20)}},
    max_depth=3,
    max_nodes=5,
)  # fmt: skip


def test_planted_signal_is_found(tmp_path):
    from ashare.factor import Evaluator
    from ashare.factor.mining import Enumerate
    from ashare.factor.mining.search import fingerprint, fingerprint_corr

    p = market(t=800, n=100, seed=21, strength=0.003)
    split = Split.recent(p.dates, holdout_years=1, valid_years=1)
    root = tmp_path / "s"
    res = mine(p, split=split, space=TINY, root=root, strategy=Enumerate(limit=2000))
    a, v, h = split.rows(p.dates)
    rows = np.unique(np.linspace(a + (v - a) // 3, v - 1, 16).astype(np.int64))
    target = fingerprint(Evaluator().evaluate(planted_expression(), p), rows)

    def corr(r):
        return fingerprint_corr(
            target, np.load(root / "fingerprints" / f"{r['trial_id']}.npy")
        )

    twins = [r for r in res.candidates if r["stage"] == "backtest" and corr(r) > 0.95]
    assert twins, "埋入的信号（或数值上等价的写法）没被回测"
    assert any(r["duplicate_of"] is None for r in twins)
    assert max(r["dsr_train"] for r in twins) > 0.95
    assert res.summary["significant"]["ic_holm"] >= 1
    assert corr(res.selected[0]) > 0.6


def test_backtest_market_is_truncated(panel, tmp_path):
    import panelbt
    from panelbt.markets import Board

    from ashare.backtest.market import BacktestMarket, market_inputs

    def bm(p):
        board = np.full(len(p.symbols), int(Board.MAIN), dtype=np.int64)
        return BacktestMarket(panelbt.Market(**market_inputs(p)), np.asarray(p.dates),
                              np.asarray(p.symbols), board, p)  # fmt: skip

    base = _run(panel, tmp_path, "m1", market=bm(panel))
    h = _split(panel).rows(panel.dates)[2]
    x = np.array(panel["open"], copy=True)
    x[h:] *= 3.0
    other = panel.replace(open=x)
    moved = _run(other, tmp_path, "m2", market=bm(other))
    assert _strip(moved) == _strip(base)
    assert base.summary["backtested"] > 0
    out = base.unlock_holdout(panel)
    assert out and out[0]["holdout"]["long"]["days"] == len(panel.dates) - h
    with pytest.raises(TypeError, match="market_from_panel"):
        _run(panel, tmp_path, "m3", market=bm(panel).market)


def test_minimal_example_runs(tmp_path):
    # 文档里的最短例子（十行以内照抄能跑）
    from ashare.factor.mining import Random, Split, mine
    from ashare.factor.mining.synthetic import market

    panel = market(t=900, n=150, strength=0.003)
    split = Split.recent(panel.dates, holdout_years=1, valid_years=1)
    result = mine(panel, split=split, strategy=Random(n=60, seed=1), root=tmp_path)
    assert "台账累计试验 60" in result.text()
