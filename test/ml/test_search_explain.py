"""超参搜索（每组都入账、Bonferroni 折算）与解释（重要性、置换重要性）。"""

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="需要可选依赖组 ml")

import ashare.ml as ml  # noqa: E402

from .synthetic import make  # noqa: E402

KW = dict(retrain=50, train_days=150, valid_days=30, start=300, threads=2)


def test_grid_search_ledgers_every_config(tmp_path):
    panel, feats, _ = make(t=420, n=30, k=3, seed=1)
    grid = {"num_leaves": [3, 7], "learning_rate": [0.05, 0.1]}
    res = ml.search(
        feats,
        "ret1",
        data=panel,
        model="lgbm",
        grid=grid,
        base={"n_estimators": 20, "min_data_in_leaf": 20},
        root=tmp_path,
        **KW,
    )
    assert len(res.table) == 4
    assert sorted(tuple(sorted(r["params"].items())) for r in res.table) == sorted(
        tuple(sorted(c.items())) for c in ml.grid_configs(grid)
    )
    book = ml.Ledger(tmp_path / "ledger.jsonl")
    trials = [r for r in book.records() if r["event"] == "trial"]
    assert len(trials) == 4 and all("search" in r["tags"] for r in trials)
    assert res.n_trials_total == 4
    assert res.threshold == pytest.approx(0.05 / 4)
    for r in res.table:
        assert r["p_adjusted"] == pytest.approx(min(1.0, r["p"] * 4))
        assert r["significant"] == (r["p"] < 0.05 / 4)
    assert "Bonferroni" in res.describe() and res.best is not None
    # 再跑一次同样的搜索：全部命中缓存，不算新的试验
    again = ml.search(
        feats,
        "ret1",
        data=panel,
        model="lgbm",
        grid=grid,
        base={"n_estimators": 20, "min_data_in_leaf": 20},
        root=tmp_path,
        **KW,
    )
    assert all(r["cached"] for r in again.table) and again.n_trials_total == 4
    # 之前的试验也计入门槛：账本里总数增加，门槛变严
    ml.walk_forward(feats, "ret1", data=panel, model="ridge", root=tmp_path, **KW)
    third = ml.search(
        feats,
        "ret1",
        data=panel,
        model="lgbm",
        grid={"num_leaves": [3]},
        base={"n_estimators": 20, "min_data_in_leaf": 20},
        root=tmp_path,
        **KW,
    )
    assert third.n_trials_total == 5


def test_random_search_reproducible():
    space = {
        "num_leaves": [7, 15, 31],
        "learning_rate": (0.01, 0.2, "log"),
        "min_data_in_leaf": (10, 100, "int"),
    }
    a, b = ml.random_configs(space, 6, seed=3), ml.random_configs(space, 6, seed=3)
    assert a == b and len(a) == 6
    assert all(
        0.01 <= c["learning_rate"] <= 0.2 and 10 <= c["min_data_in_leaf"] <= 100
        for c in a
    )
    assert ml.random_configs(space, 6, seed=4) != a


def test_search_requires_exactly_one_space(tmp_path):
    panel, feats, _ = make(t=300, n=10, k=1)
    with pytest.raises(ValueError):
        ml.search(feats, data=panel, root=tmp_path)


def test_permutation_importance_finds_real_feature(tmp_path):
    panel, feats, beta = make(t=460, n=60, k=3, beta=[1.0, 0.0, 0.0], seed=2)
    res = ml.walk_forward(
        feats,
        "ret1",
        data=panel,
        model="lgbm",
        params={"n_estimators": 40, "min_data_in_leaf": 30},
        root=tmp_path,
        ledger=False,
        **KW,
    )
    imp = ml.permutation_importance(res, seed=1)
    assert imp["folds"].shape == (len(res.folds), 3)
    assert imp["mean"][0] > 0.2
    assert np.all(np.abs(imp["mean"][1:]) < 0.05)
    gain = res.importance()
    assert gain["folds"].shape == (len(res.folds), 3)
    np.testing.assert_allclose(gain["folds"].sum(axis=1), 1.0)
    assert np.argmax(gain["mean"]) == 0
    assert len(gain["dates"]) == len(res.folds)


def test_permute_within_days_keeps_daily_sets():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(5), 7)
    x = np.arange(35.0)
    y = ml.permute_within_days(x, day, rng)
    for d in range(5):
        assert sorted(y[day == d]) == sorted(x[day == d])
    assert not np.array_equal(x, y)
