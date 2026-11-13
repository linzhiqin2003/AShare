"""模型：协议与注册（含假的第三方模型和缺可选组的提示）、内置模型、保存读回。"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="需要可选依赖组 ml")

import ashare.ml as ml  # noqa: E402
from ashare.ml import _deps  # noqa: E402
from ashare.ml import models as M  # noqa: E402

from .synthetic import make  # noqa: E402


def _batch(n_days=40, per_day=50, k=4, seed=0, signal=1.0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), per_day)
    col = np.tile(np.arange(per_day), n_days)
    X = rng.standard_normal((len(day), k)).astype(np.float32)
    y = signal * X[:, 0] - 0.5 * signal * X[:, 1] + rng.standard_normal(len(day))
    return ml.Batch.from_arrays(X, y, day, col)


def test_ridge_numpy_matches_sklearn():
    b = _batch()
    a = ml.Ridge(alpha=3.0).fit(b)
    s = ml.Ridge(alpha=3.0, solver="sklearn").fit(b)
    c = ml.Ridge(alpha=3.0, chunk=97).fit(b)  # 分块累积与一次算相同（容差内）
    np.testing.assert_allclose(a.coef_, s.coef_, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a.intercept_, s.intercept_, atol=1e-12)
    np.testing.assert_allclose(c.coef_, a.coef_, rtol=1e-10)
    np.testing.assert_allclose(a.predict(b), s.predict(b), atol=1e-10)


def test_lasso_shrinks_noise_features():
    m = ml.Lasso(alpha=0.05).fit(_batch(k=6))
    assert abs(m.coef_[0]) > 0.5 and np.all(m.coef_[2:] == 0)


def test_lgbm_early_stopping_uses_rank_ic():
    train, valid = _batch(seed=1), _batch(seed=2)
    m = ml.LGBM(
        n_estimators=300,
        learning_rate=0.2,
        min_data_in_leaf=20,
        early_stopping=10,
        threads=1,
    )
    m.fit(train, valid)
    hist = np.asarray(m.history)
    assert len(hist) < 300 and m.best_iteration == int(np.argmax(hist)) + 1
    ic = ml.daily_rank_ic(m.predict(valid), valid.y, valid.day, valid.col, valid.n_cols)
    np.testing.assert_allclose(np.nanmean(ic), hist.max(), rtol=1e-12)
    # 纯噪声验证集：提前停止很早触发
    noise = _batch(seed=3, signal=0.0)
    m2 = ml.LGBM(n_estimators=300, min_data_in_leaf=20, early_stopping=10, threads=1)
    m2.fit(_batch(seed=4, signal=0.0), noise)
    assert m2.best_iteration < 100


def test_lgbm_rank_levels_and_groups():
    b = _batch(n_days=5, per_day=40)
    m = ml.LGBMRank(levels=4, n_estimators=10, min_data_in_leaf=5, threads=1)
    lab = m._label(b)
    assert set(np.unique(lab)) == {0, 1, 2, 3}
    for d in range(5):
        seg = lab[b.day == d]
        assert np.bincount(seg.astype(int)).tolist() == [10, 10, 10, 10]
    assert m._group(b).tolist() == [40] * 5
    m.fit(b, None)
    assert np.isfinite(m.predict(b)).all()


def test_ensemble_rank_and_mean():
    train, test = _batch(seed=5), _batch(seed=6)
    e = ml.Ensemble(
        members=[
            ("ridge", {}),
            ("lgbm", {"n_estimators": 20, "min_data_in_leaf": 20, "threads": 1}),
        ],
        seeds=(0, 1),
        combine="rank",
    )
    e.fit(train, None)
    assert len(e.models) == 3  # 岭回归不吃种子，LightGBM 两个种子
    p = e.predict(test)
    parts = [
        ml.rank_within_days(m.predict(test), test.day, test.col, test.n_cols)
        for m in e.models
    ]
    np.testing.assert_allclose(p, np.mean(parts, axis=0))
    e2 = ml.Ensemble(members=["ridge"], combine="mean").fit(train, None)
    np.testing.assert_allclose(e2.predict(test), e2.models[0].predict(test))


@pytest.mark.parametrize(
    "name,params",
    [
        ("ridge", {"alpha": 2.0}),
        ("lasso", {"alpha": 0.01}),
        ("elasticnet", {"alpha": 0.01}),
        ("lgbm", {"n_estimators": 15, "min_data_in_leaf": 20, "threads": 1}),
        (
            "lgbm_rank",
            {"n_estimators": 15, "min_data_in_leaf": 20, "threads": 1, "levels": 5},
        ),
        (
            "ensemble",
            {
                "members": [
                    ("ridge", {}),
                    ("lgbm", {"n_estimators": 5, "threads": 1, "min_data_in_leaf": 20}),
                ]
            },
        ),
    ],
)
def test_save_load_roundtrip(tmp_path, name, params):
    b = _batch()
    m = ml.make_model(name, params)
    m.fit(b, _batch(seed=9) if name.startswith("lgbm") else None)
    ml.save_model(m, tmp_path / name)
    back = ml.load_model(tmp_path / name)
    np.testing.assert_array_equal(back.predict(b), m.predict(b))


def _doc_example():
    """取出模块文档里的自定义模型例子（就是用户照抄的那段）。"""
    doc = M.__doc__
    code = doc[doc.index("```python\n") + 10 : doc.index("\nml.register_model")]
    return code.strip()


def test_docstring_custom_model_runs_through_walk_forward(tmp_path):
    code = _doc_example()
    ns = {"np": np, "Path": Path}
    exec(code, ns)
    assert len(code.splitlines()) <= 20
    ml.register_model("corr_test", ns["CorrWeights"])
    try:
        panel, feats, beta = make(t=400, n=40, k=3, seed=1)
        res = ml.walk_forward(
            feats,
            "ret1",
            data=panel,
            model="corr_test",
            retrain=50,
            train_days=150,
            valid_days=0,
            start=300,
            root=tmp_path,
            threads=1,
        )
        assert res.metrics["ic_mean"] > 0.3
        w = res.models[0].w
        assert np.sign(w[0]) == np.sign(beta[0])
        rec = ml.Ledger(tmp_path / "ledger.jsonl").records()[-1]
        assert (
            rec["model"]["class"].endswith("CorrWeights")
            and rec["model"]["shrink"] == 0.5
        )
    finally:
        ml.unregister_model("corr_test")


def test_fake_third_party_model_and_missing_group(tmp_path, monkeypatch):
    """模拟以后的 nn 组：模型在外部包里，按名字登记；没装依赖时报出缺哪个组、怎么装。"""
    pkg = types.ModuleType("fake_nn_pkg")

    class TinyNet:
        def __init__(self, hidden=4, seed=0):
            self.hidden, self.seed = hidden, seed

        def fit(self, train, valid=None):
            self.w = np.linalg.lstsq(np.nan_to_num(train.X), train.y, rcond=None)[0]

        def predict(self, batch):
            return np.nan_to_num(batch.X) @ self.w

        def save(self, path):
            np.save(Path(path) / "w.npy", self.w)

        @classmethod
        def load(cls, path):
            m = cls()
            m.w = np.load(Path(path) / "w.npy")
            return m

    pkg.TinyNet = TinyNet
    monkeypatch.setitem(sys.modules, "fake_nn_pkg", pkg)
    # 1) 依赖没装：登记时不导入，用到时报 nn 组的安装方法
    monkeypatch.setitem(_deps.GROUPS, "nn", ("fake_torch_not_installed",))
    ml.register_model("tinynet", "fake_nn_pkg:TinyNet", group="nn")
    try:
        with pytest.raises(ml.MissingDependency) as err:
            ml.make_model("tinynet")
        msg = str(err.value)
        assert (
            "`nn`" in msg and "--group nn" in msg and "fake_torch_not_installed" in msg
        )
        panel, feats, _ = make(t=360, n=30, k=2)
        with pytest.raises(ml.MissingDependency):
            ml.walk_forward(
                feats,
                "ret1",
                data=panel,
                model="tinynet",
                retrain=40,
                train_days=150,
                valid_days=0,
                start=300,
                root=tmp_path,
            )
        # 2) 依赖装上（这里换成一个确实存在的模块）：同一个名字直接可用，seed 自动传入
        monkeypatch.setitem(_deps.GROUPS, "nn", ("json",))
        ml.register_model("tinynet", "fake_nn_pkg:TinyNet", group="nn", overwrite=True)
        res = ml.walk_forward(
            feats,
            "ret1",
            data=panel,
            model="tinynet",
            seed=7,
            params={"hidden": 8},
            retrain=40,
            train_days=150,
            valid_days=0,
            start=300,
            root=tmp_path,
        )
        assert res.models[0].seed == 7 and res.models[0].hidden == 8
        assert np.isfinite(res.metrics["ic_mean"])
    finally:
        ml.unregister_model("tinynet")


def test_reserved_nn_names_and_entry_points(monkeypatch):
    monkeypatch.setitem(_deps.GROUPS, "nn", ("fake_torch_not_installed",))
    with pytest.raises(ml.MissingDependency, match="--group nn"):
        ml.make_model("mlp")
    monkeypatch.setitem(_deps.GROUPS, "nn", ("json",))
    with pytest.raises(ValueError, match="还没有包登记"):
        ml.make_model("mlp")
    with pytest.raises(ValueError, match="未知模型"):
        ml.make_model("no_such_model")

    class EP:
        name, value = "ep_ridge", "ashare.ml.models:Ridge"

    monkeypatch.setattr(M, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(
        M.importlib.metadata,
        "entry_points",
        lambda group: [EP()] if group == M.ENTRY_POINT_GROUP else [],
    )
    try:
        assert isinstance(ml.make_model("ep_ridge", {"alpha": 2.0}), ml.Ridge)
        assert "ep_ridge" in ml.available_models()
    finally:
        ml.unregister_model("ep_ridge")


def test_missing_ml_group_message(monkeypatch):
    monkeypatch.setitem(_deps.GROUPS, "ml", ("lightgbm_not_here", "sklearn"))
    with pytest.raises(ml.MissingDependency) as err:
        _deps.require("ml", "ashare.ml ")
    msg = str(err.value)
    assert (
        "`ml`" in msg and "--group ml" in msg and "pip install lightgbm_not_here" in msg
    )


def test_registry_rejects_duplicates_and_bad_models():
    with pytest.raises(ValueError, match="已登记"):
        ml.register_model("ridge", ml.Ridge)
    with pytest.raises(TypeError):
        ml.make_model(object())
    with pytest.raises(ValueError):
        ml.make_model(ml.Ridge(), {"alpha": 1})


def test_params_recorded_for_custom_instances():
    class Plain:
        def __init__(self):
            self.k, self._hidden, self.arr = 3, 1, np.zeros(2)

        def fit(self, *a):
            pass

        def predict(self, b):
            return np.zeros(len(b))

    p = M.params_of(Plain())
    assert p["k"] == 3 and "_hidden" not in p and "arr" not in p
    assert json.dumps(p)
