"""账本与缓存键：试验身份的规范化、库版本、失败入账、负结果入账、不可训练时报错。"""

import importlib.util
import json
import sys

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="需要可选依赖组 ml")

import ashare.ml as ml  # noqa: E402
from ashare.ml import models as M  # noqa: E402
from ashare.ml import walk as W  # noqa: E402

from .synthetic import make  # noqa: E402

FAST = {"n_estimators": 20, "min_data_in_leaf": 20, "num_leaves": 7}
KW = dict(retrain=40, train_days=150, valid_days=30, start=300, threads=2)


def _trials(root):
    return [
        r for r in ml.Ledger(root / "ledger.jsonl").records() if r["event"] == "trial"
    ]


def test_store_key_includes_preprocess(tmp_path):
    """同一 root 下两种预处理建库：不同的库、不同的列（键里漏掉预处理会静默读到旧列）。"""
    panel, feats, _ = make(t=200, n=30, k=2, seed=1)
    universe = np.ones(panel.shape, bool)
    a = ml.build_store(feats, panel, universe, ml.Preprocess(), root=tmp_path)
    b = ml.build_store(
        feats, panel, universe, ml.Preprocess(standardize="rank"), root=tmp_path
    )
    assert a.path != b.path
    xa, xb = a.matrix(slice(0, a.n_samples)), b.matrix(slice(0, b.n_samples))
    assert not np.array_equal(xa, xb)
    np.testing.assert_allclose(np.nanmax(np.abs(xb)), 0.5, atol=1e-6)  # 排名居中到 ±0.5


class Anti:
    """反向模型：预测 = −第一个特征，在 beta[0] > 0 的数据上 Rank IC 为负。"""

    def fit(self, train, valid=None):
        return self

    def predict(self, batch):
        return -batch.X[:, 0].astype(np.float64)


def test_negative_ic_trial_is_ledgered(tmp_path):
    panel, feats, _ = make(t=420, n=40, k=2, beta=[1.0, 0.0], seed=2)
    res = ml.walk_forward(feats, "ret1", data=panel, model=Anti(), root=tmp_path, **KW)
    assert res.metrics["ic_mean"] < -0.1
    (rec,) = _trials(tmp_path)
    assert rec["status"] == "ok" and rec["metrics"]["ic_mean"] < -0.1
    assert ml.Ledger(tmp_path / "ledger.jsonl").trials() == 1


def test_seed_written_either_way_is_one_trial(tmp_path):
    panel, feats, _ = make(t=400, n=30, k=2, seed=3)
    kw = dict(KW, root=tmp_path, data=panel)
    a = ml.walk_forward(feats, "ret1", model="lgbm", params=FAST, seed=7, **kw)
    b = ml.walk_forward(feats, "ret1", model="lgbm", params=FAST | {"seed": 7}, **kw)
    assert a.trial_id == b.trial_id and b.cached
    c = ml.walk_forward(feats, "ret1", model="lgbm", params=FAST, seed=8, **kw)
    assert c.trial_id != a.trial_id
    # 岭回归没有随机性：walk_forward 的 seed 不改变定义
    r0 = ml.walk_forward(feats, "ret1", model="ridge", seed=0, **kw)
    r7 = ml.walk_forward(feats, "ret1", model="ridge", seed=7, **kw)
    assert r0.trial_id == r7.trial_id
    recs = {r["trial_id"]: r for r in _trials(tmp_path)}
    assert recs[a.trial_id]["seed"] == 7 and recs[r0.trial_id]["seed"] is None
    assert ml.Ledger(tmp_path / "ledger.jsonl").trials() == 3


SOURCE = """
import numpy as np


class Scaled:
    def __init__(self, k=1.0):
        self.k = k

    def fit(self, train, valid=None):
        return self

    def predict(self, batch):
        return {expr}
"""


def _load(path, module_name, monkeypatch):
    """把同一份源码文件按不同的模块名导入（如同从不同脚本里 import）。"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, mod)
    spec.loader.exec_module(mod)
    return mod.Scaled


def test_custom_model_identity_is_source_not_module(tmp_path, monkeypatch):
    src = tmp_path / "scaled.py"
    src.write_text(SOURCE.format(expr="self.k * batch.X[:, 0].astype(np.float64)"))
    a = _load(src, "scaled_a", monkeypatch)
    b = _load(src, "other_scaled_b", monkeypatch)
    pa, pb = M.params_of(a()), M.params_of(b())
    assert pa == pb and pa["class"] == "Scaled" and len(pa["source"]) == 16
    changed = tmp_path / "scaled2.py"
    changed.write_text(SOURCE.format(expr="-self.k * batch.X[:, 0].astype(np.float64)"))
    assert (
        M.params_of(_load(changed, "scaled_c", monkeypatch)())["source"] != pa["source"]
    )
    # 内置模型只用类名
    assert M.params_of(ml.Ridge())["class"] == "Ridge"
    assert "source" not in M.params_of(ml.Ridge())


def test_custom_model_without_source_is_marked(tmp_path):
    ns = {}
    exec(  # noqa: S102 - 动态生成的类取不到源码
        "class Dyn:\n    def fit(self, t, v=None):\n        return self\n", ns
    )
    p = M.params_of(ns["Dyn"]())
    assert p == {"class": "Dyn", "source": "unavailable"}


def test_library_versions_in_run_key_and_ledger(tmp_path, monkeypatch):
    panel, feats, _ = make(t=400, n=30, k=2, seed=4)
    kw = dict(KW, root=tmp_path, data=panel)
    a = ml.walk_forward(feats, "ret1", model="ridge", **kw)
    rec = _trials(tmp_path)[-1]
    assert set(rec["versions"]) >= {
        "numpy",
        "panelops",
        "factor",
        "lightgbm",
        "sklearn",
    }
    assert rec["versions"]["lightgbm"] and rec["versions"]["sklearn"]
    real = W.library_versions()
    monkeypatch.setattr(W, "library_versions", lambda: real | {"lightgbm": "0.0.0"})
    b = ml.walk_forward(feats, "ret1", model="ridge", **kw)
    assert not b.cached and b.run_id != a.run_id and b.trial_id == a.trial_id
    assert ml.Ledger(tmp_path / "ledger.jsonl").trials() == 1


class Broken:
    def fit(self, train, valid=None):
        raise RuntimeError(f"读不到 {self._where}")

    def predict(self, batch):
        return np.zeros(len(batch))


def test_failed_training_is_ledgered_without_paths(tmp_path):
    panel, feats, _ = make(t=400, n=30, k=2, seed=5)
    model = Broken()
    model._where = str(tmp_path / "secret" / "file.bin")
    with pytest.raises(RuntimeError):
        ml.walk_forward(feats, "ret1", data=panel, model=model, root=tmp_path, **KW)
    (rec,) = _trials(tmp_path)
    assert rec["status"] == "failed" and rec["error"] == "RuntimeError"
    assert "metrics" not in rec
    text = (tmp_path / "ledger.jsonl").read_text()
    assert str(tmp_path) not in text and "secret" not in text
    assert ml.Ledger(tmp_path / "ledger.jsonl").trials() == 1
    # 失败与成功的试验一起计数
    ml.walk_forward(feats, "ret1", data=panel, model="ridge", root=tmp_path, **KW)
    assert ml.Ledger(tmp_path / "ledger.jsonl").trials() == 2


def test_all_missing_feature_raises(tmp_path):
    panel, feats, _ = make(t=400, n=30, k=2, seed=6)
    feats = dict(feats, f1=np.full(panel.shape, np.nan))
    with pytest.raises(ValueError, match="f1 在股票池内全部缺失"):
        ml.walk_forward(feats, "ret1", data=panel, model="ridge", root=tmp_path, **KW)
    with pytest.raises(ValueError, match="全部缺失"):
        ml.build_store(feats, panel, np.ones(panel.shape, bool), memory=True)


class Zero:
    def fit(self, train, valid=None):
        return self

    def predict(self, batch):
        return np.zeros(len(batch.day))


def test_constant_prediction_raises(tmp_path):
    panel, feats, _ = make(t=400, n=30, k=2, seed=7)
    with pytest.raises(ValueError, match="同一个常数"):
        ml.walk_forward(
            feats, "ret1", data=panel, model=Zero(), root=tmp_path, ledger=False, **KW
        )


def test_fold_without_training_samples_raises(tmp_path):
    panel, feats, _ = make(t=400, n=30, k=2, seed=8)
    can = np.array(panel["can_buy_open"])
    can[:300] = False  # 第一折训练期的样本全部买不进
    with pytest.raises(ValueError, match="训练集没有可用样本"):
        ml.walk_forward(
            feats,
            "ret1",
            data=panel.replace(can_buy_open=can),
            model="ridge",
            root=tmp_path,
            ledger=False,
            **KW,
        )


def test_skipped_folds_are_logged(tmp_path):
    panel, feats, _ = make(t=400, n=30, k=2, seed=9)
    lines = []
    res = ml.walk_forward(
        feats,
        "ret1",
        data=panel,
        model="ridge",
        root=tmp_path,
        ledger=False,
        log=lines.append,
        **(KW | {"start": 120}),
    )
    skipped = res.schedule.settings["skipped"]
    assert skipped
    assert sum("跳过一折" in str(x) for x in lines) == len(skipped)


def test_carry_label_end_to_end_lookahead(tmp_path):
    panel, feats, _ = make(t=420, n=40, k=2, seed=10)
    sell = np.array(panel["can_sell_open"])
    sell[np.random.default_rng(1).random(sell.shape) < 0.2] = False
    panel = panel.replace(can_sell_open=sell)
    lab = ml.label("ret1", sell_blocked="carry", max_carry=3)
    res = ml.walk_forward(
        feats, lab, data=panel, model="ridge", root=tmp_path, ledger=False, **KW
    )
    assert res.check()
    for f in res.folds:
        assert f["last_end_day"] is not None
        assert f["last_end_day"] < f["test_rows"][0]
        assert f["last_used_day"] + lab.span < f["test_rows"][0]
    # 篡改实际卖出日：检查必须拒绝
    res.folds[1]["last_end_day"] = res.folds[1]["test_rows"][0]
    with pytest.raises(ValueError, match="实际卖出日"):
        ml.check_lookahead(res)
    json.dumps(res.folds[0]["last_end_day"])
