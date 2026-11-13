"""滚动训练：已知答案、独立标签、防前视、间隔机制、可复现、缓存与账本、与直白实现对拍。"""

import math

import numpy as np
import pytest

pytest.importorskip("lightgbm", reason="需要可选依赖组 ml")

import lightgbm as lgb  # noqa: E402
from sklearn.linear_model import Ridge as SkRidge  # noqa: E402

import ashare.ml as ml  # noqa: E402
from ashare.ml import labels as L  # noqa: E402
from ashare.ml.split import Fold, Schedule  # noqa: E402
from ashare.ml.walk import fit_folds  # noqa: E402

from .synthetic import SynthPanel, dates, make  # noqa: E402

FAST = {"n_estimators": 40, "min_data_in_leaf": 40, "num_leaves": 15}


def _walk(panel, feats, tmp_path, model="ridge", params=None, **kw):
    opts = dict(
        retrain=40,
        train_days=200,
        valid_days=40,
        start=300,
        threads=2,
        root=tmp_path,
        ledger=False,
        cache=False,
    )
    opts.update(kw)
    return ml.walk_forward(
        feats, ml.label("ret1"), data=panel, model=model, params=params, **opts
    )


def test_ridge_recovers_coefficients_and_theoretical_ic(tmp_path):
    panel, feats, beta = make(t=700, n=120, k=4, noise=1.0, seed=1)
    res = _walk(panel, feats, tmp_path, start=300, retrain=100, train_days=250)
    for m in res.models:
        np.testing.assert_allclose(m.coef_ / 0.01, beta, atol=0.06)
    rho = np.linalg.norm(beta) / math.sqrt(beta @ beta + 1.0)
    spearman = 6 / math.pi * math.asin(rho / 2)
    assert abs(res.metrics["ic_mean"] - spearman) < 0.03, (
        res.metrics["ic_mean"],
        spearman,
    )
    assert np.isfinite(res.pred[300:-2]).all()
    assert np.isnan(res.pred[:300]).all()


def test_independent_labels_center_on_zero(tmp_path):
    means = []
    for seed in range(12):
        panel, feats, _ = make(t=420, n=60, k=3, independent=True, seed=seed)
        res = _walk(
            panel,
            feats,
            tmp_path,
            model="lgbm",
            params=FAST,
            start=260,
            retrain=40,
            train_days=160,
            valid_days=30,
        )
        means.append(res.metrics["ic_mean"])
    means = np.array(means)
    se = means.std(ddof=1) / math.sqrt(len(means))
    assert abs(means.mean()) < 3 * se + 1e-3, means
    assert 2 <= (means > 0).sum() <= 10


# 预测期从第 300 行起每 40 行一折（300、340、380、420）。D 取三处：第一折中间（330）、
# 第二折中间且不在任何折边界上（361）、折边界（380）。D 落在后面的折边界时，最后一折的
# 训练集截止在 D 之前，「分箱边界用了较晚一折的训练集」这类前视改动碰不到它，测不出来；
# D 放进前面的折里才能抓到（见 bench/ml/mutations.py 的 shared_bins_use_last_fold）。
CUTS = (330, 361, 380)


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize(
    "model,params",
    [
        ("ridge", None),
        ("lgbm", FAST),
        ("lgbm", FAST | {"bins": "per_fold"}),
        ("lgbm_rank", FAST | {"levels": 5}),
        ("lasso", {"alpha": 1e-5}),
    ],
)
def test_changing_later_data_never_changes_earlier_predictions(
    tmp_path, model, params, cut
):
    """改动第 D 日及以后的全部数据（行情、可买性、特征），D 之前的预测逐位不变。"""
    panel, feats, _ = make(t=460, n=50, k=3, seed=4, blocked=0.1)
    base = _walk(panel, feats, tmp_path, model=model, params=params)
    starts = [f.test[0] for f in base.schedule.folds]
    assert starts == [300, 340, 380, 420]
    assert (cut in starts) == (cut == 380)
    rng = np.random.default_rng(99)
    arrays = {}
    for name in ("open", "close", "total_market_cap"):
        a = np.array(panel[name], dtype=float)
        a[cut:] *= np.exp(rng.normal(0, 0.3, a[cut:].shape))
        arrays[name] = a
    can = np.array(panel["can_buy_open"])
    can[cut:] = rng.random(can[cut:].shape) < 0.5
    arrays["can_buy_open"] = can
    other = {}
    for k, v in feats.items():
        v = v.copy()
        v[cut:] = rng.standard_normal(v[cut:].shape)
        other[k] = v
    changed = panel.replace(**arrays)
    assert changed.version != panel.version
    after = _walk(changed, other, tmp_path, model=model, params=params)
    np.testing.assert_array_equal(after.pred[:cut], base.pred[:cut])
    assert np.isfinite(base.pred[300:cut]).any()  # D 之前确实有预测可比
    assert not np.array_equal(after.pred[cut:], base.pred[cut:], equal_nan=True)
    assert base.check() and after.check()


def test_check_lookahead_rejects_tampered_fold(tmp_path):
    panel, feats, _ = make(t=420, n=30, k=2)
    res = _walk(panel, feats, tmp_path)
    for f in res.folds:
        assert f["last_used_day"] + res.label.span < f["test_rows"][0]
    res.folds[1]["last_used_day"] = res.folds[1]["test_rows"][0] - res.label.span
    with pytest.raises(ValueError, match="不早于预测期首日"):
        ml.check_lookahead(res)


class NearestNeighbour:
    """记忆型模型：预测 = 特征最近的那个训练样本的标签（演示间隔为什么必要）。"""

    def fit(self, train, valid=None):
        order = np.argsort(train.X[:, 0], kind="stable")
        self.x, self.y = train.X[order, 0].astype(np.float64), train.y[order]
        return self

    def predict(self, batch):
        x = batch.X[:, 0].astype(np.float64)
        i = np.clip(np.searchsorted(self.x, x), 1, len(self.x) - 1)
        left_closer = np.abs(x - self.x[i - 1]) <= np.abs(self.x[i] - x)
        return np.where(left_closer, self.y[i - 1], self.y[i])

    def save(self, path):
        pass


def _leak_setup(h=10, t=260, n=40, seed=0):
    """标签 = 独立噪声的 h 日收益（相邻信号日的标签重叠 h−1 天）；特征是一个随时间单调
    变化、各股票错开的「指纹」，和收益本身无关，只让记忆型模型找到同一只股票最近的日子。
    没有间隔时，训练集最后几天的标签覆盖了测试日的收益（未来收益经标签混进了训练），
    样本外 IC 虚高；间隔 ≥ lag + h 后就回到 0 附近。"""
    rng = np.random.default_rng(seed)
    step = rng.normal(0, 0.02, (t, n))
    logp = np.cumsum(step, axis=0)
    arrays = {
        "open": 10 * np.exp(logp),
        "adj_factor": np.ones((t, n)),
        "can_buy_open": np.ones((t, n), bool),
        "can_sell_open": np.ones((t, n), bool),
    }
    panel = SynthPanel(arrays, dates(t), [f"S{i}" for i in range(n)])
    finger = np.arange(n)[None, :] * 1000.0 + np.arange(t)[:, None] * 1.0
    universe = np.ones((t, n), bool)
    store = ml.build_store(
        {"finger": finger},
        panel,
        universe,
        ml.Preprocess(winsor=None, standardize=None),
        memory=True,
    )
    lab = ml.label("ret1", horizon=h)
    values = L.compute(panel, lab, universe)
    return store, values, universe, lab


def _ic(store, values, universe, folds, span):
    sched = Schedule(tuple(folds), store.dates, span, 0, {})
    pred, infos, _ = fit_folds(
        store,
        store.gather(values.target),
        store.gather(values.trainable(universe)),
        store.gather(values.raw),
        sched,
        NearestNeighbour,
        threads=1,
    )
    days = [f.test[0] for f in folds]
    ic = ml.panel_summary(store.scatter(pred), values.raw, 1, rows=days)["ic"]
    return float(np.nanmean(ic)), infos


def test_gap_removes_overlap_leak():
    store, values, universe, lab = _leak_setup(t=320, n=200)
    span = lab.span
    test_days = range(150, 300)
    no_gap = [Fold(k, (d - 60, d), None, (d, d + 1)) for k, d in enumerate(test_days)]
    with_gap = [
        Fold(k, (d - span - 60, d - span), None, (d, d + 1))
        for k, d in enumerate(test_days)
    ]
    leak, infos = _ic(store, values, universe, no_gap, span)
    clean, infos2 = _ic(store, values, universe, with_gap, span)
    assert all(
        i["last_used_day"] == d - 1 for i, d in zip(infos, test_days, strict=True)
    )
    assert all(
        i["last_used_day"] + span < d for i, d in zip(infos2, test_days, strict=True)
    )
    assert leak > 0.6, leak  # 虚高：重叠 h−1 天
    assert abs(clean) < 0.04, clean
    # 剂量关系而不只是开关：训练集截止在预测日前 g 天时，最后一天的标签与预测日标签
    # 重叠 h−1−g 天，理论 Rank IC ≈ (h−1−g)/h；g = 5、h = 10 时约 0.4。
    half = [
        Fold(k, (d - 5 - 60, d - 5), None, (d, d + 1)) for k, d in enumerate(test_days)
    ]
    mid, _ = _ic(store, values, universe, half, span)
    assert abs(mid - 0.4) < 0.05, mid
    # 正式入口不允许不安全的间隔
    with pytest.raises(ValueError, match="不能小于标签跨度"):
        ml.schedule(store.dates, span, start=150, gap=span - 1)


def test_reproducible_and_fold_parallel_identical(tmp_path):
    panel, feats, _ = make(t=460, n=50, k=3, seed=5)
    a = _walk(panel, feats, tmp_path, model="lgbm", params=FAST)
    b = _walk(panel, feats, tmp_path, model="lgbm", params=FAST)
    c = _walk(
        panel, feats, tmp_path, model="lgbm", params=FAST, fold_workers=2, threads=2
    )
    np.testing.assert_array_equal(a.pred, b.pred)
    np.testing.assert_array_equal(a.pred, c.pred)
    assert [f["best_iteration"] for f in a.folds] == [
        f["best_iteration"] for f in c.folds
    ]


def _naive_folds(panel, feats, sched, universe, lab, fit_predict):
    """直白实现：每折从面板逐日拼训练表、重新拟合，与正式实现不共用代码。"""
    values = L.compute(panel, lab, universe)
    names = list(feats)
    pre = ml.Preprocess()
    proc = {k: pre.apply(v, universe) for k, v in feats.items()}
    out = np.full(universe.shape, np.nan)
    for f in sched.folds:
        rows_x, rows_y = [], []
        for d in range(*f.train):
            for j in range(universe.shape[1]):
                ok = (
                    universe[d, j]
                    and np.isfinite(values.target[d, j])
                    and values.buyable[d, j]
                )
                if ok:
                    rows_x.append([proc[k][d, j] for k in names])
                    rows_y.append(values.target[d, j])
        X = np.asarray(rows_x, np.float32)
        model = fit_predict(X, np.asarray(rows_y))
        for d in range(*f.test):
            cols = np.flatnonzero(universe[d])
            xt = np.asarray([[proc[k][d, j] for k in names] for j in cols], np.float32)
            out[d, cols] = model(xt)
    return out


def test_ridge_matches_naive_sklearn(tmp_path):
    panel, feats, _ = make(t=360, n=30, k=3, seed=6, blocked=0.1)
    res = _walk(
        panel, feats, tmp_path, start=280, retrain=40, train_days=150, valid_days=0
    )
    universe = ml.default_universe(panel)

    def fit(X, y):
        m = SkRidge(alpha=1.0).fit(X.astype(np.float64), y)
        return lambda x: m.predict(x.astype(np.float64))

    ref = _naive_folds(panel, feats, res.schedule, universe, ml.label("ret1"), fit)
    np.testing.assert_allclose(res.pred, ref, rtol=1e-8, atol=1e-12, equal_nan=True)


def test_lgbm_per_fold_bitwise_matches_naive(tmp_path):
    panel, feats, _ = make(t=360, n=30, k=3, seed=7, blocked=0.1)
    params = FAST | {"bins": "per_fold", "early_stopping": 0}
    res = _walk(
        panel,
        feats,
        tmp_path,
        model="lgbm",
        params=params,
        start=280,
        retrain=40,
        train_days=150,
        valid_days=0,
    )
    universe = ml.default_universe(panel)
    lp = ml.LGBM(**params).lgb_params(2)

    def fit(X, y):
        b = lgb.train(lp, lgb.Dataset(X, label=y, params=lp), params["n_estimators"])
        return lambda x: b.predict(x, num_threads=2)

    ref = _naive_folds(panel, feats, res.schedule, universe, ml.label("ret1"), fit)
    np.testing.assert_array_equal(res.pred, ref)
    shared = _walk(
        panel,
        feats,
        tmp_path,
        model="lgbm",
        params=FAST | {"early_stopping": 0},
        start=280,
        retrain=40,
        train_days=150,
        valid_days=0,
    )
    assert not np.array_equal(shared.pred, res.pred, equal_nan=True)  # 分箱不同
    assert abs(shared.metrics["ic_mean"] - res.metrics["ic_mean"]) < 0.05


def test_blocked_samples_not_trained_but_predicted(tmp_path):
    panel, feats, _ = make(t=420, n=40, k=2, seed=8, blocked=0.3)
    res = _walk(panel, feats, tmp_path)
    values = res.labels
    universe = ml.default_universe(panel)
    blocked = universe & ~values.buyable
    rows = np.arange(300, 420 - 2)
    assert blocked[rows].any()
    assert np.isfinite(res.pred[rows][blocked[rows]]).all()
    train = values.trainable(universe)
    assert not (train & blocked).any()
    total = sum(f["n_train"] for f in res.folds)
    expect = sum(
        train[f["train_rows"][0] : f["train_rows"][1]].sum() for f in res.folds
    )
    assert total == expect


def test_cache_and_ledger(tmp_path):
    panel, feats, _ = make(t=400, n=30, k=2, seed=9)
    kw = dict(
        retrain=40, train_days=200, valid_days=40, start=300, threads=2, root=tmp_path
    )
    a = ml.walk_forward(feats, "ret1", data=panel, model="lgbm", params=FAST, **kw)
    b = ml.walk_forward(feats, "ret1", data=panel, model="lgbm", params=FAST, **kw)
    assert not a.cached and b.cached
    np.testing.assert_array_equal(a.pred, b.pred)
    assert b.models is not None and len(b.models) == len(a.folds)
    book = ml.Ledger(tmp_path / "ledger.jsonl")
    rec = book.records()
    assert [r["event"] for r in rec] == ["trial", "cache_hit"]
    assert book.trials() == 1
    r = rec[0]
    for key in (
        "data_version",
        "features",
        "label",
        "preprocess",
        "schedule",
        "model",
        "seed",
        "code_version",
        "code",
        "metrics",
        "folds",
        "seconds",
        "schedule_table",
    ):
        assert key in r, key
    assert r["features"][0][1].startswith("array:")
    ml.walk_forward(
        feats, "ret1", data=panel, model="lgbm", params=FAST | {"num_leaves": 7}, **kw
    )
    ml.walk_forward(feats, "ret1", data=panel, model="lgbm", params=FAST, seed=1, **kw)
    assert book.trials() == 3
    # 线程数不同：重新训练但不算新的一组
    ml.walk_forward(
        feats, "ret1", data=panel, model="lgbm", params=FAST, **(kw | {"threads": 1})
    )
    assert book.trials() == 3


def test_to_targets_alignment(tmp_path):
    panel, feats, _ = make(t=400, n=40, k=2, seed=10)
    res = _walk(panel, feats, tmp_path)
    tg = ml.to_targets(res, top=5, every=3)
    first = int(np.flatnonzero(np.isfinite(res.pred).any(axis=1))[0])
    assert tg.days[0] == first + res.label.lag
    assert np.all(np.diff(tg.days) == 3)
    counts = np.diff(tg.indptr)
    assert (counts == 5).all()
    d = int(tg.days[0]) - 1
    top = set(np.argsort(-res.pred[d])[:5])
    assert set(tg.cols[: tg.indptr[1]]) == top


def test_relative_label_and_features_from_expressions(tmp_path):
    from ashare.factor import call, field

    panel, _, _ = make(t=420, n=40, k=1, seed=11)
    feats = {
        "rev5": -(field("open") / call("ts_delay", field("open"), periods=5) - 1),
        "mom20": field("open") / call("ts_delay", field("open"), periods=20) - 1,
    }
    res = ml.walk_forward(
        feats,
        ml.label("ret5", relative="market", rank=True),
        data=panel,
        model="ridge",
        retrain="quarter",
        train_days=150,
        valid_days=20,
        start=300,
        root=tmp_path,
        ledger=False,
        threads=2,
    )
    assert res.check()
    assert np.isfinite(res.metrics["ic_mean"])
    assert "Rank IC" in res.summary()


def test_ensemble_of_one_equals_member_with_shared_bins(tmp_path):
    panel, feats, _ = make(t=420, n=40, k=3, seed=12)
    single = _walk(panel, feats, tmp_path, model="lgbm", params=FAST)
    ens = _walk(
        panel,
        feats,
        tmp_path,
        model="ensemble",
        params={"members": [("lgbm", FAST)], "seeds": [0], "combine": "mean"},
    )
    np.testing.assert_array_equal(single.pred, ens.pred)


def test_daily_gram_matches_direct_ridge():
    from ashare.ml.walk import DailyGram

    panel, feats, _ = make(t=200, n=30, k=3, seed=13, missing=0.1, blocked=0.2)
    universe = ml.default_universe(panel)
    store = ml.build_store(
        feats, panel, universe, ml.Preprocess(fill="none"), memory=True
    )
    values = L.compute(panel, ml.label("ret1"), universe)
    y = store.gather(values.target)
    ok = store.gather(values.trainable(universe))
    gram = DailyGram(store, y, ok, block_days=7)  # 块边界不对齐窗口
    for a, b in [(0, 50), (13, 121), (60, 198)]:
        sl = store.samples(a, b)
        idx = sl.start + np.flatnonzero(ok[sl])
        direct = ml.Batch.from_arrays(
            store.matrix(idx), y[idx], store.rows[idx], store.cols[idx]
        )
        lazy = ml.Batch(
            direct.day,
            direct.col,
            direct.n_cols,
            store.names,
            y[idx],
            loader=lambda: pytest.fail("有充分统计量时不该读特征"),
            gram=lambda a=a, b=b: gram.window(a, b),
        )
        m1, m2 = ml.Ridge(alpha=2.0).fit(direct), ml.Ridge(alpha=2.0).fit(lazy)
        np.testing.assert_allclose(m2.coef_, m1.coef_, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(m2.intercept_, m1.intercept_, rtol=1e-10, atol=1e-14)
        assert lazy._X is None
