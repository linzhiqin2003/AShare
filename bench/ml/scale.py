"""规模实测：全历史 × 全市场 × 约 150 个特征。

- 建特征库的耗时和体积（float32、只存股票池内样本）；稠密 float64 要多大作对照；
- 单折训练耗时：岭回归（NumPy 正规方程 vs sklearn）、LightGBM（跨折共享分箱 vs 每折重新分箱）；
- 整个滚动流程（岭回归按季、LightGBM 按年）的耗时和峰值驻留。

用法：MallocLargeCache=0 .venv/bin/python bench/ml/scale.py [--threads 6] [--steps store,fold,walk]
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time

import numpy as np
from mlbench import DATA, ML_ROOT, Peak, log, save

import ashare.data as ad
import ashare.ml as ml
from ashare.factor import adjusted_close, call, field, returns
from ashare.ml.walk import fit_folds


def wide_features():
    """basic_features 再加上换窗口的同族变体，凑到约 150 个。"""
    f = dict(ml.basic_features())
    price, ret = adjusted_close(), returns()
    turnover, volume, amount = field("turnover"), field("volume"), field("amount")

    def roll(op, *args, **kw):
        *panels, w = args
        return call(op, *panels, window=w, min_periods=math.ceil(0.8 * w), **kw)

    for w in (3, 10, 15, 30, 40, 90, 180):
        f[f"mom_w{w}"] = price / call("ts_delay", price, periods=w) - 1
        f[f"vol_w{w}"] = roll("ts_std", ret, max(w, 5))
        f[f"turn_w{w}"] = roll("ts_mean", turnover, w)
        f[f"turnstd_w{w}"] = roll("ts_std", turnover, max(w, 5))
        f[f"amt_w{w}"] = roll("ts_mean", amount, w)
        f[f"ma_w{w}"] = price / roll("ts_mean", price, w) - 1
        f[f"max_w{w}"] = roll("ts_max", ret, w)
        f[f"min_w{w}"] = roll("ts_min", ret, w)
        f[f"skew_w{w}"] = roll("ts_skew", ret, max(w, 10))
        f[f"cpv_w{w}"] = roll("ts_corr", price, volume, max(w, 5))
        f[f"crv_w{w}"] = roll("ts_corr", ret, call("log", volume + 1.0), max(w, 5))
        f[f"vr_w{w}"] = roll("ts_mean", volume, w) / roll("ts_mean", volume, 120)
        f[f"dd_w{w}"] = price / roll("ts_max", price, w) - 1
        f[f"rank_w{w}"] = roll("ts_rank", price, max(w, 5), normalize=True)
        f[f"slope_w{w}"] = roll("ts_slope", price, max(w, 5)) / price
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--steps", default="store,fold,walk")
    args = ap.parse_args()
    steps = set(args.steps.split(","))
    data = ad.open(root=DATA)
    feats = wide_features()
    panel = data.panel(ml.required_fields(feats, "ret5"))
    universe = ml.default_universe(panel)
    t, n = universe.shape
    out = {
        "shape": [t, n],
        "features": len(feats),
        "samples": int(universe.sum()),
        "threads": args.threads,
        "cpu": os.cpu_count(),
        "dense_float64_gb": t * n * len(feats) * 8 / 2**30,
    }
    log("规模", out)
    with Peak() as p:
        store = ml.build_store(
            feats,
            panel,
            universe,
            root=ML_ROOT,
            threads=args.threads,
            log=lambda m: None,
        )
    size = sum(
        store.path.joinpath(d.replace(":", "_") + ".f32").stat().st_size
        for d in store.digests.values()
    )
    out["store_first_call"] = p.record(size_gb=size / 2**30)
    log("特征库（首次调用，可能部分命中）", out["store_first_call"])
    if "store" in steps:
        # 冷建：换一个临时目录，从零建一遍
        import tempfile

        with tempfile.TemporaryDirectory(dir=ML_ROOT) as tmp:
            gc.collect()
            with Peak() as p:
                ml.build_store(feats, panel, universe, root=tmp, threads=args.threads)
            out["store_cold"] = p.record(size_gb=size / 2**30)
            log("特征库冷建", out["store_cold"])
        with Peak() as p:
            ml.build_store(feats, panel, universe, root=ML_ROOT, threads=args.threads)
        out["store_cached"] = p.record()
        log("特征库命中缓存", out["store_cached"])
    values = ml.compute_label(panel, "ret5", universe)
    y = store.gather(values.target)
    tr = store.gather(values.trainable(universe))
    raw = store.gather(values.raw)
    if "fold" in steps:
        sched = ml.schedule(
            store.dates,
            6,
            start="2024-01-01",
            end="2024-03-31",
            retrain="quarter",
            train_years=5,
        )
        f0 = sched.folds[0]
        sl = store.samples(*f0.train)
        idx = sl.start + np.flatnonzero(tr[sl])
        with Peak() as p:
            X = store.matrix(idx)
        out["fold_matrix"] = p.record(rows=len(idx), gb=X.nbytes / 2**30)
        b = ml.Batch(
            store.rows[idx].astype(np.int64),
            store.cols[idx].astype(np.int64),
            n,
            store.names,
            y[idx],
            _X=X,
            threads=args.threads,
        )
        for solver in ("numpy", "sklearn"):
            times = []
            for _ in range(3):
                gc.collect()
                with Peak() as p:
                    ml.Ridge(solver=solver).fit(b)
                times.append(p.record())
            out[f"ridge_{solver}"] = sorted(times, key=lambda r: r["seconds"])[1]
            log("岭回归", solver, out[f"ridge_{solver}"])
        del X, b
        gc.collect()
        for bins in ("per_fold", "shared"):
            with Peak() as p:
                _, infos, _ = fit_folds(
                    store,
                    y,
                    tr,
                    raw,
                    sched,
                    lambda b=bins: ml.LGBM(bins=b, early_stopping=0, n_estimators=200),
                    threads=args.threads,
                    shared_bins=bins == "shared",
                )
            out[f"lgbm_fold_{bins}"] = p.record(n_train=infos[0]["n_train"], rounds=200)
            log("LightGBM 单折", bins, out[f"lgbm_fold_{bins}"])
        save("scale", out)
    if "walk" in steps:
        for model, retrain, params in [
            ("ridge", "quarter", None),
            ("lgbm", "year", {"n_estimators": 200, "early_stopping": 0}),
        ]:
            gc.collect()
            with Peak() as p:
                res = ml.walk_forward(
                    store=store,
                    features=None,
                    label="ret5",
                    data=panel,
                    universe=universe,
                    model=model,
                    params=params,
                    retrain=retrain,
                    train_years=5,
                    start="2012-01-01",
                    threads=args.threads,
                    root=ML_ROOT,
                    cache=False,
                    ledger=False,
                    keep_models=False,
                )
            out[f"walk_{model}_{retrain}"] = p.record(
                folds=len(res.folds), ic=res.metrics["ic_mean"]
            )
            log("整个滚动", model, out[f"walk_{model}_{retrain}"])
            save("scale", out)
    out["load_end"] = os.getloadavg()
    print(save("scale", out))


if __name__ == "__main__":
    t = time.perf_counter()
    main()
    log("总耗时", time.perf_counter() - t)
