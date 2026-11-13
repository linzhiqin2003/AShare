"""基线对比：同一件事用最直白的写法（pandas 长表、每折重新拼表、sklearn 的
TimeSeriesSplit、每折新建 LightGBM 数据集）做一遍，与 ashare.ml 比耗时、峰值内存，
并核对预测：bins="per_fold" 时逐位一致，默认 bins="shared" 时给出差异与 IC 差。

两段分开比：
A. 截面预处理：pandas 按日 groupby（去极值、标准化、补 0）对 panelops 向量化；
B. 训练与预测：同样的特征值，pandas 长表每折过滤 + TimeSeriesSplit 对 特征库切片。

用法：MallocLargeCache=0 .venv/bin/python bench/ml/baseline.py [--start 2019-01-01] [--threads 6]
需要 pandas（不在 ml 组里；`uv sync --group compare` 或 `uv pip install pandas`）。
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
from mlbench import DATA, ML_ROOT, Peak, log, save

import ashare.data as ad
import ashare.ml as ml
from ashare.factor import Evaluator


def pandas_preprocess(long, names, lower=0.01, upper=0.99):
    """最直白的写法：按日 groupby，分位截断 → 标准化（ddof=0）→ 缺失补 0。"""
    g = long.groupby("date", sort=False)
    out = {}
    for name in names:
        x = long[name]
        lo = g[name].transform(lambda s: s.quantile(lower))
        hi = g[name].transform(lambda s: s.quantile(upper))
        x = x.clip(lo, hi)
        mean = x.groupby(long["date"]).transform("mean")
        std = x.groupby(long["date"]).transform(lambda s: s.std(ddof=0))
        out[name] = ((x - mean) / std).fillna(0.0)
    return out


def main():
    import lightgbm as lgb
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--test-start", default="2023-01-01")
    ap.add_argument("--retrain", type=int, default=63)
    ap.add_argument("--train-days", type=int, default=750)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument(
        "--pre-features", type=int, default=6, help="A 段只比前几个特征（pandas 很慢）"
    )
    args = ap.parse_args()
    data = ad.open(root=DATA)
    feats = ml.basic_features()
    fields = ml.required_fields(feats, "ret5")
    panel = data.panel(fields, start=args.start)
    universe = ml.default_universe(panel)
    names = list(feats)
    lab = ml.label("ret5")
    out = {
        "load": os.getloadavg(),
        "threads": args.threads,
        "samples": int(universe.sum()),
        "features": len(names),
        "shape": list(universe.shape),
        "start": args.start,
    }
    log("面板", universe.shape, "样本", universe.sum())

    # ---------------- A. 截面预处理
    sub = names[: args.pre_features]
    ev = Evaluator()
    raw = {n: np.ascontiguousarray(ev.evaluate(feats[n], panel)) for n in sub}
    pre = ml.Preprocess()
    with Peak() as p:
        ours = {n: pre.apply(raw[n], universe) for n in sub}
    out["preprocess_panelops"] = p.record(features=len(sub))
    rows, cols = np.nonzero(universe)
    with Peak() as p:
        long = pd.DataFrame({"date": rows, "col": cols})
        for n in sub:
            long[n] = raw[n][rows, cols]
        theirs = pandas_preprocess(long, sub)
    out["preprocess_pandas"] = p.record(features=len(sub))
    diff = max(
        float(np.nanmax(np.abs(theirs[n].to_numpy() - ours[n][rows, cols])))
        for n in sub
    )
    out["preprocess_max_abs_diff"] = diff
    log(
        "预处理",
        out["preprocess_panelops"]["seconds"],
        "vs pandas",
        out["preprocess_pandas"]["seconds"],
        "最大差",
        diff,
    )
    del long, theirs, raw, ours
    gc.collect()

    # ---------------- B. 训练与预测
    with Peak() as p:
        store = ml.build_store(
            feats, panel, universe, pre, root=ML_ROOT, threads=args.threads
        )
    out["store_build"] = p.record()
    values = ml.compute_label(panel, lab, universe)
    params = {
        "n_estimators": args.rounds,
        "early_stopping": 0,
        "bins": "per_fold",
        "threads": args.threads,
    }
    days = np.asarray(store.dates)
    t0 = int(np.searchsorted(days.astype(str), args.test_start.replace("-", "")))
    n_splits = (len(days) - lab.span - t0) // args.retrain
    test_start = len(days) - n_splits * args.retrain
    res = {}
    for bins in ("per_fold", "shared"):
        gc.collect()
        with Peak() as p:
            r = ml.walk_forward(
                store=store,
                features=None,
                label=lab,
                data=panel,
                universe=universe,
                model="lgbm",
                params=params | {"bins": bins},
                retrain=args.retrain,
                train_days=args.train_days,
                valid_days=0,
                start=test_start,
                threads=args.threads,
                root=ML_ROOT,
                ledger=False,
                cache=False,
                keep_models=False,
            )
        res[bins] = r
        out[f"ours_{bins}"] = p.record(folds=len(r.folds), ic_mean=r.metrics["ic_mean"])
        log("ours", bins, out[f"ours_{bins}"])

    # 直白写法：整张 pandas 长表（特征 float32 + 标签），TimeSeriesSplit 给日期下标，
    # 每折按日期过滤拼出 X，新建 lgb.Dataset 训练，再拼测试表预测。
    gc.collect()
    with Peak() as p:
        long = pd.DataFrame({"date": store.rows.astype(np.int64), "col": store.cols})
        for n in names:
            long[n] = np.asarray(store.column(n))
        long["y"] = store.gather(values.target)
        long["ok"] = store.gather(values.trainable(universe))
        dates_u = np.arange(len(days))
        split = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=args.retrain,
            gap=lab.span,
            max_train_size=args.train_days,
        )
        lp = ml.LGBM(**params).lgb_params(args.threads)
        preds = np.full(store.n_samples, np.nan)
        folds = 0
        for train_days, test_days in split.split(dates_u[: len(days)]):
            if test_days[0] < test_start:
                continue
            tr = long[long["date"].isin(train_days) & long["ok"]]
            booster = lgb.train(
                lp,
                lgb.Dataset(tr[names].to_numpy(), label=tr["y"].to_numpy(), params=lp),
                args.rounds,
            )
            te = long[long["date"].isin(test_days)]
            preds[te.index.to_numpy()] = booster.predict(
                te[names].to_numpy(), num_threads=args.threads
            )
            folds += 1
    out["baseline_pandas"] = p.record(folds=folds)
    log("baseline", out["baseline_pandas"])
    ours = res["per_fold"].store.gather(res["per_fold"].pred)
    mask = np.isfinite(preds)
    out["bitwise_per_fold"] = bool(np.array_equal(ours[mask], preds[mask])) and bool(
        np.array_equal(np.isfinite(ours), mask)
    )
    shared = res["shared"].store.gather(res["shared"].pred)
    out["shared_vs_per_fold_max_abs"] = float(np.nanmax(np.abs(shared - ours)))
    out["shared_vs_per_fold_ic"] = [
        res["shared"].metrics["ic_mean"],
        res["per_fold"].metrics["ic_mean"],
    ]
    out["load_end"] = os.getloadavg()
    log(
        "逐位一致",
        out["bitwise_per_fold"],
        "shared 最大差",
        out["shared_vs_per_fold_max_abs"],
    )
    print(save("baseline", out))


if __name__ == "__main__":
    t = time.perf_counter()
    main()
    log("总耗时", time.perf_counter() - t)
