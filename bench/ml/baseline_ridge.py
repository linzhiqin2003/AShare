"""基线对比（岭回归）：训练本身很快时，取数和拼表的开销占大头。

直白写法：pandas 长表 + TimeSeriesSplit + 每折按日期过滤拼 X + sklearn Ridge；
ashare.ml：特征库按折切片 + NumPy 正规方程。各跑 3 次取中位数，核对预测（容差内一致）。
用法：MallocLargeCache=0 .venv/bin/python bench/ml/baseline_ridge.py [--threads 6]
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics

import numpy as np
from mlbench import DATA, ML_ROOT, Peak, log, save

import ashare.data as ad
import ashare.ml as ml


def main():
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--test-start", default="2023-01-01")
    ap.add_argument("--retrain", type=int, default=63)
    ap.add_argument("--train-days", type=int, default=750)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    data = ad.open(root=DATA)
    feats = ml.basic_features()
    panel = data.panel(ml.required_fields(feats, "ret5"), start=args.start)
    universe = ml.default_universe(panel)
    names = list(feats)
    lab = ml.label("ret5")
    store = ml.build_store(feats, panel, universe, root=ML_ROOT, threads=args.threads)
    values = ml.compute_label(panel, lab, universe)
    days = np.asarray(store.dates)
    t0 = int(np.searchsorted(days.astype(str), args.test_start.replace("-", "")))
    n_splits = (len(days) - lab.span - t0) // args.retrain
    test_start = len(days) - n_splits * args.retrain
    out = {
        "samples": int(universe.sum()),
        "features": len(names),
        "threads": args.threads,
    }

    def ours():
        return ml.walk_forward(
            store=store,
            features=None,
            label=lab,
            data=panel,
            universe=universe,
            model="ridge",
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

    def naive():
        long = pd.DataFrame({"date": store.rows.astype(np.int64), "col": store.cols})
        for n in names:
            long[n] = np.asarray(store.column(n))
        long["y"] = store.gather(values.target)
        long["ok"] = store.gather(values.trainable(universe))
        split = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=args.retrain,
            gap=lab.span,
            max_train_size=args.train_days,
        )
        preds = np.full(store.n_samples, np.nan)
        for train_days, test_days in split.split(np.arange(len(days))):
            tr = long[long["date"].isin(train_days) & long["ok"]]
            m = Ridge(alpha=1.0).fit(tr[names].to_numpy(np.float64), tr["y"].to_numpy())
            te = long[long["date"].isin(test_days)]
            preds[te.index.to_numpy()] = m.predict(te[names].to_numpy(np.float64))
        return preds

    runs = {"ours": [], "pandas": []}
    for _ in range(args.repeat):
        for name, fn in (("ours", ours), ("pandas", naive)):
            gc.collect()
            with Peak() as p:
                result = fn()
            runs[name].append(p.record())
            log(name, runs[name][-1])
            if name == "ours":
                a = store.gather(result.pred)
            else:
                b = result
    for name, rs in runs.items():
        med = sorted(rs, key=lambda r: r["seconds"])[len(rs) // 2]
        out[name] = {
            "median": med,
            "seconds_all": [r["seconds"] for r in rs],
            "footprint_delta_all": [r["footprint_delta_gb"] for r in rs],
        }
    mask = np.isfinite(b)
    out["same_mask"] = bool(np.array_equal(mask, np.isfinite(a)))
    out["max_abs_diff"] = float(np.max(np.abs(a[mask] - b[mask])))
    out["max_rel_diff"] = float(
        np.max(np.abs(a[mask] - b[mask]) / np.maximum(np.abs(b[mask]), 1e-12))
    )
    out["seconds_median"] = {
        k: statistics.median(r["seconds"] for r in v) for k, v in runs.items()
    }
    out["load_end"] = os.getloadavg()
    log("预测差", out["max_abs_diff"], out["same_mask"], out["seconds_median"])
    print(save("baseline_ridge", out))


if __name__ == "__main__":
    main()
