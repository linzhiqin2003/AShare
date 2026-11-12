"""真实数据上的完整例子：默认因子 + 自己补的量价特征，预测未来 5 日收益，2012 年起滚动
训练到最新；报告各模型的 Rank IC、ICIR、十分组收益、top 50 等权组合扣成本后的回测，
以及「这些特征等权合成」这个朴素基准。

参数是事先定下的一组（见 docs/ML_NOTES.md），不为结果调参。
用法：MallocLargeCache=0 .venv/bin/python bench/ml/real_example.py [--models naive,ridge,lgbm,...]
每个模型跑完就把结果写进 bench/ml/results/real_<模型>.json（可分次跑）。
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import panelbt
from mlbench import DATA, ML_ROOT, Peak, log, save

import ashare.data as ad
import ashare.ml as ml
from ashare.backtest import market_from_panel

START = "2012-01-01"
RETRAIN = "halfyear"
TRAIN_YEARS = 5
VALID_DAYS = 120
TOP = 50
LGBM = {}  # 用 ml.LGBM 的默认参数（写在 ML_NOTES 里），不另调
MODELS = {
    "naive": ("signed_equal", None),
    "ridge": ("ridge", {"alpha": 1.0}),
    "lgbm": ("lgbm", LGBM),
    "lgbm_rank": ("lgbm_rank", LGBM),
    "ensemble": (
        "ensemble",
        {
            "members": [("ridge", {"alpha": 1.0}), ("lgbm", LGBM)],
            "seeds": [0, 1, 2],
            "combine": "rank",
        },
    ),
}


class SignedEqual:
    """朴素基准：各特征等权合成，方向取训练期里该特征逐日 Rank IC 均值的符号。"""

    def fit(self, train, valid=None):
        ics = [
            ml.daily_rank_ic(train.X[:, j], train.y, train.day, train.col, train.n_cols)
            for j in range(train.X.shape[1])
        ]
        self.sign = np.sign([np.nanmean(ic) for ic in ics])
        return self

    def predict(self, batch):
        return np.nan_to_num(batch.X) @ self.sign / len(self.sign)

    def feature_importance(self):
        return np.abs(self.sign)

    def save(self, path):
        np.save(f"{path}/sign.npy", self.sign)

    @classmethod
    def load(cls, path):
        m = cls()
        m.sign = np.load(f"{path}/sign.npy")
        return m


ml.register_model("signed_equal", SignedEqual, overwrite=True)


def nav_stats(nav, turnover, first):
    nav = np.asarray(nav)[first:]
    r = nav[1:] / nav[:-1] - 1
    years = len(r) / 243
    ann = (nav[-1] / nav[0]) ** (1 / years) - 1
    vol = r.std(ddof=1) * math.sqrt(243)
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    return {
        "annual_return": float(ann),
        "annual_vol": float(vol),
        "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(243)),
        "max_drawdown": dd,
        "annual_turnover": float(np.asarray(turnover)[first:].sum() / years),
        "final_nav": float(nav[-1] / nav[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--threads", type=int, default=ml.default_threads())
    ap.add_argument("--fold-workers", type=int, default=1)
    args = ap.parse_args()
    data = ad.open(root=DATA)
    feats = ml.basic_features()
    bm = market_from_panel(data, fields=ml.required_fields(feats, "ret5"))
    panel = bm.panel
    universe = ml.default_universe(panel)
    costs = bm.costs()  # 佣金万 2.5（最低 5 元）、滑点万 5、历史印花税与过户费
    config = bm.config(initial_cash=1e7)
    first_signal = int(np.searchsorted(panel.dates, START.replace("-", "")))
    # 等权持有股票池作为参照
    ones = np.where(universe, 1.0, np.nan)
    ref_targets = ml.to_targets(
        ones, top=int(universe.sum(axis=1).max()), every=5, start=first_signal
    )
    ref = panelbt.run(bm.market, ref_targets, costs, config)
    ref_stats = nav_stats(ref.nav, ref.turnover, first_signal + 1)
    log("股票池等权参照", ref_stats)
    for name in args.models.split(","):
        model, params = MODELS[name]
        log("开始", name)
        with Peak() as p:
            res = ml.walk_forward(
                feats,
                "ret5",
                data=panel,
                universe=universe,
                model=model,
                params=params,
                retrain=RETRAIN,
                train_years=TRAIN_YEARS,
                valid_days=VALID_DAYS,
                start=START,
                threads=args.threads,
                fold_workers=args.fold_workers,
                root=ML_ROOT,
                tags=("real_example",),
                log=lambda m, n=name: log(n, m),
            )
        targets = ml.to_targets(res, top=TOP)
        bt = panelbt.run(bm.market, targets, costs, config)
        stats = nav_stats(bt.nav, bt.turnover, first_signal + 1)
        excess = nav_stats(
            bt.nav / ref.nav * ref.nav[first_signal + 1], bt.turnover, first_signal + 1
        )
        m = res.metrics
        out = {
            "model": name,
            "params": params,
            "run": p.record(folds=len(res.folds)),
            "cached": res.cached,
            "run_id": res.run_id,
            "trial_id": res.trial_id,
            "ic_mean": m["ic_mean"],
            "icir": m["icir"],
            "t": m["t"],
            "days": m["days"],
            "deciles_5d": [float(g) for g in m["groups"]],
            "long_short_5d": m["long_short"],
            "top_turnover": m["top_turnover"],
            "backtest": stats,
            "excess_vs_universe": {
                "annual_return": excess["annual_return"],
                "max_drawdown": excess["max_drawdown"],
                "sharpe": excess["sharpe"],
            },
            "reference_universe": ref_stats,
            "yearly_ic": yearly(res),
            "schedule": res.schedule.describe(),
            "best_iterations": [f.get("best_iteration") for f in res.folds],
            "load": os.getloadavg(),
            "threads": args.threads,
        }
        imp = res.importance()
        if imp is not None:
            order = np.argsort(-imp["mean"])[:10]
            out["top_importance"] = [
                [imp["names"][i], float(imp["mean"][i])] for i in order
            ]
        save(f"real_{name}", out)
        log(name, res.summary().replace("\n", " | "), stats)


def yearly(res):
    ic = res.ic()
    rows = np.concatenate([np.arange(*f.test) for f in res.schedule.folds])
    years = np.array([str(res.dates[r])[:4] for r in rows])
    return {y: float(np.nanmean(ic[years == y])) for y in sorted(set(years))}


if __name__ == "__main__":
    t = time.perf_counter()
    main()
    log("总耗时", time.perf_counter() - t)
