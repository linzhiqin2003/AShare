"""解释：特征重要性（各折、随时间）和验证集上按 Rank IC 的置换重要性。

置换重要性：在每折的验证集上，把一个特征在每天的截面内随机打乱（保留它的日分布、
切断它和标签的关系），看逐日 Rank IC 均值掉多少。打乱在截面内做，所以不会把别的日子
的信息混进来。
"""

from __future__ import annotations

import numpy as np

from .metrics import mean_rank_ic


def permute_within_days(column, day, rng):
    """把 column 在每个 day 内随机重排（day 升序）；向量化，不逐日循环。"""
    order = np.lexsort((rng.random(len(day)), day))
    return column[order]


def permutation_importance(result, *, repeats=1, seed=0, on="valid"):
    """返回 {"names", "folds"（折 × 特征的 IC 下降）, "mean", "base"（每折基准 IC）}。

    on="valid"（默认）用每折的验证集；"test" 用预测期（只作参考，那段数据已经用来报告）。
    需要 `result.models`（walk_forward 的 keep_models=True）。
    """
    if not result.models:
        raise ValueError("需要训练好的模型：walk_forward(..., keep_models=True)")
    if on not in ("valid", "test"):
        raise ValueError("on: valid 或 test")
    store, values = result.store, result.labels
    y_all = store.gather(values.target if on == "valid" else values.raw)
    trainable = store.gather(values.trainable(_universe(store)))
    from .models import Batch

    rng = np.random.default_rng(seed)
    n_cols = len(store.symbols)
    drops, bases = [], []
    for info, model in zip(result.folds, result.models, strict=True):
        rows = info["valid_rows"] if on == "valid" else info["test_rows"]
        if rows is None:
            continue
        sl = store.samples(*rows)
        idx = sl.start + np.flatnonzero(trainable[sl] & np.isfinite(y_all[sl]))
        if not len(idx):
            continue
        X = store.matrix(idx)
        day, col, y = (
            store.rows[idx].astype(np.int64),
            store.cols[idx].astype(np.int64),
            y_all[idx],
        )
        batch = Batch(day, col, n_cols, store.names, y, _X=X, threads=1)
        base = mean_rank_ic(model.predict(batch), y, day, col, n_cols)
        row = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            total = 0.0
            for _ in range(repeats):
                saved = X[:, j].copy()
                X[:, j] = permute_within_days(saved, day, rng)
                total += base - mean_rank_ic(model.predict(batch), y, day, col, n_cols)
                X[:, j] = saved
            row[j] = total / repeats
        drops.append(row)
        bases.append(base)
    mat = np.vstack(drops) if drops else np.empty((0, len(store.names)))
    return {
        "names": store.names,
        "folds": mat,
        "mean": mat.mean(axis=0) if drops else None,
        "base": np.asarray(bases),
    }


def _universe(store):
    mask = np.zeros(store.shape, dtype=bool)
    mask[store.rows, store.cols] = True
    return mask
