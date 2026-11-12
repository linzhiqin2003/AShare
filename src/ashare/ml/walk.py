"""滚动训练：特征库 + 标签 + 切分 + 模型 → 一张 T×N 预测面板（预测即因子）。

```python
import ashare.data as ad, ashare.ml as ml
data = ad.open()
res = ml.walk_forward(ml.basic_features(), "ret5", data=data, model="lgbm",
                      retrain="quarter", train_years=5, start="2015-01-01")
print(res.summary())            # Rank IC、ICIR、十分组、换手
targets = ml.to_targets(res, top=50)   # 交给 panelbt.run
```
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import panelops

from . import _deps
from . import labels as _labels
from . import store as _store
from .ledger import Ledger, code_version
from .metrics import daily_rank_ic, ic_stats, panel_summary
from .models import Batch, _resolve, load_model, make_model, params_of, save_model
from .preprocess import Preprocess
from .split import check_schedule, schedule

UNIVERSE_FIELDS = ("is_listed", "is_st", "listing_age", "has_quote")


def default_threads():
    """默认训练线程数：CPU 核数的一半（机器上常有别的任务），至少 1。"""
    return max(1, (os.cpu_count() or 2) // 2)


def default_universe(panel, min_age=120):
    """默认股票池：已上市、非 ST、上市满 min_age 个交易日、当天有行情（未停牌）。"""
    listed = np.asarray(panel["is_listed"], dtype=bool)
    st = np.asarray(panel["is_st"], dtype=bool)
    age = np.asarray(panel["listing_age"])
    quote = np.asarray(panel["has_quote"], dtype=bool)
    return np.ascontiguousarray(listed & ~st & (age >= min_age) & quote)


def required_fields(features, label="ret5", preprocess=None, universe=True):
    """walk_forward 需要从面板读的字段（特征表达式、标签、预处理、默认股票池）。"""
    features = _store.normalize_features(features)
    names = set()
    for value in features.values():
        if hasattr(value, "fields"):
            names |= set(value.fields)
    names |= set(_labels.label(label).fields())
    names |= set((preprocess or Preprocess()).fields())
    if universe:
        names |= set(UNIVERSE_FIELDS)
    return sorted(names)


def _panel(data, fields):
    """Data 句柄就读需要的字段；面板（有 dates/symbols/__getitem__）原样用。"""
    if hasattr(data, "panel") and callable(data.panel):
        try:
            return data.panel(fields)
        except FileNotFoundError:
            # 旧面板没有开盘可买可卖掩码：读规则字段现算（口径同 market_from_panel）。
            rest = [f for f in fields if f not in _labels.MASKS]
            return data.panel(sorted(set(rest) | set(_labels.RULES)))
    return data


class SharedBins:
    """LightGBM 分箱数据集的跨折复用：分箱边界只用 reference 样本（第一折训练集）定，
    全部样本按这套边界分箱一次，各折取子集。"""

    def __init__(self, store, reference_index, threads):
        self.store, self.reference_index, self.threads = store, reference_index, threads
        self._full = {}
        self._lock = threading.Lock()

    def _sequence(self, index):
        import lightgbm as lgb

        store = self.store

        class _Seq(lgb.Sequence):
            batch_size = 65536

            def __len__(self):
                return (
                    len(index)
                    if not isinstance(index, slice)
                    else index.stop - index.start
                )

            def __getitem__(self, i):
                if isinstance(index, slice):
                    if isinstance(i, slice):
                        i = slice(
                            index.start + (i.start or 0), index.start + (i.stop or 0)
                        )
                    else:
                        i = index.start + i
                    rows = (
                        store.matrix(i)
                        if isinstance(i, slice)
                        else store.matrix([i])[0]
                    )
                else:
                    rows = (
                        store.matrix(index[i])
                        if isinstance(i, slice)
                        else store.matrix([index[i]])[0]
                    )
                return np.asarray(rows, dtype=np.float64)

        return _Seq()

    def full(self, dparams):
        import lightgbm as lgb

        key = json.dumps(dparams, sort_keys=True)
        with self._lock:
            if key not in self._full:
                params = dparams | {"num_threads": self.threads}
                ref = lgb.Dataset(
                    self._sequence(self.reference_index),
                    label=np.zeros(len(self.reference_index)),
                    params=params,
                    free_raw_data=True,
                ).construct()
                n = self.store.n_samples
                full = lgb.Dataset(
                    self._sequence(slice(0, n)),
                    label=np.zeros(n),
                    reference=ref,
                    params=params,
                    free_raw_data=True,
                ).construct()
                self._full[key] = full
            return self._full[key]

    def subset(self, dparams, index):
        full = self.full(dparams)
        with self._lock:
            return full.subset(np.asarray(index, dtype=np.int32)).construct()


class DailyGram:
    """线性模型的按日充分统计量：每个交易日的样本数、Σx、Σy、XᵀX、Xᵀy（float64，只算
    可训练样本，缺失特征按 0）。滚动窗口相互重叠时，各折只需把窗口内的日统计量相加，
    每个样本只读一次。第一次用到时才算（线程安全）；不用它的模型不付这份开销。"""

    def __init__(self, store, y, trainable, block_days=64):
        self.store, self.y, self.trainable = store, y, trainable
        self.block_days = block_days
        self._stats = None
        self._lock = threading.Lock()

    def _compute(self):
        store = self.store
        t, f = len(store.dates), len(store.names)
        n = np.zeros(t)
        sx, sy = np.zeros((t, f)), np.zeros(t)
        xtx, xty = np.zeros((t, f, f)), np.zeros((t, f))
        for d0 in range(0, t, self.block_days):
            d1 = min(d0 + self.block_days, t)
            sl = store.samples(d0, d1)
            ok = self.trainable[sl]
            if not ok.any():
                continue
            x = store.matrix(sl)[ok].astype(np.float64)
            x[~np.isfinite(x)] = 0.0
            yy = np.asarray(self.y[sl][ok], np.float64)
            days = store.rows[sl][ok]
            bounds = np.searchsorted(days, np.arange(d0, d1 + 1))
            for d in range(d0, d1):
                a, b = bounds[d - d0], bounds[d - d0 + 1]
                if a == b:
                    continue
                xs, ys = x[a:b], yy[a:b]
                n[d] = b - a
                sx[d] = xs.sum(axis=0)
                sy[d] = ys.sum()
                xtx[d] = xs.T @ xs
                xty[d] = xs.T @ ys
        return n, sx, sy, xtx, xty

    def window(self, first_day, last_day):
        """[first_day, last_day) 内的统计量之和 (n, Σx, Σy, XᵀX, Xᵀy)。"""
        with self._lock:
            if self._stats is None:
                self._stats = self._compute()
        n, sx, sy, xtx, xty = self._stats
        w = slice(first_day, last_day)
        return (
            float(n[w].sum()),
            sx[w].sum(axis=0),
            float(sy[w].sum()),
            xtx[w].sum(axis=0),
            xty[w].sum(axis=0),
        )


@dataclass
class WalkResult:
    """滚动训练的结果。pred：T×N 预测面板（没有预测的格子为 NaN），可直接当因子用。"""

    pred: np.ndarray
    dates: np.ndarray
    symbols: np.ndarray
    label: _labels.Label
    schedule: object
    folds: list
    metrics: dict
    run_id: str
    trial_id: str
    cached: bool = False
    feature_names: tuple = ()
    models: list | None = None
    path: Path | None = None
    raw: np.ndarray | None = field(default=None, repr=False)
    store: object = field(default=None, repr=False)
    labels: object = field(default=None, repr=False)
    seconds: float = 0.0

    def ic(self):
        """预测期每天的 Rank IC（对原始未来收益）。"""
        return self.metrics["ic"]

    def summary(self):
        m = self.metrics
        groups = " ".join(f"{g * 1e4:+.0f}" for g in m.get("groups", []))
        lines = [
            f"Rank IC 均值 {m['ic_mean']:.4f}  ICIR {m['icir']:.3f}  t {m['t']:.2f}  "
            f"天数 {m['days']}",
            f"十分组 {self.label.horizon} 日收益（万分之一，低→高）：{groups}",
            f"多空（高−低）{m.get('long_short', float('nan')) * 1e4:+.1f}  "
            f"头组换手 {m.get('top_turnover', float('nan')):.2f}",
            f"{len(self.folds)} 折，耗时 {self.seconds:.1f}s{'（缓存）' if self.cached else ''}",
        ]
        return "\n".join(lines)

    def importance(self):
        """各折特征重要性：{"folds": 折 × 特征数组（每折归一化）, "mean": 各折平均, "names"}。"""
        rows = [f.get("importance") for f in self.folds]
        rows = [np.asarray(r, np.float64) for r in rows if r is not None]
        if not rows:
            return None
        mat = np.vstack([r / r.sum() if r.sum() > 0 else r for r in rows])
        return {
            "names": self.feature_names,
            "folds": mat,
            "mean": mat.mean(axis=0),
            "dates": [f["test"][0] for f in self.folds],
        }

    def check(self):
        """防前视检查：每折用到的训练/验证样本里最晚的标签结束日 < 预测期第一天。"""
        return check_lookahead(self)


def check_lookahead(result):
    """任何一个训练（含验证）样本用到的最晚数据日期——标签结束日 = 信号日 + lag + h——
    必须早于它所服务的预测期的第一天；特征只用信号日及以前的数据（表达式求值不看未来）。
    不满足抛 ValueError，满足返回 True。"""
    span = result.label.span
    for f in result.folds:
        last = f["last_used_day"]
        if last is None:
            continue
        if last + span >= f["test_rows"][0]:
            raise ValueError(
                f"第 {f['fold']} 折：训练样本最晚信号日 {last} 的标签结束日 {last + span} "
                f"不早于预测期首日 {f['test_rows'][0]}"
            )
    check_schedule(result.schedule)
    return True


def _accepts(factory, name):
    try:
        return name in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


def _model_factory(model, params, seed):
    params = dict(params or {})
    if isinstance(model, str):
        from .models import _REGISTRY, _load_entry_points

        if model not in _REGISTRY:
            _load_entry_points()
        entry = _REGISTRY.get(model)
        factory = None
        if entry is not None and all(_deps.importable(m) for m in entry.requires):
            factory = _resolve(entry.factory)
        if factory is not None and "seed" not in params and _accepts(factory, "seed"):
            params["seed"] = seed
        return lambda: make_model(model, params), params
    if params:
        raise ValueError("传了模型实例时不能再给 params")
    import copy

    return (lambda: copy.deepcopy(model)), params


def _hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def walk_forward(
    features,
    label="ret5",
    *,
    data,
    universe=None,
    model="lgbm",
    params=None,
    retrain="quarter",
    window="rolling",
    train_years=5,
    train_days=None,
    valid_days=120,
    gap=None,
    start=None,
    end=None,
    preprocess=None,
    seed=0,
    threads=None,
    fold_workers=1,
    root=None,
    store=None,
    ledger=True,
    cache=True,
    keep_models=True,
    tags=(),
    min_train_days=None,
    log=None,
):
    """滚动训练并给出样本外预测面板。

    features：{名字: Expr 或 T×N 数组}、Expr 列表或因子登记表；label：预设名
    （"ret1"/"ret5"/"ret10"/"ret20"）或 `Label`；data：`ashare.data.open()` 的句柄或面板；
    universe：T×N 布尔股票池（默认 `default_universe`）。model：登记过的模型名
    （"ridge"、"lasso"、"elasticnet"、"lgbm"、"lgbm_rank"、"ensemble" 或自己登记的）
    或实现了协议的模型实例，params 是给它的参数。切分参数见 `ml.schedule`；start/end
    是预测期。threads：训练和预处理的总线程数（默认核数的一半）；fold_workers：同时
    训练几折（线程平分）。root：特征库、运行缓存和账本的目录（默认 data/ml）；
    ledger=False 不记账（只用于测试），cache=False 不读写运行缓存。
    """
    t0 = time.perf_counter()
    log = log or (lambda *_: None)
    label = _labels.label(label)
    preprocess = preprocess or Preprocess()
    threads = int(threads or default_threads())
    fold_workers = max(1, int(fold_workers))
    feats = _store.normalize_features(features) if store is None else None
    if store is None:
        fields = required_fields(feats, label, preprocess, universe is None)
    else:
        fields = sorted(
            set(label.fields()) | (set(UNIVERSE_FIELDS) if universe is None else set())
        )
    panel = _panel(data, fields)
    if universe is None:
        universe = default_universe(panel)
    universe = np.ascontiguousarray(universe, dtype=bool)
    root = Path(root) if root is not None else _store.default_root()
    if store is None:
        store = _store.build(
            feats, panel, universe, preprocess, root=root, threads=threads, log=log
        )
    elif (
        store.shape != universe.shape
        or store.n_samples != int(universe.sum())
        or not universe[store.rows, store.cols].all()
    ):
        raise ValueError("store 与 universe 不匹配（股票池不同的特征库不能混用）")
    values = _labels.compute(panel, label, universe)
    y = store.gather(values.target)
    trainable = store.gather(values.trainable(universe))
    sched = schedule(
        store.dates,
        label.span,
        start=start,
        end=end,
        retrain=retrain,
        window=window,
        train_years=train_years,
        train_days=train_days,
        valid_days=valid_days,
        gap=gap,
        min_train_days=min_train_days,
    )
    if not len(sched):
        raise ValueError(
            "没有可训练的折：预测期太早或训练窗口太短\n" + sched.describe()
        )
    factory, mparams = _model_factory(model, params, seed)
    probe = factory()
    definition = {
        "data_version": getattr(panel, "version", None),
        "axes": _store._axes_hash(store.dates, store.symbols),
        "universe": hashlib.sha256(np.packbits(universe).tobytes()).hexdigest(),
        "features": [[n, store.digests[n]] for n in store.names],
        "preprocess": store.preprocess.describe(),
        "label": label.describe(),
        "schedule": sched.digest,
        "model": params_of(probe),
        "seed": seed,
    }
    # 试验组按「研究定义」算：框架代码改了（修 bug、提速）不算新的一组；运行缓存则按
    # 代码哈希和线程数区分，代码一改就重算。
    code = _store.implementation_id() + ":" + _code_hash()
    trial_id = _hash(definition)
    run_id = _hash([trial_id, code, threads, fold_workers])
    run_dir = root / "runs" / run_id
    book = Ledger(root / "ledger.jsonl") if ledger else None
    raw_panel = values.raw
    if cache and (run_dir / "result.json").exists():
        result = _load_run(
            run_dir, store, label, sched, values, run_id, trial_id, keep_models
        )
        if result is not None:
            check_lookahead(result)
            result.seconds = time.perf_counter() - t0
            if book:
                book.append(
                    {
                        "event": "cache_hit",
                        "trial_id": trial_id,
                        "run_id": run_id,
                        "tags": list(tags),
                    }
                )
            return result
    shared = getattr(probe, "bins", None) == "shared"
    pred_s, infos, models = fit_folds(
        store,
        y,
        trainable,
        store.gather(values.raw),
        sched,
        factory,
        threads=threads,
        fold_workers=fold_workers,
        shared_bins=shared,
        horizon=label.horizon,
        log=log,
    )
    pred_panel = store.scatter(pred_s)
    test_rows = np.concatenate([np.arange(*f.test) for f in sched.folds])
    metrics = panel_summary(pred_panel, raw_panel, label.horizon, q=10, rows=test_rows)
    seconds = time.perf_counter() - t0
    result = WalkResult(
        pred=pred_panel,
        dates=store.dates,
        symbols=store.symbols,
        label=label,
        schedule=sched,
        folds=infos,
        metrics=metrics,
        run_id=run_id,
        trial_id=trial_id,
        feature_names=store.names,
        models=models if keep_models else None,
        path=run_dir if cache else None,
        raw=raw_panel,
        store=store,
        labels=values,
        seconds=seconds,
    )
    check_lookahead(result)
    if cache:
        _save_run(run_dir, result, pred_s, models if keep_models else [])
    if book:
        book.append(
            {
                "event": "trial",
                "trial_id": trial_id,
                "run_id": run_id,
                "tags": list(tags),
                **definition,
                "code": code,
                "code_version": code_version(),
                "model_name": model if isinstance(model, str) else type(model).__name__,
                "schedule_table": sched.table(),
                "schedule_settings": sched.settings,
                "threads": threads,
                "fold_workers": fold_workers,
                "metrics": {k: v for k, v in metrics.items() if k != "ic"},
                "folds": [
                    {k: v for k, v in f.items() if k != "importance"} for f in infos
                ],
                "seconds": seconds,
            }
        )
    return result


def fit_folds(
    store,
    y,
    trainable,
    raw_s,
    sched,
    factory,
    *,
    threads,
    fold_workers=1,
    shared_bins=False,
    horizon=1,
    log=None,
):
    """按日程逐折训练和预测（walk_forward 的核心，也供实验直接调用）。

    y：样本向量形式的训练目标；trainable：可进训练集的样本；raw_s：原始未来收益（算
    每折的样本外 IC）；factory：无参可调用，返回一个新模型。返回 (样本预测向量, 各折信息,
    各折模型)。这里不检查日程是否安全——walk_forward 只会传 `schedule()` 生成并通过
    `check_schedule` 的日程。
    """
    log = log or (lambda *_: None)
    _raw_s = raw_s
    per_fold = max(1, threads // fold_workers)
    shared = None
    if shared_bins:
        f0 = sched.folds[0]
        sl = store.samples(*f0.train)
        ref_idx = sl.start + np.flatnonzero(trainable[sl])
        shared = SharedBins(store, ref_idx, per_fold)
    n_cols = len(store.symbols)
    gram = DailyGram(store, y, trainable)

    def batch(index, with_y=True):
        day, col = store.rows[index], store.cols[index]
        return Batch(
            day.astype(np.int64),
            col.astype(np.int64),
            n_cols,
            store.names,
            y[index].astype(np.float64) if with_y else None,
            loader=lambda: store.matrix(index),
            index=index,
            shared=shared,
            threads=per_fold,
        )

    def run(fold):
        started = time.perf_counter()
        sl = store.samples(*fold.train)
        train_idx = sl.start + np.flatnonzero(trainable[sl])
        valid_idx = None
        if fold.valid is not None:
            sv = store.samples(*fold.valid)
            valid_idx = sv.start + np.flatnonzero(trainable[sv])
        test = store.samples(*fold.test)
        m = factory()
        train_b = batch(train_idx)
        train_b.gram = lambda: gram.window(*fold.train)
        valid_b = batch(valid_idx) if valid_idx is not None and len(valid_idx) else None
        m.fit(train_b, valid_b)
        train_b.release()
        if valid_b is not None:
            valid_b.release()
        test_b = batch(test, with_y=False)
        pred = np.asarray(m.predict(test_b), dtype=np.float64)
        if pred.shape != (test.stop - test.start,):
            raise ValueError("模型 predict 返回的长度与样本数不一致")
        test_b.release()
        used = [train_idx] + ([valid_idx] if valid_idx is not None else [])
        used = np.concatenate(used) if used else np.empty(0, np.int64)
        ic = daily_rank_ic(
            pred, _raw_s[test], store.rows[test], store.cols[test], n_cols
        )
        imp = m.feature_importance() if hasattr(m, "feature_importance") else None
        info = {
            "fold": fold.index,
            "train_rows": list(fold.train),
            "valid_rows": None if fold.valid is None else list(fold.valid),
            "test_rows": list(fold.test),
            "train": [
                str(store.dates[fold.train[0]]),
                str(store.dates[fold.train[1] - 1]),
            ],
            "test": [
                str(store.dates[fold.test[0]]),
                str(store.dates[fold.test[1] - 1]),
            ],
            "n_train": int(len(train_idx)),
            "n_valid": 0 if valid_idx is None else int(len(valid_idx)),
            "n_test": int(test.stop - test.start),
            "last_used_day": int(store.rows[used].max()) if len(used) else None,
            "best_iteration": getattr(m, "best_iteration", None),
            "seconds": time.perf_counter() - started,
            "importance": None if imp is None else np.asarray(imp, np.float64).tolist(),
            **{k: v for k, v in ic_stats(ic, horizon).items()},
        }
        log(
            f"第 {fold.index} 折 {info['test'][0]}–{info['test'][1]}：Rank IC "
            f"{info['ic_mean']:.4f}，{info['seconds']:.1f}s"
        )
        return fold, pred, info, m

    old = panelops.get_num_threads()
    panelops.set_num_threads(per_fold)
    try:
        if fold_workers > 1:
            with ThreadPoolExecutor(fold_workers) as pool:
                outputs = list(pool.map(run, sched.folds))
        else:
            outputs = [run(f) for f in sched.folds]
    finally:
        panelops.set_num_threads(old)
    pred_s = np.full(store.n_samples, np.nan)
    infos, models = [], []
    for fold, pred, info, m in outputs:
        pred_s[store.samples(*fold.test)] = pred
        infos.append(info)
        models.append(m)
    return pred_s, infos, models


@functools.cache
def _code_hash():
    here = Path(__file__).parent
    h = hashlib.sha256()
    for name in ("labels.py", "split.py", "models.py", "walk.py", "metrics.py"):
        h.update(here.joinpath(name).read_bytes())
    return h.hexdigest()


def _save_run(run_dir, result, pred_s, models):
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "pred.npy.tmp"
    with open(tmp, "wb") as f:
        np.save(f, pred_s)
    os.replace(tmp, run_dir / "pred.npy")
    saved = []
    for i, m in enumerate(models):
        try:
            save_model(m, run_dir / f"fold{i}")
            saved.append(True)
        except Exception:  # 模型不支持保存时只缓存预测
            saved.append(False)
    meta = {
        "folds": result.folds,
        "models": saved,
        "run_id": result.run_id,
        "metrics": {k: v for k, v in result.metrics.items() if k != "ic"},
    }
    body = json.dumps(
        meta, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)
    )
    (run_dir / "result.json.tmp").write_text(body)
    os.replace(run_dir / "result.json.tmp", run_dir / "result.json")


def _load_run(run_dir, store, label, sched, values, run_id, trial_id, keep_models):
    raw_panel = values.raw
    try:
        meta = json.loads((run_dir / "result.json").read_text())
        pred_s = np.load(run_dir / "pred.npy")
        if pred_s.shape != (store.n_samples,):
            return None
        models = None
        if keep_models and all(meta["models"]) and meta["models"]:
            models = [
                load_model(run_dir / f"fold{i}") for i in range(len(meta["models"]))
            ]
    except (OSError, ValueError, KeyError):
        return None
    pred_panel = store.scatter(pred_s)
    test_rows = np.concatenate([np.arange(*f.test) for f in sched.folds])
    metrics = panel_summary(pred_panel, raw_panel, label.horizon, q=10, rows=test_rows)
    return WalkResult(
        pred=pred_panel,
        dates=store.dates,
        symbols=store.symbols,
        label=label,
        schedule=sched,
        folds=meta["folds"],
        metrics=metrics,
        run_id=run_id,
        trial_id=trial_id,
        cached=True,
        feature_names=store.names,
        models=models,
        path=run_dir,
        raw=raw_panel,
        store=store,
        labels=values,
    )


def to_targets(pred, top=50, *, every=None, lag=None, start=None, validate=True):
    """预测（`WalkResult` 或 T×N 面板）→ 等权买入前 top 只的 `panelbt.Targets`。

    每 every 行（默认等于标签持有期）在信号日选股，成交日按 lag（默认标签的 lag）由
    `ashare.backtest.execution_targets` 错位。start：第一个信号日的行号（默认第一个有
    预测的行）。返回值直接交给 `panelbt.run(market, targets, costs, config)`。
    """
    from ..backtest import execution_targets

    if isinstance(pred, WalkResult):
        every = every or pred.label.horizon
        lag = lag or pred.label.lag
        pred = pred.pred
    every, lag = int(every or 5), int(lag or 1)
    score = np.ascontiguousarray(pred, dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(score).any(axis=1))
    if not len(rows):
        raise ValueError("预测面板全是 NaN")
    first = int(rows[0]) if start is None else int(start)
    days = np.arange(first, int(rows[-1]) + 1, every, dtype=np.int64)
    sparse = panelops.targets(score, k=int(top), days=days)
    return execution_targets(sparse, t=score.shape[0], lag=lag, validate=validate)


_code_hash()
