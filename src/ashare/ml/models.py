"""模型协议、注册表和内置模型。

协议很小（`Model`）：`fit(train, valid)`、`predict(batch)`、`save(path)`、类方法
`load(path)`，可选 `feature_importance()` 和 `get_params()`。`train`、`valid`、`batch`
都是 `Batch`：`X`（float32 样本 × 特征，取用时才读盘）、`y`（训练目标）、`day`（每个
样本的信号日行号，升序）、`col`（证券列号）。第三方模型（包括以后的 PyTorch 模型）只要
实现这个协议，再 `register_model("名字", 工厂)` 或在入口点组 `ashare.ml.models` 里登记，
就能用 `walk_forward(..., model="名字")` 调用，不用改本包的代码。

```python
class CorrWeights:                         # 二十行以内的自定义模型
    def __init__(self, shrink=0.5):
        self.shrink = shrink
    def fit(self, train, valid=None):
        c = np.corrcoef(np.nan_to_num(train.X).T, train.y)[-1, :-1]
        self.w = self.shrink * np.nan_to_num(c)
    def predict(self, batch):
        return np.nan_to_num(batch.X) @ self.w
    def save(self, path):
        np.save(Path(path) / "w.npy", self.w)
    @classmethod
    def load(cls, path):
        m = cls(); m.w = np.load(Path(path) / "w.npy"); return m

ml.register_model("corr", CorrWeights)
ml.walk_forward(features, "ret5", data=panel, model="corr")
```
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import _deps
from .metrics import mean_rank_ic, rank_within_days

# ---------------------------------------------------------------- 数据批次


@dataclass
class Batch:
    """一段样本。X 在第一次访问时才从特征库读出（float32，C 连续），之后缓存。"""

    day: np.ndarray
    col: np.ndarray
    n_cols: int
    feature_names: tuple
    y: np.ndarray | None = None
    loader: Any = None
    index: Any = None
    shared: Any = None
    threads: int = 1
    gram: Any = None
    _X: np.ndarray | None = field(default=None, repr=False)

    @property
    def X(self):
        if self._X is None:
            self._X = self.loader()
        return self._X

    def __len__(self):
        return len(self.day)

    def release(self):
        self._X = None

    @classmethod
    def from_arrays(cls, X, y=None, day=None, col=None, feature_names=None, threads=1):
        """直接用数组造一个批次（测试、自定义流程用）。"""
        X = np.ascontiguousarray(X, dtype=np.float32)
        day = np.zeros(len(X), np.int64) if day is None else np.asarray(day, np.int64)
        col = np.arange(len(X)) if col is None else np.asarray(col, np.int64)
        names = tuple(feature_names or (f"f{i}" for i in range(X.shape[1])))
        return cls(
            day,
            col,
            int(col.max()) + 1 if len(col) else 0,
            names,
            None if y is None else np.asarray(y, np.float64),
            _X=X,
            threads=threads,
        )


@runtime_checkable
class Model(Protocol):
    def fit(self, train: Batch, valid: Batch | None) -> None: ...

    def predict(self, batch: Batch) -> np.ndarray: ...

    def save(self, path) -> None: ...


def params_of(model):
    """模型参数（进账本和缓存键）；没有 get_params 时用类名 + __dict__ 里的标量。"""
    if hasattr(model, "get_params"):
        return {"class": type(model).__name__, **model.get_params()}
    simple = {
        k: v
        for k, v in vars(model).items()
        if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))
    }
    return {"class": f"{type(model).__module__}.{type(model).__qualname__}", **simple}


# ---------------------------------------------------------------- 注册表


@dataclass(frozen=True)
class _Entry:
    factory: Any  # 可调用对象，或 "模块:属性" 字符串（用到时才导入）
    group: str | None = None
    requires: tuple = ()
    description: str = ""


_REGISTRY: dict[str, _Entry] = {}
_LOCK = threading.Lock()
# 预留给 nn 可选组的名字：没装时给出装哪个组的提示。
RESERVED = {"mlp": "nn", "lstm": "nn", "gru": "nn", "transformer": "nn"}
ENTRY_POINT_GROUP = "ashare.ml.models"


def register_model(
    name, factory, *, group=None, requires=None, description="", overwrite=False
):
    """登记模型名。factory：类或返回模型实例的可调用对象，也可以是 "模块:属性" 字符串
    （用到时才导入，登记时不需要装依赖）。group：所需的可选依赖组（如 "nn"），用到时
    检查；requires：额外要能导入的模块名。"""
    if not isinstance(name, str) or not name:
        raise ValueError("模型名必须是非空字符串")
    with _LOCK:
        if name in _REGISTRY and not overwrite:
            raise ValueError(f"模型名 {name!r} 已登记；要替换请传 overwrite=True")
        if requires is None:
            requires = _deps.GROUPS.get(group, ()) if group else ()
        _REGISTRY[name] = _Entry(factory, group, tuple(requires), description)


def unregister_model(name):
    with _LOCK:
        _REGISTRY.pop(name, None)


def available_models():
    _load_entry_points()
    return sorted(_REGISTRY)


_ENTRY_POINTS_LOADED = False


def _load_entry_points():
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.name not in _REGISTRY:
            # 入口点的值就是 "模块:属性"，延迟到用时导入。
            register_model(ep.name, ep.value, group=None, description="入口点")


def _resolve(factory):
    if isinstance(factory, str):
        module, _, attr = factory.partition(":")
        obj = importlib.import_module(module)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj
    return factory


def make_model(spec, params=None):
    """名字（+参数字典）或模型实例 → 模型实例。"""
    params = dict(params or {})
    if not isinstance(spec, str):
        if params:
            raise ValueError("传了模型实例时不能再给 params")
        if not (hasattr(spec, "fit") and hasattr(spec, "predict")):
            raise TypeError("模型需要实现 fit(train, valid) 和 predict(batch)")
        return spec
    if spec not in _REGISTRY:
        _load_entry_points()
    entry = _REGISTRY.get(spec)
    if entry is None:
        if spec in RESERVED:
            group = RESERVED[spec]
            _deps.require(group, f"模型 {spec!r} ")
            raise ValueError(
                f"模型 {spec!r} 属于可选组 `{group}`，依赖已装，但还没有包登记这个名字；"
                f"安装提供它的包，或用 ml.register_model({spec!r}, 你的类) 登记"
            )
        raise ValueError(f"未知模型 {spec!r}；已登记：{available_models()}")
    if entry.requires:
        absent = [m for m in entry.requires if not _deps.importable(m)]
        if absent:
            group = entry.group or "?"
            if entry.group in _deps.GROUPS:
                _deps.require(entry.group, f"模型 {spec!r} ")
            raise _deps.MissingDependency(
                f"模型 {spec!r} 需要可选依赖组 `{group}`（缺 {'、'.join(absent)}）。"
                f"运行 `uv sync --all-packages --group dev --group {group} --inexact` 安装。"
            )
    return _resolve(entry.factory)(**params)


def load_model(path):
    """读回 `save` 存下的模型（目录里有 model.json 记着类）。"""
    meta = json.loads((Path(path) / "model.json").read_text())
    cls = _resolve(meta["class"])
    return cls.load(path)


def save_model(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save(path)
    cls = type(model)
    (path / "model.json").write_text(
        json.dumps(
            {
                "class": f"{cls.__module__}:{cls.__qualname__}",
                "params": params_of(model),
            },
            default=str,
        )
    )


# ---------------------------------------------------------------- 线性模型


def _finite(X):
    return np.where(np.isfinite(X), X, 0)


class Ridge:
    """岭回归（带截距）。solver="numpy"：按块累积正规方程的充分统计量再解，内存只占
    一块的 float64；"sklearn"：sklearn.linear_model.Ridge（参考实现）。特征缺失按 0。"""

    def __init__(self, alpha=1.0, solver="numpy", chunk=1 << 20):
        self.alpha, self.solver, self.chunk = float(alpha), solver, int(chunk)
        self.coef_ = self.intercept_ = None

    def get_params(self):
        return {"alpha": self.alpha, "solver": self.solver}

    def fit(self, train, valid=None):
        if self.solver == "numpy" and train.gram is not None:
            X, y = None, None
        else:
            X, y = train.X, np.asarray(train.y, np.float64)
        if self.solver == "sklearn":
            from sklearn.linear_model import Ridge as SkRidge

            m = SkRidge(alpha=self.alpha).fit(_finite(X).astype(np.float64), y)
            self.coef_, self.intercept_ = m.coef_, float(m.intercept_)
            return self
        if self.solver != "numpy":
            raise ValueError("solver: numpy 或 sklearn")
        f = X.shape[1] if train.gram is None else len(train.feature_names)
        if train.gram is not None:
            # 滚动训练时按日预先累积的充分统计量：每个样本只读一次，各折只把窗口内的
            # 日统计量相加（walk_forward 提供，见 ashare.ml.walk.DailyGram）。
            n, sx, sy, xtx, xty = train.gram()
        else:
            n, sx, sy = 0, np.zeros(f), 0.0
            xtx, xty = np.zeros((f, f)), np.zeros(f)
            for a in range(0, len(X), self.chunk):
                xb = X[a : a + self.chunk].astype(np.float64)
                xb[~np.isfinite(xb)] = 0.0
                yb = y[a : a + self.chunk]
                n += len(xb)
                sx += xb.sum(axis=0)
                sy += yb.sum()
                xtx += xb.T @ xb
                xty += xb.T @ yb
        mx, my = sx / n, sy / n
        c = xtx - n * np.outer(mx, mx)
        b = xty - n * mx * my
        self.coef_ = np.linalg.solve(c + self.alpha * np.eye(f), b)
        self.intercept_ = float(my - mx @ self.coef_)
        return self

    def predict(self, batch):
        X = batch.X if isinstance(batch, Batch) else batch
        return _finite(X).astype(np.float64) @ self.coef_ + self.intercept_

    def feature_importance(self):
        return np.abs(self.coef_)

    def save(self, path):
        np.savez(Path(path) / "linear.npz", coef=self.coef_, intercept=self.intercept_)
        (Path(path) / "params.json").write_text(json.dumps(self.get_params()))

    @classmethod
    def load(cls, path):
        m = cls(**json.loads((Path(path) / "params.json").read_text()))
        z = np.load(Path(path) / "linear.npz")
        m.coef_, m.intercept_ = z["coef"], float(z["intercept"])
        return m


class _SkLinear(Ridge):
    _cls = None

    def __init__(self, alpha=1e-4, l1_ratio=1.0, max_iter=2000, tol=1e-4):
        self.alpha, self.l1_ratio = float(alpha), float(l1_ratio)
        self.max_iter, self.tol = int(max_iter), float(tol)
        self.coef_ = self.intercept_ = None

    def get_params(self):
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def fit(self, train, valid=None):
        from sklearn.linear_model import ElasticNet as SkEN

        m = SkEN(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        m.fit(_finite(train.X).astype(np.float64), np.asarray(train.y, np.float64))
        self.coef_, self.intercept_ = m.coef_, float(m.intercept_)
        return self


class Lasso(_SkLinear):
    """Lasso（sklearn 坐标下降），alpha 是 sklearn 的口径。"""

    def __init__(self, alpha=1e-4, max_iter=2000, tol=1e-4):
        super().__init__(alpha, 1.0, max_iter, tol)

    def get_params(self):
        return {"alpha": self.alpha, "max_iter": self.max_iter, "tol": self.tol}


class ElasticNet(_SkLinear):
    """ElasticNet（sklearn），l1_ratio 介于 0 和 1。"""


# ---------------------------------------------------------------- LightGBM

_BIN_KEYS = (
    "max_bin",
    "min_data_in_bin",
    "bin_construct_sample_cnt",
    "data_random_seed",
    "use_missing",
    "zero_as_missing",
    "feature_pre_filter",
)


class LGBM:
    """LightGBM 回归。提前停止看验证集上的逐日 Rank IC 均值（不是损失）。

    bins="shared"（默认）：分箱边界只用第一折的训练数据定，全部样本分箱一次，各折取子集
    （跨折复用，且不用到第一折预测期以后的数据）；"per_fold"：每折用自己的训练数据重新
    分箱（与最直白的写法逐位一致，慢）。固定 seed 和 threads 时结果逐位可复现。
    """

    objective = "regression"

    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=1000,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=10.0,
        max_bin=63,
        early_stopping=50,
        seed=0,
        threads=None,
        bins="shared",
        **extra,
    ):
        self.n_estimators, self.learning_rate = int(n_estimators), float(learning_rate)
        self.num_leaves, self.min_data_in_leaf = int(num_leaves), int(min_data_in_leaf)
        self.feature_fraction, self.bagging_fraction = (
            feature_fraction,
            bagging_fraction,
        )
        self.bagging_freq, self.lambda_l2, self.max_bin = (
            bagging_freq,
            lambda_l2,
            max_bin,
        )
        self.early_stopping, self.seed, self.threads = early_stopping, seed, threads
        if bins not in ("shared", "per_fold"):
            raise ValueError("bins: shared 或 per_fold")
        self.bins, self.extra = bins, dict(extra)
        self.booster = None
        self.best_iteration = None
        self.history = []

    def get_params(self):
        return {
            k: getattr(self, k)
            for k in (
                "n_estimators",
                "learning_rate",
                "num_leaves",
                "min_data_in_leaf",
                "feature_fraction",
                "bagging_fraction",
                "bagging_freq",
                "lambda_l2",
                "max_bin",
                "early_stopping",
                "seed",
                "bins",
            )
        } | {"objective": self.objective, **self.extra}

    def lgb_params(self, threads):
        return {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "max_bin": self.max_bin,
            "seed": self.seed,
            "num_threads": int(self.threads or threads),
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
            "metric": "None",
            "feature_pre_filter": False,
            **self.extra,
        }

    def dataset_params(self):
        p = self.lgb_params(1)
        return {k: p[k] for k in _BIN_KEYS if k in p} | {"verbosity": -1}

    def _label(self, batch):
        return np.asarray(batch.y, np.float64)

    def _group(self, batch):
        return None

    def _dataset(self, batch, params, reference=None):
        import lightgbm as lgb

        label, group = self._label(batch), self._group(batch)
        if self.bins == "shared" and batch.shared is not None:
            ds = batch.shared.subset(self.dataset_params(), batch.index)
            ds.set_label(label)
            if group is not None:
                ds.set_group(group)
            return ds
        return lgb.Dataset(
            batch.X,
            label=label,
            group=group,
            params=params,
            reference=reference,
            free_raw_data=True,
        )

    def fit(self, train, valid=None):
        import lightgbm as lgb

        params = self.lgb_params(train.threads)
        dtrain = self._dataset(train, params)
        valid_sets, callbacks, feval = [], [], None
        if valid is not None and len(valid) and self.early_stopping:
            dvalid = self._dataset(valid, params, reference=dtrain)
            y, day, col, n = valid.y, valid.day, valid.col, valid.n_cols

            def feval(preds, _data):
                return "rank_ic", mean_rank_ic(preds, y, day, col, n), True

            valid_sets = [dvalid]
            callbacks = [
                lgb.early_stopping(
                    self.early_stopping, first_metric_only=True, verbose=False
                ),
                lgb.record_evaluation(record := {}),
            ]
        self.booster = lgb.train(
            params,
            dtrain,
            self.n_estimators,
            valid_sets=valid_sets,
            valid_names=["valid"] if valid_sets else None,
            feval=feval,
            callbacks=callbacks,
        )
        self.best_iteration = (
            self.booster.best_iteration or self.booster.current_iteration()
        )
        if valid_sets:
            self.history = list(record.get("valid", {}).get("rank_ic", []))
        return self

    def predict(self, batch):
        X = batch.X if isinstance(batch, Batch) else batch
        return self.booster.predict(
            X,
            num_iteration=self.best_iteration,
            num_threads=int(self.threads or getattr(batch, "threads", 1)),
        )

    def feature_importance(self, kind="gain"):
        return self.booster.feature_importance(kind, iteration=self.best_iteration)

    def save(self, path):
        self.booster.save_model(
            str(Path(path) / "booster.txt"), num_iteration=self.best_iteration
        )
        (Path(path) / "params.json").write_text(
            json.dumps(
                {"params": self.get_params(), "best_iteration": self.best_iteration}
            )
        )

    @classmethod
    def load(cls, path):
        import lightgbm as lgb

        meta = json.loads((Path(path) / "params.json").read_text())
        params = dict(meta["params"])
        params.pop("objective", None)
        m = cls(**params)
        m.booster = lgb.Booster(model_file=str(Path(path) / "booster.txt"))
        m.best_iteration = meta["best_iteration"]
        return m


class LGBMRank(LGBM):
    """LightGBM 排序（lambdarank，按日分组）。训练目标按日分成 levels 个等级（0 最差），
    等级增益线性；预测值只有截面上的相对大小有意义。"""

    objective = "lambdarank"

    def __init__(self, levels=20, **kwargs):
        kwargs.setdefault("min_data_in_leaf", 1000)
        super().__init__(**kwargs)
        self.levels = int(levels)

    def get_params(self):
        return super().get_params() | {"levels": self.levels}

    def lgb_params(self, threads):
        return super().lgb_params(threads) | {
            "label_gain": list(range(self.levels)),
            "lambdarank_truncation_level": 50,
        }

    def _label(self, batch):
        r = rank_within_days(batch.y, batch.day, batch.col, batch.n_cols) + 0.5
        return np.minimum(np.floor(r * self.levels), self.levels - 1).astype(np.float64)

    def _group(self, batch):
        _, counts = np.unique(batch.day, return_counts=True)
        return counts

    @classmethod
    def load(cls, path):
        meta = json.loads((Path(path) / "params.json").read_text())
        params = dict(meta["params"])
        params.pop("objective", None)
        levels = params.pop("levels", 20)
        import lightgbm as lgb

        m = cls(levels=levels, **params)
        m.booster = lgb.Booster(model_file=str(Path(path) / "booster.txt"))
        m.best_iteration = meta["best_iteration"]
        return m


# ---------------------------------------------------------------- 集成


class Ensemble:
    """多模型、多随机种子的平均。members：模型名或 (名字, 参数) 列表；seeds：每个接受
    seed 参数的成员按这些种子各训练一份。combine="mean" 直接平均预测值，"rank" 先按日
    截面排名再平均（量纲不同的模型合在一起时用）。"""

    def __init__(self, members=("lgbm",), seeds=(0,), combine="rank"):
        if combine not in ("mean", "rank"):
            raise ValueError("combine: mean 或 rank")
        self.members = [m if isinstance(m, (list, tuple)) else (m, {}) for m in members]
        self.members = [(str(n), dict(p)) for n, p in self.members]
        self.seeds, self.combine = tuple(int(s) for s in seeds), combine
        self.models = []

    def get_params(self):
        return {
            "members": self.members,
            "seeds": list(self.seeds),
            "combine": self.combine,
        }

    def _instances(self):
        out = []
        for name, params in self.members:
            probe = make_model(name, params)
            if "seed" in getattr(probe, "get_params", dict)():
                out += [make_model(name, params | {"seed": s}) for s in self.seeds]
            else:
                out.append(probe)
        return out

    @property
    def bins(self):
        """有成员用共享分箱时，让 walk_forward 准备跨折共享的分箱数据集。"""
        kinds = {getattr(m, "bins", None) for m in self._instances()}
        return "shared" if "shared" in kinds else None

    def fit(self, train, valid=None):
        self.models = [m.fit(train, valid) or m for m in self._instances()]
        return self

    def predict(self, batch):
        preds = [np.asarray(m.predict(batch), np.float64) for m in self.models]
        if self.combine == "rank":
            preds = [
                rank_within_days(p, batch.day, batch.col, batch.n_cols) for p in preds
            ]
        return np.mean(preds, axis=0)

    def feature_importance(self):
        parts = [
            m.feature_importance()
            for m in self.models
            if hasattr(m, "feature_importance")
        ]
        parts = [p / p.sum() if p.sum() > 0 else p for p in parts]
        return np.mean(parts, axis=0) if parts else None

    def save(self, path):
        for i, m in enumerate(self.models):
            save_model(m, Path(path) / f"member{i}")
        (Path(path) / "params.json").write_text(json.dumps(self.get_params()))

    @classmethod
    def load(cls, path):
        p = json.loads((Path(path) / "params.json").read_text())
        e = cls([tuple(m) for m in p["members"]], p["seeds"], p["combine"])
        k = 0
        while (Path(path) / f"member{k}").exists():
            e.models.append(load_model(Path(path) / f"member{k}"))
            k += 1
        return e


register_model("ridge", Ridge, description="岭回归（NumPy 正规方程）")
register_model("lasso", Lasso, description="Lasso（sklearn）")
register_model("elasticnet", ElasticNet, description="ElasticNet（sklearn）")
register_model("lgbm", LGBM, description="LightGBM 回归")
register_model("lgbm_rank", LGBMRank, description="LightGBM lambdarank，按日分组")
register_model("ensemble", Ensemble, description="多模型 / 多种子平均")
