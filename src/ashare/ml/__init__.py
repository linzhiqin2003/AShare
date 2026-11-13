"""机器学习：因子当特征 → 滚动训练 → 预测值当新因子。

需要可选依赖组 `ml`（LightGBM、scikit-learn）：
`uv sync --all-packages --group dev --group ml --inexact`。神经网络模型以后走单独的
`nn` 组，默认不装。

```python
import ashare.data as ad
import ashare.ml as ml
data = ad.open()
res = ml.walk_forward(ml.basic_features(), "ret5", data=data, model="lgbm",
                      retrain="quarter", train_years=5, start="2015-01-01")
print(res.summary())
targets = ml.to_targets(res, top=50)        # 交给 panelbt.run
```
"""

from ._deps import MissingDependency, require

require("ml", "ashare.ml ")

from .explain import permutation_importance, permute_within_days  # noqa: E402
from .features import basic_features  # noqa: E402
from .labels import PRESETS, Label, LabelValues, label  # noqa: E402
from .labels import compute as compute_label  # noqa: E402
from .ledger import Ledger  # noqa: E402
from .metrics import daily_rank_ic, ic_stats, panel_summary, rank_within_days  # noqa: E402
from .models import (  # noqa: E402
    LGBM,
    Batch,
    ElasticNet,
    Ensemble,
    Lasso,
    LGBMRank,
    Model,
    Ridge,
    available_models,
    load_model,
    make_model,
    register_model,
    save_model,
    unregister_model,
)
from .preprocess import Preprocess  # noqa: E402
from .search import SearchResult, grid_configs, random_configs, search  # noqa: E402
from .split import Fold, Schedule, check_schedule, schedule  # noqa: E402
from .store import FeatureStore, default_root  # noqa: E402
from .store import build as build_store  # noqa: E402
from .walk import (  # noqa: E402
    WalkResult,
    check_lookahead,
    default_threads,
    default_universe,
    required_fields,
    to_targets,
    walk_forward,
)

__all__ = [
    "Batch",
    "ElasticNet",
    "Ensemble",
    "FeatureStore",
    "Fold",
    "LGBM",
    "LGBMRank",
    "Label",
    "LabelValues",
    "Lasso",
    "Ledger",
    "MissingDependency",
    "Model",
    "PRESETS",
    "Preprocess",
    "Ridge",
    "Schedule",
    "SearchResult",
    "WalkResult",
    "available_models",
    "basic_features",
    "build_store",
    "check_lookahead",
    "check_schedule",
    "compute_label",
    "daily_rank_ic",
    "default_root",
    "default_threads",
    "default_universe",
    "grid_configs",
    "ic_stats",
    "label",
    "load_model",
    "make_model",
    "panel_summary",
    "permutation_importance",
    "permute_within_days",
    "random_configs",
    "rank_within_days",
    "register_model",
    "required_fields",
    "save_model",
    "schedule",
    "search",
    "to_targets",
    "unregister_model",
    "walk_forward",
]
