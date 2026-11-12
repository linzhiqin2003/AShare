"""超参搜索：网格或随机，用同一套时间切分评估；每一组都进账本。

显著性折算先做简单可靠的：Bonferroni。门槛 = α / N，N 取账本里到现在为止的试验组数
（不只本次搜索），最好那组的 p 值乘以 N 再和 α 比。更精细的方法（Deflated Sharpe、
White Reality Check、Romano–Wolf 等）留到因子挖掘那条线。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from . import store as _store
from .ledger import Ledger
from .walk import _panel, default_universe, required_fields, walk_forward  # noqa: F401


def grid_configs(grid):
    """{参数: [取值...]} 的笛卡尔积，按键名排序后展开，顺序固定。"""
    keys = sorted(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def random_configs(space, n_iter, seed=0):
    """space：{参数: 取值列表} 或 {参数: (low, high, "uniform"|"log"|"int")}。"""
    rng = np.random.default_rng(seed)
    keys = sorted(space)
    out = []
    for _ in range(int(n_iter)):
        cfg = {}
        for k in keys:
            v = space[k]
            if isinstance(v, tuple) and len(v) == 3 and isinstance(v[2], str):
                lo, hi, kind = v
                if kind == "log":
                    cfg[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
                elif kind == "int":
                    cfg[k] = int(rng.integers(lo, hi + 1))
                elif kind == "uniform":
                    cfg[k] = float(rng.uniform(lo, hi))
                else:
                    raise ValueError(f"{k}: 分布只能是 uniform、log、int")
            else:
                options = list(v)
                cfg[k] = options[int(rng.integers(len(options)))]
        out.append(cfg)
    return out


@dataclass
class SearchResult:
    table: list
    n_trials_total: int
    alpha: float

    @property
    def best(self):
        rows = [r for r in self.table if math.isfinite(r["ic_mean"])]
        return max(rows, key=lambda r: r["ic_mean"]) if rows else None

    @property
    def threshold(self):
        """Bonferroni 门槛：α / 账本里的总试验组数。"""
        return self.alpha / max(self.n_trials_total, 1)

    def describe(self):
        lines = [
            f"共 {len(self.table)} 组；账本累计 {self.n_trials_total} 组；"
            f"Bonferroni 门槛 p < {self.threshold:.2e}（α = {self.alpha}）"
        ]
        for r in sorted(
            self.table,
            key=lambda r: -r["ic_mean"] if math.isfinite(r["ic_mean"]) else 1,
        ):
            flag = "显著" if r["significant"] else "不显著"
            lines.append(
                f"IC {r['ic_mean']:+.4f}  ICIR {r['icir']:+.3f}  p {r['p']:.2e}  "
                f"折算 p {r['p_adjusted']:.2e} {flag}  {r['params']}"
            )
        best = self.best
        if best is not None:
            lines.append(
                f"试了 {self.n_trials_total} 组之后，最好那组（IC {best['ic_mean']:+.4f}）"
                f"折算后 p = {best['p_adjusted']:.2e}，"
                + (
                    "仍显著"
                    if best["significant"]
                    else "不能排除是多次尝试挑出来的运气"
                )
            )
        return "\n".join(lines)


def search(
    features,
    label="ret5",
    *,
    data,
    model="lgbm",
    grid=None,
    space=None,
    n_iter=20,
    seed=0,
    base=None,
    alpha=0.05,
    root=None,
    universe=None,
    preprocess=None,
    tag="search",
    log=None,
    end,
    **walk_kwargs,
):
    """网格（给 grid）或随机（给 space 和 n_iter）搜索模型参数。

    每组参数跑一次 `walk_forward`（同一特征库、同一切分），结果都进账本（带 tag）。
    end（必填）：搜索用到的最后一个预测日。设在最终检验期之前，留出的那一段不参与任何
    一组的评估，选定参数后只在它上面跑一次报告结果。不给 end 会报错，避免默认用到
    全部日期、没有留出段。
    返回 `SearchResult`：table 每行有 params、ic_mean、icir、p、p_adjusted、significant。
    """
    if end is None:
        raise ValueError(
            "search 必须给 end：把最后一段日期留作最终检验，搜索只用到 end 为止"
        )
    walk_kwargs["end"] = end
    if (grid is None) == (space is None):
        raise ValueError("给 grid（网格）或 space（随机）其中之一")
    configs = (
        grid_configs(grid) if grid is not None else random_configs(space, n_iter, seed)
    )
    base = dict(base or {})
    root = root if root is not None else _store.default_root()
    panel = _panel(data, required_fields(features, label, preprocess, universe is None))
    if universe is None:
        universe = default_universe(panel)
    from .preprocess import Preprocess

    store = _store.build(
        _store.normalize_features(features),
        panel,
        universe,
        preprocess or Preprocess(),
        root=root,
        threads=walk_kwargs.get("threads"),
    )
    rows = []
    for k, cfg in enumerate(configs):
        res = walk_forward(
            features,
            label,
            data=panel,
            universe=universe,
            model=model,
            params=base | cfg,
            store=store,
            root=root,
            seed=seed,
            keep_models=False,
            tags=(tag,),
            log=log,
            **walk_kwargs,
        )
        m = res.metrics
        rows.append(
            {
                "params": cfg,
                "ic_mean": m["ic_mean"],
                "icir": m["icir"],
                "t": m["t"],
                "p": m["p"],
                "long_short": m.get("long_short"),
                "run_id": res.run_id,
                "trial_id": res.trial_id,
                "cached": res.cached,
            }
        )
        if log:
            log(f"搜索 {k + 1}/{len(configs)}：{cfg} → IC {m['ic_mean']:.4f}")
    total = Ledger(_store.Path(root) / "ledger.jsonl").trials()
    total = max(total, len(rows))
    for r in rows:
        r["p_adjusted"] = (
            min(1.0, r["p"] * total) if math.isfinite(r["p"]) else float("nan")
        )
        r["significant"] = bool(math.isfinite(r["p"]) and r["p"] < alpha / total)
    return SearchResult(rows, total, alpha)
