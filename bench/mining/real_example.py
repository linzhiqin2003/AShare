"""真实数据上的小规模搜索例子：默认空间、随机采样加一轮进化。

只读主目录的面板。台账、指纹、具体表达式和它们的指标都写到 --work（必须在仓库之外：
私有仓库或临时目录）；--summary 只写汇总统计（试验数、通过数、耗时、折算前后的显著个数
分布），这一份可以进仓库。不解锁留出段。

    uv run --no-sync python bench/mining/real_example.py --root <数据根目录> \\
        --work <仓库外的目录> --summary bench/mining/results/real_example.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from ashare.factor.evaluate import FactorEvaluator
from ashare.factor.mining import Evolution, Random, Space, Split, mine

BINS = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0000001)


def _hist(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    counts, _ = np.histogram(v, bins=BINS)
    labels = ["[0,0.5)", "[0.5,0.9)", "[0.9,0.95)", "[0.95,0.99)", "[0.99,1]"]
    return dict(zip(labels, counts.tolist(), strict=True))


def _stats(res):
    s = res.summary
    distinct = [
        r
        for r in res.candidates
        if r["stage"] == "backtest" and r["duplicate_of"] is None
    ]
    return {
        "strategy": s["strategy"],
        "proposed": s["proposed"],
        "unique": s["unique"],
        "screened_out": s["screened_out"],
        "backtested": s["backtested"],
        "duplicates": s["duplicates"],
        "failed": s["failed"],
        "cache_hits": s["cache_hits"],
        "trials_total": s["trials_total"],
        "effective_trials": s["effective_trials"],
        "pbo": s["pbo"],
        "pbo_note": s.get("pbo_note"),
        "significant": s["significant"],
        "psr_train_hist": _hist(r["psr_train"] for r in distinct),
        "dsr_train_hist": _hist(r["dsr_train"] for r in distinct),
        "dsr_valid_hist": _hist(r["dsr_valid"] for r in distinct),
        "seconds": s["seconds"],
        "per_second": s["per_second"],
    }


def main(argv=None):
    import ashare.data as ad
    from ashare.backtest import market_from_panel

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--random", type=int, default=1000)
    ap.add_argument("--population", type=int, default=100)
    ap.add_argument("--generations", type=int, default=5)
    args = ap.parse_args(argv)
    work = Path(args.work)
    started = time.time()
    data = ad.open(root=args.root)
    space = Space.default()
    fields = sorted(set(FactorEvaluator.required_fields()) | set(space.fields))
    panel = data.panel(fields, start=args.start)
    market = market_from_panel(data, panel=panel, validate=False)
    split = Split.recent(panel.dates, holdout_years=2, valid_years=2)
    load = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    out = {
        "panel": {"rows": len(panel.dates), "symbols": len(panel.symbols),
                  "first": str(panel.dates[0])[:4], "last": str(panel.dates[-1])[:4]},
        "split": {"valid_start": split.valid_start[:4], "holdout_start": split.holdout_start[:4]},
        "load_before": load.split("load averages:")[-1].strip(),
        "settings": {"space": "Space.default()", "rebalance": 5, "top_fraction": 0.1,
                     "screen_t": 1.0, "sr_var": "null", "alpha": 0.05,
                     "benchmark": "universe", "autocorr": True,
                     "random": args.random, "population": args.population,
                     "generations": args.generations},
    }  # fmt: skip
    common = dict(split=split, space=space, root=work, market=market)
    first = mine(panel, strategy=Random(n=args.random, seed=1), **common)
    (work / "random.txt").write_text(first.text(top=20))
    second = mine(
        panel,
        strategy=Evolution(
            population=args.population, generations=args.generations, seed=1
        ),
        **common,
    )
    (work / "evolution.txt").write_text(second.text(top=20))
    (work / "evolution.json").write_text(
        json.dumps(second.to_dict(), ensure_ascii=False, default=str)
    )
    out["random"] = _stats(first)
    out["evolution"] = _stats(second)
    out["seconds_total"] = time.time() - started
    out["load_after"] = (
        subprocess.run(["uptime"], capture_output=True, text=True)
        .stdout.split("load averages:")[-1]
        .strip()
    )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
