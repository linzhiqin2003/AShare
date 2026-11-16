"""C8 真实挖掘期演示；研究文件只写 tmp，不提供验证或确认开关。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ashare.data import crypto
from ashare.factor.timing import Factor, Settings, Study


def momentum(view, hours):
    frame = view.hourly()
    px = frame["close"].to_numpy()
    quality = frame["quality"].to_numpy()
    px = np.where((quality & 3721) == 0, px, np.nan)
    value = np.full(len(px), np.nan)
    value[hours:] = px[hours:] / px[:-hours] - 1
    return frame["available_at"].to_numpy(), value


def funding_mean(view, hours):
    frame = view.hourly()
    value = (
        frame.select(
            pl.col("funding_rate")
            .fill_nan(0.0)
            .fill_null(0.0)
            .rolling_sum(window_size=hours, min_samples=hours)
            * 8760
            / hours
        )
        .to_series()
        .to_numpy()
    )
    return frame["available_at"].to_numpy(), value


def run(root, folder, *, minute_root=None, metrics_root=None):
    folder = Path(folder)
    # 演示路径必须留在当前 worktree tmp；不接受主目录 mining 或外部私有目录。
    base = (Path(__file__).resolve().parents[1] / "tmp").resolve()
    if not folder.resolve().is_relative_to(base):
        raise ValueError("演示仅允许写本 worktree 的 tmp 子目录")
    folder.mkdir(parents=True, exist_ok=True)
    value = dict(
        ledger=str((folder / "ledger.jsonl").resolve()),
        symbol="BTCUSDT",
        trade_hour=0,
        horizon=7,
        stages=dict(
            mining=["2020-01-01", "2025-05-01"],
            validation=["2025-05-01", "2026-01-01"],
            confirmation=["2026-01-01", "2026-11-01"],
        ),
        validation_hash=None,
        confirmation_hash=None,
        repetitions=4000,
    )
    settings = Settings.from_dict(value)
    (folder / "study.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
    study = Study(
        settings,
        crypto.open(root),
        ledger=folder / "ledger.jsonl",
        minute_root=minute_root,
        metrics_root=metrics_root,
    )
    output = {}
    definitions = {}
    began = time.perf_counter()
    for name, factor in (
        ("momentum_7d", Factor(momentum, params=dict(hours=168), family="momentum")),
        ("momentum_30d", Factor(momentum, params=dict(hours=720), family="momentum")),
        (
            "funding_mean_7d",
            Factor(funding_mean, params=dict(hours=168), family="funding"),
        ),
    ):
        t = time.perf_counter()
        result = study.evaluate(factor)
        output[name] = {
            k: result[k]
            for k in (
                "n",
                "alpha",
                "beta",
                "se",
                "t",
                "meat_terms",
                "p",
                "lookahead_checks",
            )
        }
        output[name].update(
            spearman=result["descriptive"]["spearman"], seconds=time.perf_counter() - t
        )
        definitions[name] = study.test_hash(factor)
        print(name + " complete", flush=True)
    summary = study.summary()
    for name, key in definitions.items():
        output[name]["q"] = summary["q"][key]
    records = study.ledger.records()
    if any(r["stage"] != "mining" for r in records):
        raise ValueError("真实演示出现非挖掘记录")
    return dict(
        attempts=summary["attempts"],
        validation_calls=0,
        confirmation_calls=0,
        seconds=time.perf_counter() - began,
        factors=output,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--minute-root")
    p.add_argument("--metrics-root")
    p.add_argument("--folder", default="tmp/c8-timing-demo")
    p.add_argument("--output", default="bench/results/timing_demo.json")
    a = p.parse_args()
    result = run(
        a.root, a.folder, minute_root=a.minute_root, metrics_root=a.metrics_root
    )
    Path(a.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
