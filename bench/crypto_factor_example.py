"""C6 回溯探索：冻结的资金费过热减仓规则，不调参。"""

import argparse
import json
from pathlib import Path

import numpy as np
from crypto_examples import signals, summary

from ashare.backtest.crypto import crypto_market, daily_targets
from ashare.data import crypto
from ashare.factor.crypto import crypto_factors, daily_factor


def overheating(weights, z):
    """z>2 多头减半，z<−2 空头减半，缺因子不变。"""
    hot = ((weights > 0) & (z > 2)) | ((weights < 0) & (z < -2))
    return np.where(hot, weights * 0.5, weights)


def run(ds):
    f = crypto_factors(
        ds,
        None,
        ["BTCUSDT", "ETHUSDT"],
        ds.meta["start"],
        ds.meta["end"] + 1,
        names=["funding_z_90d"],
    )
    factor_days, z = daily_factor(f["funding_z_90d"], 0)
    output = {}
    for kind in ("usdm", "spot"):
        days, w = signals(ds, kind)
        indices = np.searchsorted(factor_days, days)
        if not np.array_equal(factor_days[indices], days):
            raise ValueError("策略与因子日轴不一致")
        adjusted = overheating(w, z[indices])
        m = crypto_market(
            ds,
            ["BTCUSDT", "ETHUSDT"],
            ds.meta["start"],
            ds.meta["end"],
            kind=kind,
            initial_cash=1000,
            leverage=2,
            assume="conservative" if kind == "usdm" else None,
            allow_missing=True,
        )
        base = m.run(daily_targets((days, w)))
        overlay = m.run(daily_targets((days, adjusted)))
        output[kind] = dict(
            base=summary(base, 1000, len(m.times)),
            overlay=summary(overlay, 1000, len(m.times)),
            changed_targets=int(np.count_nonzero(w != adjusted)),
            days=len(days),
            warmup_days=120,
        )
    return dict(
        classification="回溯探索；确认需要 2026-11-15 之后新数据",
        data_version=ds.version,
        start=ds.meta["start"],
        end=ds.meta["end"],
        thresholds=[-2, 2],
        multiplier=0.5,
        strategies=output,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="tmp/crypto")
    p.add_argument("--output", default="bench/results/crypto_factor_example.json")
    a = p.parse_args()
    result = run(crypto.open(a.data))
    Path(a.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
