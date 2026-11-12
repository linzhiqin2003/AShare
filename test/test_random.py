import json
from pathlib import Path

import numpy as np
import panelbt as bt
from panelbt.models import COUNTERS, METRICS
from panelbt.reference import run as reference_run

from .generators import case

assert Path(bt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


def test_2000_random_cases(tmp_path):
    counts = dict.fromkeys(COUNTERS, 0)
    for seed in range(2000):
        market, target, costs, config = case(seed)
        actual = bt.run(market, target, costs, config)
        expected = reference_run(market, target, costs, config)
        for key in (*METRICS, "positions"):
            np.testing.assert_allclose(
                getattr(actual, key),
                getattr(expected, key),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"seed={seed}, field={key}",
            )
        assert actual.diagnostics == expected.diagnostics, (
            seed,
            actual.diagnostics,
            expected.diagnostics,
        )
        assert len(actual.trades) == len(expected.trades), seed
        for a, b in zip(actual.trades, expected.trades, strict=True):
            assert (a.day, a.col) == (b.day, b.col), seed
            np.testing.assert_allclose(
                [a.qty, a.price, a.commission, a.tax, a.other_fee, a.slippage_cost],
                [b.qty, b.price, b.commission, b.tax, b.other_fee, b.slippage_cost],
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"seed={seed}",
            )
        assert len(actual.delist_events) == len(expected.delist_events), seed
        for a, b in zip(actual.delist_events, expected.delist_events, strict=True):
            assert (a.day, a.col) == (b.day, b.col)
            np.testing.assert_allclose(
                [a.qty, a.price, a.proceeds],
                [b.qty, b.price, b.proceeds],
                rtol=1e-9,
                atol=1e-9,
            )
        assert np.all(actual.cash >= 0), seed
        for name, value in actual.diagnostics.items():
            counts[name] += value
    assert all(value > 0 for value in counts.values()), counts
    output = tmp_path / "coverage.json"
    output.write_text(json.dumps(counts, indent=2) + "\n")
    print("\n随机 2000 例覆盖计数:", json.dumps(counts, ensure_ascii=False))
