from pathlib import Path

import portopt

import ashare
from ashare.risk.validate import synthetic

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(portopt.__file__).resolve().is_relative_to(ROOT)


def test_synthetic_recovery():
    result = synthetic()
    for name, report in result["models"].items():
        recovery = report["recovery"]
        assert recovery["common_covariance_relative_frobenius"] < 0.16
        assert recovery["specific_relative_mae"] < 0.08
        assert recovery["portfolio_variance_relative_mae"] < 0.12
        assert recovery["total_variance_relative_mae"] < 0.08
        if name == "fundamental":
            assert recovery["factor_covariance_relative_frobenius"] < 0.16
            assert recovery["factor_returns_rmse"] < 0.002
        for row in report["bias"].values():
            assert 0.85 < row["mean"] < 1.15
            assert row["within_95pct"] > 0.65
