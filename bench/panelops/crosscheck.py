# ruff: noqa: E402
"""External numerical checks; discrepancies are assertions or explicit semantics."""

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib")
)

import alphalens.performance as ap
import alphalens.utils as au
import numpy as np
import pandas as pd
import panelops as po
from additional_baselines import NEW_TS
from additional_baselines import pandas_extra as pandas_new
from baselines import pandas_ts

ROOT = Path(__file__).resolve().parents[2]
assert Path(po.__file__).resolve().is_relative_to(ROOT)


def pandas_extra(x, op, window, mp, y):
    """Use pandas rolling for unsupported reductions, only for correctness."""
    frame = pd.DataFrame(x)
    if op in ("delay", "delta"):
        shifted = frame.shift(window)
        return (shifted if op == "delay" else frame - shifted).to_numpy()
    if op in ("regression_slope", "regression_residual"):
        other = pd.DataFrame(y)
        valid = frame.notna() & other.notna()
        frame, other = frame.where(valid), other.where(valid)
        rx, ry = (
            frame.rolling(window, min_periods=mp),
            other.rolling(window, min_periods=mp),
        )
        slope = rx.cov(other, ddof=0) / ry.var(ddof=0)
        result = (
            slope
            if op == "regression_slope"
            else frame - rx.mean() - slope * (other - ry.mean())
        )
        return result.to_numpy()

    def reduce(values):
        good = np.isfinite(values)
        v = values[good]
        idx = np.arange(len(values))[good]
        if op in ("argmin", "argmax"):
            filled = np.where(good, values, np.inf if op == "argmin" else -np.inf)
            return float(getattr(np, op)(filled[::-1]))
        if op == "product":
            return 0.0 if np.any(v == 0) else np.prod(v)
        if op == "decay_linear":
            weights = idx + window - len(values) + 1
            return np.average(v, weights=weights)
        if len(v) < 2:
            return np.nan
        design = np.column_stack([np.ones(len(v)), idx])
        intercept, slope = np.linalg.lstsq(design, v, rcond=None)[0]
        return (
            slope
            if op == "slope"
            else values[-1] - intercept - slope * (len(values) - 1)
        )

    return frame.rolling(window, min_periods=mp).apply(reduce, raw=True).to_numpy()


def main():
    rng = np.random.default_rng(881)
    x, y = rng.normal(size=(2, 67, 9))
    x[rng.random(x.shape) < 0.1] = np.nan
    y[rng.random(y.shape) < 0.1] = np.nan
    checks = []
    for window in (1, 5, 20, 80):
        for mp in sorted({1, window}):
            for op in po.api.TS_OPS:
                paired = op in ("corr", "cov")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    if op in NEW_TS:
                        expected = pandas_new(x, op, window, mp, y)
                    elif op in (
                        "delay",
                        "delta",
                        "argmin",
                        "argmax",
                        "product",
                        "decay_linear",
                        "slope",
                        "residual",
                        "regression_slope",
                        "regression_residual",
                    ):
                        expected = pandas_extra(x, op, window, mp, y)
                    else:
                        expected = pandas_ts(x, op, window, mp, y=y if paired else None)
                if op in ("delay", "delta"):
                    actual = getattr(po, "ts_" + op)(x, periods=window)
                elif op == "ffill":
                    actual = po.ts_ffill(x)
                else:
                    actual = getattr(po, "ts_" + op)(
                        x,
                        *(
                            [y]
                            if op
                            in (
                                "corr",
                                "cov",
                                "regression_slope",
                                "regression_residual",
                                "sumif",
                            )
                            else []
                        ),
                        window=window,
                        min_periods=mp,
                    )
                np.testing.assert_allclose(
                    actual, expected, rtol=2e-10, atol=2e-10, equal_nan=True
                )
                differences = abs(actual - expected)
                checks.append(
                    dict(
                        op=op,
                        window=window,
                        min_periods=mp,
                        max_abs=float(
                            np.max(differences[np.isfinite(differences)], initial=0)
                        ),
                    )
                )
    # Divisible, untied observations make ordinal buckets exactly match qcut.
    scores, returns = rng.normal(size=(2, 20, 30))
    dates = pd.date_range("2025-01-01", periods=20)
    index = pd.MultiIndex.from_product(
        [dates, [f"s{i}" for i in range(30)]], names=["date", "asset"]
    )
    frame = pd.DataFrame({"factor": scores.ravel(), "1D": returns.ravel()}, index=index)
    frame["factor_quantile"] = au.quantize_factor(frame, quantiles=5)
    external_ic = ap.factor_information_coefficient(frame)["1D"].to_numpy()
    external_returns, _ = ap.mean_return_by_quantile(
        frame, by_date=True, demeaned=False
    )
    external_returns = (
        external_returns["1D"].unstack("factor_quantile").reindex(dates).to_numpy()
    )
    native_ic = po.rank_ic(scores, returns)
    native_returns = po.group_returns(scores, returns)
    np.testing.assert_allclose(native_ic, external_ic, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(native_returns, external_returns, rtol=2e-10, atol=2e-10)
    turnover = po.group_turnover(scores)
    for group in range(1, 6):
        expected = (
            ap.quantile_turnover(frame["factor_quantile"], group)
            .reindex(dates)
            .to_numpy()
        )
        np.testing.assert_allclose(
            turnover[:, group - 1], expected, rtol=2e-10, atol=2e-10, equal_nan=True
        )
    # Alphalens' IC is Spearman, so Pearson IC is checked through pandas directly.
    pearson = np.array(
        [pd.Series(a).corr(pd.Series(b)) for a, b in zip(scores, returns, strict=True)]
    )
    np.testing.assert_allclose(po.ic(scores, returns), pearson, rtol=2e-10, atol=2e-10)
    tied = frame.copy()
    tied["factor"] = 1.0
    tied_error = None
    try:
        au.quantize_factor(tied, quantiles=5)
    except ValueError as error:
        tied_error = str(error).splitlines()[0]
    assert tied_error is not None
    result = dict(
        pandas=checks,
        alphalens=dict(
            rows=20,
            securities=30,
            q=5,
            rank_ic_max_abs=float(np.max(abs(native_ic - external_ic))),
            group_returns_max_abs=float(np.max(abs(native_returns - external_returns))),
            turnover="all five groups passed",
            tied_qcut_error=tied_error,
        ),
        differences=[
            "Alphalens IC means Spearman; panelops exposes ic (Pearson) and rank_ic (Spearman).",
            "Alphalens defaults to demeaned group returns; comparison explicitly sets demeaned=False.",
            "qcut rejects repeated quantile boundaries; panelops ordinal groups break score ties by column index.",
            "For counts not divisible by q, qcut interpolated edges and floor(i*q/count) can allocate different boundary observations.",
            "Missing returns do not change panelops score buckets; only within-group return averaging excludes them.",
            "Pandas rolling skew/kurt are bias-corrected; panelops uses biased central moments (separately specified).",
        ],
    )
    path = ROOT / ".cache/panelops/crosscheck.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(
        f"{len(checks)} pandas cases; Alphalens rank IC, five return groups and five turnover groups passed."
    )


if __name__ == "__main__":
    main()
