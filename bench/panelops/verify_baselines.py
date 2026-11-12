"""Check every benchmark adapter against a direct reference, outside timing."""

import json
import warnings
from pathlib import Path

import numpy as np
import panelops as po
from additional_baselines import ELEMENTS, PAIRED
from panelops import reference
from run import candidate, methods

ROOT = Path(__file__).resolve().parents[2]
assert Path(po.__file__).resolve().is_relative_to(ROOT)


def main():
    rng = np.random.default_rng(812)
    records = []
    unavailable = []
    for shape in ((17, 5), (67, 13)):
        x, y = rng.normal(size=(2, *shape))
        x[rng.random(shape) < 0.2] = np.nan
        y[rng.random(shape) < 0.2] = np.nan
        groups = rng.integers(-1, 4, size=shape, dtype=np.int64)
        operations = [
            *("ts_" + op for op in po.api.TS_OPS),
            *("cs_" + op for op in po.api.CS_OPS),
            "ic",
            "rank_ic",
            "factor_autocorr",
            "group_returns",
            "group_turnover",
            "targets",
            *ELEMENTS,
        ]
        for op in operations:
            for window in (5, 20, 60) if op.startswith("ts_") else (20,):
                mp = max(1, window // 2)
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    if op.startswith("ts_"):
                        name = op[3:]
                        kw = (
                            {"periods": window}
                            if name in ("delay", "delta")
                            else {}
                            if name == "ffill"
                            else {"window": window, "min_periods": mp}
                        )
                        expected = reference.ts(
                            x, name, y=y if name in PAIRED else None, **kw
                        )
                    elif op.startswith("cs_"):
                        expected = reference.cs(
                            x,
                            op[3:],
                            groups=groups
                            if op in ("cs_group_demean", "cs_neutralize")
                            else None,
                            exposures=[y] if op == "cs_neutralize" else [],
                        )
                    elif op in ("ic", "rank_ic"):
                        expected = reference.ic(x, y, rank=op == "rank_ic")
                    elif op == "factor_autocorr":
                        expected = reference.ic(x, reference.ts(x, "delay"), rank=True)
                    elif op == "group_returns":
                        expected = reference.group_returns(x, y)
                    elif op == "group_turnover":
                        expected = reference.group_turnover(x)
                    elif op == "targets":
                        expected = reference.targets(x)[1:]
                    else:
                        binary = {
                            "gt",
                            "ge",
                            "lt",
                            "le",
                            "eq",
                            "ne",
                            "and_",
                            "or_",
                            "maximum",
                            "minimum",
                            "power",
                            "signed_power",
                            "fill_nan",
                        }
                        args = (
                            (x, y, -y)
                            if op == "where"
                            else (x, y)
                            if op in binary
                            else (x,)
                        )
                        expected = reference.elementwise(op, *args)
                    for method in methods(op):
                        if method.startswith("rust") or method in (
                            "numpy_1",
                            "polars_default",
                            "polars_long_default",
                        ):
                            continue
                        config = {"op": op, "method": method, "window": window}
                        try:
                            actual = candidate(
                                config,
                                x,
                                y,
                                groups
                                if op in ("cs_group_demean", "cs_neutralize")
                                else None,
                            )()
                        except NotImplementedError:
                            unavailable.append({"op": op, "method": method})
                            continue
                        if op == "targets":
                            for a, b in zip(actual, expected, strict=True):
                                np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-10)
                        else:
                            np.testing.assert_allclose(
                                actual,
                                expected,
                                rtol=2e-10,
                                atol=2e-10,
                                equal_nan=True,
                                err_msg=f"{op} {method} W={window} shape={shape}",
                            )
                        records.append(
                            {
                                "op": op,
                                "method": method,
                                "window": window,
                                "shape": shape,
                            }
                        )
    output = ROOT / ".cache/panelops/baseline-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"passed": len(records), "cases": records, "unsupported": unavailable},
            indent=2,
        )
    )
    print(len(records), "baseline numerical comparisons passed")


if __name__ == "__main__":
    main()
