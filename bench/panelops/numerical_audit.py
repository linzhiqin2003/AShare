"""Reproducible adversarial and optional read-only real-panel checks; store no observations."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import panelops as po
from panelops import reference

ROOT = Path(__file__).resolve().parents[2]
assert Path(po.__file__).resolve().is_relative_to(ROOT)
sys.path.insert(0, str(ROOT))
from test.panelops.test_adversarial import PAIRED, corpus, invoke  # noqa: E402


def errors(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-10, equal_nan=True)
    finite = np.isfinite(a) & np.isfinite(b)
    delta = abs(a[finite] - b[finite])
    expected = abs(b[finite])
    nonzero = expected > 0
    return dict(
        max_absolute=float(delta.max(initial=0)),
        max_relative=float((delta[nonzero] / expected[nonzero]).max(initial=0)),
        max_tolerance_units=float((delta / (2e-10 + 2e-10 * expected)).max(initial=0)),
        finite=int(finite.sum()),
        nan=int(np.isnan(b).sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/panelops/numerical.json"
    )
    args = parser.parse_args()
    x, y, mask = corpus()
    report = {"build_id": po.build_id, "adversarial": [], "real": []}
    with np.errstate(all="ignore"):
        for op in po.api.TS_OPS:
            tests = []
            for w in (2, 5, 20, 60, 250):
                a, b = invoke(op, x, y, w, mask)
                tests.append({"window": w, **errors(a, b)})
            report["adversarial"].append({"op": op, "cases": tests})
            print(op, max(c["max_tolerance_units"] for c in tests), flush=True)
        if args.real:
            generation = json.loads((args.real / "current.json").read_text())["version"]
            base = args.real / generation
            mapped = {
                f: np.load(base / (f + ".npy"), mmap_mode="r", allow_pickle=False)
                for f in ("adj_factor", "close", "amount")
            }
            shape = mapped["close"].shape
            cols = np.sort(
                np.random.default_rng(0).choice(shape[1], 600, replace=False)
            )
            panels = {
                f: np.ascontiguousarray(a[:, cols], dtype=np.float64)
                for f, a in mapped.items()
            }
            rows = np.unique(
                np.r_[
                    0,
                    1,
                    4,
                    19,
                    127,
                    128,
                    249,
                    250,
                    511,
                    512,
                    np.linspace(600, shape[0] - 1, 12, dtype=int),
                ]
            )
            rows = rows[rows < shape[0]]
            report["real_metadata"] = {
                "generation": generation,
                "shape": shape,
                "sample_columns": 600,
                "column_seed": 0,
                "audit_rows": rows.tolist(),
                "windows": [20, 250],
                "read_only_mmap": True,
            }
            for field, x in panels.items():
                y = panels["amount"] if field != "amount" else panels["close"]
                for op in po.api.TS_OPS:
                    for w in (20, 250):
                        kw = (
                            {"periods": w}
                            if op in ("delay", "delta")
                            else {"limit": w}
                            if op == "ffill"
                            else {"window": w, "min_periods": max(1, w // 2)}
                        )
                        a = getattr(po, "ts_" + op)(
                            x, *([y] if op in PAIRED else []), **kw
                        )
                        b = reference.ts(
                            x, op, y=y if op in PAIRED else None, rows=rows, **kw
                        )
                        report["real"].append(
                            {
                                "field": field,
                                "op": op,
                                "window": w,
                                **errors(a[rows], b[rows]),
                            }
                        )
                        if w == 20 and op in (
                            "std",
                            "var",
                            "skew",
                            "kurt",
                            "zscore",
                            "rsquare",
                        ):
                            windows = np.lib.stride_tricks.sliding_window_view(
                                x, w, axis=0
                            )
                            const = np.isfinite(windows).all(-1) & (
                                np.max(windows, axis=-1) == np.min(windows, axis=-1)
                            )
                            values = a[w - 1 :][const]
                            assert (
                                np.all(values == 0)
                                if op in ("std", "var")
                                else np.isnan(values).all()
                            )
                            report["real"][-1]["exact_constant_windows"] = int(
                                const.sum()
                            )
                    print("real", field, op, "passed", flush=True)
            # Reproduce the review's specific real-data failure: constant adj_factor predictor.
            dependent, predictor = panels["close"], panels["adj_factor"]
            view = np.lib.stride_tricks.sliding_window_view(predictor, 20, axis=0)
            constant = np.isfinite(view).all(-1) & (
                np.max(view, axis=-1) == np.min(view, axis=-1)
            )
            for op in ("corr", "cov", "regression_slope", "regression_residual"):
                for w in (20, 250):
                    a = getattr(po, "ts_" + op)(
                        dependent, predictor, w, min_periods=w // 2
                    )
                    b = reference.ts(
                        dependent,
                        op,
                        y=predictor,
                        window=w,
                        min_periods=w // 2,
                        rows=rows,
                    )
                    row = {
                        "field": "close",
                        "predictor": "adj_factor",
                        "op": op,
                        "window": w,
                        **errors(a[rows], b[rows]),
                    }
                    if w == 20 and op != "cov":
                        assert np.isnan(a[19:][constant]).all()
                        row["exact_constant_windows"] = int(constant.sum())
                    report["real"].append(row)
                print("real close~adj_factor", op, "passed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
