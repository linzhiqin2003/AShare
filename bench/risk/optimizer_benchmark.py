"""端到端调用基准；顺序调仓与独立问题并行吞吐明确分开。"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import clarabel
import cvxpy
import numpy as np
import osqp
import portopt
import scipy
from baseline import metadata
from threadpoolctl import threadpool_limits

from ashare.portfolio import (
    Constraints,
    PortfolioProblem,
    SolverSession,
    VerificationError,
    objective,
    optimize,
    violations,
)
from ashare.portfolio.canonical import canonicalize
from ashare.portfolio.solvers import csc_tuple
from ashare.risk import RiskModel

METHODS = ("cvxpy_clarabel", "cvxpy_osqp", "clarabel", "osqp", "rust")


def fixture(n, te=False):
    rng = np.random.default_rng(442 + n)
    industry = np.eye(31)[np.arange(n) % 31]
    styles = rng.normal(size=(n, 13))
    b = np.column_stack((np.ones(n), industry, styles))
    f = np.diag(np.r_[0.0001, np.full(31, 0.000025), np.full(13, 0.00001)])
    model = RiskModel(
        tuple(map(str, range(n))),
        b,
        tuple(f"f{i}" for i in range(45)),
        f,
        rng.uniform(0.0001, 0.0004, n),
    )
    cap = np.full(n, 1 / n)
    can_buy, can_sell = np.ones((2, n), bool)
    can_buy[::97] = False
    can_sell[::101] = False
    return PortfolioProblem(
        rng.normal(0, 0.0001, n),
        model,
        cap,
        cap,
        risk_aversion=10,
        constraints=Constraints(
            max_weight=4 / n,
            active_bound=3 / n,
            industry_bound=0.02,
            style_bound=0.03,
            turnover=0.1,
            tracking_error=0.0006 if te else None,
        ),
        buy_cost=0.00001,
        sell_cost=0.00002,
        can_buy=can_buy,
        can_sell=can_sell,
        industries=industry,
        styles=styles,
    )


def bench_one(n, method, periods, te):
    p = fixture(n, te)
    session = SolverSession(method)
    start = perf_counter()
    try:
        cold = optimize(p, backend=method, session=session)
        cold_status = cold.status
    except VerificationError:
        cold, cold_status = None, "verification_rejected"
    cold_seconds = perf_counter() - start
    samples = []
    single_statuses = []
    for _ in range(3):
        start = perf_counter()
        try:
            r = optimize(p, backend=method, session=session)
            single_statuses.append(r.status)
        except VerificationError:
            single_statuses.append("verification_rejected")
        samples.append(perf_counter() - start)
    session = SolverSession(method)
    rng = np.random.default_rng(994)
    alphas = rng.normal(0, 0.0001, (periods, n))
    previous = p.current
    max_violation = 0.0
    failures = []
    numerical_recoveries = []
    start = perf_counter()
    for day in range(periods):
        problem = replace(p, current=previous, alpha=alphas[day])
        try:
            r = optimize(problem, backend=method, session=session)
        except VerificationError as exc:
            failures.append(
                {
                    "day": day,
                    "status": "verification_rejected",
                    "diagnostics": {"error": str(exc)},
                }
            )
            continue
        if not r.success:
            failures.append(
                {"day": day, "status": r.status, "diagnostics": r.diagnostics}
            )
        if r.success and "fallback_backend" in r.diagnostics:
            numerical_recoveries.append({"day": day, **r.diagnostics})
        previous = r.weights
        max_violation = max(max_violation, max(r.violations.values()))
        if (day + 1) % 25 == 0:
            print(
                f"progress N={n} {method} te={te}: {day + 1}/{periods}, {perf_counter() - start:.2f}s, rejected={len(failures)}",
                flush=True,
            )
    total = perf_counter() - start
    return {
        "cold_seconds": cold_seconds,
        "cold_status": cold_status,
        "single_statuses": single_statuses,
        "single_samples": samples,
        "single_seconds": float(np.median(samples)),
        "periods": periods,
        "sequential_seconds": total,
        "max_violation": max_violation,
        "status": "optimal"
        if not failures
        and cold_status == "optimal"
        and all(s == "optimal" for s in single_statuses)
        else "contains_explicit_fallbacks",
        "failures": failures,
        "numerical_recoveries": numerical_recoveries,
        "objective_single": cold.objective if cold is not None else None,
    }


def bench_parallel(n, periods, te):
    p = fixture(n, te)
    rng = np.random.default_rng(994)
    alphas = rng.normal(0, 0.0001, (periods, n))
    start = perf_counter()
    data = canonicalize(p)
    qs = np.broadcast_to(data.q, (periods, len(data.q))).copy()
    qs[:, :n] = -1000 * alphas
    bs = np.broadcast_to(data.b, (periods, len(data.b)))
    result = portopt.solve_many(
        csc_tuple(data.P), qs, csc_tuple(data.A), bs, data.cones, threads=4
    )
    max_violation = 0.0
    max_residual = max_gap = max_objective_error = 0.0
    objectives = []
    for day, row in enumerate(result):
        x, status, obj, dual, primal, dual_residual, *_ = row
        if status != "Solved":
            raise RuntimeError(status)
        assert np.isfinite([obj, dual, primal, dual_residual]).all()
        weights = np.asarray(x[:n]) / n + p.benchmark
        value = objective(p, weights) + (alphas[day] - p.alpha) @ weights
        objectives.append(float(value))
        max_residual = max(max_residual, primal, dual_residual)
        max_gap = max(max_gap, abs(obj - dual) / data.scale)
        max_objective_error = max(
            max_objective_error,
            abs(obj / data.scale + value - alphas[day] @ p.benchmark),
        )
        max_violation = max(
            max_violation,
            max(violations(p, weights).values()),
        )
    assert max_violation <= 1e-7
    assert max_residual <= 1e-7 and max_gap <= 1e-7 and max_objective_error <= 1e-8
    return {
        "independent_seconds": perf_counter() - start,
        "periods": periods,
        "threads": 4,
        "max_violation": max_violation,
        "max_residual": max_residual,
        "max_gap": max_gap,
        "max_objective_error": max_objective_error,
        "objectives": objectives,
        "dependent_rebalances": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 3000, 5000])
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/risk-bench") / "optimizers.json"
    )
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "metadata": metadata(),
        "versions": {
            "cvxpy": cvxpy.__version__,
            "clarabel": clarabel.__version__,
            "osqp": osqp.__version__,
            "scipy": scipy.__version__,
        },
        "fixture": "N varies, K=45; alpha changes each period; prior solved holdings feed next; B/F/D fixed to exercise decomposition reuse.",
        "rows": {},
    }
    if args.output.exists():
        old = json.loads(args.output.read_text())
        result["rows"] = old.get("rows", {})
    with threadpool_limits(limits=1):
        for n in args.sizes:
            for te in (False, True):
                scenario = "socp" if te else "qp"
                for method in args.methods:
                    key = f"{n}/{scenario}/{method}"
                    if args.skip_existing and key in result["rows"]:
                        continue
                    if te and "osqp" in method:
                        row = {
                            "status": "unsupported",
                            "reason": "OSQP has linear constraints only; no own nonlinear solver added.",
                        }
                    else:
                        row = bench_one(n, method, args.periods, te)
                    result["rows"][key] = row
                    args.output.write_text(json.dumps(result, indent=2))
                    print(key, row, flush=True)
                if not args.no_parallel:
                    key = f"{n}/{scenario}/rust_parallel_independent"
                    if args.skip_existing and key in result["rows"]:
                        continue
                    result["rows"][key] = bench_parallel(n, args.periods, te)
                    args.output.write_text(json.dumps(result, indent=2))
                    print(key, result["rows"][key], flush=True)


if __name__ == "__main__":
    main()
