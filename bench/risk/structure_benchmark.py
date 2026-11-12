"""N 行/y 行、向量复用、实用 OSQP 容差和回退 RSS 的配对实测。"""

import argparse
import json
import resource
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from baseline import metadata
from optimizer_benchmark import fixture

import ashare.portfolio.solvers as solvers
from ashare.portfolio import SolverSession, optimize, violations
from ashare.portfolio.canonical import canonicalize


def memory_child(n):
    session = SolverSession("auto")
    session.primary = SolverSession("clarabel")
    session.primary.solve = lambda *args: (None, "NumericalError", {})
    begin = perf_counter()
    result = optimize(fixture(n, True), session=session)
    assert result.success and result.diagnostics["fallback_backend"] == "cvxpy_clarabel"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        N=n,
        rss_bytes=rss if sys.platform == "darwin" else rss * 1024,
        seconds=perf_counter() - begin,
        status=result.status,
    )


def run(output):
    result = dict(
        metadata=metadata(),
        factor_constraints={},
        vector_reuse={},
        practical_osqp={},
        fallback_memory=[],
    )
    for n in (1000, 3000, 5000):
        for te in (False, True):
            p = fixture(n, te)
            row = {}
            weights = []
            for form in (False, True):
                samples = []
                for _ in range(3):
                    with patch.object(
                        solvers,
                        "canonicalize",
                        lambda p, form=form: canonicalize(p, factor_constraints=form),
                    ):
                        begin = perf_counter()
                        value = optimize(p, backend="clarabel")
                        samples.append(perf_counter() - begin)
                    assert value.success
                weights.append(value.weights)
                data = canonicalize(p, factor_constraints=form)
                row["factor_y" if form else "stock_rows"] = dict(
                    samples=samples,
                    median=float(np.median(samples)),
                    P_nnz=data.P.nnz,
                    A_nnz=data.A.nnz,
                )
            row["max_weight_difference"] = float(
                np.max(np.abs(weights[0] - weights[1]))
            )
            np.testing.assert_allclose(weights[0], weights[1], atol=3e-6, rtol=1e-5)
            result["factor_constraints"][f"{n}/{'socp' if te else 'qp'}"] = row
        p = fixture(n)
        alphas = np.random.default_rng(994).normal(0, 0.0001, (15, n))
        for reuse in (False, True):
            session = SolverSession("clarabel")
            previous = p.current
            reused = 0
            begin = perf_counter()
            for alpha in alphas:
                if not reuse:
                    session = SolverSession("clarabel")
                r = optimize(replace(p, alpha=alpha, current=previous), session=session)
                assert r.success
                reused += r.diagnostics["vectors_only"]
                previous = r.weights
            result["vector_reuse"][f"{n}/{reuse}"] = dict(
                seconds=perf_counter() - begin, periods=15, reused=reused
            )
        for backend, tolerance in (
            ("clarabel", 1e-9),
            ("osqp", 1e-9),
            ("osqp", 1e-6),
            ("osqp", 1e-5),
        ):
            session = SolverSession(backend)
            previous = p.current
            errors = []
            statuses = []
            begin = perf_counter()
            for alpha in alphas:
                problem = replace(p, alpha=alpha, current=previous)
                w, status, info = session.solve(problem, tolerance)
                statuses.append(status)
                if w is not None:
                    errors.append(max(violations(problem, w).values()))
                    previous = w
            result["practical_osqp"][f"{n}/{backend}/{tolerance}"] = dict(
                seconds=perf_counter() - begin,
                periods=15,
                statuses=statuses,
                max_violation=max(errors, default=0),
                strict_selfcheck_failures=sum(e > 1e-7 for e in errors),
            )
        child = subprocess.run(
            [sys.executable, __file__, "--memory-child", str(n)],
            check=True,
            capture_output=True,
            text=True,
        )
        result["fallback_memory"].append(json.loads(child.stdout))
        Path(output).write_text(json.dumps(result, indent=2))
        print(n, "done", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/risk-bench/structure.json")
    )
    parser.add_argument("--memory-child", type=int)
    args = parser.parse_args()
    if args.memory_child:
        print(json.dumps(memory_child(args.memory_child)))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run(args.output)
