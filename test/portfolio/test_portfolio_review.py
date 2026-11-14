"""审查回归测试：经济约束的手算预言机和内存边界。"""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import portopt
import pytest

import ashare
from ashare.portfolio import (
    Constraints,
    PortfolioProblem,
    Relaxation,
    SolverSession,
    VerificationError,
    optimize,
    optimize_sequence,
    prune_and_resolve,
)
from ashare.portfolio.canonical import canonicalize
from ashare.risk import RiskModel

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(portopt.__file__).resolve().is_relative_to(ROOT)


def problem(n=4):
    b = np.column_stack((np.ones(n), np.arange(n) % 2 == 0, np.arange(n) % 2 == 1))
    risk = RiskModel(
        tuple(map(str, range(n))),
        b,
        ("country", "even", "odd"),
        np.zeros((3, 3)),
        np.full(n, 0.1),
    )
    return PortfolioProblem(
        np.zeros(n),
        risk,
        np.full(n, 1 / n),
        np.full(n, 1 / n),
        risk_aversion=1,
        constraints=Constraints(max_weight=1),
    )


@pytest.mark.parametrize(
    "backend", ("clarabel", "rust", "osqp", "cvxpy_clarabel", "cvxpy_osqp")
)
def test_directional_box_relaxation(backend):
    p = problem(6)
    p.benchmark = np.array([0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
    p.current = np.array([0.6, 0, 0.1, 0.1, 0.1, 0.1])
    p.can_buy[:2] = False
    p.can_sell[0] = False
    p.constraints = Constraints(max_weight=0.3, active_bound=0.12)
    r = optimize(p, backend=backend)
    assert r.success and r.relaxations == []
    np.testing.assert_allclose(r.weights[:2], [0.6, 0], atol=1e-10)
    assert {
        (x["index"], x["constraint"], x["reason"]) for x in r.bound_relaxations
    } == {
        (0, "max_weight", "cannot_sell"),
        (0, "active_bound", "cannot_sell"),
        (1, "active_bound", "cannot_buy"),
    }
    assert all(x["symbol"] == p.risk.symbols[x["index"]] for x in r.bound_relaxations)
    # 可以卖出时，上限仍须执行。
    p.can_sell[0] = True
    r = optimize(p, backend=backend)
    assert r.success and r.weights[0] <= 0.3 + 1e-8
    assert all(x["index"] != 0 for x in r.bound_relaxations)


@pytest.mark.parametrize(
    "status", ("MaxIterations", "NumericalError", "AlmostSolved", "solver_error")
)
def test_numerical_failure_never_relaxes(status):
    p = problem()
    p.constraints = Constraints(max_weight=0.4, turnover=0.02)
    calls = []

    class Failed:
        backend = "fake"

        def solve(self, pp, tolerance):
            calls.append((pp.constraints.turnover, tolerance))
            return None, status, {}

    r = optimize(p, session=Failed(), relaxations=(Relaxation("turnover", None),))
    assert not r.success and r.status == "fallback_previous" and r.relaxations == []
    np.testing.assert_array_equal(r.weights, p.current)
    assert len(calls) >= 2 and all(limit == 0.02 for limit, _ in calls)
    assert r.diagnostics["reason"] == status


def test_numerical_retry_preserves_problem():
    p = problem()
    p.alpha[0] = 1
    p.constraints = Constraints(max_weight=1, turnover=0.02)

    class Flaky(SolverSession):
        def __init__(self):
            super().__init__("clarabel")
            self.calls = 0

        def solve(self, pp, tolerance):
            self.calls += 1
            if self.calls == 1:
                return None, "MaxIterations", {}
            return super().solve(pp, tolerance)

    r = optimize(p, session=Flaky(), relaxations=(Relaxation("turnover", None),))
    assert r.success and not r.relaxations
    assert np.maximum(r.weights - p.current, 0).sum() == pytest.approx(0.02, abs=1e-8)


@pytest.mark.parametrize(
    "backend", ("clarabel", "rust", "cvxpy_clarabel", "osqp", "cvxpy_osqp")
)
def test_asymmetric_cost_hand_solution(backend):
    p = problem(2)
    p.alpha[:] = [0.2, 0]
    p.current[:] = [0.3, 0.7]
    p.buy_cost[:] = [0.04, 0.30]
    p.sell_cost[:] = [0.02, 0.12]
    r = optimize(p, backend=backend)
    assert r.success
    # x>.3 时导数 .2 - .4(x-.5) - (.04+.12)=0，故 x=.6。
    np.testing.assert_allclose(r.weights, [0.6, 0.4], atol=1e-7)
    assert r.objective == pytest.approx(0.07, abs=1e-10)


@pytest.mark.parametrize("factor_form", (True, False))
@pytest.mark.parametrize("kind", ("industries", "styles"))
def test_lower_exposure_bound(factor_form, kind):
    p = problem()
    matrix = p.risk.B[:, 1:2].copy()
    if not factor_form:
        matrix *= 1.3
    setattr(p, kind, matrix)
    p.alpha = -matrix[:, 0]
    p.constraints = replace(
        p.constraints,
        **{"industry_bound" if kind == "industries" else "style_bound": 0.07},
    )
    r = optimize(p)
    assert r.success
    assert (matrix.T @ (r.weights - p.benchmark))[0] == pytest.approx(-0.07, abs=1e-8)


def test_relaxation_order_and_previous_holdings():
    p = problem()
    p.current = np.array([0.5, 0, 0.5, 0])
    p.industries = p.risk.B[:, 1:]
    p.constraints = Constraints(max_weight=1, industry_bound=0.01, turnover=0.01)
    r = optimize(
        p, relaxations=(Relaxation("turnover", 0.6), Relaxation("industry_bound", 1))
    )
    assert r.success and [x["constraint"] for x in r.relaxations] == ["turnover"]
    assert [x["status"] for x in r.attempts] == ["PrimalInfeasible", "optimal"]
    r = optimize(
        p, relaxations=(Relaxation("industry_bound", 1), Relaxation("turnover", 0.6))
    )
    assert r.success and [x["constraint"] for x in r.relaxations] == ["industry_bound"]
    r = optimize(replace(p, constraints=Constraints(max_weight=0.1)))
    assert not r.success
    np.testing.assert_array_equal(r.weights, p.current)
    assert not np.array_equal(r.weights, p.benchmark)


def test_independent_postsolve_violation_rejection():
    p = problem()
    p.industries = p.risk.B[:, 1:]
    p.constraints = Constraints(max_weight=1, industry_bound=0.02)

    class Bad:
        backend = "fake"

        def solve(self, *_):
            return np.array([0.4, 0.1, 0.25, 0.25]), "optimal", {"gap": 0.0}

    with pytest.raises(VerificationError, match="约束超限"):
        optimize(p, session=Bad())


def test_sequence_has_actual_recursion_and_prune_preserves_frozen():
    p = problem()
    p.current = np.array([0.1, 0.3, 0.3, 0.3])
    p.alpha[0] = 1
    p.constraints = Constraints(max_weight=1, turnover=0.03)
    results, _ = optimize_sequence([1, 2, 3], [p, p, p])
    np.testing.assert_allclose(
        [r.weights[0] for r in results], [0.13, 0.16, 0.19], atol=1e-8
    )
    q = problem()
    q.current = np.array([0.001, 0.333, 0.333, 0.333])
    q.alpha[0] = -1
    q.can_sell[0] = False
    initial = optimize(q)
    refined, report = prune_and_resolve(q, initial, 0.01)
    assert refined.success and report["removed"] == 0
    assert refined.weights[0] == pytest.approx(0.001, abs=1e-9)


def test_factor_form_and_vector_only_reuse():
    p = problem(100)
    p = replace(
        p,
        industry_factors=(1, 2),
        constraints=Constraints(max_weight=0.03, industry_bound=0.01),
    )
    small, dense = canonicalize(p), canonicalize(p, factor_constraints=False)
    assert small.A.nnz < dense.A.nnz
    assert small.P.nnz <= 100 + 9
    session = SolverSession("clarabel")
    r = optimize(p, session=session)
    first = session.canonical
    second = optimize(
        replace(p, alpha=np.linspace(-0.001, 0.001, 100), current=r.weights),
        session=session,
    )
    assert (
        second.success
        and second.diagnostics["vectors_only"]
        and session.canonical is first
    )
    # 矩阵变化必须重建，不能误用旧 y 映射或旧 Hessian。
    risk = RiskModel(p.risk.symbols, p.risk.B, p.risk.factors, p.risk.F, p.risk.D * 1.2)
    third = optimize(replace(p, risk=risk), session=session)
    assert (
        third.success
        and not third.diagnostics["vectors_only"]
        and session.canonical is not first
    )


def test_auto_fallback_peak_memory():
    script = r"""
import json, resource, sys
from pathlib import Path
import ashare, portopt
root = Path(sys.argv[1])
assert Path(ashare.__file__).resolve().is_relative_to(root)
assert Path(portopt.__file__).resolve().is_relative_to(root)
sys.path.insert(0, str(root/'bench/risk'))
from optimizer_benchmark import fixture
from ashare.portfolio import SolverSession, optimize
import cvxpy as cp
original = cp.Problem.solve
def guarded(self, **kwargs):
    assert kwargs.get('ignore_dpp') is True
    return original(self, **kwargs)
cp.Problem.solve = guarded
def peak():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == 'darwin' else rss * 1024
s = SolverSession('auto')
s.primary = SolverSession('clarabel')
s.primary.solve = lambda *args: (None, 'NumericalError', {})
p = fixture(1000, True)
before = peak()
r = optimize(p, session=s)
assert r.success and r.diagnostics['fallback_backend'] == 'cvxpy_clarabel'
print(json.dumps({'before': before, 'after': peak()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    # 只量 optimize 期间峰值内存的增量。原来断言进程总峰值 < 1 GB：macOS 上约 160 MB，
    # GitHub 的 Linux 机器上导入和夹具就接近 1 GB，测出 1.04 GB（2026-11-14 CI）。
    # 要守的回归是 DPP 编译把 N=1000 的回退撑到 8.11 GB（docs/RISK_NOTES.md P1-1）。
    peak = json.loads(result.stdout)
    grown = peak["after"] - peak["before"]
    assert grown < 1_000_000_000, (peak, grown)


def test_reused_vectors_follow_benchmark_bounds_costs_and_masks():
    p = problem(80)
    session = SolverSession("clarabel")
    first = optimize(p, session=session)
    assert first.success
    benchmark = np.linspace(1, 3, 80)
    benchmark /= benchmark.sum()
    changed = replace(
        p,
        alpha=p.alpha[::-1],
        benchmark=benchmark,
        current=first.weights,
        buy_cost=0.0007,
        sell_cost=0.0013,
        constraints=Constraints(max_weight=0.06, active_bound=0.04, turnover=0.2),
    )
    for problem_ in (
        changed,
        replace(
            changed, can_buy=np.arange(80) % 3 != 0, can_sell=np.arange(80) % 7 != 0
        ),
    ):
        reused = optimize(problem_, session=session)
        fresh = optimize(problem_, backend="clarabel")
        assert reused.success and fresh.success
        np.testing.assert_allclose(reused.weights, fresh.weights, atol=1e-8, rtol=1e-8)
        assert reused.objective == pytest.approx(fresh.objective, abs=1e-10)
