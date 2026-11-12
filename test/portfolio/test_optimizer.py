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
    objective,
    optimize,
    optimize_sequence,
    prune_and_resolve,
    to_targets,
    violations,
    woodbury_solution,
)
from ashare.risk import RiskModel

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(portopt.__file__).resolve().is_relative_to(ROOT)


def fixture(seed=1, n=20):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n, 4))
    f = np.diag([0.0001, 0.0002, 0.0003, 0.0004])
    m = RiskModel(
        tuple(map(str, range(n))), b, tuple("abcd"), f, rng.uniform(0.0001, 0.0006, n)
    )
    return PortfolioProblem(
        rng.normal(0, 0.001, n),
        m,
        np.full(n, 1 / n),
        np.full(n, 1 / n),
        constraints=Constraints(max_weight=0.3),
    )


@pytest.mark.parametrize(
    "backend", ["clarabel", "osqp", "cvxpy_clarabel", "cvxpy_osqp", "rust"]
)
def test_closed_form(backend):
    p = fixture()
    p.constraints = Constraints(long_only=False, max_weight=None)
    r = optimize(p, backend=backend)
    assert r.success
    np.testing.assert_allclose(r.weights, woodbury_solution(p), atol=2e-7, rtol=2e-7)


def test_risk_specific_and_active():
    p = fixture()
    p.alpha[:] = 0
    r = optimize(p)
    np.testing.assert_allclose(r.weights, p.benchmark, atol=1e-7)
    p.constraints = Constraints(long_only=False, max_weight=None)
    p.alpha = np.linspace(-0.001, 0.001, len(p.alpha))
    r = optimize(p)
    np.testing.assert_allclose(r.weights, woodbury_solution(p), atol=2e-7)


def test_turnover_and_direction():
    p = fixture(n=8)
    p.alpha = np.arange(8) * 0.1
    p.constraints = Constraints(max_weight=1, turnover=0.08)
    r = optimize(p)
    assert r.success
    assert np.maximum(r.weights - p.current, 0).sum() == pytest.approx(0.08, abs=2e-7)
    assert np.maximum(p.current - r.weights, 0).sum() == pytest.approx(0.08, abs=2e-7)
    p.can_buy[-1] = False
    p.can_sell[0] = False
    p.can_buy[3] = p.can_sell[3] = False
    r = optimize(p)
    assert r.weights[-1] <= p.current[-1] + 1e-7
    assert r.weights[0] >= p.current[0] - 1e-7
    assert r.weights[3] == pytest.approx(p.current[3], abs=1e-7)


def test_all_industries_and_tracking_error():
    p = fixture(n=30)
    p.industries = np.eye(3)[np.arange(30) % 3]
    p.styles = p.risk.B
    p.alpha = p.industries[:, -1] * 0.1
    p.constraints = Constraints(
        max_weight=0.2, industry_bound=0.01, style_bound=0.02, tracking_error=0.0005
    )
    for backend in ("clarabel", "cvxpy_clarabel", "rust"):
        r = optimize(p, backend=backend)
        assert r.success
        assert abs((r.weights - p.benchmark) @ p.industries[:, -1]) <= 0.0100001
        assert p.risk.volatility(r.weights, p.benchmark) <= 0.0005001


def test_infeasible_relaxation_and_fallback():
    p = fixture(n=8)
    p.constraints = Constraints(max_weight=0.1, turnover=0)
    r = optimize(p)
    assert not r.success and r.status == "fallback_previous"
    np.testing.assert_array_equal(r.weights, p.current)
    r = optimize(p, relaxations=(Relaxation("max_weight", 0.3),))
    assert r.success and len(r.relaxations) == 1
    assert r.relaxations[0]["before"] == 0.1
    assert r.relaxations[0]["after"] == 0.3


def test_properties_and_pruning():
    from dataclasses import replace

    p = fixture(n=15)
    p.buy_cost = p.sell_cost = np.full(15, 100.0)
    r = optimize(p)
    np.testing.assert_allclose(r.weights, p.current, atol=1e-8)
    p.buy_cost = p.sell_cost = np.zeros(15)
    r1 = optimize(p)
    r2 = optimize(replace(p, risk_aversion=100))
    assert (
        p.risk.variance(r2.weights, p.benchmark)
        <= p.risk.variance(r1.weights, p.benchmark) + 1e-12
    )
    tight = replace(p, constraints=Constraints(max_weight=0.1))
    r3 = optimize(tight)
    assert r3.objective <= r1.objective + 1e-10
    refined, report = prune_and_resolve(p, r1, 0.01)
    assert refined.success
    assert report["objective_difference"] <= 1e-8


def test_random_dual_solver():
    from dataclasses import replace

    counts = dict(infeasible=0, relaxed=0, frozen=0, turnover_active=0, tangent=0)
    for seed in range(240):
        p = fixture(seed + 100, n=8 + seed % 9)
        n = len(p.alpha)
        p.alpha *= 20
        p.buy_cost = np.full(n, 0.0001)
        p.sell_cost = np.full(n, 0.0002)
        p.industries = np.eye(3)[np.arange(n) % 3]
        p.styles = p.risk.B
        limit = 1 / n if seed % 12 == 0 else 0.5
        p.constraints = Constraints(
            max_weight=limit,
            active_bound=0.3,
            industry_bound=0.15,
            style_bound=0.3,
            turnover=0.05,
        )
        p.can_buy[seed % n] = False
        p.can_sell[(seed + 1) % n] = False
        p.can_buy[0] = p.can_sell[0] = False
        rules = ()
        if seed % 10 == 0:
            p.constraints = replace(p.constraints, max_weight=0.5 / n)
            if seed % 20 == 0:
                rules = (Relaxation("max_weight", 0.5),)
        a = optimize(p, backend="clarabel", relaxations=rules)
        b = optimize(p, backend="osqp", relaxations=rules)
        assert a.success == b.success, (seed, a.status, b.status)
        if not a.success:
            assert seed % 10 == 0 and seed % 20 != 0
            counts["infeasible"] += 1
            assert a.status == b.status == "fallback_previous"
            continue
        assert abs(a.objective - b.objective) < 2e-8
        np.testing.assert_allclose(a.weights, b.weights, atol=3e-5, rtol=1e-5)
        counts["relaxed"] += bool(a.relaxations)
        counts["frozen"] += abs(a.weights[0] - p.current[0]) < 1e-7
        counts["turnover_active"] += (
            abs(np.maximum(a.weights - p.current, 0).sum() - 0.05) < 1e-6
        )
        counts["tangent"] += limit == 1 / n and not rules
    assert all(v > 0 for v in counts.values()), counts
    print("random coverage", counts)


def test_sequence_targets_and_reuse():
    p = fixture(n=12)
    p.constraints = Constraints(max_weight=0.3, turnover=0.02)
    results, targets = optimize_sequence([0, 5, 10], [p, p, p], backend="osqp")
    assert all(r.success for r in results)
    assert results[-1].diagnostics["reused"]
    assert targets.days.tolist() == [0, 5, 10]
    assert len(targets.indptr) == 4 and targets.indptr[-1] == len(targets.cols)
    with pytest.raises(ValueError):
        to_targets([2, 1], np.full((2, 3), 1 / 3))
    assert objective(p, results[0].weights) == pytest.approx(results[0].objective)
    assert max(violations(p, results[0].weights).values()) < 1e-7
    assert isinstance(SolverSession("clarabel"), SolverSession)


def test_self_check_rejects_bad_or_inaccurate_solution():
    from ashare.portfolio import VerificationError

    p = fixture(n=8)

    class FakeSession:
        backend = "fake"

        def __init__(self, status, weights, diagnostics):
            self.status, self.weights, self.diagnostics = status, weights, diagnostics

        def solve(self, *_):
            return self.weights, self.status, self.diagnostics

    with pytest.raises(VerificationError):
        optimize(p, session=FakeSession("optimal", np.ones(8), {}))
    with pytest.raises(VerificationError):
        optimize(
            p, session=FakeSession("optimal", p.current, {"primal_residual": 1e-3})
        )
    r = optimize(p, session=FakeSession("optimal_inaccurate", p.current, {}))
    assert not r.success and r.status == "fallback_previous"


def test_changing_risk_and_mask_updates():
    from dataclasses import replace

    p = fixture(n=25)
    session = SolverSession("osqp")
    first = optimize(p, session=session)
    risk = RiskModel(
        p.risk.symbols, p.risk.B * 1.2, p.risk.factors, p.risk.F * 1.3, p.risk.D * 0.8
    )
    can_buy = p.can_buy.copy()
    can_buy[5] = False
    second = replace(
        p, current=first.weights, risk=risk, alpha=-p.alpha, can_buy=can_buy
    )
    a = optimize(second, session=session)
    b = optimize(second, backend="clarabel")
    assert a.success and b.success
    np.testing.assert_allclose(a.weights, b.weights, atol=3e-5, rtol=1e-5)
    assert abs(a.objective - b.objective) < 2e-8


def test_global_sparse_columns():
    targets = to_targets([2], np.array([[0.2, 0.3, 0.5]]), columns=[8, 2, 5])
    np.testing.assert_array_equal(targets.cols, [2, 5, 8])
    np.testing.assert_array_equal(targets.weights, [0.3, 0.5, 0.2])
    with pytest.raises(ValueError):
        to_targets([2], np.array([[0.2, 0.3, 0.5]]), columns=[8, 2, 2])


def test_all_frozen_and_cash_budget():
    from dataclasses import replace

    p = fixture(n=25)
    frozen = replace(p, can_buy=np.zeros(25, bool), can_sell=np.zeros(25, bool))
    r = optimize(frozen)
    assert r.success
    np.testing.assert_array_equal(r.weights, p.current)
    cash = replace(p, constraints=Constraints(invested=0, max_weight=1))
    r = optimize(cash)
    assert r.success
    np.testing.assert_array_equal(r.weights, np.zeros(25))


def test_reference_solver_exception_is_marked(monkeypatch):
    import cvxpy as cp

    def fail(*args, **kwargs):
        raise cp.error.SolverError("synthetic numerical failure")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    p = fixture(n=8)
    for backend in ("cvxpy_clarabel", "cvxpy_osqp"):
        r = optimize(p, backend=backend)
        assert not r.success and r.status == "fallback_previous"
        assert r.diagnostics["reason"] == "solver_error"
        np.testing.assert_array_equal(r.weights, p.current)


def test_cvxpy_quadratic_formulations_and_changing_covariance():
    from dataclasses import replace

    from ashare.portfolio.reference import CvxpySession

    p = fixture(n=25)
    p.impact = np.linspace(0.001, 0.02, 25)
    p.buy_cost = np.full(25, 0.0001)
    p.sell_cost = np.full(25, 0.0002)
    for expanded in (False, True):
        session = SolverSession("cvxpy_clarabel")
        session.reference = CvxpySession("CLARABEL", expanded_quadratic=expanded)
        for scale in (1.0, 1.3):
            model = replace(p.risk, F=scale * p.risk.F)
            problem = replace(p, risk=model, risk_aversion=10 * scale)
            a, b = (
                optimize(problem, session=session),
                optimize(problem, backend="clarabel"),
            )
            assert a.success and b.success
            np.testing.assert_allclose(a.weights, b.weights, atol=2e-7, rtol=2e-7)
            assert abs(a.objective - b.objective) < 1e-10


@pytest.mark.parametrize("backend", ["clarabel", "cvxpy_clarabel", "rust"])
def test_tracking_error_is_active(backend):
    p = fixture(n=30)
    p.alpha *= 10
    p.constraints = Constraints(max_weight=0.2, tracking_error=0.0002)
    r = optimize(p, backend=backend)
    assert r.success
    assert p.risk.volatility(r.weights, p.benchmark) == pytest.approx(0.0002, abs=2e-9)
