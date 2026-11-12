"""直接调用求解器；OSQP 会复用同结构问题的分解和热启动。"""

import clarabel
import numpy as np
import osqp

from .canonical import canonicalize, structure_key
from .reference import CvxpySession


def csc_tuple(matrix):
    return (
        matrix.shape[0],
        matrix.shape[1],
        matrix.indptr.astype(np.uint64),
        matrix.indices.astype(np.uint64),
        matrix.data,
    )


class SolverSession:
    def __init__(self, backend="auto"):
        self.backend = backend
        self.solver = None
        self.previous = None
        self.primary = None
        self.canonical = None
        self.canonical_key = None
        self.clarabel_solvers = {}
        self.fallback = None
        self.reference = (
            CvxpySession(backend.removeprefix("cvxpy_").upper())
            if backend.startswith("cvxpy_")
            else None
        )

    def solve(self, p, tolerance=1e-9):
        if self.backend == "auto":
            primary_backend = "clarabel"
            if self.primary is None or self.primary.backend != primary_backend:
                self.primary = SolverSession(primary_backend)
            weights, status, info = self.primary.solve(p, tolerance)
            info["primary_backend"] = primary_backend
            if status != "optimal" and status not in (
                "PrimalInfeasible",
                "DualInfeasible",
                "infeasible",
                "unbounded",
            ):
                if self.fallback is None:
                    self.fallback = CvxpySession("CLARABEL", expanded_quadratic=True)
                primary_status = status
                primary_statuses = info.get("numerical_statuses", [status])
                weights, status, fallback_info = self.fallback.solve(p, tolerance)
                info = {
                    **fallback_info,
                    "primary_backend": primary_backend,
                    "primary_status": primary_status,
                    "primary_numerical_statuses": primary_statuses,
                    "fallback_backend": "cvxpy_clarabel",
                }
            return weights, status, info
        if self.reference:
            return self.reference.solve(p, tolerance)
        key = structure_key(p)
        vectors_only = self.canonical is not None and key == self.canonical_key
        if vectors_only:
            self.canonical.update_vectors(p)
        else:
            self.canonical = canonicalize(p)
            self.canonical_risk = p.risk
            self.canonical_key = key
            self.clarabel_solvers.clear()
        data = self.canonical
        if self.backend == "rust":
            import portopt

            x, status, obj, dual, rp, rd, iterations, statuses = portopt.solve(
                csc_tuple(data.P),
                data.q,
                csc_tuple(data.A),
                data.b,
                data.cones,
                tolerance,
            )
            return (
                np.asarray(x[: data.n]) / data.n + p.benchmark
                if status == "Solved"
                else None,
                "optimal" if status == "Solved" else status,
                {
                    "primal_residual": rp,
                    "dual_residual": rd,
                    "gap": abs(obj - dual) / data.scale,
                    "iterations": iterations,
                    "numerical_statuses": statuses,
                },
            )
        if self.backend == "clarabel":
            cones = [
                (
                    (
                        clarabel.ZeroConeT,
                        clarabel.NonnegativeConeT,
                        clarabel.SecondOrderConeT,
                    )[kind]
                )(size)
                for kind, size in data.cones
            ]
            statuses = []
            for equilibrate, regularization in (
                (True, 1e-10),
                (False, 1e-10),
                (True, 1e-12),
                (False, 1e-12),
            ):
                settings = clarabel.DefaultSettings()
                settings.verbose = False
                settings.max_iter = 200
                settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = (
                    tolerance
                )
                settings.max_threads = 1
                settings.static_regularization_constant = regularization
                settings.equilibrate_enable = equilibrate
                config = (equilibrate, regularization, tolerance)
                solver = self.clarabel_solvers.get(config)
                if solver is None:
                    solver = clarabel.DefaultSolver(
                        data.P, data.q, data.A, data.b, cones, settings
                    )
                    self.clarabel_solvers[config] = solver
                else:
                    solver.update(q=data.q, b=data.b)
                sol = solver.solve()
                status = str(sol.status)
                statuses.append(status)
                if status in ("Solved", "PrimalInfeasible", "DualInfeasible"):
                    break
            return (
                np.array(sol.x[: data.n]) / data.n + p.benchmark
                if status == "Solved"
                else None,
                "optimal" if status == "Solved" else status,
                {
                    "primal_residual": sol.r_prim,
                    "dual_residual": sol.r_dual,
                    "gap": abs(sol.obj_val - sol.obj_val_dual) / data.scale,
                    "iterations": sol.iterations,
                    "vectors_only": vectors_only,
                    "numerical_statuses": statuses,
                },
            )
        if self.backend != "osqp":
            raise ValueError(f"未知求解器 {self.backend}")
        if p.constraints.tracking_error is not None:
            raise ValueError("OSQP 不支持跟踪误差二阶锥约束")
        eq = data.cones[0][1]
        lower = np.full(len(data.b), -np.inf)
        lower[:eq] = data.b[:eq]
        reusable = self.previous is not None
        if reusable:
            for new, old in ((data.P, self.previous.P), (data.A, self.previous.A)):
                reusable &= (
                    new.shape == old.shape
                    and np.array_equal(new.indptr, old.indptr)
                    and np.array_equal(new.indices, old.indices)
                )
        if reusable:
            self.solver.update_settings(eps_abs=tolerance, eps_rel=tolerance)
            changes = {"q": data.q, "l": lower, "u": data.b}
            if not np.array_equal(data.P.data, self.previous.P.data):
                changes["Px"] = data.P.data
            if not np.array_equal(data.A.data, self.previous.A.data):
                changes["Ax"] = data.A.data
            self.solver.update(**changes)
        else:
            self.solver = osqp.OSQP()
            self.solver.setup(
                P=data.P,
                q=data.q,
                A=data.A,
                l=lower,
                u=data.b,
                verbose=False,
                eps_abs=tolerance,
                eps_rel=tolerance,
                max_iter=100000,
                polishing=True,
            )
        self.previous = data
        sol = self.solver.solve(raise_error=False)
        status = sol.info.status
        return (
            sol.x[: data.n] / data.n + p.benchmark if status == "solved" else None,
            "optimal"
            if status == "solved"
            else ("infeasible" if status == "primal infeasible" else status),
            {
                "primal_residual": sol.info.prim_res,
                "dual_residual": sol.info.dual_res,
                "iterations": sol.info.iter,
                "reused": bool(reusable),
            },
        )
