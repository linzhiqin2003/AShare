"""独立的 cvxpy 参考建模，保留参数化图供连续调仓对拍。"""

import cvxpy as cp
import numpy as np

from .problem import box_limits, factor_indices


class CvxpySession:
    def __init__(self, solver, *, expanded_quadratic=False):
        self.solver = solver
        self.expanded_quadratic = expanded_quadratic
        self.key = None

    def solve(self, p, tolerance):
        c = p.constraints
        if c.tracking_error is not None and self.solver == "OSQP":
            raise ValueError("OSQP 不支持跟踪误差二阶锥约束")
        n, k = p.risk.B.shape
        flags = tuple(
            getattr(c, name) is not None
            for name in (
                "invested",
                "max_weight",
                "active_bound",
                "industry_bound",
                "style_bound",
                "turnover",
                "tracking_error",
            )
        )
        mappings = tuple(
            None if matrix is None else factor_indices(p, matrix)
            for matrix in (p.industries, p.styles)
        )
        key = (
            tuple(None if ix is None else tuple(ix) for ix in mappings),
            n,
            k,
            c.long_only,
            p.can_buy.tobytes(),
            p.can_sell.tobytes(),
            flags,
            None if p.industries is None else p.industries.shape,
            None if p.styles is None else p.styles.shape,
            p.risk.F.tobytes(),
            p.risk_aversion,
        )
        if key != self.key:
            self.key = key
            self.parameters = {}

            def param(name, shape, nonneg=False):
                x = cp.Parameter(shape, nonneg=nonneg)
                self.parameters[name] = x
                return x

            a, y, buy, sell = (
                cp.Variable(n),
                cp.Variable(k),
                cp.Variable(n),
                cp.Variable(n),
            )
            self.a = a
            B = param("B", (n, k))
            fr = param("fr", (k, k)) if self.expanded_quadratic else None
            dr = param("dr", n, True)
            current = param("current", n)
            benchmark = param("benchmark", n)
            alpha = param("alpha", n)
            ir = param("ir", n, True)
            io = param("io", n)
            bc, sc = param("bc", n, True), param("sc", n, True)
            constraints = [y == B.T @ a, a - buy + sell == current, buy >= 0, sell >= 0]
            fixed = ~p.can_buy & ~p.can_sell
            if fixed.any():
                constraints.append(a[fixed] == current[fixed])
            only_buy_blocked = ~p.can_buy & ~fixed
            only_sell_blocked = ~p.can_sell & ~fixed
            if only_buy_blocked.any():
                constraints.append(a[only_buy_blocked] <= current[only_buy_blocked])
            if only_sell_blocked.any():
                constraints.append(a[only_sell_blocked] >= current[only_sell_blocked])
            if c.invested is not None:
                constraints.append(cp.sum(a) == param("budget", ()))
            if c.long_only:
                constraints.append(a >= -benchmark)
            if c.max_weight is not None:
                constraints.append(a <= param("upper", n))
            if c.active_bound is not None:
                constraints.extend(
                    (a <= param("active_upper", n), a >= param("active_lower", n))
                )
            for name, matrix, limit in (
                ("industries", p.industries, c.industry_bound),
                ("styles", p.styles, c.style_bound),
            ):
                if limit is not None:
                    ix = mappings[0 if name == "industries" else 1]
                    exposure = param(name, matrix.shape).T @ a if ix is None else y[ix]
                    bound = param(name + "_limit", matrix.shape[1], True)
                    constraints.extend((exposure <= bound, -exposure <= bound))
            if c.turnover is not None:
                limit = param("turnover", (), True)
                constraints.extend((cp.sum(buy) <= limit, cp.sum(sell) <= limit))
            if c.tracking_error is not None:
                constraints.append(
                    cp.norm(
                        cp.hstack(
                            (
                                param("te_fr", (k, k)) @ y,
                                cp.multiply(param("te_dr", n, True), a),
                            )
                        )
                    )
                    <= param("te", (), True)
                )
            # 默认直接保留 QP Hessian；F 变动时重建图，避免平方项辅助等式的病态标度。
            # 展开形式仅作数值失败时的独立重表达，不改变问题和容差。
            if self.expanded_quadratic:
                risk = cp.sum_squares(fr @ y) + cp.sum_squares(cp.multiply(dr, a))
                impact = cp.sum_squares(cp.multiply(ir, a) + io)
            else:
                risk = cp.quad_form(
                    y, cp.psd_wrap(1000 * p.risk_aversion / n * p.risk.F)
                ) + cp.sum(cp.multiply(dr, cp.square(a)))
                impact = cp.sum(cp.multiply(ir, cp.square(a))) + io @ a
            loss = risk + impact + bc @ buy + sc @ sell - alpha @ a
            self.problem = cp.Problem(cp.Minimize(loss), constraints)
            if not self.problem.is_dpp():
                raise RuntimeError("参考建模应满足 DPP")
        vals, vec = np.linalg.eigh(p.risk.F)
        root = np.sqrt(np.maximum(vals, 0))[:, None] * vec.T
        current = n * (p.current - p.benchmark)
        values = {
            "B": p.risk.B,
            "dr": 1000 * p.risk_aversion * p.risk.D / n,
            "current": current,
            "benchmark": n * p.benchmark,
            "alpha": p.alpha * 1000,
            "ir": 1000 * p.impact / n,
            "io": -2 * (1000 * p.impact / n) * current,
            "bc": p.buy_cost * 1000,
            "sc": p.sell_cost * 1000,
        }
        if self.expanded_quadratic:
            values.update(
                fr=np.sqrt(1000 * p.risk_aversion / n) * root,
                dr=np.sqrt(values["dr"]),
                ir=np.sqrt(values["ir"]),
                io=-np.sqrt(values["ir"]) * current,
            )
        if c.invested is not None:
            values["budget"] = n * (c.invested - p.benchmark.sum())
        upper, active_lower, active_upper, _ = box_limits(p)
        if c.max_weight is not None:
            values["upper"] = n * (upper - p.benchmark)
        if c.active_bound is not None:
            values["active_lower"] = n * (active_lower - p.benchmark)
            values["active_upper"] = n * (active_upper - p.benchmark)
        for name, matrix, limit in (
            ("industries", p.industries, c.industry_bound),
            ("styles", p.styles, c.style_bound),
        ):
            if limit is not None:
                if name in self.parameters:
                    values[name] = matrix
                values[name + "_limit"] = n * np.broadcast_to(limit, (matrix.shape[1],))
        if c.turnover is not None:
            values["turnover"] = n * c.turnover
        if c.tracking_error is not None:
            values.update(te_fr=root, te_dr=np.sqrt(p.risk.D), te=n * c.tracking_error)
        for name, value in values.items():
            self.parameters[name].value = value
        # 参数化 Hessian 的 DPP 编译映射在大 N 时可能远大于原始稀疏问题。
        opts = {
            "verbose": False,
            "warm_start": True,
            "ignore_dpp": True,
        }
        if self.solver == "CLARABEL":
            opts.update(
                tol_gap_abs=tolerance,
                tol_gap_rel=tolerance,
                tol_feas=tolerance,
                max_iter=200,
                max_threads=1,
                static_regularization_constant=1e-10,
            )
        else:
            opts.update(
                eps_abs=tolerance, eps_rel=tolerance, max_iter=100000, polishing=True
            )
        statuses = []
        error = None
        if self.solver == "CLARABEL":
            for equilibrate, regularization in (
                (True, 1e-10),
                (False, 1e-10),
                (True, 1e-12),
                (False, 1e-12),
            ):
                opts.update(
                    equilibrate_enable=equilibrate,
                    static_regularization_constant=regularization,
                    warm_start=False,
                )
                try:
                    self.problem.solve(solver=self.solver, **opts)
                except cp.error.SolverError as exc:
                    error = str(exc)
                    statuses.append("solver_error")
                    continue
                error = None
                statuses.append(self.problem.status)
                if self.problem.status in ("optimal", "infeasible", "unbounded"):
                    break
        else:
            try:
                self.problem.solve(solver=self.solver, **opts)
            except cp.error.SolverError as exc:
                return None, "solver_error", {"error": str(exc)}
            statuses.append(self.problem.status)
        if error is not None:
            return (
                None,
                "solver_error",
                {"error": error, "numerical_statuses": statuses},
            )
        status = self.problem.status
        info = {
            "iterations": self.problem.solver_stats.num_iters,
            "numerical_statuses": statuses,
            "solve_time": self.problem.solver_stats.solve_time,
        }
        stats = self.problem.solver_stats.extra_stats
        if self.solver == "OSQP" and stats is not None:
            info.update(
                primal_residual=stats.info.prim_res, dual_residual=stats.info.dual_res
            )
        # cvxpy 的 Clarabel 适配器只公开 solve stats；从缓存的原生解读取残差。
        if self.solver == "CLARABEL":
            native = self.problem._solver_cache.get("CLARABEL")
            if native is not None:
                sol = native.get_solution()
                info.update(
                    primal_residual=sol.r_prim,
                    dual_residual=sol.r_dual,
                    gap=abs(sol.obj_val - sol.obj_val_dual) / (1000 * n),
                )
        weights = None if self.a.value is None else self.a.value / n + p.benchmark
        return weights, status, info
