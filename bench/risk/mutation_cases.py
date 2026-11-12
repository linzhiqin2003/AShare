"""审查 46 处及第二轮新增逻辑的真实源码变异规格。"""

CASES = [
    (
        "T1 回归估计池用当日而非前一日",
        [
            (
                "src/ashare/risk/series.py",
                "eligible = ex.astype(bool) & np.isfinite(returns[future])",
                "eligible = prepared[future][2].astype(bool) & np.isfinite(returns[future])",
                1,
            )
        ],
    ),
    (
        "T2 DASTD 窗口含 t+1",
        [
            (
                "src/ashare/risk/fundamental.py",
                "vol_hist = returns[vol_start : day + 1]",
                "vol_hist = returns[vol_start : day + 2]",
                1,
            )
        ],
    ),
    (
        "T3 动量窗口后移一天",
        [
            (
                "src/ashare/risk/fundamental.py",
                "end = day + 1 - c.momentum_skip",
                "end = day + 2 - c.momentum_skip",
                1,
            )
        ],
    ),
    (
        "T4 统计模型窗口含 t+1",
        [
            (
                "src/ashare/risk/series.py",
                "returns[max(0, day - c.history + 1) : day + 1],",
                "returns[max(0, day - c.history + 1) : day + 2],",
                1,
            )
        ],
    ),
    (
        "T5 因子收益历史窗口多含一天",
        [
            (
                "src/ashare/risk/series.py",
                "fs = np.asarray((history_f + [f])[-c.history :])",
                "fs = np.asarray((history_f + [f])[-c.history - 1 :])",
                1,
            )
        ],
    ),
    (
        "T6 估计池偷看次日停牌",
        [
            (
                "src/ashare/risk/fundamental.py",
                '& ~source["is_suspended"][day].astype(bool)',
                '& ~source["is_suspended"][min(day + 1, len(source.dates) - 1)].astype(bool)',
                1,
            )
        ],
    ),
    (
        "T7 贝塔的市场收益用当日市值加权",
        [
            (
                "src/ashare/risk/fundamental.py",
                'prev_cap = np.asarray(source["total_mv"][start - 1 : day], dtype=float)',
                'prev_cap = np.asarray(source["total_mv"][start : day + 1], dtype=float)',
                1,
            )
        ],
    ),
    (
        "C1 缓存键漏 version",
        [
            (
                "src/ashare/risk/series.py",
                '"version": source.version,\n            "symbols": source.symbols,',
                '"symbols": source.symbols,',
                1,
            )
        ],
    ),
    (
        "C2 缓存键前缀摘要漏字段取值",
        [
            (
                "src/ashare/risk/series.py",
                "h.update(field.encode() + value.dtype.str.encode() + value.tobytes())",
                "h.update(field.encode() + value.dtype.str.encode())",
                1,
            )
        ],
    ),
    (
        "C3 缓存键漏 config",
        [("src/ashare/risk/series.py", '"config": asdict(c),', '"config": 0,', 1)],
    ),
    (
        "W1 Gram 路径丢掉行业约束",
        [
            (
                "src/ashare/risk/kernels.py",
                "length = np.linalg.norm(constraint)",
                "length = 0.0",
                1,
            )
        ],
    ),
    (
        "W2 标准化用等权均值",
        [
            (
                "src/ashare/risk/fundamental.py",
                "mean = np.average(x[est], weights=cap[est])",
                "mean = np.mean(x[est])",
                1,
            )
        ],
    ),
    (
        "W3 标准化的标准差用全体而非估计池",
        [
            (
                "src/ashare/risk/fundamental.py",
                "sd = np.std(x[est])",
                "sd = np.std(x)",
                1,
            )
        ],
    ),
    (
        "W4 行业约束漏掉 unknown 行业",
        [
            (
                "src/ashare/risk/series.py",
                "np.arange(1, len(c.industry_codes) + 2),",
                "np.arange(1, len(c.industry_codes) + 1),",
                1,
            )
        ],
    ),
    (
        "N1 NW 协方差少一阶滞后",
        [
            (
                "src/ashare/risk/kernels.py",
                "for lag in range(1, lags + 1):\n        cross = v[lag:].T @ v[:-lag]",
                "for lag in range(1, lags):\n        cross = v[lag:].T @ v[:-lag]",
                1,
            )
        ],
    ),
    (
        "N2 EWMA 不去均值",
        [
            (
                "src/ashare/risk/kernels.py",
                "mean = (np.nan_to_num(x) * w).sum(axis=0)",
                "mean = np.zeros(x.shape[1])",
                1,
            )
        ],
    ),
    (
        "N3 半衰期用 e 而非 2（仅 numpy）",
        [
            (
                "src/ashare/risk/kernels.py",
                "w = np.exp2(-np.arange(length - 1, -1, -1, dtype=float) / half_life)",
                "w = np.exp(-np.arange(length - 1, -1, -1, dtype=float) / half_life)",
                1,
            )
        ],
    ),
    (
        "N4 因子波动率用相关系数的半衰期",
        [
            (
                "src/ashare/risk/engine.py",
                "vol = np.sqrt(ew_variance(returns, vol_half_life, vol_lags))",
                "vol = np.sqrt(ew_variance(returns, corr_half_life, vol_lags))",
                1,
            )
        ],
    ),
    (
        "N5 波动状态乘子开方后再乘协方差",
        [
            (
                "src/ashare/risk/series.py",
                "covariance *= max(multiplier, 1e-6)",
                "covariance *= max(np.sqrt(multiplier), 1e-6)",
                1,
            )
        ],
    ),
    (
        "L1 LW 收缩强度减半",
        [
            (
                "src/ashare/risk/statistical.py",
                "shrinkage = float(np.clip((pi - rho) / delta, 0, 1)) if delta > 1e-30 else 1.0",
                "shrinkage = float(np.clip((pi - rho) / (2 * delta), 0, 1)) if delta > 1e-30 else 1.0",
                1,
            )
        ],
    ),
    (
        "L2 PCA 特质方差漏掉收缩目标的对角项",
        [
            (
                "src/ashare/risk/statistical.py",
                "np.sum(u * u, axis=1) + d - np.sum(b * b * values, axis=1), 1e-12",
                "np.sum(u * u, axis=1) - np.sum(b * b * values, axis=1), 1e-12",
                1,
            )
        ],
    ),
    (
        "S1 特质方差去掉下限",
        [
            (
                "src/ashare/risk/fundamental.py",
                "return np.maximum(shrunk**2, c.variance_floor), base",
                "return shrunk**2, base",
                1,
            )
        ],
    ),
    (
        "S2 短历史不混合结构化模型",
        [
            (
                "src/ashare/risk/fundamental.py",
                "gamma = np.minimum(count / c.structural_days, 1.0)",
                "gamma = np.ones(len(count))",
                1,
            )
        ],
    ),
    (
        "S3 贝叶斯收缩权重写反",
        [
            (
                "src/ashare/risk/fundamental.py",
                "shrunk[ix] = weight * mean + (1 - weight) * vol[ix]",
                "shrunk[ix] = (1 - weight) * mean + weight * vol[ix]",
                1,
            )
        ],
    ),
    (
        "R1 组合方差漏 F",
        [
            (
                "src/ashare/risk/model.py",
                "return float(y @ self.F @ y + np.dot(self.D, w * w))",
                "return float(np.dot(self.D, w * w))",
                1,
            )
        ],
    ),
    (
        "R2 matvec 漏 D",
        [
            (
                "src/ashare/risk/model.py",
                "return self.B @ (self.F @ (self.B.T @ w)) + self.D * w",
                "return self.B @ (self.F @ (self.B.T @ w))",
                1,
            )
        ],
    ),
    (
        "O1 预算约束写在主动权重上",
        [
            (
                "src/ashare/portfolio/canonical.py",
                "rhs.append(np.array([c.invested - p.benchmark.sum()]))",
                "rhs.append(np.array([c.invested]))",
                1,
            )
        ],
    ),
    (
        "O2 买卖费率对调（仅实现）",
        [
            (
                "src/ashare/portfolio/canonical.py",
                "                p.buy_cost,\n                p.sell_cost,",
                "                p.sell_cost,\n                p.buy_cost,",
                1,
            )
        ],
    ),
    (
        "O3 不能卖的冻结被丢掉",
        [
            (
                "src/ashare/portfolio/problem.py",
                "lower[~p.can_sell] = np.maximum(lower[~p.can_sell], p.current[~p.can_sell])",
                "lower = lower",
                1,
            )
        ],
    ),
    (
        "O4 行业/风格偏离只约束上侧",
        [
            (
                "src/ashare/portfolio/canonical.py",
                "add_weights(-matrix.T, limit)",
                "add_weights(matrix.T, limit)",
                1,
            ),
            (
                "src/ashare/portfolio/canonical.py",
                "inequalities.extend((selection, -selection))",
                "inequalities.extend((selection, selection))",
                1,
            ),
        ],
    ),
    (
        "O5 TE 锥漏掉特质部分",
        [
            (
                "src/ashare/portfolio/canonical.py",
                "sparse.hstack((sparse.diags(np.sqrt(p.risk.D)), Z(n, k + 2 * n))),",
                "sparse.hstack((sparse.diags(0 * np.sqrt(p.risk.D)), Z(n, k + 2 * n))),",
                1,
            )
        ],
    ),
    (
        "O6 放松顺序反转",
        [
            (
                "src/ashare/portfolio/optimize.py",
                "rule = relaxations[step]",
                "rule = relaxations[len(relaxations) - 1 - step]",
                1,
            )
        ],
    ),
    (
        "O7 求解后自检被关掉",
        [
            (
                "src/ashare/portfolio/optimize.py",
                "    if max(raw.values(), default=0) > tolerance:",
                "    if False:",
                1,
            ),
            (
                "src/ashare/portfolio/optimize.py",
                "            if max(check.values(), default=0) > tolerance:",
                "            if False:",
                1,
            ),
        ],
    ),
    (
        "O8 只做多写在主动权重上",
        [
            (
                "src/ashare/portfolio/problem.py",
                "np.maximum(0.0 if p.constraints.long_only else -np.inf, active_lower)",
                "np.maximum(p.benchmark if p.constraints.long_only else -np.inf, active_lower)",
                1,
            )
        ],
    ),
    (
        "O9 批量求解不递推持仓",
        [
            (
                "src/ashare/portfolio/construction.py",
                "        current = result.weights\n    return results",
                "        current = current\n    return results",
                1,
            )
        ],
    ),
    (
        "O10 裁剪重解删掉不能卖的持仓",
        [
            (
                "src/ashare/portfolio/optimize.py",
                "remove = (result.weights < threshold) & problem.can_sell",
                "remove = result.weights < threshold",
                1,
            )
        ],
    ),
    (
        "O11 无解回退不返回上一期权重",
        [
            (
                "src/ashare/portfolio/optimize.py",
                "    weights = problem.current.copy()\n    return OptimizationResult(",
                "    weights = problem.benchmark.copy()\n    return OptimizationResult(",
                1,
            )
        ],
    ),
    (
        "D3 双改：买卖费率对调（实现+cvxpy 参考）",
        [
            (
                "src/ashare/portfolio/canonical.py",
                "                p.buy_cost,\n                p.sell_cost,",
                "                p.sell_cost,\n                p.buy_cost,",
                1,
            ),
            (
                "src/ashare/portfolio/reference.py",
                "loss = risk + impact + bc @ buy + sc @ sell - alpha @ a",
                "loss = risk + impact + sc @ buy + bc @ sell - alpha @ a",
                1,
            ),
        ],
    ),
    (
        "D4 双改：冻结方向反转（实现+参考+自检+清理）",
        [
            ("src/ashare/portfolio/reference.py", "p.can_buy", "§TMP§", -1),
            ("src/ashare/portfolio/reference.py", "p.can_sell", "p.can_buy", -1),
            ("src/ashare/portfolio/reference.py", "§TMP§", "p.can_sell", -1),
            ("src/ashare/portfolio/problem.py", "p.can_buy", "§TMP§", -1),
            ("src/ashare/portfolio/problem.py", "p.can_sell", "p.can_buy", -1),
            ("src/ashare/portfolio/problem.py", "§TMP§", "p.can_sell", -1),
        ],
    ),
    (
        "D6 双改：换手按双边（实现+参考+自检）",
        [
            (
                "src/ashare/portfolio/canonical.py",
                "row[0, n + k : n + k + n] = 1\n        row[1, n + k + n :] = 1",
                "row[:, n + k :] = 1",
                1,
            ),
            (
                "src/ashare/portfolio/reference.py",
                "constraints.extend((cp.sum(buy) <= limit, cp.sum(sell) <= limit))",
                "constraints.append(cp.sum(buy) + cp.sum(sell) <= limit)",
                1,
            ),
            (
                "src/ashare/portfolio/problem.py",
                "max(np.maximum(trade, 0).sum(), np.maximum(-trade, 0).sum())",
                "np.abs(trade).sum()",
                1,
            ),
        ],
    ),
    (
        "RU1 Rust NW 方差滞后项漏系数 2",
        [
            (
                "crates/portopt-core/src/lib.rs",
                "variance += 2. * (1. - lag as f64 / (lags + 1) as f64) * cross;",
                "variance += (1. - lag as f64 / (lags + 1) as f64) * cross;",
                1,
            )
        ],
    ),
    (
        "RU2 Rust 偏差统计用总体标准差(ddof=0)",
        [
            (
                "crates/portopt-core/src/lib.rs",
                "((square - sum * sum / window as f64) / (window - 1) as f64)",
                "((square - sum * sum / window as f64) / window as f64)",
                1,
            )
        ],
    ),
    (
        "D1 双改：NW 滞后项取负（numpy 两处 + Rust 两处）",
        [
            (
                "src/ashare/risk/kernels.py",
                "result += (1 - lag / (lags + 1)) * (cross + cross.T)",
                "result -= (1 - lag / (lags + 1)) * (cross + cross.T)",
                1,
            ),
            (
                "src/ashare/risk/kernels.py",
                "result += 2 * (1 - lag / (lags + 1)) * np.sum(v[lag:] * v[:-lag], axis=0)",
                "result -= 2 * (1 - lag / (lags + 1)) * np.sum(v[lag:] * v[:-lag], axis=0)",
                1,
            ),
            (
                "crates/portopt-core/src/lib.rs",
                "variance += 2. * (1. - lag as f64 / (lags + 1) as f64) * cross;",
                "variance -= 2. * (1. - lag as f64 / (lags + 1) as f64) * cross;",
                1,
            ),
            (
                "crates/portopt-core/src/lib.rs",
                "result += (1. - lag as f64 / (lags + 1) as f64) * cross;",
                "result -= (1. - lag as f64 / (lags + 1) as f64) * cross;",
                1,
            ),
        ],
    ),
    (
        "D2 双改：WLS 权重用市值（numpy 两处 + Rust）",
        [
            (
                "src/ashare/risk/kernels.py",
                "weights = np.sqrt(c)\n",
                "weights = c * 1.0\n",
                2,
            ),
            (
                "crates/portopt-core/src/lib.rs",
                "weight_sum += cap[i].sqrt();",
                "weight_sum += cap[i];",
                1,
            ),
            (
                "crates/portopt-core/src/lib.rs",
                "let root = (cap[i].sqrt() / mean).sqrt();",
                "let root = (cap[i] / mean).sqrt();",
                1,
            ),
        ],
    ),
    (
        "D5 双改：行业约束用根号市值加权（numpy 两处 + Rust）",
        [
            (
                "src/ashare/risk/kernels.py",
                "constraint[np.asarray(industry_indices)] = c @ x[:, industry_indices]",
                "constraint[np.asarray(industry_indices)] = np.sqrt(c) @ x[:, industry_indices]",
                2,
            ),
            (
                "crates/portopt-core/src/lib.rs",
                "constraint[j] += cap[i] * b[i * k + j];",
                "constraint[j] += cap[i].sqrt() * b[i * k + j];",
                1,
            ),
        ],
    ),
    (
        "D7 双改：半衰期用 e 而非 2（numpy + Rust 两处）",
        [
            (
                "src/ashare/risk/kernels.py",
                "w = np.exp2(-np.arange(length - 1, -1, -1, dtype=float) / half_life)",
                "w = np.exp(-np.arange(length - 1, -1, -1, dtype=float) / half_life)",
                1,
            ),
            (
                "crates/portopt-core/src/lib.rs",
                ".map(|i| (-((t - 1 - i) as f64) / half_life).exp2())",
                ".map(|i| (-((t - 1 - i) as f64) / half_life).exp())",
                2,
            ),
        ],
    ),
    (
        "X01 日期轴不参与缓存键",
        [
            (
                "src/ashare/risk/series.py",
                "h.update(str(source.dates[day]).encode())",
                "pass",
                1,
            )
        ],
    ),
    (
        "X02 回退重新启用 DPP",
        [
            (
                "src/ashare/portfolio/reference.py",
                '"ignore_dpp": True',
                '"ignore_dpp": False',
                1,
            )
        ],
    ),
    (
        "X03 数值失败错误进入放松",
        [
            (
                "src/ashare/portfolio/optimize.py",
                'if status not in ("PrimalInfeasible", "infeasible") or step == len(relaxations):',
                "if step == len(relaxations):",
                1,
            )
        ],
    ),
    (
        "X04 个股冻结上限不自动放宽",
        [
            (
                "src/ashare/portfolio/problem.py",
                '("max_weight", "upper", upper, ~p.can_sell),',
                '("max_weight", "upper", upper, np.zeros_like(p.can_sell)), ',
                1,
            )
        ],
    ),
    (
        "X05 禁买主动下界不自动放宽",
        [
            (
                "src/ashare/portfolio/problem.py",
                '("active_bound", "lower", active_lower, ~p.can_buy),',
                '("active_bound", "lower", active_lower, np.zeros_like(p.can_buy)), ',
                1,
            )
        ],
    ),
    (
        "X06 缺失逐只放宽记录",
        [
            (
                "src/ashare/portfolio/problem.py",
                "return upper, active_lower, active_upper, records",
                "return upper, active_lower, active_upper, []",
                1,
            )
        ],
    ),
    (
        "X07 市值分组混入非估计池",
        [
            (
                "src/ashare/risk/fundamental.py",
                "eligible = np.flatnonzero(estimation)",
                "eligible = np.arange(len(cap))",
                1,
            )
        ],
    ),
    (
        "X08 零残差不使用结构化先验",
        [
            (
                "src/ashare/risk/fundamental.py",
                "gamma[base <= c.variance_floor] = 0.0",
                "pass",
                1,
            )
        ],
    ),
    (
        "X09 单股行业不屏蔽残差",
        [
            (
                "src/ashare/risk/series.py",
                "] = np.nan\n                            regressed",
                "] = 0.0\n                            regressed",
                1,
            )
        ],
    ),
    (
        "X10 偏差统计混入非估计池",
        [
            (
                "src/ashare/risk/series.py",
                'np.where(eligible, prev_state["cap"], 0.0)',
                'prev_state["cap"]',
                1,
            )
        ],
    ),
    (
        "X11 短历史统计风险不收缩",
        [
            (
                "src/ashare/risk/statistical.py",
                "reliability = np.minimum(count / max(threshold, 1), 1.0)",
                "reliability = np.ones(len(count))",
                1,
            )
        ],
    ),
    (
        "X12 递推遗漏移出窗口项",
        [("src/ashare/risk/rolling.py", "sums -= decay**window * old[0]", "pass", 1)],
    ),
    (
        "X13 递推市场收益偷用当日市值",
        [
            (
                "src/ashare/risk/rolling.py",
                'cap = self.source["total_mv"][t - 1]',
                'cap = self.source["total_mv"][t]',
                1,
            )
        ],
    ),
    (
        "X14 不恢复递推检查点",
        [("src/ashare/risk/series.py", "rolling.restore(checkpoint)", "pass", 1)],
    ),
    (
        "X15 PSD 投影不精确对称化",
        [
            (
                "src/ashare/risk/kernels.py",
                "return (result + result.T) * 0.5",
                "return result",
                2,
            )
        ],
    ),
    (
        "X16 Rust 病态设计不走确定回退",
        [("crates/portopt-core/src/lib.rs", "if well_conditioned {", "if true {", 1)],
    ),
    (
        "X17 忽略线程数设置",
        [
            (
                "packages/portopt/python/portopt/__init__.py",
                "_core.set_num_threads(_threads(threads))",
                "_core.set_num_threads(1)",
                1,
            )
        ],
    ),
    (
        "X18 主动风险漏掉特质项",
        [
            (
                "src/ashare/portfolio/canonical.py",
                "2 * p.risk_aversion * p.risk.D + 2 * p.impact",
                "0 * p.risk_aversion * p.risk.D + 2 * p.impact",
                1,
            )
        ],
    ),
    (
        "X19 无解回退错误标成功",
        [
            (
                "src/ashare/portfolio/optimize.py",
                '        False,\n        "fallback_previous",',
                '        True,\n        "fallback_previous",',
                1,
            )
        ],
    ),
]

CASES.extend(
    [
        (
            "X20 主动风险写成绝对风险",
            [
                (
                    "src/ashare/portfolio/canonical.py",
                    "rhs = [np.zeros(k), p.current - p.benchmark]",
                    "q[:n] += 2 * p.risk_aversion * p.risk.D * p.benchmark\n    rhs = [p.risk.B.T @ p.benchmark, p.current - p.benchmark]",
                    1,
                )
            ],
        ),
        (
            "X21 正式回归绕过 Rust",
            [
                (
                    "src/ashare/risk/engine.py",
                    "return portopt.regress(\n        returns, exposures, market_cap, estimation, industry_indices, threads\n    )",
                    "return kernels.cross_section_batch(\n        returns, exposures, market_cap, estimation, industry_indices\n    )",
                    1,
                )
            ],
        ),
        (
            "X22 外部 F 不对称化",
            [
                (
                    "src/ashare/risk/model.py",
                    'object.__setattr__(self, "F", immutable((self.F + self.F.T) * 0.5, 2, "F"))',
                    'object.__setattr__(self, "F", immutable(self.F, 2, "F"))',
                    1,
                )
            ],
        ),
        (
            "X23 复用时不更新约束向量",
            [
                (
                    "src/ashare/portfolio/canonical.py",
                    "self.b = (\n            np.concatenate([fn(p, lower, upper) for fn in self.vector_blocks]) * self.n\n        )",
                    "self.b = self.b",
                    1,
                )
            ],
        ),
        (
            "X24 回归到 N 维行业风格行",
            [
                (
                    "src/ashare/portfolio/canonical.py",
                    "ix = factor_indices(p, matrix) if factor_constraints else None",
                    "ix = None",
                    1,
                )
            ],
        ),
        (
            "X25 漏掉最后一个行业约束",
            [
                (
                    "src/ashare/portfolio/canonical.py",
                    'limit = vector(bound, matrix.shape[1], "exposure_bound")',
                    'limit = vector(bound, matrix.shape[1], "exposure_bound")\n            if name == "industry_bound":\n                matrix, limit = matrix[:, :-1], limit[:-1]',
                    1,
                )
            ],
        ),
    ]
)

CASES.extend(
    [
        (
            "X26 CMRA 回看端点错一天",
            [
                (
                    "src/ashare/risk/rolling.py",
                    "(day - np.arange(1, 13) * 21) % 253",
                    "(day - np.arange(1, 13) * 21 + 1) % 253",
                    1,
                )
            ],
        ),
        (
            "X27 CMRA 检查点遗漏历史前缀",
            [("src/ashare/risk/rolling.py", '                "cmra_prefix",\n', "", 1)],
        ),
    ]
)

CASES.extend(
    [
        (
            "X28 清理缓存未删除旧文件",
            [
                (
                    "src/ashare/risk/cache.py",
                    "                path.unlink()",
                    "                pass",
                    1,
                )
            ],
        ),
        (
            "X29 历史快照重新保存大矩阵 B",
            [
                (
                    "src/ashare/risk/cache.py",
                    "                symbols=np.asarray(model.symbols),",
                    "                B=model.B,\n                symbols=np.asarray(model.symbols),",
                    1,
                )
            ],
        ),
    ]
)
