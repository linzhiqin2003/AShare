"""C6 日频标签、正向 Hodrick 1B 与 Berger–Boos 模拟检验。

数组约定：x[i] 在入场 τ_i 可知，g[i] 是 (τ_i, τ_{i+1}] 的单日推断收益。
同日相关指 x[i] 的创新与 g[i-1]；g[i] 与 x[i] 在零假设下独立。
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

from ashare.data.crypto.signals import BAD_PRICE
from ashare.data.crypto.timeutil import DAY, HOUR, millis
from ashare.factor.crypto import daily_factor, rolling_sum
from ashare.factor.stats import adjust

BOUNDARY = millis("2024-01-01")
GAMMA = 0.001
PHI_GRID = np.r_[
    np.arange(0, 901, 50) / 1000,
    np.arange(910, 991, 10) / 1000,
    np.arange(991, 1000) / 1000,
]
CONFIRM_FACTORS = ("funding_mean_7d", "funding_z_90d", "premium_7d", "basis_q")
CONFIRM_FAMILY = tuple((s, f) for s in ("BTCUSDT", "ETHUSDT") for f in CONFIRM_FACTORS)


def forward_sum(g, horizon):
    """从当前入场起未来 h 个完整日；任一日缺失即缺失。"""
    g = np.asarray(g, dtype=float)
    if horizon < 1 or int(horizon) != horizon:
        raise ValueError("horizon 必须为正整数")
    out = np.full_like(g, np.nan)
    if len(g) >= horizon:
        total = rolling_sum(g, horizon)
        count = rolling_sum(np.isfinite(g).astype(float), horizon)
        out[: len(g) - horizon + 1] = np.where(
            count[horizon - 1 :] == horizon, total[horizon - 1 :], np.nan
        )
    return out


def crypto_labels(ds, symbols, values, *, trade_hour=0, horizons=(1, 7, 30)):
    """先在连续日历轴计算收益再映射请求日；缺信号不删收益，缺覆盖不顺延。"""
    horizons = tuple(horizons)
    if not horizons or any(h < 1 or int(h) != h for h in horizons):
        raise ValueError("horizons 必须是非空的正整数序列")
    days, _ = daily_factor(values, trade_hour)
    entries = days + DAY
    instruments = [prefix + s for s in symbols for prefix in ("spot:", "usdm:")]
    a = ds.arrays(
        ["open", "tradable", "quality", "funding_rate", "funding_price"],
        instruments=instruments,
    )
    t = np.asarray(a["time"])
    if t.ndim != 1 or np.any(t % HOUR) or np.any(np.diff(t) != HOUR):
        raise ValueError("标签需要矩形完整小时轴，缺口须保留占位")
    if any(
        a[k].shape != (len(t), len(instruments))
        for k in ("open", "tradable", "quality", "funding_rate", "funding_price")
    ):
        raise ValueError("标签面板字段不是同轴矩形")
    valid = (
        a["tradable"].astype(bool)
        & ((a["quality"].astype(int) & BAD_PRICE) == 0)
        & (a["open"] > 0)
    )
    cash = np.where(
        np.isfinite(a["funding_rate"]), a["funding_rate"] * a["funding_price"], 0
    )
    # funding 位于结算前一小时；累加 [入场行, 退出行) 正好是 (入场,退出]。
    cumulative = np.vstack([np.zeros(len(instruments)), np.cumsum(cash, axis=0)])

    def locate(moments):
        indices = np.searchsorted(t, moments)
        inside = indices < len(t)
        exact = np.zeros(len(moments), bool)
        exact[inside] = t[indices[inside]] == moments[inside]
        return indices, exact

    def returns_at(moments, horizon):
        loc, in_covered = locate(moments)
        outloc, out_covered = locate(moments + horizon * DAY)
        ok = in_covered & out_covered
        arr = np.full((len(moments), len(instruments)), np.nan)
        ii = np.flatnonzero(ok)
        if len(ii):
            left, right = loc[ii], outloc[ii]
            price_in, price_out = a["open"][left], a["open"][right]
            paid = cumulative[right] - cumulative[left]
            arr[ii] = np.divide(
                price_out - price_in - paid,
                price_in,
                out=np.full_like(price_in, np.nan),
                where=valid[left] & valid[right],
            )
        return {"spot": arr[:, 0::2], "usdm": arr[:, 1::2]}

    desc = {h: returns_at(entries, h) for h in horizons}
    # 从面板完整读取，日历收益延伸至末个请求入场 + 最大持有期。
    # 收益轴与信号是否存在无关；最后一个单日收益的退出时刻也须精确匹配。
    calendar_entries = (
        np.arange(entries[0], entries[-1] + max(horizons) * DAY, DAY)
        if len(entries)
        else np.empty(0, dtype=np.int64)
    )
    daily_returns = returns_at(calendar_entries, 1)
    calendar_g = {
        kind: np.log1p(np.where(daily_returns[kind] > -1, daily_returns[kind], np.nan))
        for kind in ("spot", "usdm")
    }
    requested = np.searchsorted(calendar_entries, entries)
    g = {kind: v[requested] for kind, v in calendar_g.items()}
    inf = {
        h: {kind: forward_sum(v, h)[requested] for kind, v in calendar_g.items()}
        for h in horizons
    }
    return dict(
        day=days,
        entry=entries,
        descriptive={h: desc[h] for h in horizons},
        daily=g,
        inference=inf,
    )


def sample_mask(entries, horizon, split, *, boundary=BOUNDARY):
    """C6 默认边界不变；其他研究可以显式传入自己的 UTC 边界。"""
    entries = np.asarray(entries)
    boundary = millis(boundary)
    if split == "in":
        return (entries < boundary) & (entries + horizon * DAY < boundary)
    if split == "out":
        return entries >= boundary
    raise ValueError("split 只能为 in/out")


def hodrick_1b(x, g, horizon, *, sample=None):
    """正向 OLS；S 是求和。q 过去窗口缺任一 z 即不进入 S。"""
    x, g = np.asarray(x, float), np.asarray(g, float)
    if x.ndim != 1 or g.shape != x.shape:
        raise ValueError("x/g 必须为同轴一维日数组")
    mask = np.ones(len(x), bool) if sample is None else np.asarray(sample, bool)
    g = np.where(mask, g, np.nan)
    y = forward_sum(g, horizon)
    use = mask & np.isfinite(x) & np.isfinite(y)
    n = int(use.sum())
    result = dict(n=n, beta=np.nan, se=np.nan, t=np.nan, alpha=np.nan, meat_terms=0)
    if n < 3:
        return result
    # 先减一个有效锚点再求均值，避免近常数 x 的均值消减；只在有效 OLS 域定变换。
    anchor = x[use][0]
    centered = x - anchor
    shift = np.mean(centered[use])
    centered -= shift
    amplitude = np.max(np.abs(centered[use]))
    if amplitude == 0:
        return result
    scale = amplitude * np.sqrt(np.mean((centered[use] / amplitude) ** 2))
    standardized = centered / scale
    z = np.column_stack([np.ones(len(x)), standardized])
    orthogonal, upper = np.linalg.qr(z[use], mode="reduced")
    coef = np.linalg.solve(upper, orthogonal.T @ y[use])
    inverse_upper = np.linalg.solve(upper, np.eye(2))
    bread = inverse_upper @ inverse_upper.T
    # Z 的行域与 OLS 完全相同：因子和 h 日标签均有效；不压缩时间轴。
    available = use
    qx = rolling_sum(np.where(available, standardized, np.nan), horizon)
    count = rolling_sum(available.astype(float), horizon)
    valid_g = mask & np.isfinite(g)
    e = g - np.mean(g[valid_g])
    meat_mask = valid_g & (count == horizon)
    q = np.column_stack([np.full(len(x), horizon), qx])
    weighted = q[meat_mask] * e[meat_mask, None]
    # 同一标准化坐标下的 sandwich；直接平方和开根，避免非正方差被 max(0) 隐藏。
    influence = weighted @ bread
    standard_error = np.linalg.norm(influence[:, 1])
    beta = float(coef[1] / scale)
    se = float(standard_error / scale)
    return dict(
        n=n,
        alpha=float(coef[0] - beta * (anchor + shift)),
        beta=beta,
        se=se,
        t=float(coef[1] / standard_error) if standard_error > 0 else np.nan,
        meat_terms=int(meat_mask.sum()),
    )


def calibrate(x, g, entries, *, boundary=BOUNDARY):
    """只用样本内单日标签；AR 创新 η_d 与同日 g_d（数组 g[d−1]）相关。"""
    x, g = np.asarray(x, float), np.asarray(g, float)
    ins = sample_mask(entries, 1, "in", boundary=boundary)
    good = (
        ins[1:]
        & ins[:-1]
        & np.isfinite(x[1:])
        & np.isfinite(x[:-1])
        & np.isfinite(g[:-1])
    )
    n = int(good.sum())
    if n < 30:
        raise ValueError("样本内校准不足 30 对")
    previous, current, returns = x[:-1][good], x[1:][good], g[:-1][good]
    if np.std(previous) == 0 or np.std(returns) == 0:
        raise ValueError("样本内因子/收益退化")
    design = np.column_stack([np.ones(n), previous])
    c, phi = np.linalg.lstsq(design, current, rcond=None)[0]
    eta = current - c - phi * previous
    if np.std(eta) <= 1e-14:
        raise ValueError("样本内因子创新退化")
    delta = float(np.corrcoef(returns, eta)[0, 1])
    half = NormalDist().inv_cdf(1 - GAMMA / 2) / np.sqrt(n - 3)
    center = np.arctanh(np.clip(delta, -0.999999999999, 0.999999999999))
    interval = np.tanh([center - half, center + half])
    eps = g[ins & np.isfinite(g)]
    eps = eps - eps.mean()
    return dict(
        n=n,
        phi=float(phi),
        delta=delta,
        delta_interval=interval,
        delta_grid=np.linspace(*interval, 11),
        sd_eta=float(np.std(eta, ddof=1)),
        sd_x=float(np.std(x[ins & np.isfinite(x)], ddof=1)),
        eps=eps,
    )


def empirical_p(null_abs_t, observed):
    values = np.asarray(null_abs_t)
    if not np.isfinite(values).all():
        raise ValueError("模拟统计量退化，不能悄悄删除重复")
    p = (1 + np.count_nonzero(values >= abs(observed), axis=-1)) / (
        values.shape[-1] + 1
    )
    return np.where(np.isfinite(observed), p, np.nan)


def holm_confirm(pvalues):
    """固定八名额，包括无法估计的 NaN。"""
    p = np.array([pvalues.get(key, np.nan) for key in CONFIRM_FAMILY])
    return dict(zip(CONFIRM_FAMILY, adjust(p, "holm"), strict=True))


def quantile_description(x, y, entries, horizon):
    x, y, entries = np.asarray(x), np.asarray(y), np.asarray(entries)
    group = np.zeros(len(x), int)
    for i in range(len(x)):
        history = x[(entries >= entries[i] - 365 * DAY) & (entries < entries[i])]
        history = history[np.isfinite(history)]
        if len(history) >= 180 and np.isfinite(x[i]):
            group[i] = (
                np.searchsorted(
                    np.quantile(history, [0.2, 0.4, 0.6, 0.8]), x[i], side="left"
                )
                + 1
            )
    rows = []
    for q in range(1, 6):
        use = (group == q) & np.isfinite(y)
        dates = entries[use]
        rows.append(
            dict(
                group=q,
                n=int(use.sum()),
                mean=float(np.mean(y[use])) if use.any() else np.nan,
                overlapping_pairs=int(np.count_nonzero(np.diff(dates) < horizon * DAY)),
            )
        )
    return rows


def descriptive(x, y):
    use = np.isfinite(x) & np.isfinite(y)
    return dict(
        n=int(use.sum()),
        spearman=float(spearmanr(x[use], y[use]).statistic)
        if use.sum() > 2 and np.std(x[use]) > 0
        else np.nan,
    )


@dataclass
class NullTable:
    """每格预排序 |t*|；功效只查表，不进行嵌套模拟。"""

    points: list
    distributions: np.ndarray
    gamma: float = GAMMA

    def pvalue(self, statistic):
        query = np.abs(np.asarray(statistic))
        worst = np.zeros_like(query, dtype=float)
        b = self.distributions.shape[1]
        for row in self.distributions:
            count = b - np.searchsorted(row, query, side="left")
            worst = np.maximum(worst, (1 + count) / (b + 1))
        return np.where(np.isfinite(query), np.minimum(1, worst + self.gamma), np.nan)

    def critical(self, alpha=0.05 / 8):
        """查出完整保守检验的严格临界值；相等仍按 >= 计数。"""
        b = self.distributions.shape[1]
        allowed = int(np.floor((alpha - self.gamma) * (b + 1) - 1 + 1e-12))
        if allowed < 0:
            return np.inf
        return float(np.max(self.distributions[:, b - allowed - 1]))


class SimulationBank:
    """公共随机数，x 的两条 AR 基分量允许 δ 网格以二次型精确合成。

    平稳初值的非高斯部分用足够长的后向创新和，截断尾方差 ≤ 1e−20；
    正态分量直接抽精确平稳分布。分批生成初值，避免巨型时间×重复数组。
    """

    def __init__(
        self, eps, length, *, repetitions=4000, seed=6182026, phi_grid=PHI_GRID
    ):
        eps = np.asarray(eps, float)
        if len(eps) < 3 or np.std(eps) == 0 or not np.isfinite(eps).all():
            raise ValueError("模拟需要非退化的样本内收益创新")
        self.eps = eps - eps.mean()
        self.sd = float(np.std(self.eps))
        self.length, self.repetitions, self.seed = length, repetitions, seed
        self.phi_grid = np.asarray(phi_grid)
        rng = np.random.default_rng(seed)
        # 第 0 列创新与 x[0] 同日；日收益 g[0] 对应第 1 列创新。
        self.returns = rng.choice(self.eps, (repetitions, length + 1))
        self.normal = rng.standard_normal((repetitions, length + 1))
        self.initial_normal = rng.standard_normal(repetitions)
        self.initial = {}
        self._stationary(self.phi_grid)
        self._last = None
        self.parts_cache = {}
        self.power_cache = {}

    def _stationary(self, phis):
        phis = [float(p) for p in phis if float(p) not in self.initial]
        if not phis:
            return
        burn = int(np.ceil(np.log(1e-10) / np.log(0.999)))
        weights = np.asarray(phis)[:, None] ** np.arange(burn)[None, :]
        out = np.empty((len(phis), self.repetitions))
        # 固定独立流：增添网格点不会改变原有公共随机数。
        rng = np.random.default_rng(self.seed + 1)
        for lo in range(0, self.repetitions, 128):
            hi = min(lo + 128, self.repetitions)
            history = rng.choice(self.eps / self.sd, (hi - lo, burn))
            out[:, lo:hi] = weights @ history.T
        for p, row in zip(phis, out, strict=True):
            self.initial[p] = row

    def basis(self, phi):
        phi = float(phi)
        if self._last is not None and self._last[0] == phi:
            return self._last[1:]
        self._stationary([phi])
        # 初值是 x[-1]；先注入 ε_0/η_0，再得到 x[0]。
        a = lfilter(
            [1],
            [1, -phi],
            self.returns[:, :-1] / self.sd,
            axis=1,
            zi=(phi * self.initial[phi])[:, None],
        )[0]
        b = lfilter(
            [1],
            [1, -phi],
            self.normal[:, :-1],
            axis=1,
            zi=(phi * self.initial_normal / np.sqrt(1 - phi**2))[:, None],
        )[0]
        self._last = (phi, a, b)
        return a, b


def _batch_components(a, b, g, xmask, gmask, horizon):
    """t=N/sqrt(V)，N 对两条 x 基分量线性，V 二次；不依赖 δ。"""

    # 沿最后一维前向求和，保留原缺失位置和时间间隔。
    def sums(v):
        c = np.pad(np.cumsum(v, axis=1), ((0, 0), (1, 0)))
        return c[:, horizon:] - c[:, :-horizon]

    total = np.full_like(g, 0)
    total[:, : g.shape[1] - horizon + 1] = sums(g)
    complete = np.zeros(len(gmask), bool)
    complete[: len(gmask) - horizon + 1] = (
        np.convolve(gmask.astype(int), np.ones(horizon, int), "valid") == horizon
    )
    use = xmask & complete
    if use.sum() < 3:
        raise ValueError("模拟有效样本不足")
    means = [v[:, use].mean(axis=1) for v in (a, b)]
    numerator = np.array(
        [
            np.sum((v[:, use] - m[:, None]) * total[:, use], axis=1)
            for v, m in zip((a, b), means, strict=True)
        ]
    )
    past = (
        np.convolve(use.astype(int), np.ones(horizon, int), "full")[: len(xmask)]
        == horizon
    )
    meat = past & gmask
    qs = []
    for v, mean in zip((a, b), means, strict=True):
        q = np.zeros_like(v)
        q[:, horizon - 1 :] = sums(np.where(use[None, :], v, 0))
        qs.append(q[:, meat] - horizon * mean[:, None])
    e = g[:, meat] - g[:, gmask].mean(axis=1)[:, None]
    q1, q2 = qs
    e2 = e * e
    variance = np.array(
        [
            np.sum(e2 * q1 * q1, axis=1),
            np.sum(e2 * q1 * q2, axis=1),
            np.sum(e2 * q2 * q2, axis=1),
        ]
    )
    return numerator, variance


def _component_t(parts, delta):
    n, v = parts
    d, s = delta, np.sqrt(1 - delta**2)
    denominator = d * d * v[0] + 2 * d * s * v[1] + s * s * v[2]
    return (d * n[0] + s * n[1]) / np.sqrt(np.maximum(denominator, 1e-300))


def conservative_test(
    bank,
    xmask,
    gmask,
    horizon,
    delta_grid,
    observed,
    *,
    refine=True,
    full_refinement=False,
):
    points, distributions, surface = [], [], []
    g = bank.returns[:, 1:]
    for phi in bank.phi_grid:
        key = (float(phi), horizon, xmask.tobytes(), gmask.tobytes())
        if key not in bank.parts_cache:
            bank.parts_cache[key] = _batch_components(
                *bank.basis(phi), g, xmask, gmask, horizon
            )
        parts = bank.parts_cache[key]
        row = []
        for delta in delta_grid:
            stat = np.abs(_component_t(parts, float(delta)))
            if not np.isfinite(stat).all():
                raise ValueError("模拟出现非有限 t")
            points.append((float(phi), float(delta)))
            distributions.append(np.sort(stat))
            row.append(float(empirical_p(stat, observed)))
        surface.append(row)
    surface = np.array(surface)
    i, j = np.unravel_index(np.argmax(surface), surface.shape)
    # 平台最大值只要包含内点，也加密该内点；不能因 argmax 恰取边界而漏掉。
    maxima = np.argwhere(surface == surface.max())
    interior = [
        (int(a), int(b))
        for a, b in maxima
        if 0 < a < len(bank.phi_grid) - 1 and 0 < b < len(delta_grid) - 1
    ]
    if refine and (interior or full_refinement):
        if full_refinement:
            # 确认及功效共用预先冻结的加密全集；覆盖任何备择 t 的内点加密，避免嵌套模拟。
            fine_phi = np.unique(
                np.concatenate(
                    [
                        np.linspace(bank.phi_grid[a - 1], bank.phi_grid[a + 1], 5)
                        for a in range(1, len(bank.phi_grid) - 1)
                    ]
                )
            )
            fine_delta = np.unique(
                np.round(
                    np.concatenate(
                        [
                            np.linspace(delta_grid[b - 1], delta_grid[b + 1], 5)
                            for b in range(1, len(delta_grid) - 1)
                        ]
                    ),
                    15,
                )
            )
        else:
            i, j = interior[0]
            fine_phi = np.linspace(bank.phi_grid[i - 1], bank.phi_grid[i + 1], 5)
            fine_delta = np.linspace(delta_grid[j - 1], delta_grid[j + 1], 5)
        bank._stationary(fine_phi)
        for phi in fine_phi:
            key = (float(phi), horizon, xmask.tobytes(), gmask.tobytes())
            if key not in bank.parts_cache:
                bank.parts_cache[key] = _batch_components(
                    *bank.basis(phi), g, xmask, gmask, horizon
                )
            parts = bank.parts_cache[key]
            for delta in fine_delta:
                point = (float(phi), float(delta))
                if point in points:
                    continue
                points.append(point)
                distributions.append(np.sort(np.abs(_component_t(parts, delta))))
    table = NullTable(points, np.asarray(distributions))
    order = np.argsort(np.abs(delta_grid))

    def monotone(diffs):
        return bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    details = dict(
        p=float(table.pvalue(observed)),
        grid_max=float(surface.max()),
        grid_points=len(points),
        monotone_phi=monotone(np.diff(surface, axis=0)),
        monotone_abs_delta=monotone(np.diff(surface[:, order], axis=1)),
        maximizer=points[
            int(np.argmax([empirical_p(v, observed) for v in table.distributions]))
        ],
        repetitions=bank.repetitions,
        gamma=GAMMA,
        refined=len(points) > surface.size,
        full_refinement=full_refinement,
    )
    return table, details


def _power_components(a, b, g, xmask, gmask, horizon):
    """备择 t 的充分统计量：分子二次，meat 到四次；δ/效应网格只算标量。"""

    def forward(v):
        c = np.pad(np.cumsum(v, axis=1), ((0, 0), (1, 0)))
        out = np.zeros_like(v)
        out[:, : v.shape[1] - horizon + 1] = c[:, horizon:] - c[:, :-horizon]
        return out

    complete = np.zeros(len(gmask), bool)
    complete[: len(gmask) - horizon + 1] = (
        np.convolve(gmask.astype(int), np.ones(horizon, int), "valid") == horizon
    )
    use = xmask & complete
    past = (
        np.convolve(use.astype(int), np.ones(horizon, int), "full")[: len(xmask)]
        == horizon
    )
    meat = past & gmask
    means = [v[:, use].mean(axis=1) for v in (a, b)]
    centered = [v[:, use] - m[:, None] for v, m in zip((a, b), means, strict=True)]
    fa, fb = forward(a)[:, use], forward(b)[:, use]
    f0 = forward(g)[:, use]
    n0 = np.array([np.sum(v * f0, axis=1) for v in centered])
    n1 = np.array(
        [
            np.sum(centered[0] * fa, axis=1),
            np.sum(centered[0] * fb + centered[1] * fa, axis=1),
            np.sum(centered[1] * fb, axis=1),
        ]
    )
    qs = []
    for v, m in zip((a, b), means, strict=True):
        past_sum = np.zeros_like(v)
        past_sum[:, horizon - 1 :] = forward(np.where(use[None, :], v, 0))[
            :, : v.shape[1] - horizon + 1
        ]
        qs.append(past_sum[:, meat] - horizon * m[:, None])
    u = [v[:, meat] - v[:, gmask].mean(axis=1)[:, None] for v in (a, b)]
    e = g[:, meat] - g[:, gmask].mean(axis=1)[:, None]
    qa, qb = qs
    ua, ub = u
    # (d*qa+s*qb)^2 * (e + slope*(d*ua+s*ub))^2。
    qpoly = [qa * qa, 2 * qa * qb, qb * qb]
    upoly = [ua * ua, 2 * ua * ub, ub * ub]
    v0 = np.array([np.sum(q * e * e, axis=1) for q in qpoly])
    v1 = np.zeros((4, len(a)))
    v2 = np.zeros((5, len(a)))
    for i, q in enumerate(qpoly):
        for j, v in enumerate((ua, ub)):
            v1[i + j] += 2 * np.sum(q * e * v, axis=1)
        for j, v in enumerate(upoly):
            v2[i + j] += np.sum(q * v, axis=1)
    return n0, n1, v0, v1, v2


def power_curve(bank, table, xmask, gmask, horizon, calibration, *, effects=None):
    """逐日备择后聚合、完整保守 p 查表、各格最低功效；正负分别搜索。"""
    effects = np.asarray(
        effects
        if effects is not None
        else [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        float,
    )
    curves = {sign: np.ones(len(effects)) for sign in (-1, 1)}
    critical = table.critical()
    for phi in sorted(set(p[0] for p in table.points)):
        key = (float(phi), horizon, xmask.tobytes(), gmask.tobytes())
        if key not in bank.power_cache:
            bank.power_cache[key] = _power_components(
                *bank.basis(phi), bank.returns[:, 1:], xmask, gmask, horizon
            )
        parts = bank.power_cache[key]
        for delta in sorted(set(d for p, d in table.points if p == phi)):
            d, s = delta, np.sqrt(1 - delta**2)

            def polynomial(coef, d=d, s=s):
                degree = len(coef) - 1
                return sum(v * d ** (degree - i) * s**i for i, v in enumerate(coef))

            n0, n1, v0, v1, v2 = [polynomial(c) for c in parts]
            # x 的创新尺度；因子均值对有截距 t 无影响。
            scale = calibration["sd_eta"]
            for sign in (-1, 1):
                for i, effect in enumerate(effects):
                    slope = sign * effect * scale / (horizon * calibration["sd_x"])
                    stat = np.abs(
                        (n0 + slope * n1)
                        / np.sqrt(
                            np.maximum(v0 + slope * v1 + slope * slope * v2, 1e-300)
                        )
                    )
                    curves[sign][i] = min(
                        curves[sign][i], float(np.mean(stat > critical))
                    )
    result = {}
    for sign, rates in curves.items():
        detected = np.flatnonzero(rates >= 0.8)
        reached = int(detected[0]) if len(detected) else None
        previous = reached - 1 if reached is not None else len(effects) - 1

        def point(index, rates=rates):
            return (
                dict(effect=float(effects[index]), worst_power=float(rates[index]))
                if index is not None and index >= 0
                else None
            )

        result["positive" if sign == 1 else "negative"] = dict(
            effects=effects.tolist(),
            worst_power=rates.tolist(),
            minimum_detectable=float(effects[detected[0]]) if len(detected) else None,
            search_bracket=dict(previous=point(previous), reached=point(reached)),
            status="detectable" if len(detected) else "not_detectable_in_search",
            daily_slope=float(
                sign * effects[detected[0]] / (horizon * calibration["sd_x"])
            )
            if len(detected)
            else None,
            single_initial_shift_effect_range=[
                float(effects[detected[0]] / horizon),
                float(
                    effects[detected[0]]
                    * sum(0.999**j for j in range(horizon))
                    / horizon
                ),
            ]
            if len(detected)
            else None,
        )
    return dict(
        alpha=0.00625,
        target=0.8,
        effect_unit="h*b*sd_x, sustained standardized factor; see single_initial_shift_effect_range",
        effect_unit_label=("持续七日" if horizon == 7 else f"持续{horizon}日")
        + "增加 1 sd 的等价效应（对数收益）",
        mde_is_discrete=True,
        search_upper_bound=float(effects[-1]),
        **result,
    )
