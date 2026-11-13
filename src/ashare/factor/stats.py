"""多重试验下的显著性：概率夏普、折算夏普、多重检验校正、回测过拟合概率、有效试验数。

这个模块不依赖搜索器，手工研究也能直接用；只用 NumPy 和标准库。夏普一律是**每期**口径
（日频就是日夏普，不年化），样本长度 T 是收益期数。每个函数的出处写在文档串里，
公式与对拍见 `docs/MINING_NOTES.md`。

```python
from ashare.factor import stats
stats.deflated_sharpe(sr=0.08, t=1000, skew=0, kurt=3, n_trials=200, sr_var=0.001)
stats.adjust([0.001, 0.02, 0.4], method="bh")      # 校正后的 p 值
```
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329
METHODS = ("bonferroni", "holm", "bh", "by")


# ---- 单条收益序列的矩 ------------------------------------------------------------


def moments(r):
    """每期夏普、偏度、峰度（非超额，正态为 3）。偏度与峰度用总体矩（ddof=0）。

    样本少于 2 或标准差为 0 时返回三个 NaN。"""
    x = np.asarray(r, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return math.nan, math.nan, math.nan
    sd = float(x.std(ddof=1))
    sd0 = float(x.std(ddof=0))
    if not sd > 0:
        return math.nan, math.nan, math.nan
    z = (x - x.mean()) / sd0
    return float(x.mean() / sd), float((z**3).mean()), float((z**4).mean())


def sharpe_variance(sr, t, skew=0.0, kurt=3.0):
    """夏普估计量的渐近方差（每期口径）：(1 − γ3·SR + (γ4 − 1)/4·SR²) / (T − 1)。

    出处：Mertens (2002)；Bailey & López de Prado (2012) "The Sharpe Ratio Efficient
    Frontier" 式 (4) 的分母部分。下限 1e-12 防止极端偏峰让方差变负。"""
    if t < 2:
        return math.nan
    return max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr) / (t - 1)


# ---- 概率夏普与折算夏普 ------------------------------------------------------------


def probabilistic_sharpe(sr, t, skew=0.0, kurt=3.0, sr_star=0.0):
    """概率夏普 PSR：真实夏普大于 sr_star 的概率。

    PSR = Φ[(SR − SR*)·√(T − 1) / √(1 − γ3·SR + (γ4 − 1)/4·SR²)]。
    出处：Bailey & López de Prado (2012)，J. of Risk 15(2)，式 (11)；
    又见 Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"，式 (2)。
    sr、sr_star 每期口径；kurt 为非超额峰度。"""
    if not (math.isfinite(sr) and t >= 2):
        return math.nan
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return _N.cdf((sr - sr_star) * math.sqrt(t - 1) / denom)


def expected_max_sharpe(n_trials, sr_var):
    """N 次独立试验、试验间夏普方差为 sr_var 时，纯噪声里挑出的最大夏普的期望。

    SR0 = √V·[(1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))]，γ 为欧拉常数。
    出处：Bailey & López de Prado (2014)，式 (3)（「False Strategy Theorem」，
    又见 Bailey, Borwein, López de Prado & Zhu (2014) Notices AMS）。N < 2 时为 0。"""
    n = float(n_trials)
    if n < 2:
        return 0.0
    if not (sr_var >= 0):
        return math.nan
    a = _N.inv_cdf(1 - 1 / n)
    b = _N.inv_cdf(1 - 1 / (n * math.e))
    return math.sqrt(sr_var) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(sr, t, skew=0.0, kurt=3.0, *, n_trials, sr_var):
    """折算夏普 DSR：以「N 次试验里纯噪声的期望最大夏普」SR0 为门槛的 PSR。

    DSR = PSR(SR0)，SR0 = expected_max_sharpe(N, V)。出处：Bailey & López de Prado
    (2014) "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting and Non-Normality"，J. of Portfolio Management 40(5)，式 (2)(3)。
    n_trials：到现在为止的试验总数（含被筛掉、被判重复和失败的）；sr_var：各试验
    夏普的方差（每期口径）。返回概率；> 0.95 相当于 5% 水平显著。"""
    sr0 = expected_max_sharpe(n_trials, sr_var)
    return probabilistic_sharpe(sr, t, skew, kurt, sr0)


def deflated_sharpe_of(r, *, n_trials, sr_var):
    """对一条收益序列直接算 DSR，返回 (DSR, SR, SR0)。"""
    sr, g3, g4 = moments(r)
    x = np.asarray(r, dtype=np.float64)
    t = int(np.isfinite(x).sum())
    sr0 = expected_max_sharpe(n_trials, sr_var)
    return probabilistic_sharpe(sr, t, g3, g4, sr0), sr, sr0


def trial_sharpe_variance(srs, *, t=None, floor=True):
    """各试验夏普的截面方差 V（ddof=1）。

    floor=True 且给了 t 时取 max(经验方差, 1/(T−1))：后者是零假设下单个夏普估计量
    的方差（正态、SR=0），两段式筛选只回测了一部分候选、经验方差可能偏小，取大者让
    SR0 只会更高、结论只会更保守。有效样本少于 2 个时只返回下限（没给 t 时为 NaN）。"""
    x = np.asarray([s for s in srs if s is not None], dtype=np.float64)
    x = x[np.isfinite(x)]
    v = float(x.var(ddof=1)) if len(x) >= 2 else math.nan
    if floor and t is not None and t >= 2:
        null = 1.0 / (t - 1)
        return null if not math.isfinite(v) else max(v, null)
    return v


# ---- t 值与 p 值 -----------------------------------------------------------------


def _betacf(a, b, x):
    """正则化不完全贝塔函数的连分式（Lentz 法；Press 等《Numerical Recipes》6.4 节）。"""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )  # fmt: skip
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_sf(t, df):
    """Student t 分布的上尾概率 P(T > t)（df > 0，可为非整数）。"""
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def p_from_t(t, *, two_sided=False, df=None):
    """t 值的 p 值。默认单侧（备择：均值 > 0，做多头因子时只关心正向）。

    df 为 None 时用正态近似；给 df 时用 Student t（小样本、HAC 方差估计偏小时更稳妥）。"""
    t = np.asarray(t, dtype=np.float64)

    def upper_one(v):
        if not math.isfinite(v) and math.isnan(v):
            return math.nan
        return 0.5 * math.erfc(v / math.sqrt(2)) if df is None else student_sf(v, df)

    upper = np.vectorize(upper_one, otypes=[float])(t) if t.size else t
    p = 2 * np.minimum(upper, 1 - upper) if two_sided else upper
    return float(p) if np.ndim(p) == 0 else p


def hac_t(x, lags, kernel="bartlett"):
    """均值为 0 的 t 值，标准误用 HAC 长期方差（滞后 lags）。返回 (t, 有效样本数)。

    - ``bartlett``：Newey & West (1987) Econometrica 55(3)，权重 1 − j/(lags+1)；
    - ``uniform``：Hansen & Hodrick (1980) J. Political Economy 88(5)，滞后 1..lags 权重全为 1。
      持有 h 期的日 IC 在零假设下恰是 MA(h − 1)（收益只在 h 天内重叠），取 lags = h − 1
      的均匀核正好；Bartlett 核会把这几阶自协方差打折、低估方差（h = 5 时 t 值约偏大
      1.2 倍）。均匀核的方差估计可能为负，此时退回 Bartlett。
    lags = 0 时就是普通 t 值（方差用 1/T 归一）。"""
    if kernel not in ("bartlett", "uniform"):
        raise ValueError("kernel: 只能是 bartlett 或 uniform")
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return math.nan, n
    mu = v.mean()
    e = v - mu
    gamma = [float(e @ e) / n]
    for j in range(1, min(int(lags), n - 1) + 1):
        gamma.append(float(e[j:] @ e[:-j]) / n)

    def lrv(uniform):
        s = gamma[0]
        for j in range(1, len(gamma)):
            s += 2 * (1.0 if uniform else 1 - j / (lags + 1)) * gamma[j]
        return s

    s = lrv(kernel == "uniform")
    if kernel == "uniform" and not s > 0:
        s = lrv(False)
    if not s > 0:
        return math.nan, n
    return float(mu / math.sqrt(s / n)), n


def newey_west_t(x, lags):
    """Bartlett 核的 HAC t 值（Newey & West 1987），见 `hac_t`。"""
    return hac_t(x, lags, "bartlett")


# ---- 多重检验校正 ----------------------------------------------------------------


def adjust(p, method="holm"):
    """校正后的 p 值（与输入同顺序）。NaN（无效检验）不被拒绝，但仍占分母 m。

    - ``bonferroni``：min(1, m·p)。控制族错误率（FWER）。
    - ``holm``：Holm (1979) Scand. J. Statist. 6(2) 逐步下降法，
      p̃_(i) = max_{j≤i} min(1, (m − j + 1)·p_(j))。控制 FWER，任意相关结构下成立。
    - ``bh``：Benjamini & Hochberg (1995) JRSS-B 57(1) 逐步上升法，
      p̃_(i) = min_{j≥i} min(1, m·p_(j)/j)。控制错误发现率（FDR），独立或正相关
      （PRDS）时成立。
    - ``by``：Benjamini & Yekutieli (2001) Ann. Statist. 29(4) 定理 1.3，BH 再乘
      c(m) = Σ_{k≤m} 1/k，任意相关结构下控制 FDR。候选里同时有 x 与 −x 这类负相关的
      单侧检验时 PRDS 不成立，BH 可能超出名义水平，用它。
    """
    if method not in METHODS:
        raise ValueError(f"method: 只能是 {METHODS}")
    p = np.asarray(p, dtype=np.float64).ravel()
    m = len(p)
    out = np.full(m, np.nan)
    if m == 0:
        return out
    valid = np.isfinite(p)
    if np.any((p[valid] < 0) | (p[valid] > 1)):
        raise ValueError("p 值必须在 [0, 1] 内")
    # 无效检验排在最后，按 p = 1 处理，校正值输出 NaN
    q = np.where(valid, p, 1.0)
    order = np.argsort(q, kind="stable")
    sorted_p = q[order]
    if method == "bonferroni":
        adj = np.minimum(1.0, m * sorted_p)
    elif method == "holm":
        adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * sorted_p))
    else:
        ranks = np.arange(1, m + 1)
        scale = float(np.sum(1.0 / ranks)) if method == "by" else 1.0
        adj = np.minimum.accumulate((scale * m * sorted_p / ranks)[::-1])[::-1]
        adj = np.minimum(1.0, adj)
    out[order] = adj
    out[~valid] = np.nan
    return out


def reject(p, alpha=0.05, method="holm"):
    """哪些假设在 alpha 水平被拒绝（布尔数组）。"""
    adj = adjust(p, method)
    return np.isfinite(adj) & (adj <= alpha)


# ---- 回测过拟合概率 --------------------------------------------------------------


def pbo_cscv(returns, n_blocks=10):
    """组合对称交叉验证（CSCV）下的回测过拟合概率 PBO。

    returns：T×N 的每期收益（N 个候选，同一区间；含 NaN 的行整行去掉）。按时间切成
    n_blocks（偶数）块，取全部 C(S, S/2) 种组合作训练集、其余作测试集；训练集夏普最高的
    候选在测试集里的相对名次 ω = rank/(N + 1)，logit λ = ln(ω/(1 − ω))，
    PBO = P(λ ≤ 0)。出处：Bailey, Borwein, López de Prado & Zhu (2017)
    "The Probability of Backtest Overfitting"，J. of Computational Finance 20(4)，
    第 2 节 CSCV 第 1–5 步。名次 = 测试集夏普严格更低的候选数 + 1（1..N，越大越好）。
    返回 {"pbo", "n_splits", "median_logit", "logits"}。"""
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("returns: 需要 T×N 矩阵")
    x = x[np.isfinite(x).all(axis=1)]
    t, n = x.shape
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks: 必须是不小于 2 的偶数")
    if n < 2 or t < n_blocks * 2:
        raise ValueError("候选少于 2 个或样本太短，无法做 CSCV")
    edges = np.linspace(0, t, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]

    def sr(m):
        sd = m.std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, m.mean(axis=0) / sd, np.nan)

    logits = []
    for train in combinations(range(n_blocks), n_blocks // 2):
        tr = np.concatenate([blocks[i] for i in train])
        te = np.concatenate([blocks[i] for i in range(n_blocks) if i not in train])
        s_tr, s_te = sr(x[tr]), sr(x[te])
        if not np.isfinite(s_tr).any():
            continue
        best = int(np.nanargmax(s_tr))
        if not np.isfinite(s_te[best]):
            continue
        rank = int(np.sum(np.nan_to_num(s_te, nan=-np.inf) < s_te[best])) + 1
        omega = rank / (n + 1)
        logits.append(math.log(omega / (1 - omega)))
    lam = np.asarray(logits)
    return {
        "pbo": float((lam <= 0).mean()) if len(lam) else math.nan,
        "n_splits": len(lam),
        "median_logit": float(np.median(lam)) if len(lam) else math.nan,
        "logits": lam,
    }


# ---- 有效试验数 ------------------------------------------------------------------

EFFECTIVE_METHODS = ("avg_corr", "nyholt", "li_ji")


def correlation(returns):
    """T×N 收益的两两相关矩阵，按成对完整样本算；方差为 0 或重叠不足 3 期的为 NaN。"""
    x = np.asarray(returns, dtype=np.float64)
    n = x.shape[1]
    out = np.full((n, n), np.nan)
    ok = np.isfinite(x)
    for i in range(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            both = ok[:, i] & ok[:, j]
            if both.sum() < 3:
                continue
            a, b = x[both, i], x[both, j]
            if a.std() > 0 and b.std() > 0:
                out[i, j] = out[j, i] = float(np.corrcoef(a, b)[0, 1])
    return out


def effective_trials(returns=None, *, corr=None, method="li_ji"):
    """按候选之间的相关结构估计有效独立试验数（1 ≤ M_eff ≤ N）。

    输入 T×N 收益（或直接给 N×N 相关矩阵 corr；缺失相关按 0 处理）。
    - ``avg_corr``：M = ρ̄ + (1 − ρ̄)·N，ρ̄ 为非对角相关的平均（截到 [0, 1]）。出处：
      Bailey & López de Prado (2014) DSR 论文第 3 节对相关试验的建议；
      López de Prado (2018)《Advances in Financial ML》第 14 章。
    - ``nyholt``：M = 1 + (N − 1)·(1 − Var(λ)/N)，λ 为相关矩阵特征值。出处：
      Nyholt (2004) Am. J. Hum. Genet. 74(4)。
    - ``li_ji``：M = Σ f(|λ_i|)，f(x) = 1[x ≥ 1] + (x − ⌊x⌋)。出处：Li & Ji (2005)
      Heredity 95(3)。
    局限：都只看线性相关，把相关结构当成平稳；候选少、样本短时特征值噪声大，
    结果应当作量级参考。用于折算时更保守的做法是直接用原始试验数。"""
    if method not in EFFECTIVE_METHODS:
        raise ValueError(f"method: 只能是 {EFFECTIVE_METHODS}")
    if corr is None:
        if returns is None:
            raise ValueError("给 returns 或 corr")
        corr = correlation(returns)
    c = np.asarray(corr, dtype=np.float64)
    n = c.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    c = np.where(np.isfinite(c), c, 0.0)
    np.fill_diagonal(c, 1.0)
    if method == "avg_corr":
        rho = float(c[~np.eye(n, dtype=bool)].mean())
        rho = min(max(rho, 0.0), 1.0)
        m = rho + (1 - rho) * n
    else:
        lam = np.linalg.eigvalsh((c + c.T) / 2)
        if method == "nyholt":
            m = 1 + (n - 1) * (1 - float(np.var(lam, ddof=1)) / n)
        else:
            a = np.abs(lam)
            m = float(np.sum((a >= 1).astype(float) + (a - np.floor(a))))
    return float(min(max(m, 1.0), n))
