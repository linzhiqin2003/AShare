"""ashare.factor.stats：对手算例子、直白参考实现和模拟的对拍。"""

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pytest

from ashare.factor import stats

N = NormalDist()


def test_dsr_matches_paper_example():
    # Bailey & López de Prado (2014) 第 4 节的数值例子：年化夏普 2.5、T = 1250 日、
    # 偏度 −3、峰度 10、N = 100 次试验、试验间年化夏普方差 0.5 → SR0 ≈ 0.1132、DSR ≈ 0.9004。
    sr = 2.5 / math.sqrt(250)
    v = 0.5 / 250
    sr0 = stats.expected_max_sharpe(100, v)
    assert sr0 == pytest.approx(0.1132, abs=5e-5)
    dsr = stats.deflated_sharpe(sr, 1250, -3, 10, n_trials=100, sr_var=v)
    assert dsr == pytest.approx(0.9004, abs=5e-4)


def test_psr_hand_formula():
    sr, t, g3, g4, star = 0.1, 500, -0.5, 6.0, 0.02
    z = (sr - star) * math.sqrt(t - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert stats.probabilistic_sharpe(sr, t, g3, g4, star) == pytest.approx(
        N.cdf(z), rel=1e-15
    )
    # 正态、门槛 0：PSR 就是 Φ(SR·√(T−1)/√(1+SR²/2))
    assert stats.probabilistic_sharpe(0.05, 101) == pytest.approx(
        N.cdf(0.05 * 10 / math.sqrt(1 + 0.05**2 / 2))
    )


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(3)
    v = 0.01
    for n in (10, 100, 1000):
        draws = rng.standard_normal((4000, n)) * math.sqrt(v)
        simulated = draws.max(axis=1).mean()
        # 近似公式（极值分布）在 N 小时略偏，N ≥ 10 时相对误差在几个百分点内
        assert stats.expected_max_sharpe(n, v) == pytest.approx(simulated, rel=0.05)
    assert stats.expected_max_sharpe(1, v) == 0.0
    # 试验越多门槛越高
    assert stats.expected_max_sharpe(1000, v) > stats.expected_max_sharpe(100, v)


def test_deflated_sharpe_uses_trial_count():
    a = stats.deflated_sharpe(0.08, 1000, n_trials=10, sr_var=0.001)
    b = stats.deflated_sharpe(0.08, 1000, n_trials=1000, sr_var=0.001)
    assert a > b
    # N = 1 退化为门槛 0 的 PSR
    assert stats.deflated_sharpe(0.08, 1000, n_trials=1, sr_var=0.001) == (
        stats.probabilistic_sharpe(0.08, 1000)
    )


def test_moments_reference():
    rng = np.random.default_rng(1)
    r = rng.standard_t(5, 800) * 0.01 + 0.0005
    sr, g3, g4 = stats.moments(r)
    m = r.mean()
    sd1 = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
    sd0 = math.sqrt(sum((x - m) ** 2 for x in r) / len(r))
    assert sr == pytest.approx(m / sd1, rel=1e-12)
    assert g3 == pytest.approx(sum(((x - m) / sd0) ** 3 for x in r) / len(r), rel=1e-9)
    assert g4 == pytest.approx(sum(((x - m) / sd0) ** 4 for x in r) / len(r), rel=1e-9)
    dsr, sr2, sr0 = stats.deflated_sharpe_of(r, n_trials=50, sr_var=0.002)
    assert sr2 == sr and sr0 == stats.expected_max_sharpe(50, 0.002)
    assert dsr == stats.deflated_sharpe(sr, 800, g3, g4, n_trials=50, sr_var=0.002)


def test_trial_sharpe_variance_floor():
    srs = [0.01, 0.011, 0.012]
    raw = stats.trial_sharpe_variance(srs, floor=False)
    assert raw == pytest.approx(np.var(srs, ddof=1))
    assert stats.trial_sharpe_variance(srs, t=101) == pytest.approx(1 / 100)
    wide = [0.0, 0.5, -0.5]
    assert stats.trial_sharpe_variance(wide, t=101) == pytest.approx(0.25)
    assert stats.trial_sharpe_variance([0.1], t=11) == pytest.approx(0.1)


def test_student_sf_closed_forms():
    # df = 1（柯西）：0.5 − atan(t)/π；df = 2：0.5 − t / (2√(t² + 2))；t = 0 → 0.5
    for t in (-3.0, -0.4, 0.0, 0.7, 2.0, 12.0):
        assert stats.student_sf(t, 1) == pytest.approx(
            0.5 - math.atan(t) / math.pi, rel=1e-12
        )
        assert stats.student_sf(t, 2) == pytest.approx(
            0.5 - t / (2 * math.sqrt(t * t + 2)), rel=1e-12
        )
    # 自由度很大时趋于正态
    assert stats.student_sf(2.0, 1e6) == pytest.approx(1 - N.cdf(2.0), rel=1e-5)
    assert stats.p_from_t(3.2, df=49) > stats.p_from_t(3.2)


def test_p_from_t():
    assert stats.p_from_t(1.6448536269514722) == pytest.approx(0.05, rel=1e-9)
    assert stats.p_from_t(1.959963984540054, two_sided=True) == pytest.approx(0.05)
    assert stats.p_from_t(-3.0, two_sided=True) == pytest.approx(2 * (1 - N.cdf(3.0)))
    out = stats.p_from_t(np.array([0.0, 2.0]))
    assert out[0] == pytest.approx(0.5) and out[1] == pytest.approx(1 - N.cdf(2))


def _hac_reference(x, lags, uniform=False):
    n = len(x)
    m = sum(x) / n
    e = [v - m for v in x]
    s = sum(v * v for v in e) / n
    for j in range(1, lags + 1):
        g = sum(e[i] * e[i - j] for i in range(j, n)) / n
        s += 2 * (1 if uniform else 1 - j / (lags + 1)) * g
    return m / math.sqrt(s / n)


def test_newey_west_reference():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(300) * 0.05 + 0.01
    x = np.convolve(x, np.ones(5) / 5, mode="valid")  # 人为重叠
    for lags in (0, 1, 4):
        t, n = stats.newey_west_t(x, lags)
        assert n == len(x)
        assert t == pytest.approx(_hac_reference(list(x), lags), rel=1e-12)
        tu, _ = stats.hac_t(x, lags, "uniform")
        assert tu == pytest.approx(_hac_reference(list(x), lags, True), rel=1e-12)
    # 重叠序列的 HAC t 值明显小于普通 t 值
    assert stats.newey_west_t(x, 4)[0] < 0.7 * stats.newey_west_t(x, 0)[0]
    assert stats.hac_t(x, 4, "uniform")[0] < stats.newey_west_t(x, 4)[0]


def test_hac_uniform_calibrated_on_overlapping_sums():
    # 零假设下 h 期重叠和的均值：均匀核滞后 h−1 的 t 值拒绝率接近名义水平，Bartlett 偏高
    rng = np.random.default_rng(8)
    h, n, reps = 5, 400, 1500
    rej_u = rej_b = 0
    for _ in range(reps):
        e = rng.standard_normal(n + h - 1)
        x = np.convolve(e, np.ones(h), mode="valid")
        rej_u += stats.hac_t(x, h - 1, "uniform")[0] > 1.6448536269514722
        rej_b += stats.hac_t(x, h - 1, "bartlett")[0] > 1.6448536269514722
    assert rej_u / reps < 0.075
    assert rej_b / reps > rej_u / reps


def _holm_reference(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out, running = [0.0] * m, 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def _bh_reference(p):
    m = len(p)
    out = []
    for i in range(m):
        # p̃_i = min over j with p_j >= p_i of m p_j / rank_j
        ranks = sorted(p)
        best = 1.0
        for r, pj in enumerate(ranks, start=1):
            if pj >= p[i]:
                best = min(best, m * pj / r)
        out.append(min(1.0, best))
    return out


def test_adjust_hand_example():
    p = [0.01, 0.04, 0.03, 0.005]
    # 手算：Bonferroni = 4p；Holm 排序 0.005,0.01,0.03,0.04 → 0.02,0.03,0.06,0.06；
    # BH：4·0.005/1=0.02，4·0.01/2=0.02，4·0.03/3=0.04，4·0.04/4=0.04
    assert stats.adjust(p, "bonferroni") == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert stats.adjust(p, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert stats.adjust(p, "bh") == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert list(stats.reject(p, 0.05, "holm")) == [True, False, False, True]
    assert list(stats.reject(p, 0.05, "bh")) == [True, True, True, True]
    # BY：BH × c(4)，c(4) = 1 + 1/2 + 1/3 + 1/4 = 25/12
    c = 25 / 12
    assert stats.adjust(p, "by") == pytest.approx(
        [0.02 * c, 0.04 * c, 0.04 * c, 0.02 * c]
    )


@pytest.mark.parametrize("seed", range(5))
def test_adjust_matches_reference(seed):
    rng = np.random.default_rng(seed)
    p = list(rng.random(37) ** 3)
    assert stats.adjust(p, "holm") == pytest.approx(_holm_reference(p), rel=1e-12)
    assert stats.adjust(p, "bh") == pytest.approx(_bh_reference(p), rel=1e-12)
    # 性质：bonferroni ≥ holm ≥ bh ≥ 原值
    b, h, q = (stats.adjust(p, m) for m in ("bonferroni", "holm", "bh"))
    assert np.all(b >= h - 1e-15) and np.all(h >= q - 1e-15)
    assert np.all(q >= np.asarray(p) - 1e-15)


def test_adjust_nan_occupies_denominator():
    out = stats.adjust([0.01, float("nan")], "bonferroni")
    assert out[0] == pytest.approx(0.02) and math.isnan(out[1])
    assert not stats.reject([float("nan")]).any()
    with pytest.raises(ValueError):
        stats.adjust([1.5])


def test_mht_under_null_simulation():
    # 完全零假设下：Holm 的族错误率、BH 的错误发现率（此时等于族错误率）都不超过 α。
    rng = np.random.default_rng(11)
    reps, m, alpha = 2000, 50, 0.1
    hits = {"holm": 0, "bh": 0, "bonferroni": 0, "by": 0}
    for _ in range(reps):
        p = rng.random(m)
        for k in hits:
            hits[k] += bool(stats.reject(p, alpha, k).any())
    for k, v in hits.items():
        assert v / reps <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / reps), k


def _pbo_reference(x, s):
    t, n = x.shape
    edges = np.linspace(0, t, s + 1).astype(int)
    lam = []
    for train in combinations(range(s), s // 2):
        tr = [i for b in train for i in range(edges[b], edges[b + 1])]
        te = [
            i for b in range(s) if b not in train for i in range(edges[b], edges[b + 1])
        ]

        def sr(rows, j):
            v = [x[i, j] for i in rows]
            mu = sum(v) / len(v)
            sd = math.sqrt(sum((a - mu) ** 2 for a in v) / (len(v) - 1))
            return mu / sd

        s_tr = [sr(tr, j) for j in range(n)]
        best = max(range(n), key=lambda j: s_tr[j])
        s_te = [sr(te, j) for j in range(n)]
        rank = sum(v < s_te[best] for v in s_te) + 1
        w = rank / (n + 1)
        lam.append(math.log(w / (1 - w)))
    return sum(v <= 0 for v in lam) / len(lam)


def test_pbo_reference_and_extremes():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((120, 6)) * 0.01
    assert stats.pbo_cscv(x, 6)["pbo"] == pytest.approx(_pbo_reference(x, 6))
    assert stats.pbo_cscv(x, 6)["n_splits"] == 20
    # 一个候选在每一段都显著更好 → 不会过拟合
    y = x.copy()
    y[:, 0] += 0.02
    assert stats.pbo_cscv(y, 6)["pbo"] == 0.0
    # 纯噪声、很多候选：PBO 应在 0.5 附近
    z = rng.standard_normal((400, 50)) * 0.01
    assert 0.2 < stats.pbo_cscv(z, 8)["pbo"] < 0.8
    with pytest.raises(ValueError):
        stats.pbo_cscv(x, 5)


def test_effective_trials():
    rng = np.random.default_rng(4)
    base = rng.standard_normal((500, 1))
    same = np.repeat(base, 10, axis=1) + 1e-9 * rng.standard_normal((500, 10))
    indep = rng.standard_normal((500, 10))
    for method in stats.EFFECTIVE_METHODS:
        assert stats.effective_trials(same, method=method) == pytest.approx(1, abs=0.05)
        assert stats.effective_trials(indep, method=method) == pytest.approx(
            10, rel=0.15
        )
    # 平均相关公式手算：ρ̄ = 0.5、N = 4 → 0.5 + 0.5·4 = 2.5
    c = np.full((4, 4), 0.5)
    np.fill_diagonal(c, 1)
    assert stats.effective_trials(corr=c, method="avg_corr") == pytest.approx(2.5)
    # Nyholt 手算：特征值 2.5, 0.5, 0.5, 0.5 → Var = Σ(λ−1)²/(N−1) = 1 → 1 + 3·(1 − 1/4) = 3.25
    assert stats.effective_trials(corr=c, method="nyholt") == pytest.approx(3.25)
    # Li–Ji 手算：f(2.5) + 3·f(0.5) = (1 + 0.5) + 1.5 = 3.0
    assert stats.effective_trials(corr=c, method="li_ji") == pytest.approx(3.0)


def test_variance_ratio_reference():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(500)
    e = x - x.mean()
    g = [e[j:] @ e[: len(e) - j] / len(e) for j in range(4)]
    ref = 1 + 2 * sum((1 - j / 4) * g[j] / g[0] for j in range(1, 4))
    assert stats.variance_ratio(x, 3, floor=False) == pytest.approx(ref, rel=1e-12)
    assert stats.variance_ratio(x, 3) == max(1.0, ref)
    # AR(1) φ：长期方差比 → (1+φ)/(1−φ)（Bartlett 窗足够长时接近）
    phi, n = 0.5, 200_000
    y = np.empty(n)
    z = rng.standard_normal(n)
    y[0] = z[0]
    for i in range(1, n):
        y[i] = phi * y[i - 1] + z[i]
    assert stats.variance_ratio(y, 200) == pytest.approx(3.0, rel=0.05)
    assert stats.variance_ratio([1.0, 1.0, 1.0]) == 1.0
    # 负自相关：原始比值小于 1，默认截在 1（不拿来放松检验）
    z = rng.standard_normal(5000)
    neg = z[1:] - 0.8 * z[:-1]
    assert stats.variance_ratio(neg, 20, floor=False) < 0.5
    assert stats.variance_ratio(neg, 20) == 1.0
    assert stats.variance_ratio(neg) == 1.0


def test_sharpe_with_variance_ratio():
    sr, t, q = 0.06, 800, 2.5
    base = stats.probabilistic_sharpe(sr, t)
    scaled = stats.probabilistic_sharpe(sr, t, q=q)
    z = sr * math.sqrt(t - 1) / math.sqrt(q * (1 + sr * sr / 2))
    assert scaled == pytest.approx(N.cdf(z), rel=1e-14) and scaled < base
    assert stats.sharpe_variance(sr, t, q=q) == pytest.approx(
        q * stats.sharpe_variance(sr, t), rel=1e-15
    )
    a = stats.deflated_sharpe(sr, t, n_trials=50, sr_var=q / (t - 1), q=q)
    sr0 = stats.expected_max_sharpe(50, q / (t - 1))
    assert a == pytest.approx(stats.probabilistic_sharpe(sr, t, sr_star=sr0, q=q))


def test_overlap_t_bound_and_df():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(301)
    t, df, n = stats.overlap_t(x, 7)
    assert n == 301 and df == 301 // 7 - 1
    assert t == pytest.approx(x.mean() / math.sqrt(x.var(ddof=1) * 7 / 301), rel=1e-14)
    assert math.isnan(stats.overlap_t([1.0], 1)[0])


def test_overlap_ic_holm_calibrated():
    """零假设下的重叠持有期日 IC（持有 h 天的收益在相邻信号日之间重叠，因子本身高度
    持续）：Holm 的族错误率在名义水平的抽样误差以内（完整网格见 bench/mining/ic_grid.py）。"""
    from ashare.factor.evaluate import ic_summary

    rng = np.random.default_rng(8)
    reps, m, alpha, h, n = 300, 10, 0.05, 20, 250
    hits = 0
    for _ in range(reps):
        # 日收益 iid；h 日持有收益 = 未来 h 天之和；因子 = 持续不变的随机分数 → 日 IC 是
        # 截面上「分数与 h 日收益」的相关，近似为日 IC 的 h 天滑动和
        daily = rng.standard_normal((n + h, m))
        ic = np.lib.stride_tricks.sliding_window_view(daily, h, axis=0).sum(-1)[:n]
        ic = ic / math.sqrt(h) * 0.05
        p = [ic_summary(ic[:, k], h)["p"] for k in range(m)]
        hits += bool(stats.reject(p, alpha, "holm").any())
    assert hits / reps <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / reps)
