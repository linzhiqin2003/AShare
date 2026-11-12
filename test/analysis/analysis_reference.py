"""直白的参考实现：逐日、逐股、逐因子的 Python 循环，不调用 ashare.analysis 的任何代码。"""

from __future__ import annotations

import math


def ref_total_returns(close, adj):
    t, n = len(close), len(close[0])
    out = [[math.nan] * n for _ in range(t)]
    for i in range(n):
        last_c = last_a = math.nan
        prev = math.nan
        for day in range(t):
            c, a = close[day][i], adj[day][i]
            if not math.isnan(c):
                last_c = c
            if not math.isnan(a):
                last_a = a
            price = last_c * last_a
            if day and not math.isnan(prev) and prev > 0:
                out[day][i] = price / prev - 1
            prev = price
    return out


def ref_attribution(
    held, bench, returns, exposures, factor_returns, rp=None, cost=None
):
    """逐日：返回 [(因子贡献列表, 特质, 交易, 成本, Rp, Rb)]，第 0 日全 0。"""
    t, n = len(held), len(held[0])
    k = len(factor_returns[0])
    out = []
    for day in range(t):
        r = [0.0 if math.isnan(x) else x for x in returns[day]]
        hp = sum(held[day][i] * r[i] for i in range(n))
        rb = sum(bench[day][i] * r[i] for i in range(n))
        contrib = [0.0] * k
        if day > 0 and all(not math.isnan(x) for x in factor_returns[day]):
            for j in range(k):
                x = 0.0
                for i in range(n):
                    row = exposures[day - 1][i]
                    if any(math.isnan(v) for v in row):
                        continue
                    x += (held[day][i] - bench[day][i]) * row[j]
                contrib[j] = x * factor_returns[day][j]
        specific = hp - rb - sum(contrib)
        if rp is None:
            out.append((contrib, specific, 0.0, 0.0, hp, rb))
        else:
            c = -cost[day]
            out.append((contrib, specific, rp[day] - hp - c, c, rp[day], rb))
    return out


def ref_grap_total(effects, rp, rb):
    """GRAP 直接公式：Σ_t a_t Π_{s<t}(1+Rp_s) Π_{s>t}(1+Rb_s)，只算期末。"""
    t = len(rp)
    m = len(effects[0])
    total = [0.0] * m
    for day in range(t):
        before = 1.0
        for s in range(day):
            before *= 1 + rp[s]
        after = 1.0
        for s in range(day + 1, t):
            after *= 1 + rb[s]
        for j in range(m):
            total[j] += effects[day][j] * before * after
    return total


def ref_carino_total(effects, rp, rb):
    gp = gb = 1.0
    for a, b in zip(rp, rb, strict=True):
        gp *= 1 + a
        gb *= 1 + b
    big = (math.log(gp) - math.log(gb)) / (gp - gb) if gp != gb else 1 / gp
    m = len(effects[0])
    total = [0.0] * m
    for day in range(len(rp)):
        d = rp[day] - rb[day]
        k = (
            (math.log(1 + rp[day]) - math.log(1 + rb[day])) / d
            if d != 0
            else 1 / (1 + rp[day])
        )
        for j in range(m):
            total[j] += effects[day][j] * k / big
    return total


def ref_brinson(wp, wb, rp, rb, bhb=False, known=()):
    """单期、按行业的字典：{行业: (配置, 选择, 交互)}。权重为 0 的行业收益忽略；
    known 里的行业（现金）两边收益都已知，不论权重是否为 0 都直接用。"""
    sectors = sorted(set(wp) | set(wb), key=str)
    total_b = sum(wb.get(s, 0) * rb.get(s, 0) for s in sectors)
    out = {}
    for s in sectors:
        p, b = wp.get(s, 0.0), wb.get(s, 0.0)
        r_b = rb[s] if (b != 0 or s in known) else total_b
        r_p = rp[s] if (p != 0 or s in known) else r_b
        alloc = (p - b) * r_b if bhb else (p - b) * (r_b - total_b)
        out[s] = (alloc, b * (r_p - r_b), (p - b) * (r_p - r_b))
    return out


def ref_risk(a, b, f, d):
    """稠密协方差 Σ = B F Bᵀ + D 的直接计算：返回 (σ, 每只股票对 σ 的贡献)。"""
    n, k = len(b), len(f)
    sigma = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for p in range(k):
                for q in range(k):
                    s += b[i][p] * f[p][q] * b[j][q]
            sigma[i][j] = s + (d[i] if i == j else 0.0)
    sa = [sum(sigma[i][j] * a[j] for j in range(n)) for i in range(n)]
    var = sum(a[i] * sa[i] for i in range(n))
    vol = math.sqrt(var)
    return vol, [a[i] * sa[i] / vol for i in range(n)]


def ref_factor_risk(a, b, f, d):
    """因子与特质的方差贡献（逐元素循环）：返回 (因子列表, 特质)。"""
    n, k = len(b), len(f)
    x = [sum(b[i][p] * a[i] for i in range(n)) for p in range(k)]
    contrib = []
    for p in range(k):
        contrib.append(sum(x[p] * f[p][q] * x[q] for q in range(k)))
    return contrib, sum(d[i] * a[i] * a[i] for i in range(n))


def ref_trade_cost(
    amount, side, price, volume, k, e, slip, rate, minimum, tax_sell, fee
):
    """单笔成交的成本，逐项与 panelbt 规格一致（不调用任何库）。"""
    q = amount / price
    part = q / volume
    slipped = price * (1 + side * slip)
    effective = slipped * (1 + side * k * part**e)
    impact = q * abs(effective - slipped)
    gross = q * effective
    commission = max(rate * gross, minimum)
    tax = tax_sell * gross if side < 0 else 0.0
    return {
        "impact": impact,
        "slippage": q * price * slip,
        "commission": commission,
        "tax": tax,
        "fee": fee * gross,
        "participation": part,
    }
