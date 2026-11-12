"""多期链接：把逐日的分项贡献累计成「各项之和恰好等于累计主动收益」的累计贡献。

累计主动收益定义为算术差 `Π(1+Rp) − Π(1+Rb)`；无基准时 Rb = 0，就是累计收益。

默认 GRAP（Frongello 递推的等价形式）：第 t 日的贡献乘以此前的组合增长、此后的基准增长，

    C_T = C_{T-1} · (1 + Rb_T) + a_T · Π_{s<T}(1 + Rp_s)

逐项相加时可直接伸缩相消，得到 Π(1+Rp) − Π(1+Rb)；不取对数、不做除法，
组合与基准逐日收益相等或接近时也没有奇点，递推一次就给出每一天的累计值。
缺点是同一贡献发生在不同日子时权重不同（这是复利本来的样子）。

可选 Carino 对数链接：系数 k_t = ln((1+Rp)/(1+Rb)) / (Rp − Rb)，再除以全期系数 K。
各期贡献被同一个规则缩放，与发生顺序无关，但要取对数，累计收益到 −100% 时无定义。
"""

from __future__ import annotations

import numpy as np

METHODS = ("grap", "carino")


def link(effects, portfolio, benchmark=None, method="grap"):
    """effects: T×M 逐日贡献（每行和 = Rp − Rb）；返回 T×M 的逐日累计值。"""
    a = np.asarray(effects, dtype=float)
    rp = np.asarray(portfolio, dtype=float)
    rb = np.zeros_like(rp) if benchmark is None else np.asarray(benchmark, float)
    if a.ndim != 2 or a.shape[0] != len(rp) or rb.shape != rp.shape:
        raise ValueError("link: effects 须为 T×M，与收益序列等长")
    if method == "grap":
        out = np.empty_like(a)
        acc = np.zeros(a.shape[1])
        growth = 1.0
        for t in range(len(rp)):
            acc = acc * (1.0 + rb[t]) + a[t] * growth
            growth *= 1.0 + rp[t]
            out[t] = acc
        return out
    if method == "carino":
        lp, lb = np.log1p(rp), np.log1p(rb)
        diff = rp - rb
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(diff != 0, (lp - lb) / diff, 1.0 / (1.0 + rp))
        running = np.cumsum(a * k[:, None], axis=0)
        cp, cb = np.cumsum(lp), np.cumsum(lb)
        total = np.expm1(cp) - np.expm1(cb)
        with np.errstate(invalid="ignore", divide="ignore"):
            big_k = np.where(total != 0, (cp - cb) / total, np.exp(-cp))
        return running / big_k[:, None]
    raise ValueError(f"method: 只支持 {METHODS}")


def cumulative_active(portfolio, benchmark=None):
    """逐日的 Π(1+Rp) − Π(1+Rb)，与 grap 用同样的乘法顺序。"""
    rp = np.asarray(portfolio, dtype=float)
    rb = np.zeros_like(rp) if benchmark is None else np.asarray(benchmark, float)
    return np.cumprod(1.0 + rp) - np.cumprod(1.0 + rb)
