"""国泰君安 191 的独立参考实现（收 186 条；030、143、165、166、183 不收）。

照研报原文（``docs/factors/sources/gtja191.jsonl`` 的 ``formula`` 字段）逐层直写，不化简、
不合并、不调换运算顺序；每个函数上方的注释是原文公式。口径照
``docs/CONVENTIONS.md`` 和负责人裁决（2026-11-13），原文疑点的处理写在函数体里的
注释和 ``docs/REFERENCE_gtja191_NOTES.md``。算子在 ``gtja191_ref_ops``，不经过表达式
引擎和 ``panelops``。

输入是 ``reference_frames`` 给出的字典（``docs/REFERENCE_INPUTS.md``）；每个函数返回与
``data["close"]`` 同索引、同列的 DataFrame。字段记号：``OP H L C`` 后复权开高低收、``VW`` 均价、
``V`` 成交量（除以复权因子）、``AMT`` 成交额、``RET`` 收益、``BO BC`` 基准指数开盘、收盘
（T × 1，与股票广播）；``DTM DBM TR HD LD`` 按研报定义表由上面的字段算出。
"""

import numpy as np
import pandas as pd

from .gtja191_ref_ops import (
    absv,
    and_,
    clean,
    count,
    delay,
    delta,
    div,
    emax,
    emin,
    eq,
    filter_,
    ge,
    gt,
    highday,
    iff,
    le,
    log,
    lowday,
    lt,
    or_,
    pw,
    rank,
    regbeta_pairs,
    regbeta_seq,
    sign,
    sma,
    sumif,
    ts_prod,
    ts_sum,
    wma,
)
from .gtja191_ref_ops import decay_linear as decaylinear
from .gtja191_ref_ops import ts_corr as corr
from .gtja191_ref_ops import ts_cov as cov
from .gtja191_ref_ops import ts_max as tsmax
from .gtja191_ref_ops import ts_mean as mean
from .gtja191_ref_ops import ts_min as tsmin
from .gtja191_ref_ops import ts_rank as tsrank
from .gtja191_ref_ops import ts_std as std

EXCLUDED = ("gtja191_030", "gtja191_143", "gtja191_165", "gtja191_166", "gtja191_183")


class _Fields:
    """一份输入数据的 NumPy 视图；派生字段按需计算并缓存。"""

    def __init__(self, data):
        close = data["close"]
        self.index = close.index
        self.columns = close.columns
        self.shape = close.shape

        def arr(name):
            return np.asarray(data[name], dtype=float)

        self.C = arr("close")
        self.OP = arr("open")
        self.H = arr("high")
        self.L = arr("low")
        self.V = arr("volume")
        self.VW = arr("vwap")
        self.AMT = arr("amount")
        self.RET = arr("returns")
        self._data = data
        self._cache = {}

    def _bench(self, name):
        if name not in self._data:
            raise KeyError(f"输入里没有 {name}（基准指数）")
        values = np.asarray(self._data[name], dtype=float)
        return values.reshape(-1, 1)

    @property
    def BO(self):
        return self._bench("benchmark_open")

    @property
    def BC(self):
        return self._bench("benchmark_close")

    def _derived(self, name, build):
        if name not in self._cache:
            self._cache[name] = build()
        return self._cache[name]

    # DTM = (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
    @property
    def DTM(self):
        OP, H = self.OP, self.H
        return self._derived(
            "DTM", lambda: iff(le(OP, delay(OP, 1)), 0, emax(H - OP, OP - delay(OP, 1)))
        )

    # DBM = (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    @property
    def DBM(self):
        OP, L = self.OP, self.L
        return self._derived(
            "DBM", lambda: iff(ge(OP, delay(OP, 1)), 0, emax(OP - L, OP - delay(OP, 1)))
        )

    # TR = MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
    @property
    def TR(self):
        H, L, C = self.H, self.L, self.C
        return self._derived(
            "TR",
            lambda: emax(emax(H - L, absv(H - delay(C, 1))), absv(L - delay(C, 1))),
        )

    # HD = HIGH-DELAY(HIGH,1)
    @property
    def HD(self):
        return self._derived("HD", lambda: self.H - delay(self.H, 1))

    # LD = DELAY(LOW,1)-LOW
    @property
    def LD(self):
        return self._derived("LD", lambda: delay(self.L, 1) - self.L)


def _fields(data):
    return data if isinstance(data, _Fields) else _Fields(data)


def _out(f, value):
    """广播成 T × N、非有限值换成 NaN，包成与 close 同索引同列的 DataFrame。"""
    values = clean(np.broadcast_to(np.asarray(value, dtype=float), f.shape))
    return pd.DataFrame(values, index=f.index, columns=f.columns)


# (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))
def gtja191_001(data, **params):
    f = _fields(data)
    C, OP, V = f.C, f.OP, f.V
    return _out(f, -1 * corr(rank(delta(log(V), 1)), rank(div(C - OP, OP)), 6))


# (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
def gtja191_002(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    # Q-G-002：以研报原文为准。
    return _out(f, -1 * delta(div((C - L) - (H - C), H - L), 1))


# SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
def gtja191_003(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    # Q-G-003：单个 `=` 按相等比较。
    return _out(
        f,
        ts_sum(
            iff(
                eq(C, delay(C, 1)),
                0,
                C - iff(gt(C, delay(C, 1)), emin(L, delay(C, 1)), emax(H, delay(C, 1))),
            ),
            6,
        ),
    )


# ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME / MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
def gtja191_004(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(
        f,
        iff(
            lt(div(ts_sum(C, 8), 8) + std(C, 8), div(ts_sum(C, 2), 2)),
            -1 * 1,
            iff(
                lt(div(ts_sum(C, 2), 2), div(ts_sum(C, 8), 8) - std(C, 8)),
                1,
                iff(
                    or_(lt(1, div(V, mean(V, 20))), eq(div(V, mean(V, 20)), 1)),
                    1,
                    -1 * 1,
                ),
            ),
        ),
    )


# (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))
def gtja191_005(data, **params):
    f = _fields(data)
    H, V = f.H, f.V
    return _out(f, -1 * tsmax(corr(tsrank(V, 5), tsrank(H, 5), 5), 3))


# (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
def gtja191_006(data, **params):
    f = _fields(data)
    OP, H = f.OP, f.H
    return _out(f, rank(sign(delta(OP * 0.85 + H * 0.15, 4))) * -1)


# ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
def gtja191_007(data, **params):
    f = _fields(data)
    C, V, VW = f.C, f.V, f.VW
    # Q-G-02：MAX/MIN(x, 3) 按 TSMAX/TSMIN。
    return _out(
        f, (rank(tsmax(VW - C, 3)) + rank(tsmin(VW - C, 3))) * rank(delta(V, 3))
    )


# RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
def gtja191_008(data, **params):
    f = _fields(data)
    H, L, VW = f.H, f.L, f.VW
    return _out(f, rank(delta(div(H + L, 2) * 0.2 + VW * 0.8, 4) * -1))


# SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
def gtja191_009(data, **params):
    f = _fields(data)
    H, L, V = f.H, f.L, f.V
    x = div((div(H + L, 2) - div(delay(H, 1) + delay(L, 1), 2)) * (H - L), V)
    return _out(f, sma(x, 7, 2))


# (RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))
def gtja191_010(data, **params):
    f = _fields(data)
    C, RET = f.C, f.RET
    # Q-G-010：按 RANK(TSMAX((...)^2, 5))。
    return _out(f, rank(tsmax(pw(iff(lt(RET, 0), std(RET, 20), C), 2), 5)))


# SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
def gtja191_011(data, **params):
    f = _fields(data)
    C, H, L, V = f.C, f.H, f.L, f.V
    # Q-G-011：./ 与 .* 按 / 与 *。
    return _out(f, ts_sum(div((C - L) - (H - C), H - L) * V, 6))


# (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))
def gtja191_012(data, **params):
    f = _fields(data)
    C, OP, VW = f.C, f.OP, f.VW
    return _out(f, rank(OP - div(ts_sum(VW, 10), 10)) * (-1 * rank(absv(C - VW))))


# (((HIGH * LOW)^0.5) - VWAP)
def gtja191_013(data, **params):
    f = _fields(data)
    H, L, VW = f.H, f.L, f.VW
    return _out(f, pw(H * L, 0.5) - VW)


# CLOSE-DELAY(CLOSE,5)
def gtja191_014(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, C - delay(C, 5))


# OPEN/DELAY(CLOSE,1)-1
def gtja191_015(data, **params):
    f = _fields(data)
    C, OP = f.C, f.OP
    return _out(f, div(OP, delay(C, 1)) - 1)


# (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
def gtja191_016(data, **params):
    f = _fields(data)
    V, VW = f.V, f.VW
    return _out(f, -1 * tsmax(rank(corr(rank(V), rank(VW), 5)), 5))


# RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
def gtja191_017(data, **params):
    f = _fields(data)
    C, VW = f.C, f.VW
    # Q-G-02：MAX(VWAP, 15) 按 TSMAX。
    return _out(f, pw(rank(VW - tsmax(VW, 15)), delta(C, 5)))


# CLOSE/DELAY(CLOSE,5)
def gtja191_018(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(C, delay(C, 5)))


# (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
def gtja191_019(data, **params):
    f = _fields(data)
    C = f.C
    return _out(
        f,
        iff(
            lt(C, delay(C, 5)),
            div(C - delay(C, 5), delay(C, 5)),
            iff(eq(C, delay(C, 5)), 0, div(C - delay(C, 5), C)),
        ),
    )


# (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
def gtja191_020(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(C - delay(C, 6), delay(C, 6)) * 100)


# REGBETA(MEAN(CLOSE,6),SEQUENCE(6))
def gtja191_021(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-021：REGBETA 的窗口取 SEQUENCE(6) 的 6。
    return _out(f, regbeta_seq(mean(C, 6), 6))


# SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
def gtja191_022(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-022：SMEAN 按 SMA。
    x = div(C - mean(C, 6), mean(C, 6))
    return _out(f, sma(x - delay(x, 3), 12, 1))


# SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
def gtja191_023(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-023：STD(CLOSE:20) 按 STD(CLOSE,20)。
    up = sma(iff(gt(C, delay(C, 1)), std(C, 20), 0), 20, 1)
    dn = sma(iff(le(C, delay(C, 1)), std(C, 20), 0), 20, 1)
    return _out(f, div(up, up + dn) * 100)


# SMA(CLOSE-DELAY(CLOSE,5),5,1)
def gtja191_024(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, sma(C - delay(C, 5), 5, 1))


# ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))
def gtja191_025(data, **params):
    f = _fields(data)
    C, V, RET = f.C, f.V, f.RET
    return _out(
        f,
        (-1 * rank(delta(C, 7) * (1 - rank(decaylinear(div(V, mean(V, 20)), 9)))))
        * (1 + rank(ts_sum(RET, 250))),
    )


# ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
def gtja191_026(data, **params):
    f = _fields(data)
    C, VW = f.C, f.VW
    return _out(f, (div(ts_sum(C, 7), 7) - C) + corr(VW, delay(C, 5), 230))


# WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
def gtja191_027(data, **params):
    f = _fields(data)
    C = f.C
    return _out(
        f,
        wma(
            div(C - delay(C, 3), delay(C, 3)) * 100
            + div(C - delay(C, 6), delay(C, 6)) * 100,
            12,
        ),
    )


# 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
def gtja191_028(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    # Q-G-028：第二个分母按 (TSMAX(HIGH,9)-TSMIN(LOW,9))。
    k1 = div(C - tsmin(L, 9), tsmax(H, 9) - tsmin(L, 9)) * 100
    k2 = div(C - tsmin(L, 9), tsmax(H, 9) - tsmin(L, 9)) * 100
    return _out(f, 3 * sma(k1, 3, 1) - 2 * sma(sma(k2, 3, 1), 3, 1))


# (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
def gtja191_029(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(f, div(C - delay(C, 6), delay(C, 6)) * V)


# (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
def gtja191_031(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(C - mean(C, 12), mean(C, 12)) * 100)


# (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
def gtja191_032(data, **params):
    f = _fields(data)
    H, V = f.H, f.V
    return _out(f, -1 * ts_sum(rank(corr(rank(H), rank(V), 3)), 3))


# ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) * TSRANK(VOLUME, 5))
def gtja191_033(data, **params):
    f = _fields(data)
    L, V, RET = f.L, f.V, f.RET
    return _out(
        f,
        (
            (-1 * tsmin(L, 5) + delay(tsmin(L, 5), 5))
            * rank(div(ts_sum(RET, 240) - ts_sum(RET, 20), 220))
        )
        * tsrank(V, 5),
    )


# MEAN(CLOSE,12)/CLOSE
def gtja191_034(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(mean(C, 12), C))


# (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN *0.35)), 17),7))) * -1)
def gtja191_035(data, **params):
    f = _fields(data)
    OP, V = f.OP, f.V
    # 照抄：(OPEN * 0.65) + (OPEN * 0.35) 两项是同一个字段，不化简。
    return _out(
        f,
        emin(
            rank(decaylinear(delta(OP, 1), 15)),
            rank(decaylinear(corr(V, OP * 0.65 + OP * 0.35, 17), 7)),
        )
        * -1,
    )


# RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)
def gtja191_036(data, **params):
    f = _fields(data)
    V, VW = f.V, f.VW
    # Q-G-036：按 RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))。
    return _out(f, rank(ts_sum(corr(rank(V), rank(VW), 6), 2)))


# (-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))
def gtja191_037(data, **params):
    f = _fields(data)
    OP, RET = f.OP, f.RET
    return _out(
        f,
        -1
        * rank(
            ts_sum(OP, 5) * ts_sum(RET, 5) - delay(ts_sum(OP, 5) * ts_sum(RET, 5), 10)
        ),
    )


# (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
def gtja191_038(data, **params):
    f = _fields(data)
    H = f.H
    return _out(f, iff(lt(div(ts_sum(H, 20), 20), H), -1 * delta(H, 2), 0))


# ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)
def gtja191_039(data, **params):
    f = _fields(data)
    C, OP, V, VW = f.C, f.OP, f.V, f.VW
    return _out(
        f,
        (
            rank(decaylinear(delta(C, 2), 8))
            - rank(
                decaylinear(corr(VW * 0.3 + OP * 0.7, ts_sum(mean(V, 180), 37), 14), 12)
            )
        )
        * -1,
    )


# SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
def gtja191_040(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(
        f,
        div(
            ts_sum(iff(gt(C, delay(C, 1)), V, 0), 26),
            ts_sum(iff(le(C, delay(C, 1)), V, 0), 26),
        )
        * 100,
    )


# (RANK(MAX(DELTA((VWAP), 3), 5))* -1)
def gtja191_041(data, **params):
    f = _fields(data)
    VW = f.VW
    # Q-G-02：MAX(DELTA(VWAP,3), 5) 按 TSMAX。
    return _out(f, rank(tsmax(delta(VW, 3), 5)) * -1)


# ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
def gtja191_042(data, **params):
    f = _fields(data)
    H, V = f.H, f.V
    return _out(f, (-1 * rank(std(H, 10))) * corr(H, V, 10))


# SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
def gtja191_043(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(
        f,
        ts_sum(iff(gt(C, delay(C, 1)), V, iff(lt(C, delay(C, 1)), -V, 0)), 6),
    )


# (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))
def gtja191_044(data, **params):
    f = _fields(data)
    L, V, VW = f.L, f.V, f.VW
    return _out(
        f,
        tsrank(decaylinear(corr(L, mean(V, 10), 7), 6), 4)
        + tsrank(decaylinear(delta(VW, 3), 10), 15),
    )


# (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
def gtja191_045(data, **params):
    f = _fields(data)
    C, OP, V, VW = f.C, f.OP, f.V, f.VW
    return _out(
        f, rank(delta(C * 0.6 + OP * 0.4, 1)) * rank(corr(VW, mean(V, 150), 15))
    )


# (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
def gtja191_046(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(mean(C, 3) + mean(C, 6) + mean(C, 12) + mean(C, 24), 4 * C))


# SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
def gtja191_047(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(f, sma(div(tsmax(H, 6) - C, tsmax(H, 6) - tsmin(L, 6)) * 100, 9, 1))


# (-1*((RANK(((SIGN((CLOSE -DELAY(CLOSE,1))) + SIGN((DELAY(CLOSE, 1) -DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))
def gtja191_048(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    s = (sign(C - delay(C, 1)) + sign(delay(C, 1) - delay(C, 2))) + sign(
        delay(C, 2) - delay(C, 3)
    )
    return _out(f, div(-1 * (rank(s) * ts_sum(V, 5)), ts_sum(V, 20)))


# SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
def gtja191_049(data, **params):
    f = _fields(data)
    H, L = f.H, f.L
    m = emax(absv(H - delay(H, 1)), absv(L - delay(L, 1)))
    a = ts_sum(iff(ge(H + L, delay(H, 1) + delay(L, 1)), 0, m), 12)
    b = ts_sum(iff(le(H + L, delay(H, 1) + delay(L, 1)), 0, m), 12)
    return _out(f, div(a, a + b))


# SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
def gtja191_050(data, **params):
    f = _fields(data)
    H, L = f.H, f.L
    m = emax(absv(H - delay(H, 1)), absv(L - delay(L, 1)))
    a = ts_sum(iff(ge(H + L, delay(H, 1) + delay(L, 1)), 0, m), 12)
    b = ts_sum(iff(le(H + L, delay(H, 1) + delay(L, 1)), 0, m), 12)
    return _out(f, div(b, b + a) - div(a, a + b))


# SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
def gtja191_051(data, **params):
    f = _fields(data)
    H, L = f.H, f.L
    m = emax(absv(H - delay(H, 1)), absv(L - delay(L, 1)))
    a = ts_sum(iff(ge(H + L, delay(H, 1) + delay(L, 1)), 0, m), 12)
    b = ts_sum(iff(le(H + L, delay(H, 1) + delay(L, 1)), 0, m), 12)
    return _out(f, div(b, b + a))


# SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100
def gtja191_052(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    # Q-G-052：`L` 按 LOW；MAX(0, x) 是逐元素。
    tp = div(H + L + C, 3)
    return _out(
        f,
        div(
            ts_sum(emax(0, H - delay(tp, 1)), 26),
            ts_sum(emax(0, delay(tp, 1) - L), 26),
        )
        * 100,
    )


# COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
def gtja191_053(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(count(gt(C, delay(C, 1)), 12), 12) * 100)


# (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
def gtja191_054(data, **params):
    f = _fields(data)
    C, OP = f.C, f.OP
    # Q-G-054：STD 补参数 n = 5。
    return _out(f, -1 * rank((std(absv(C - OP), 5) + (C - OP)) + corr(C, OP, 10)))


# SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
def gtja191_055(data, **params):
    f = _fields(data)
    C, OP, H, L = f.C, f.OP, f.H, f.L
    a = absv(H - delay(C, 1))
    b = absv(L - delay(C, 1))
    c = absv(H - delay(L, 1))
    e = absv(delay(C, 1) - delay(OP, 1))
    x = iff(
        and_(gt(a, b), gt(a, c)),
        a + div(b, 2) + div(e, 4),
        iff(and_(gt(b, c), gt(b, a)), b + div(a, 2) + div(e, 4), c + div(e, 4)),
    )
    num = 16 * (C - delay(C, 1) + div(C - OP, 2) + delay(C, 1) - delay(OP, 1))
    return _out(f, ts_sum(div(num, x) * emax(a, b), 20))


# (RANK((OPEN -TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH+ LOW) / 2), 19), SUM(MEAN(VOLUME,40), 19), 13))^5)))
def gtja191_056(data, **params):
    f = _fields(data)
    OP, H, L, V = f.OP, f.H, f.L, f.V
    return _out(
        f,
        lt(
            rank(OP - tsmin(OP, 12)),
            rank(
                pw(
                    rank(corr(ts_sum(div(H + L, 2), 19), ts_sum(mean(V, 40), 19), 13)),
                    5,
                )
            ),
        ),
    )


# SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
def gtja191_057(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(f, sma(div(C - tsmin(L, 9), tsmax(H, 9) - tsmin(L, 9)) * 100, 3, 1))


# COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
def gtja191_058(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(count(gt(C, delay(C, 1)), 20), 20) * 100)


# SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
def gtja191_059(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(
        f,
        ts_sum(
            iff(
                eq(C, delay(C, 1)),
                0,
                C - iff(gt(C, delay(C, 1)), emin(L, delay(C, 1)), emax(H, delay(C, 1))),
            ),
            20,
        ),
    )


# SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)
def gtja191_060(data, **params):
    f = _fields(data)
    C, H, L, V = f.C, f.H, f.L, f.V
    # Q-G-060：./ 与 .* 按 / 与 *。
    return _out(f, ts_sum(div((C - L) - (H - C), H - L) * V, 20))


# (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)), RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)
def gtja191_061(data, **params):
    f = _fields(data)
    L, V, VW = f.L, f.V, f.VW
    return _out(
        f,
        emax(
            rank(decaylinear(delta(VW, 1), 12)),
            rank(decaylinear(rank(corr(L, mean(V, 80), 8)), 17)),
        )
        * -1,
    )


# (-1 * CORR(HIGH, RANK(VOLUME), 5))
def gtja191_062(data, **params):
    f = _fields(data)
    H, V = f.H, f.V
    return _out(f, -1 * corr(H, rank(V), 5))


# SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
def gtja191_063(data, **params):
    f = _fields(data)
    C = f.C
    return _out(
        f,
        div(sma(emax(C - delay(C, 1), 0), 6, 1), sma(absv(C - delay(C, 1)), 6, 1))
        * 100,
    )


# (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)
def gtja191_064(data, **params):
    f = _fields(data)
    C, V, VW = f.C, f.V, f.VW
    # Q-G-02：内层 MAX(CORR(...), 13) 按 TSMAX；外层两个序列逐元素。
    return _out(
        f,
        emax(
            rank(decaylinear(corr(rank(VW), rank(V), 4), 4)),
            rank(decaylinear(tsmax(corr(rank(C), rank(mean(V, 60)), 4), 13), 14)),
        )
        * -1,
    )


# MEAN(CLOSE,6)/CLOSE
def gtja191_065(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(mean(C, 6), C))


# (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
def gtja191_066(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(C - mean(C, 6), mean(C, 6)) * 100)


# SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
def gtja191_067(data, **params):
    f = _fields(data)
    C = f.C
    return _out(
        f,
        div(
            sma(emax(C - delay(C, 1), 0), 24, 1),
            sma(absv(C - delay(C, 1)), 24, 1),
        )
        * 100,
    )


# SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
def gtja191_068(data, **params):
    f = _fields(data)
    H, L, V = f.H, f.L, f.V
    x = div((div(H + L, 2) - div(delay(H, 1) + delay(L, 1), 2)) * (H - L), V)
    return _out(f, sma(x, 15, 2))


# (SUM(DTM,20)>SUM(DBM,20)？(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)：(SUM(DTM,20)=SUM(DBM,20)？0：(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
def gtja191_069(data, **params):
    f = _fields(data)
    DTM, DBM = f.DTM, f.DBM
    # Q-G-069：全角 ？：按 ?:；单个 `=` 按相等比较。
    s1 = ts_sum(DTM, 20)
    s2 = ts_sum(DBM, 20)
    return _out(
        f,
        iff(gt(s1, s2), div(s1 - s2, s1), iff(eq(s1, s2), 0, div(s1 - s2, s2))),
    )


# STD(AMOUNT,6)
def gtja191_070(data, **params):
    f = _fields(data)
    AMT = f.AMT
    return _out(f, std(AMT, 6))


# (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
def gtja191_071(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(C - mean(C, 24), mean(C, 24)) * 100)


# SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
def gtja191_072(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(f, sma(div(tsmax(H, 6) - C, tsmax(H, 6) - tsmin(L, 6)) * 100, 15, 1))


# ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) -RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)
def gtja191_073(data, **params):
    f = _fields(data)
    C, V, VW = f.C, f.V, f.VW
    return _out(
        f,
        (
            tsrank(decaylinear(decaylinear(corr(C, V, 10), 16), 4), 5)
            - rank(decaylinear(corr(VW, mean(V, 30), 4), 3))
        )
        * -1,
    )


# (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
def gtja191_074(data, **params):
    f = _fields(data)
    L, V, VW = f.L, f.V, f.VW
    return _out(
        f,
        rank(corr(ts_sum(L * 0.35 + VW * 0.65, 20), ts_sum(mean(V, 40), 20), 7))
        + rank(corr(rank(VW), rank(V), 6)),
    )


# COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)
def gtja191_075(data, **params):
    f = _fields(data)
    C, OP, BO, BC = f.C, f.OP, f.BO, f.BC
    return _out(f, div(count(and_(gt(C, OP), lt(BC, BO)), 50), count(lt(BC, BO), 50)))


# STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
def gtja191_076(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    x = div(absv(div(C, delay(C, 1)) - 1), V)
    return _out(f, div(std(x, 20), mean(x, 20)))


# MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) -(VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
def gtja191_077(data, **params):
    f = _fields(data)
    H, L, V, VW = f.H, f.L, f.V, f.VW
    return _out(
        f,
        emin(
            rank(decaylinear((div(H + L, 2) + H) - (VW + H), 20)),
            rank(decaylinear(corr(div(H + L, 2), mean(V, 40), 3), 6)),
        ),
    )


# ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
def gtja191_078(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    # Q-G-078：MA 按 MEAN。
    tp = div(H + L + C, 3)
    return _out(f, div(tp - mean(tp, 12), 0.015 * mean(absv(C - mean(tp, 12)), 12)))


# SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
def gtja191_079(data, **params):
    f = _fields(data)
    C = f.C
    return _out(
        f,
        div(
            sma(emax(C - delay(C, 1), 0), 12, 1),
            sma(absv(C - delay(C, 1)), 12, 1),
        )
        * 100,
    )


# (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
def gtja191_080(data, **params):
    f = _fields(data)
    V = f.V
    return _out(f, div(V - delay(V, 5), delay(V, 5)) * 100)


# SMA(VOLUME,21,2)
def gtja191_081(data, **params):
    f = _fields(data)
    V = f.V
    return _out(f, sma(V, 21, 2))


# SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
def gtja191_082(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(f, sma(div(tsmax(H, 6) - C, tsmax(H, 6) - tsmin(L, 6)) * 100, 20, 1))


# (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
def gtja191_083(data, **params):
    f = _fields(data)
    H, V = f.H, f.V
    return _out(f, -1 * rank(cov(rank(H), rank(V), 5)))


# SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
def gtja191_084(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(
        f,
        ts_sum(iff(gt(C, delay(C, 1)), V, iff(lt(C, delay(C, 1)), -V, 0)), 20),
    )


# (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
def gtja191_085(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(f, tsrank(div(V, mean(V, 20)), 20) * tsrank(-1 * delta(C, 7), 8))


# ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) : (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))
def gtja191_086(data, **params):
    f = _fields(data)
    C = f.C
    z = div(delay(C, 20) - delay(C, 10), 10) - div(delay(C, 10) - C, 10)
    return _out(
        f,
        iff(lt(0.25, z), -1 * 1, iff(lt(z, 0), 1, (-1 * 1) * (C - delay(C, 1)))),
    )


# ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)
def gtja191_087(data, **params):
    f = _fields(data)
    OP, H, L, VW = f.OP, f.H, f.L, f.VW
    # 照抄：(LOW * 0.9) + (LOW * 0.1) 两项是同一个字段，不化简。
    return _out(
        f,
        (
            rank(decaylinear(delta(VW, 4), 7))
            + tsrank(
                decaylinear(div((L * 0.9 + L * 0.1) - VW, OP - div(H + L, 2)), 11), 7
            )
        )
        * -1,
    )


# (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
def gtja191_088(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(C - delay(C, 20), delay(C, 20)) * 100)


# 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
def gtja191_089(data, **params):
    f = _fields(data)
    C = f.C
    return _out(
        f,
        2 * (sma(C, 13, 2) - sma(C, 27, 2) - sma(sma(C, 13, 2) - sma(C, 27, 2), 10, 2)),
    )


# ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
def gtja191_090(data, **params):
    f = _fields(data)
    V, VW = f.V, f.VW
    return _out(f, rank(corr(rank(VW), rank(V), 5)) * -1)


# ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
def gtja191_091(data, **params):
    f = _fields(data)
    C, L, V = f.C, f.L, f.V
    # Q-G-02：MAX(CLOSE, 5) 按 TSMAX。
    return _out(f, (rank(C - tsmax(C, 5)) * rank(corr(mean(V, 40), L, 5))) * -1)


# (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)), TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)
def gtja191_092(data, **params):
    f = _fields(data)
    C, V, VW = f.C, f.V, f.VW
    return _out(
        f,
        emax(
            rank(decaylinear(delta(C * 0.35 + VW * 0.65, 2), 3)),
            tsrank(decaylinear(absv(corr(mean(V, 180), C, 13)), 5), 15),
        )
        * -1,
    )


# SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
def gtja191_093(data, **params):
    f = _fields(data)
    OP, L = f.OP, f.L
    return _out(
        f,
        ts_sum(iff(ge(OP, delay(OP, 1)), 0, emax(OP - L, OP - delay(OP, 1))), 20),
    )


# SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
def gtja191_094(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(
        f,
        ts_sum(iff(gt(C, delay(C, 1)), V, iff(lt(C, delay(C, 1)), -V, 0)), 30),
    )


# STD(AMOUNT,20)
def gtja191_095(data, **params):
    f = _fields(data)
    AMT = f.AMT
    return _out(f, std(AMT, 20))


# SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
def gtja191_096(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(
        f,
        sma(sma(div(C - tsmin(L, 9), tsmax(H, 9) - tsmin(L, 9)) * 100, 3, 1), 3, 1),
    )


# STD(VOLUME,10)
def gtja191_097(data, **params):
    f = _fields(data)
    V = f.V
    return _out(f, std(V, 10))


# ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
def gtja191_098(data, **params):
    f = _fields(data)
    C = f.C
    x = div(delta(div(ts_sum(C, 100), 100), 100), delay(C, 100))
    return _out(
        f,
        iff(
            or_(lt(x, 0.05), eq(x, 0.05)),
            -1 * (C - tsmin(C, 100)),
            -1 * delta(C, 3),
        ),
    )


# (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
def gtja191_099(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(f, -1 * rank(cov(rank(C), rank(V), 5)))


# STD(VOLUME,20)
def gtja191_100(data, **params):
    f = _fields(data)
    V = f.V
    return _out(f, std(V, 20))


# ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))), RANK(VOLUME), 11))) * -1)
def gtja191_101(data, **params):
    f = _fields(data)
    C, H, V, VW = f.C, f.H, f.V, f.VW
    return _out(
        f,
        lt(
            rank(corr(C, ts_sum(mean(V, 30), 37), 15)),
            rank(corr(rank(H * 0.1 + VW * 0.9), rank(V), 11)),
        )
        * -1,
    )


# SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
def gtja191_102(data, **params):
    f = _fields(data)
    V = f.V
    return _out(
        f,
        div(sma(emax(V - delay(V, 1), 0), 6, 1), sma(absv(V - delay(V, 1)), 6, 1))
        * 100,
    )


# ((20-LOWDAY(LOW,20))/20)*100
def gtja191_103(data, **params):
    f = _fields(data)
    L = f.L
    return _out(f, div(20 - lowday(L, 20), 20) * 100)


# (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
def gtja191_104(data, **params):
    f = _fields(data)
    C, H, V = f.C, f.H, f.V
    return _out(f, -1 * (delta(corr(H, V, 5), 5) * rank(std(C, 20))))


# (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
def gtja191_105(data, **params):
    f = _fields(data)
    OP, V = f.OP, f.V
    return _out(f, -1 * corr(rank(OP), rank(V), 10))


# CLOSE-DELAY(CLOSE,20)
def gtja191_106(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, C - delay(C, 20))


# (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))
def gtja191_107(data, **params):
    f = _fields(data)
    C, OP, H, L = f.C, f.OP, f.H, f.L
    return _out(
        f,
        ((-1 * rank(OP - delay(H, 1))) * rank(OP - delay(C, 1)))
        * rank(OP - delay(L, 1)),
    )


# ((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)
def gtja191_108(data, **params):
    f = _fields(data)
    H, V, VW = f.H, f.V, f.VW
    # Q-G-02：MIN(HIGH, 2) 按 TSMIN。
    return _out(f, pw(rank(H - tsmin(H, 2)), rank(corr(VW, mean(V, 120), 6))) * -1)


# SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
def gtja191_109(data, **params):
    f = _fields(data)
    H, L = f.H, f.L
    return _out(f, div(sma(H - L, 10, 2), sma(sma(H - L, 10, 2), 10, 2)))


# SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
def gtja191_110(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(
        f,
        div(
            ts_sum(emax(0, H - delay(C, 1)), 20),
            ts_sum(emax(0, delay(C, 1) - L), 20),
        )
        * 100,
    )


# SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
def gtja191_111(data, **params):
    f = _fields(data)
    C, H, L, V = f.C, f.H, f.L, f.V
    # Q-G-111：VOL 按 VOLUME。
    x = div(V * ((C - L) - (H - C)), H - L)
    return _out(f, sma(x, 11, 2) - sma(x, 4, 2))


# (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
def gtja191_112(data, **params):
    f = _fields(data)
    C = f.C
    up = ts_sum(iff(gt(C - delay(C, 1), 0), C - delay(C, 1), 0), 12)
    dn = ts_sum(iff(lt(C - delay(C, 1), 0), absv(C - delay(C, 1)), 0), 12)
    return _out(f, div(up - dn, up + dn) * 100)


# (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))
def gtja191_113(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(
        f,
        -1
        * (
            (rank(div(ts_sum(delay(C, 5), 20), 20)) * corr(C, V, 2))
            * rank(corr(ts_sum(C, 5), ts_sum(C, 20), 2))
        ),
    )


# ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
def gtja191_114(data, **params):
    f = _fields(data)
    C, H, L, V, VW = f.C, f.H, f.L, f.V, f.VW
    x = div(H - L, div(ts_sum(C, 5), 5))
    return _out(f, div(rank(delay(x, 2)) * rank(rank(V)), div(x, VW - C)))


# (RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) / 2), 4), TSRANK(VOLUME, 10), 7)))
def gtja191_115(data, **params):
    f = _fields(data)
    C, H, L, V = f.C, f.H, f.L, f.V
    return _out(
        f,
        pw(
            rank(corr(H * 0.9 + C * 0.1, mean(V, 30), 10)),
            rank(corr(tsrank(div(H + L, 2), 4), tsrank(V, 10), 7)),
        ),
    )


# REGBETA(CLOSE,SEQUENCE,20)
def gtja191_116(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-116：SEQUENCE 按 SEQUENCE(20)。
    return _out(f, regbeta_seq(C, 20))


# ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
def gtja191_117(data, **params):
    f = _fields(data)
    C, H, L, V, RET = f.C, f.H, f.L, f.V, f.RET
    return _out(
        f,
        (tsrank(V, 32) * (1 - tsrank((C + H) - L, 16))) * (1 - tsrank(RET, 32)),
    )


# SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
def gtja191_118(data, **params):
    f = _fields(data)
    OP, H, L = f.OP, f.H, f.L
    return _out(f, div(ts_sum(H - OP, 20), ts_sum(OP - L, 20)) * 100)


# (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) -RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
def gtja191_119(data, **params):
    f = _fields(data)
    OP, V, VW = f.OP, f.V, f.VW
    # Q-G-02：MIN(CORR(...), 9) 按 TSMIN。
    return _out(
        f,
        rank(decaylinear(corr(VW, ts_sum(mean(V, 5), 26), 5), 7))
        - rank(
            decaylinear(tsrank(tsmin(corr(rank(OP), rank(mean(V, 15)), 21), 9), 7), 8)
        ),
    )


# (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
def gtja191_120(data, **params):
    f = _fields(data)
    C, VW = f.C, f.VW
    return _out(f, div(rank(VW - C), rank(VW + C)))


# ((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) * -1)
def gtja191_121(data, **params):
    f = _fields(data)
    V, VW = f.V, f.VW
    # Q-G-02：MIN(VWAP, 12) 按 TSMIN。
    return _out(
        f,
        pw(
            rank(VW - tsmin(VW, 12)),
            tsrank(corr(tsrank(VW, 20), tsrank(mean(V, 60), 2), 18), 3),
        )
        * -1,
    )


# (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
def gtja191_122(data, **params):
    f = _fields(data)
    C = f.C
    s = sma(sma(sma(log(C), 13, 2), 13, 2), 13, 2)
    return _out(f, div(s - delay(s, 1), delay(s, 1)))


# ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
def gtja191_123(data, **params):
    f = _fields(data)
    H, L, V = f.H, f.L, f.V
    return _out(
        f,
        lt(
            rank(corr(ts_sum(div(H + L, 2), 20), ts_sum(mean(V, 60), 20), 9)),
            rank(corr(L, V, 6)),
        )
        * -1,
    )


# (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
def gtja191_124(data, **params):
    f = _fields(data)
    C, VW = f.C, f.VW
    # Q-G-124：照抄研报的 TSMAX。
    return _out(f, div(C - VW, decaylinear(rank(tsmax(C, 30)), 2)))


# (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))
def gtja191_125(data, **params):
    f = _fields(data)
    C, V, VW = f.C, f.V, f.VW
    return _out(
        f,
        div(
            rank(decaylinear(corr(VW, mean(V, 80), 17), 20)),
            rank(decaylinear(delta(C * 0.5 + VW * 0.5, 3), 16)),
        ),
    )


# (CLOSE+HIGH+LOW)/3
def gtja191_126(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(f, div(C + H + L, 3))


# (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
def gtja191_127(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-127：MEAN 补参数 n = 12；MAX(CLOSE, 12) 按 TSMAX。
    return _out(
        f,
        pw(
            mean(pw(div(100 * (C - tsmax(C, 12)), tsmax(C, 12)), 2), 12),
            div(1, 2),
        ),
    )


# 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
def gtja191_128(data, **params):
    f = _fields(data)
    C, H, L, V = f.C, f.H, f.L, f.V
    tp = div(H + L + C, 3)
    return _out(
        f,
        100
        - div(
            100,
            1
            + div(
                ts_sum(iff(gt(tp, delay(tp, 1)), tp * V, 0), 14),
                ts_sum(iff(lt(tp, delay(tp, 1)), tp * V, 0), 14),
            ),
        ),
    )


# SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
def gtja191_129(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, ts_sum(iff(lt(C - delay(C, 1), 0), absv(C - delay(C, 1)), 0), 12))


# (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))
def gtja191_130(data, **params):
    f = _fields(data)
    H, L, V, VW = f.H, f.L, f.V, f.VW
    return _out(
        f,
        div(
            rank(decaylinear(corr(div(H + L, 2), mean(V, 40), 9), 10)),
            rank(decaylinear(corr(rank(VW), rank(V), 7), 3)),
        ),
    )


# (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
def gtja191_131(data, **params):
    f = _fields(data)
    C, V, VW = f.C, f.V, f.VW
    # Q-G-131：DELAT 按 DELTA。
    return _out(f, pw(rank(delta(VW, 1)), tsrank(corr(C, mean(V, 50), 18), 18)))


# MEAN(AMOUNT,20)
def gtja191_132(data, **params):
    f = _fields(data)
    AMT = f.AMT
    return _out(f, mean(AMT, 20))


# ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
def gtja191_133(data, **params):
    f = _fields(data)
    H, L = f.H, f.L
    return _out(
        f,
        div(20 - highday(H, 20), 20) * 100 - div(20 - lowday(L, 20), 20) * 100,
    )


# (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
def gtja191_134(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(f, div(C - delay(C, 12), delay(C, 12)) * V)


# SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
def gtja191_135(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-135：以研报原文为准（聚宽版本开头多一个负号）。
    return _out(f, sma(delay(div(C, delay(C, 20)), 1), 20, 1))


# ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
def gtja191_136(data, **params):
    f = _fields(data)
    OP, V, RET = f.OP, f.V, f.RET
    return _out(f, (-1 * rank(delta(RET, 3))) * corr(OP, V, 10))


# 16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
def gtja191_137(data, **params):
    f = _fields(data)
    C, OP, H, L = f.C, f.OP, f.H, f.L
    a = absv(H - delay(C, 1))
    b = absv(L - delay(C, 1))
    c = absv(H - delay(L, 1))
    e = absv(delay(C, 1) - delay(OP, 1))
    x = iff(
        and_(gt(a, b), gt(a, c)),
        a + div(b, 2) + div(e, 4),
        iff(and_(gt(b, c), gt(b, a)), b + div(a, 2) + div(e, 4), c + div(e, 4)),
    )
    num = 16 * (C - delay(C, 1) + div(C - OP, 2) + delay(C, 1) - delay(OP, 1))
    return _out(f, div(num, x) * emax(a, b))


# ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) -TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)
def gtja191_138(data, **params):
    f = _fields(data)
    L, V, VW = f.L, f.V, f.VW
    return _out(
        f,
        (
            rank(decaylinear(delta(L * 0.7 + VW * 0.3, 3), 20))
            - tsrank(
                decaylinear(
                    tsrank(corr(tsrank(L, 8), tsrank(mean(V, 60), 17), 5), 19), 16
                ),
                7,
            )
        )
        * -1,
    )


# (-1 * CORR(OPEN, VOLUME, 10))
def gtja191_139(data, **params):
    f = _fields(data)
    OP, V = f.OP, f.V
    return _out(f, -1 * corr(OP, V, 10))


# MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) -(RANK(HIGH) +RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))
def gtja191_140(data, **params):
    f = _fields(data)
    C, OP, H, L, V = f.C, f.OP, f.H, f.L, f.V
    return _out(
        f,
        emin(
            rank(decaylinear((rank(OP) + rank(L)) - (rank(H) + rank(C)), 8)),
            tsrank(decaylinear(corr(tsrank(C, 8), tsrank(mean(V, 60), 20), 8), 7), 3),
        ),
    )


# (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)
def gtja191_141(data, **params):
    f = _fields(data)
    H, V = f.H, f.V
    return _out(f, rank(corr(rank(H), rank(mean(V, 15)), 9)) * -1)


# (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME /MEAN(VOLUME,20)), 5)))
def gtja191_142(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(
        f,
        ((-1 * rank(tsrank(C, 10))) * rank(delta(delta(C, 1), 1)))
        * rank(tsrank(div(V, mean(V, 20)), 5)),
    )


# SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
def gtja191_144(data, **params):
    f = _fields(data)
    C, AMT = f.C, f.AMT
    return _out(
        f,
        div(
            sumif(div(absv(div(C, delay(C, 1)) - 1), AMT), 20, lt(C, delay(C, 1))),
            count(lt(C, delay(C, 1)), 20),
        ),
    )


# (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
def gtja191_145(data, **params):
    f = _fields(data)
    V = f.V
    return _out(f, div(mean(V, 9) - mean(V, 26), mean(V, 12)) * 100)


# MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60);
def gtja191_146(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-146：末尾分号删去；最后一个 SMA 补参数 m = 2。
    r = div(C - delay(C, 1), delay(C, 1))
    x = r - sma(div(C - delay(C, 1), delay(C, 1)), 61, 2)
    return _out(f, div(mean(x, 20) * x, sma(pw(r - x, 2), 60, 2)))


# REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
def gtja191_147(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-147：REGBETA 的窗口取 SEQUENCE(12) 的 12。
    return _out(f, regbeta_seq(mean(C, 12), 12))


# ((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
def gtja191_148(data, **params):
    f = _fields(data)
    OP, V = f.OP, f.V
    return _out(
        f,
        lt(rank(corr(OP, ts_sum(mean(V, 60), 9), 6)), rank(OP - tsmin(OP, 14))) * -1,
    )


# REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)
def gtja191_149(data, **params):
    f = _fields(data)
    C, BC = f.C, f.BC
    # Q-G-149：FILTER 掉的日子为 NaN；252 行窗口内有效配对不少于 20 对才出数，
    # 面板开头不足 252 行时用已有的行。R-G-02：股票自己当天的收益
    # （公式里的 CLOSE/DELAY(CLOSE,1)-1，筛选前）为 NaN 时输出 NaN。
    cond = lt(BC, delay(BC, 1))
    r = div(C, delay(C, 1)) - 1
    y = filter_(r, cond)
    x = filter_(div(BC, delay(BC, 1)) - 1, cond)
    beta = regbeta_pairs(y, x, 252, min_pairs=20)
    return _out(f, np.where(np.isnan(r), np.nan, beta))


# (CLOSE+HIGH+LOW)/3*VOLUME
def gtja191_150(data, **params):
    f = _fields(data)
    C, H, L, V = f.C, f.H, f.L, f.V
    return _out(f, div(C + H + L, 3) * V)


# SMA(CLOSE-DELAY(CLOSE,20),20,1)
def gtja191_151(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, sma(C - delay(C, 20), 20, 1))


# SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
def gtja191_152(data, **params):
    f = _fields(data)
    C = f.C
    a = delay(sma(delay(div(C, delay(C, 9)), 1), 9, 1), 1)
    return _out(f, sma(mean(a, 12) - mean(a, 26), 9, 1))


# (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
def gtja191_153(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, div(mean(C, 3) + mean(C, 6) + mean(C, 12) + mean(C, 24), 4))


# (((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))
def gtja191_154(data, **params):
    f = _fields(data)
    V, VW = f.V, f.VW
    # Q-G-02：MIN(VWAP, 16) 按 TSMIN。
    return _out(f, lt(VW - tsmin(VW, 16), corr(VW, mean(V, 180), 18)))


# SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
def gtja191_155(data, **params):
    f = _fields(data)
    V = f.V
    return _out(
        f,
        sma(V, 13, 2) - sma(V, 27, 2) - sma(sma(V, 13, 2) - sma(V, 27, 2), 10, 2),
    )


# (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)
def gtja191_156(data, **params):
    f = _fields(data)
    OP, L, VW = f.OP, f.L, f.VW
    return _out(
        f,
        emax(
            rank(decaylinear(delta(VW, 5), 3)),
            rank(
                decaylinear(
                    div(delta(OP * 0.15 + L * 0.85, 2), OP * 0.15 + L * 0.85) * -1, 3
                )
            ),
        )
        * -1,
    )


# (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))
def gtja191_157(data, **params):
    f = _fields(data)
    C, RET = f.C, f.RET
    # Q-G-02：MIN(PROD(...), 5) 按 TSMIN。
    return _out(
        f,
        tsmin(
            ts_prod(
                rank(
                    rank(
                        log(ts_sum(tsmin(rank(rank(-1 * rank(delta(C - 1, 5)))), 2), 1))
                    )
                ),
                1,
            ),
            5,
        )
        + tsrank(delay(-1 * RET, 6), 5),
    )


# ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
def gtja191_158(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(f, div((H - sma(C, 15, 2)) - (L - sma(C, 15, 2)), C))


# ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
def gtja191_159(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    # Q-G-159：HGIH 按 HIGH。
    lo = emin(L, delay(C, 1))
    hi = emax(H, delay(C, 1))
    t6 = div(C - ts_sum(lo, 6), ts_sum(hi - lo, 6))
    t12 = div(C - ts_sum(lo, 12), ts_sum(hi - lo, 12))
    t24 = div(C - ts_sum(lo, 24), ts_sum(hi - lo, 24))
    return _out(
        f,
        div(
            (t6 * 12 * 24 + t12 * 6 * 24 + t24 * 6 * 24) * 100,
            6 * 12 + 6 * 24 + 12 * 24,
        ),
    )


# SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
def gtja191_160(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, sma(iff(le(C, delay(C, 1)), std(C, 20), 0), 20, 1))


# MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
def gtja191_161(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(
        f,
        mean(emax(emax(H - L, absv(delay(C, 1) - H)), absv(delay(C, 1) - L)), 12),
    )


# (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
def gtja191_162(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-02：MAX/MIN(x, 12) 按 TSMAX/TSMIN；MAX(x, 0) 是逐元素。
    r = (
        div(
            sma(emax(C - delay(C, 1), 0), 12, 1),
            sma(absv(C - delay(C, 1)), 12, 1),
        )
        * 100
    )
    return _out(f, div(r - tsmin(r, 12), tsmax(r, 12) - tsmin(r, 12)))


# RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
def gtja191_163(data, **params):
    f = _fields(data)
    C, H, V, VW, RET = f.C, f.H, f.V, f.VW, f.RET
    return _out(f, rank((((-1 * RET) * mean(V, 20)) * VW) * (H - C)))


# SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)
def gtja191_164(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    # Q-G-02：MIN(x, 12) 按 TSMIN。
    x = iff(gt(C, delay(C, 1)), div(1, C - delay(C, 1)), 1)
    return _out(f, sma(div(x - tsmin(x, 12), H - L) * 100, 13, 2))


# SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
def gtja191_167(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-167：以研报原文为准。
    return _out(f, ts_sum(iff(gt(C - delay(C, 1), 0), C - delay(C, 1), 0), 12))


# (-1*VOLUME/MEAN(VOLUME,20))
def gtja191_168(data, **params):
    f = _fields(data)
    V = f.V
    return _out(f, div(-1 * V, mean(V, 20)))


# SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)
def gtja191_169(data, **params):
    f = _fields(data)
    C = f.C
    a = delay(sma(C - delay(C, 1), 9, 1), 1)
    return _out(f, sma(mean(a, 12) - mean(a, 26), 10, 1))


# ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))
def gtja191_170(data, **params):
    f = _fields(data)
    C, H, V, VW = f.C, f.H, f.V, f.VW
    return _out(
        f,
        (
            div(rank(div(1, C)) * V, mean(V, 20))
            * div(H * rank(H - C), div(ts_sum(H, 5), 5))
        )
        - rank(VW - delay(VW, 5)),
    )


# ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
def gtja191_171(data, **params):
    f = _fields(data)
    C, OP, H, L = f.C, f.OP, f.H, f.L
    return _out(f, div(-1 * ((L - C) * pw(OP, 5)), (C - H) * pw(C, 5)))


# MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
def gtja191_172(data, **params):
    f = _fields(data)
    TR, HD, LD = f.TR, f.HD, f.LD
    a = div(ts_sum(iff(and_(gt(LD, 0), gt(LD, HD)), LD, 0), 14) * 100, ts_sum(TR, 14))
    b = div(ts_sum(iff(and_(gt(HD, 0), gt(HD, LD)), HD, 0), 14) * 100, ts_sum(TR, 14))
    return _out(f, mean(div(absv(a - b), a + b) * 100, 6))


# 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2);
def gtja191_173(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-173：末尾分号删去。
    return _out(
        f,
        3 * sma(C, 13, 2)
        - 2 * sma(sma(C, 13, 2), 13, 2)
        + sma(sma(sma(log(C), 13, 2), 13, 2), 13, 2),
    )


# SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
def gtja191_174(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, sma(iff(gt(C, delay(C, 1)), std(C, 20), 0), 20, 1))


# MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
def gtja191_175(data, **params):
    f = _fields(data)
    C, H, L = f.C, f.H, f.L
    return _out(
        f,
        mean(emax(emax(H - L, absv(delay(C, 1) - H)), absv(delay(C, 1) - L)), 6),
    )


# CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
def gtja191_176(data, **params):
    f = _fields(data)
    C, H, L, V = f.C, f.H, f.L, f.V
    return _out(
        f,
        corr(rank(div(C - tsmin(L, 12), tsmax(H, 12) - tsmin(L, 12))), rank(V), 6),
    )


# ((20-HIGHDAY(HIGH,20))/20)*100
def gtja191_177(data, **params):
    f = _fields(data)
    H = f.H
    return _out(f, div(20 - highday(H, 20), 20) * 100)


# (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
def gtja191_178(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    return _out(f, div(C - delay(C, 1), delay(C, 1)) * V)


# (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
def gtja191_179(data, **params):
    f = _fields(data)
    L, V, VW = f.L, f.V, f.VW
    return _out(f, rank(corr(VW, V, 4)) * rank(corr(rank(L), rank(mean(V, 50)), 12)))


# ((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 * VOLUME)))
def gtja191_180(data, **params):
    f = _fields(data)
    C, V = f.C, f.V
    # 新疑点 Q-G-180：`? ((...) * SIGN(...) : (...)))` 的冒号落在问号后面那对括号里，
    # 括号错位；唯一能成式的读法是 cond ? (-1*TSRANK(...))*SIGN(...) : (-1*VOLUME)。
    return _out(
        f,
        iff(
            lt(mean(V, 20), V),
            (-1 * tsrank(absv(delta(C, 7)), 60)) * sign(delta(C, 7)),
            -1 * V,
        ),
    )


# SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)
def gtja191_181(data, **params):
    f = _fields(data)
    C, BC = f.C, f.BC
    # Q-G-181：第二个 SUM 补参数 n = 20；BANCHMARKINDEXCLOSE 为基准收盘。
    r = div(C, delay(C, 1)) - 1
    num = ts_sum((r - mean(r, 20)) - pw(BC - mean(BC, 20), 2), 20)
    den = ts_sum(pw(BC - mean(BC, 20), 3), 20)
    return _out(f, div(num, den))


# COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20
def gtja191_182(data, **params):
    f = _fields(data)
    C, OP, BO, BC = f.C, f.OP, f.BO, f.BC
    # Q-G-182：OR 按 ||。
    return _out(
        f,
        div(
            count(or_(and_(gt(C, OP), gt(BC, BO)), and_(lt(C, OP), lt(BC, BO))), 20),
            20,
        ),
    )


# (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
def gtja191_184(data, **params):
    f = _fields(data)
    C, OP = f.C, f.OP
    return _out(f, rank(corr(delay(OP - C, 1), C, 200)) + rank(OP - C))


# RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
def gtja191_185(data, **params):
    f = _fields(data)
    C, OP = f.C, f.OP
    return _out(f, rank(-1 * pw(1 - div(OP, C), 2)))


# (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
def gtja191_186(data, **params):
    f = _fields(data)
    TR, HD, LD = f.TR, f.HD, f.LD
    a = div(ts_sum(iff(and_(gt(LD, 0), gt(LD, HD)), LD, 0), 14) * 100, ts_sum(TR, 14))
    b = div(ts_sum(iff(and_(gt(HD, 0), gt(HD, LD)), HD, 0), 14) * 100, ts_sum(TR, 14))
    x = mean(div(absv(a - b), a + b) * 100, 6)
    return _out(f, div(x + delay(x, 6), 2))


# SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
def gtja191_187(data, **params):
    f = _fields(data)
    OP, H = f.OP, f.H
    return _out(
        f,
        ts_sum(iff(le(OP, delay(OP, 1)), 0, emax(H - OP, OP - delay(OP, 1))), 20),
    )


# ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
def gtja191_188(data, **params):
    f = _fields(data)
    H, L = f.H, f.L
    # Q-G-188：en dash 按减号。
    return _out(f, div((H - L) - sma(H - L, 11, 2), sma(H - L, 11, 2)) * 100)


# MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
def gtja191_189(data, **params):
    f = _fields(data)
    C = f.C
    return _out(f, mean(absv(C - mean(C, 6)), 6))


# LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
def gtja191_190(data, **params):
    f = _fields(data)
    C = f.C
    # Q-G-190：DELAY(CLOSE) 按 DELAY(CLOSE,1)。
    # 照抄：第一个 SUMIF 的被加项字面是 CLOSE/DELAY(CLOSE,1)-1-(...)^(1/20)-1（多减一个 1），
    # 第二个是 ...-((...)^(1/20)-1)，两处写法不同，不改；几何平均的窗口写作 DELAY(CLOSE,19)。
    r = div(C, delay(C, 1)) - 1
    g = pw(div(C, delay(C, 19)), div(1, 20)) - 1
    first = div(C, delay(C, 1)) - 1 - pw(div(C, delay(C, 19)), div(1, 20)) - 1
    return _out(
        f,
        log(
            div(
                (count(gt(r, g), 20) - 1) * sumif(pw(first, 2), 20, lt(r, g)),
                count(lt(r, g), 20) * sumif(pw(r - g, 2), 20, gt(r, g)),
            )
        ),
    )


# ((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)
def gtja191_191(data, **params):
    f = _fields(data)
    C, H, L, V = f.C, f.H, f.L, f.V
    return _out(f, (corr(mean(V, 20), L, 5) + div(H + L, 2)) - C)


REFERENCES = {
    name: globals()[name]
    for name in sorted(globals())
    if name.startswith("gtja191_") and name[8:].isdigit()
}


def compute_many(data, names):
    """一次算一批；输入只转换一次。返回 ``{名字: ndarray[T, N]}``。"""
    fields = _fields(data)
    return {name: REFERENCES[name](fields).to_numpy(float) for name in names}
