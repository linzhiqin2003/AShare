"""国泰君安 191 的翻译（第二步逐条填写，规范见 docs/DSL_REFERENCE.md）。

每条一个函数 ``gtja191_001`` … ``gtja191_191``，返回 ``Expr``，文档字符串写研报原文公式。
不收的条目不写函数，理由在 docs/CONVENTIONS.md 第 4 节。
"""

from ashare.factor.formula import dsl as d


def gtja191_001():
    """(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))"""
    return -1 * d.correlation(
        d.rank(d.delta(d.log(d.volume), 1)),
        d.rank((d.close - d.open) / d.open),
        6,
    )


def gtja191_002():
    """(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))"""
    return -1 * d.delta(
        (((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)),
        1,
    )


def gtja191_003():
    """SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)"""
    return d.ts_sum(
        d.where(
            d.eq(d.close, d.delay(d.close, 1)),
            0,
            d.close
            - d.where(
                d.gt(d.close, d.delay(d.close, 1)),
                d.minimum(d.low, d.delay(d.close, 1)),
                d.maximum(d.high, d.delay(d.close, 1)),
            ),
        ),
        6,
    )


def gtja191_004():
    """((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME / MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))"""
    return d.where(
        d.lt(
            (d.ts_sum(d.close, 8) / 8) + d.stddev(d.close, 8),
            d.ts_sum(d.close, 2) / 2,
        ),
        (-1 * 1),
        d.where(
            d.lt(
                d.ts_sum(d.close, 2) / 2,
                (d.ts_sum(d.close, 8) / 8) - d.stddev(d.close, 8),
            ),
            1,
            d.where(
                d.or_(
                    d.lt(1, d.volume / d.ts_mean(d.volume, 20)),
                    d.eq(d.volume / d.ts_mean(d.volume, 20), 1),
                ),
                1,
                (-1 * 1),
            ),
        ),
    )


def gtja191_005():
    """(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))"""
    return -1 * d.ts_max(
        d.correlation(
            d.ts_rank(d.volume, 5),
            d.ts_rank(d.high, 5),
            5,
        ),
        3,
    )


def gtja191_006():
    """(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)"""
    return (
        d.rank(
            d.sign(
                d.delta(
                    ((d.open * 0.85) + (d.high * 0.15)),
                    4,
                )
            )
        )
        * -1
    )


def gtja191_007():
    """((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))"""
    return (
        d.rank(d.ts_max(d.vwap - d.close, 3)) + d.rank(d.ts_min(d.vwap - d.close, 3))
    ) * d.rank(d.delta(d.volume, 3))


def gtja191_008():
    """RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)"""
    return d.rank(
        d.delta(
            ((((d.high + d.low) / 2) * 0.2) + (d.vwap * 0.8)),
            4,
        )
        * -1
    )


def gtja191_009():
    """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)"""
    return d.sma(
        ((d.high + d.low) / 2 - (d.delay(d.high, 1) + d.delay(d.low, 1)) / 2)
        * (d.high - d.low)
        / d.volume,
        7,
        2,
    )


def gtja191_010():
    """(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))"""
    return d.rank(
        d.ts_max(
            d.power(d.where(d.lt(d.returns, 0), d.stddev(d.returns, 20), d.close), 2),
            5,
        )
    )


def gtja191_011():
    """SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)"""
    return d.ts_sum(
        ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low) * d.volume, 6
    )


def gtja191_012():
    """(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))"""
    return d.rank(d.open - d.ts_sum(d.vwap, 10) / 10) * (
        -1 * d.rank(d.abs(d.close - d.vwap))
    )


def gtja191_013():
    """(((HIGH * LOW)^0.5) - VWAP)"""
    return d.power(d.high * d.low, 0.5) - d.vwap


def gtja191_014():
    """CLOSE-DELAY(CLOSE,5)"""
    return d.close - d.delay(d.close, 5)


def gtja191_015():
    """OPEN/DELAY(CLOSE,1)-1"""
    return d.open / d.delay(d.close, 1) - 1


def gtja191_016():
    """(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))"""
    return -1 * d.ts_max(d.rank(d.correlation(d.rank(d.volume), d.rank(d.vwap), 5)), 5)


def gtja191_017():
    """RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)"""
    return d.power(d.rank(d.vwap - d.ts_max(d.vwap, 15)), d.delta(d.close, 5))


def gtja191_018():
    """CLOSE/DELAY(CLOSE,5)"""
    return d.close / d.delay(d.close, 5)


def gtja191_019():
    """(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))"""
    return d.where(
        d.lt(d.close, d.delay(d.close, 5)),
        (d.close - d.delay(d.close, 5)) / d.delay(d.close, 5),
        d.where(
            d.eq(d.close, d.delay(d.close, 5)),
            0,
            (d.close - d.delay(d.close, 5)) / d.close,
        ),
    )


def gtja191_020():
    """(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100"""
    return (d.close - d.delay(d.close, 6)) / d.delay(d.close, 6) * 100


def gtja191_021():
    """REGBETA(MEAN(CLOSE,6),SEQUENCE(6))"""
    return d.ts_slope(d.ts_mean(d.close, 6), 6)


def gtja191_022():
    """SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)"""
    return d.sma(
        (d.close - d.ts_mean(d.close, 6)) / d.ts_mean(d.close, 6)
        - d.delay((d.close - d.ts_mean(d.close, 6)) / d.ts_mean(d.close, 6), 3),
        12,
        1,
    )


def gtja191_023():
    """SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100"""
    return (
        d.sma(
            d.where(
                d.gt(d.close, d.delay(d.close, 1)),
                d.stddev(d.close, 20),
                0,
            ),
            20,
            1,
        )
        / (
            d.sma(
                d.where(
                    d.gt(d.close, d.delay(d.close, 1)),
                    d.stddev(d.close, 20),
                    0,
                ),
                20,
                1,
            )
            + d.sma(
                d.where(
                    d.le(d.close, d.delay(d.close, 1)),
                    d.stddev(d.close, 20),
                    0,
                ),
                20,
                1,
            )
        )
        * 100
    )


def gtja191_024():
    """SMA(CLOSE-DELAY(CLOSE,5),5,1)"""
    return d.sma(d.close - d.delay(d.close, 5), 5, 1)


def gtja191_025():
    """((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))"""
    return (
        -1
        * d.rank(
            d.delta(d.close, 7)
            * (1 - d.rank(d.decay_linear(d.volume / d.ts_mean(d.volume, 20), 9)))
        )
    ) * (1 + d.rank(d.ts_sum(d.returns, 250)))


def gtja191_026():
    """((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))"""
    return (d.ts_sum(d.close, 7) / 7 - d.close) + d.correlation(
        d.vwap, d.delay(d.close, 5), 230
    )


def gtja191_027():
    """WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)"""
    return d.wma(
        (d.close - d.delay(d.close, 3)) / d.delay(d.close, 3) * 100
        + (d.close - d.delay(d.close, 6)) / d.delay(d.close, 6) * 100,
        12,
    )


def gtja191_028():
    """3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)"""
    return 3 * d.sma(
        (d.close - d.ts_min(d.low, 9))
        / (d.ts_max(d.high, 9) - d.ts_min(d.low, 9))
        * 100,
        3,
        1,
    ) - 2 * d.sma(
        d.sma(
            (d.close - d.ts_min(d.low, 9))
            / (d.ts_max(d.high, 9) - d.ts_min(d.low, 9))
            * 100,
            3,
            1,
        ),
        3,
        1,
    )


def gtja191_029():
    """(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME"""
    return (d.close - d.delay(d.close, 6)) / d.delay(d.close, 6) * d.volume


def gtja191_031():
    """(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100"""
    return (d.close - d.ts_mean(d.close, 12)) / d.ts_mean(d.close, 12) * 100


def gtja191_032():
    """(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))"""
    return -1 * d.ts_sum(d.rank(d.correlation(d.rank(d.high), d.rank(d.volume), 3)), 3)


def gtja191_033():
    """((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) * TSRANK(VOLUME, 5))"""
    return (
        (-1 * d.ts_min(d.low, 5) + d.delay(d.ts_min(d.low, 5), 5))
        * d.rank((d.ts_sum(d.returns, 240) - d.ts_sum(d.returns, 20)) / 220)
        * d.ts_rank(d.volume, 5)
    )


def gtja191_034():
    """MEAN(CLOSE,12)/CLOSE"""
    return d.ts_mean(d.close, 12) / d.close


def gtja191_035():
    """(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN *0.35)), 17),7))) * -1)"""
    return (
        d.minimum(
            d.rank(d.decay_linear(d.delta(d.open, 1), 15)),
            d.rank(
                d.decay_linear(
                    d.correlation(d.volume, (d.open * 0.65) + (d.open * 0.35), 17),
                    7,
                )
            ),
        )
        * -1
    )


def gtja191_036():
    """RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)"""
    return d.rank(d.ts_sum(d.correlation(d.rank(d.volume), d.rank(d.vwap), 6), 2))


def gtja191_037():
    """(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))"""
    return -1 * d.rank(
        (d.ts_sum(d.open, 5) * d.ts_sum(d.returns, 5))
        - d.delay(d.ts_sum(d.open, 5) * d.ts_sum(d.returns, 5), 10)
    )


def gtja191_038():
    """(((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)"""
    return d.where(
        d.lt(d.ts_sum(d.high, 20) / 20, d.high),
        (-1 * d.delta(d.high, 2)),
        0,
    )


def gtja191_039():
    """((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)"""
    return (
        d.rank(d.decay_linear(d.delta(d.close, 2), 8))
        - d.rank(
            d.decay_linear(
                d.correlation(
                    (d.vwap * 0.3) + (d.open * 0.7),
                    d.ts_sum(d.ts_mean(d.volume, 180), 37),
                    14,
                ),
                12,
            )
        )
    ) * -1


def gtja191_040():
    """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100"""
    return (
        d.ts_sum(d.where(d.gt(d.close, d.delay(d.close, 1)), d.volume, 0), 26)
        / d.ts_sum(d.where(d.le(d.close, d.delay(d.close, 1)), d.volume, 0), 26)
        * 100
    )


def gtja191_041():
    """(RANK(MAX(DELTA((VWAP), 3), 5))* -1)"""
    return d.rank(d.ts_max(d.delta(d.vwap, 3), 5)) * -1


def gtja191_042():
    """((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))"""
    return (-1 * d.rank(d.stddev(d.high, 10))) * d.correlation(d.high, d.volume, 10)


def gtja191_043():
    """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)"""
    return d.ts_sum(
        d.where(
            d.gt(d.close, d.delay(d.close, 1)),
            d.volume,
            d.where(d.lt(d.close, d.delay(d.close, 1)), -d.volume, 0),
        ),
        6,
    )


def gtja191_044():
    """(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))"""
    return d.ts_rank(
        d.decay_linear(d.correlation(d.low, d.ts_mean(d.volume, 10), 7), 6), 4
    ) + d.ts_rank(d.decay_linear(d.delta(d.vwap, 3), 10), 15)


def gtja191_045():
    """(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))"""
    return d.rank(d.delta((d.close * 0.6) + (d.open * 0.4), 1)) * d.rank(
        d.correlation(d.vwap, d.ts_mean(d.volume, 150), 15)
    )


def gtja191_046():
    """(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)"""
    return (
        d.ts_mean(d.close, 3)
        + d.ts_mean(d.close, 6)
        + d.ts_mean(d.close, 12)
        + d.ts_mean(d.close, 24)
    ) / (4 * d.close)


def gtja191_047():
    """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)"""
    return d.sma(
        (d.ts_max(d.high, 6) - d.close)
        / (d.ts_max(d.high, 6) - d.ts_min(d.low, 6))
        * 100,
        9,
        1,
    )


def gtja191_048():
    """(-1*((RANK(((SIGN((CLOSE -DELAY(CLOSE,1))) + SIGN((DELAY(CLOSE, 1) -DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))"""
    return -1 * (
        (
            d.rank(
                d.sign(d.close - d.delay(d.close, 1))
                + d.sign(d.delay(d.close, 1) - d.delay(d.close, 2))
                + d.sign(d.delay(d.close, 2) - d.delay(d.close, 3))
            )
            * d.ts_sum(d.volume, 5)
        )
        / d.ts_sum(d.volume, 20)
    )


def gtja191_049():
    """SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))"""
    return d.ts_sum(
        d.where(
            d.ge(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
            0,
            d.maximum(
                d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
            ),
        ),
        12,
    ) / (
        d.ts_sum(
            d.where(
                d.ge(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
                0,
                d.maximum(
                    d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
                ),
            ),
            12,
        )
        + d.ts_sum(
            d.where(
                d.le(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
                0,
                d.maximum(
                    d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
                ),
            ),
            12,
        )
    )


def gtja191_050():
    """SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))"""
    return d.ts_sum(
        d.where(
            d.le(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
            0,
            d.maximum(
                d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
            ),
        ),
        12,
    ) / (
        d.ts_sum(
            d.where(
                d.le(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
                0,
                d.maximum(
                    d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
                ),
            ),
            12,
        )
        + d.ts_sum(
            d.where(
                d.ge(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
                0,
                d.maximum(
                    d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
                ),
            ),
            12,
        )
    ) - d.ts_sum(
        d.where(
            d.ge(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
            0,
            d.maximum(
                d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
            ),
        ),
        12,
    ) / (
        d.ts_sum(
            d.where(
                d.ge(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
                0,
                d.maximum(
                    d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
                ),
            ),
            12,
        )
        + d.ts_sum(
            d.where(
                d.le(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
                0,
                d.maximum(
                    d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
                ),
            ),
            12,
        )
    )


def gtja191_051():
    """SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))"""
    return d.ts_sum(
        d.where(
            d.le(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
            0,
            d.maximum(
                d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
            ),
        ),
        12,
    ) / (
        d.ts_sum(
            d.where(
                d.le(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
                0,
                d.maximum(
                    d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
                ),
            ),
            12,
        )
        + d.ts_sum(
            d.where(
                d.ge(d.high + d.low, d.delay(d.high, 1) + d.delay(d.low, 1)),
                0,
                d.maximum(
                    d.abs(d.high - d.delay(d.high, 1)), d.abs(d.low - d.delay(d.low, 1))
                ),
            ),
            12,
        )
    )


def gtja191_052():
    """SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100"""
    return (
        d.ts_sum(
            d.maximum(0, d.high - d.delay((d.high + d.low + d.close) / 3, 1)),
            26,
        )
        / d.ts_sum(
            d.maximum(0, d.delay((d.high + d.low + d.close) / 3, 1) - d.low),
            26,
        )
        * 100
    )


def gtja191_053():
    """COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100"""
    return d.ts_count(d.gt(d.close, d.delay(d.close, 1)), 12) / 12 * 100


def gtja191_054():
    """(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))"""
    return -1 * d.rank(
        (d.stddev(d.abs(d.close - d.open), 5) + (d.close - d.open))
        + d.correlation(d.close, d.open, 10)
    )


def gtja191_055():
    """SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)"""
    return d.ts_sum(
        16
        * (
            d.close
            - d.delay(d.close, 1)
            + (d.close - d.open) / 2
            + d.delay(d.close, 1)
            - d.delay(d.open, 1)
        )
        / d.where(
            d.and_(
                d.gt(
                    d.abs(d.high - d.delay(d.close, 1)),
                    d.abs(d.low - d.delay(d.close, 1)),
                ),
                d.gt(
                    d.abs(d.high - d.delay(d.close, 1)),
                    d.abs(d.high - d.delay(d.low, 1)),
                ),
            ),
            d.abs(d.high - d.delay(d.close, 1))
            + d.abs(d.low - d.delay(d.close, 1)) / 2
            + d.abs(d.delay(d.close, 1) - d.delay(d.open, 1)) / 4,
            d.where(
                d.and_(
                    d.gt(
                        d.abs(d.low - d.delay(d.close, 1)),
                        d.abs(d.high - d.delay(d.low, 1)),
                    ),
                    d.gt(
                        d.abs(d.low - d.delay(d.close, 1)),
                        d.abs(d.high - d.delay(d.close, 1)),
                    ),
                ),
                d.abs(d.low - d.delay(d.close, 1))
                + d.abs(d.high - d.delay(d.close, 1)) / 2
                + d.abs(d.delay(d.close, 1) - d.delay(d.open, 1)) / 4,
                d.abs(d.high - d.delay(d.low, 1))
                + d.abs(d.delay(d.close, 1) - d.delay(d.open, 1)) / 4,
            ),
        )
        * d.maximum(
            d.abs(d.high - d.delay(d.close, 1)),
            d.abs(d.low - d.delay(d.close, 1)),
        ),
        20,
    )


def gtja191_056():
    """(RANK((OPEN -TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH+ LOW) / 2), 19), SUM(MEAN(VOLUME,40), 19), 13))^5)))"""
    return d.lt(
        d.rank(d.open - d.ts_min(d.open, 12)),
        d.rank(
            d.power(
                d.rank(
                    d.correlation(
                        d.ts_sum((d.high + d.low) / 2, 19),
                        d.ts_sum(d.ts_mean(d.volume, 40), 19),
                        13,
                    )
                ),
                5,
            )
        ),
    )


def gtja191_057():
    """SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)"""
    return d.sma(
        (d.close - d.ts_min(d.low, 9))
        / (d.ts_max(d.high, 9) - d.ts_min(d.low, 9))
        * 100,
        3,
        1,
    )


def gtja191_058():
    """COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100"""
    return d.ts_count(d.gt(d.close, d.delay(d.close, 1)), 20) / 20 * 100


def gtja191_059():
    """SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)"""
    return d.ts_sum(
        d.where(
            d.eq(d.close, d.delay(d.close, 1)),
            0,
            d.close
            - d.where(
                d.gt(d.close, d.delay(d.close, 1)),
                d.minimum(d.low, d.delay(d.close, 1)),
                d.maximum(d.high, d.delay(d.close, 1)),
            ),
        ),
        20,
    )


def gtja191_060():
    """SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)"""
    return d.ts_sum(
        ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low) * d.volume, 20
    )


def gtja191_061():
    """(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)), RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)"""
    return (
        d.maximum(
            d.rank(d.decay_linear(d.delta(d.vwap, 1), 12)),
            d.rank(
                d.decay_linear(
                    d.rank(d.correlation(d.low, d.ts_mean(d.volume, 80), 8)),
                    17,
                )
            ),
        )
        * -1
    )


def gtja191_062():
    """(-1 * CORR(HIGH, RANK(VOLUME), 5))"""
    return -1 * d.correlation(d.high, d.rank(d.volume), 5)


def gtja191_063():
    """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100"""
    return (
        d.sma(d.maximum(d.close - d.delay(d.close, 1), 0), 6, 1)
        / d.sma(d.abs(d.close - d.delay(d.close, 1)), 6, 1)
        * 100
    )


def gtja191_064():
    """(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)"""
    return (
        d.maximum(
            d.rank(
                d.decay_linear(
                    d.correlation(d.rank(d.vwap), d.rank(d.volume), 4),
                    4,
                )
            ),
            d.rank(
                d.decay_linear(
                    d.ts_max(
                        d.correlation(
                            d.rank(d.close),
                            d.rank(d.ts_mean(d.volume, 60)),
                            4,
                        ),
                        13,
                    ),
                    14,
                )
            ),
        )
        * -1
    )


def gtja191_065():
    """MEAN(CLOSE,6)/CLOSE"""
    return d.ts_mean(d.close, 6) / d.close


def gtja191_066():
    """(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100"""
    return (d.close - d.ts_mean(d.close, 6)) / d.ts_mean(d.close, 6) * 100


def gtja191_067():
    """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100"""
    return (
        d.sma(d.maximum(d.close - d.delay(d.close, 1), 0), 24, 1)
        / d.sma(d.abs(d.close - d.delay(d.close, 1)), 24, 1)
        * 100
    )


def gtja191_068():
    """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)"""
    return d.sma(
        ((d.high + d.low) / 2 - (d.delay(d.high, 1) + d.delay(d.low, 1)) / 2)
        * (d.high - d.low)
        / d.volume,
        15,
        2,
    )


def gtja191_069():
    """(SUM(DTM,20)>SUM(DBM,20)？(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)：(SUM(DTM,20)=SUM(DBM,20)？0：(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))"""
    return d.where(
        d.gt(d.ts_sum(d.dtm, 20), d.ts_sum(d.dbm, 20)),
        (d.ts_sum(d.dtm, 20) - d.ts_sum(d.dbm, 20)) / d.ts_sum(d.dtm, 20),
        d.where(
            d.eq(d.ts_sum(d.dtm, 20), d.ts_sum(d.dbm, 20)),
            0,
            (d.ts_sum(d.dtm, 20) - d.ts_sum(d.dbm, 20)) / d.ts_sum(d.dbm, 20),
        ),
    )


def gtja191_070():
    """STD(AMOUNT,6)"""
    return d.stddev(d.amount, 6)


def gtja191_071():
    """(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100"""
    return (d.close - d.ts_mean(d.close, 24)) / d.ts_mean(d.close, 24) * 100


def gtja191_072():
    """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)"""
    return d.sma(
        (d.ts_max(d.high, 6) - d.close)
        / (d.ts_max(d.high, 6) - d.ts_min(d.low, 6))
        * 100,
        15,
        1,
    )


def gtja191_073():
    """((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) -RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)"""
    return (
        d.ts_rank(
            d.decay_linear(d.decay_linear(d.correlation(d.close, d.volume, 10), 16), 4),
            5,
        )
        - d.rank(d.decay_linear(d.correlation(d.vwap, d.ts_mean(d.volume, 30), 4), 3))
    ) * -1


def gtja191_074():
    """(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))"""
    return d.rank(
        d.correlation(
            d.ts_sum((d.low * 0.35) + (d.vwap * 0.65), 20),
            d.ts_sum(d.ts_mean(d.volume, 40), 20),
            7,
        )
    ) + d.rank(d.correlation(d.rank(d.vwap), d.rank(d.volume), 6))


def gtja191_075():
    """COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)"""
    return d.ts_count(
        d.and_(d.gt(d.close, d.open), d.lt(d.benchmark_close, d.benchmark_open)), 50
    ) / d.ts_count(d.lt(d.benchmark_close, d.benchmark_open), 50)


def gtja191_076():
    """STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)"""
    return d.stddev(
        d.abs((d.close / d.delay(d.close, 1) - 1)) / d.volume, 20
    ) / d.ts_mean(d.abs((d.close / d.delay(d.close, 1) - 1)) / d.volume, 20)


def gtja191_077():
    """MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) -(VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))"""
    return d.minimum(
        d.rank(
            d.decay_linear(((((d.high + d.low) / 2) + d.high) - (d.vwap + d.high)), 20)
        ),
        d.rank(
            d.decay_linear(
                d.correlation(((d.high + d.low) / 2), d.ts_mean(d.volume, 40), 3), 6
            )
        ),
    )


def gtja191_078():
    """((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))"""
    return (
        (d.high + d.low + d.close) / 3 - d.ts_mean((d.high + d.low + d.close) / 3, 12)
    ) / (
        0.015
        * d.ts_mean(d.abs(d.close - d.ts_mean((d.high + d.low + d.close) / 3, 12)), 12)
    )


def gtja191_079():
    """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100"""
    return (
        d.sma(d.maximum(d.close - d.delay(d.close, 1), 0), 12, 1)
        / d.sma(d.abs(d.close - d.delay(d.close, 1)), 12, 1)
        * 100
    )


def gtja191_080():
    """(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100"""
    return (d.volume - d.delay(d.volume, 5)) / d.delay(d.volume, 5) * 100


def gtja191_081():
    """SMA(VOLUME,21,2)"""
    return d.sma(d.volume, 21, 2)


def gtja191_082():
    """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)"""
    return d.sma(
        (d.ts_max(d.high, 6) - d.close)
        / (d.ts_max(d.high, 6) - d.ts_min(d.low, 6))
        * 100,
        20,
        1,
    )


def gtja191_083():
    """(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))"""
    return -1 * d.rank(d.covariance(d.rank(d.high), d.rank(d.volume), 5))


def gtja191_084():
    """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)"""
    return d.ts_sum(
        d.where(
            d.gt(d.close, d.delay(d.close, 1)),
            d.volume,
            d.where(d.lt(d.close, d.delay(d.close, 1)), -d.volume, 0),
        ),
        20,
    )


def gtja191_085():
    """(TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))"""
    return d.ts_rank(d.volume / d.ts_mean(d.volume, 20), 20) * d.ts_rank(
        -1 * d.delta(d.close, 7), 8
    )


def gtja191_086():
    """((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) : (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))"""
    return d.where(
        d.lt(
            0.25,
            ((d.delay(d.close, 20) - d.delay(d.close, 10)) / 10)
            - ((d.delay(d.close, 10) - d.close) / 10),
        ),
        (-1 * 1),
        d.where(
            d.lt(
                ((d.delay(d.close, 20) - d.delay(d.close, 10)) / 10)
                - ((d.delay(d.close, 10) - d.close) / 10),
                0,
            ),
            1,
            (-1 * 1) * (d.close - d.delay(d.close, 1)),
        ),
    )


def gtja191_087():
    """((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)"""
    return (
        d.rank(d.decay_linear(d.delta(d.vwap, 4), 7))
        + d.ts_rank(
            d.decay_linear(
                (((d.low * 0.9) + (d.low * 0.1)) - d.vwap)
                / (d.open - ((d.high + d.low) / 2)),
                11,
            ),
            7,
        )
    ) * -1


def gtja191_088():
    """(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100"""
    return (d.close - d.delay(d.close, 20)) / d.delay(d.close, 20) * 100


def gtja191_089():
    """2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))"""
    return 2 * (
        d.sma(d.close, 13, 2)
        - d.sma(d.close, 27, 2)
        - d.sma(d.sma(d.close, 13, 2) - d.sma(d.close, 27, 2), 10, 2)
    )


def gtja191_090():
    """( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)"""
    return d.rank(d.correlation(d.rank(d.vwap), d.rank(d.volume), 5)) * -1


def gtja191_091():
    """((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)"""
    return (
        d.rank(d.close - d.ts_max(d.close, 5))
        * d.rank(d.correlation(d.ts_mean(d.volume, 40), d.low, 5))
        * -1
    )


def gtja191_092():
    """(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)), TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)"""
    return (
        d.maximum(
            d.rank(
                d.decay_linear(
                    d.delta((d.close * 0.35) + (d.vwap * 0.65), 2),
                    3,
                )
            ),
            d.ts_rank(
                d.decay_linear(
                    d.abs(d.correlation(d.ts_mean(d.volume, 180), d.close, 13)),
                    5,
                ),
                15,
            ),
        )
        * -1
    )


def gtja191_093():
    """SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)"""
    return d.ts_sum(
        d.where(
            d.ge(d.open, d.delay(d.open, 1)),
            0,
            d.maximum(d.open - d.low, d.open - d.delay(d.open, 1)),
        ),
        20,
    )


def gtja191_094():
    """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)"""
    return d.ts_sum(
        d.where(
            d.gt(d.close, d.delay(d.close, 1)),
            d.volume,
            d.where(d.lt(d.close, d.delay(d.close, 1)), -d.volume, 0),
        ),
        30,
    )


def gtja191_095():
    """STD(AMOUNT,20)"""
    return d.stddev(d.amount, 20)


def gtja191_096():
    """SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)"""
    return d.sma(
        d.sma(
            (d.close - d.ts_min(d.low, 9))
            / (d.ts_max(d.high, 9) - d.ts_min(d.low, 9))
            * 100,
            3,
            1,
        ),
        3,
        1,
    )


def gtja191_097():
    """STD(VOLUME,10)"""
    return d.stddev(d.volume, 10)


def gtja191_098():
    """((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))"""
    return d.where(
        d.or_(
            d.lt(
                d.delta(d.ts_sum(d.close, 100) / 100, 100) / d.delay(d.close, 100),
                0.05,
            ),
            d.eq(
                d.delta(d.ts_sum(d.close, 100) / 100, 100) / d.delay(d.close, 100),
                0.05,
            ),
        ),
        (-1 * (d.close - d.ts_min(d.close, 100))),
        (-1 * d.delta(d.close, 3)),
    )


def gtja191_099():
    """(-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))"""
    return -1 * d.rank(d.covariance(d.rank(d.close), d.rank(d.volume), 5))


def gtja191_100():
    """STD(VOLUME,20)"""
    return d.stddev(d.volume, 20)


def gtja191_101():
    """((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))), RANK(VOLUME), 11))) * -1)"""
    return (
        d.lt(
            d.rank(d.correlation(d.close, d.ts_sum(d.ts_mean(d.volume, 30), 37), 15)),
            d.rank(
                d.correlation(
                    d.rank((d.high * 0.1) + (d.vwap * 0.9)),
                    d.rank(d.volume),
                    11,
                )
            ),
        )
        * -1
    )


def gtja191_102():
    """SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100"""
    return (
        d.sma(d.maximum(d.volume - d.delay(d.volume, 1), 0), 6, 1)
        / d.sma(d.abs(d.volume - d.delay(d.volume, 1)), 6, 1)
        * 100
    )


def gtja191_103():
    """((20-LOWDAY(LOW,20))/20)*100"""
    return ((20 - d.lowday(d.low, 20)) / 20) * 100


def gtja191_104():
    """(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))"""
    return -1 * (
        d.delta(d.correlation(d.high, d.volume, 5), 5) * d.rank(d.stddev(d.close, 20))
    )


def gtja191_105():
    """(-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))"""
    return -1 * d.correlation(d.rank(d.open), d.rank(d.volume), 10)


def gtja191_106():
    """CLOSE-DELAY(CLOSE,20)"""
    return d.close - d.delay(d.close, 20)


def gtja191_107():
    """(((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))"""
    return (
        (-1 * d.rank(d.open - d.delay(d.high, 1)))
        * d.rank(d.open - d.delay(d.close, 1))
    ) * d.rank(d.open - d.delay(d.low, 1))


def gtja191_108():
    """((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)"""
    return (
        d.power(
            d.rank(d.high - d.ts_min(d.high, 2)),
            d.rank(d.correlation(d.vwap, d.ts_mean(d.volume, 120), 6)),
        )
        * -1
    )


def gtja191_109():
    """SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)"""
    return d.sma(d.high - d.low, 10, 2) / d.sma(d.sma(d.high - d.low, 10, 2), 10, 2)


def gtja191_110():
    """SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100"""
    return (
        d.ts_sum(d.maximum(0, d.high - d.delay(d.close, 1)), 20)
        / d.ts_sum(d.maximum(0, d.delay(d.close, 1) - d.low), 20)
        * 100
    )


def gtja191_111():
    """SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)"""
    return d.sma(
        d.volume * ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low), 11, 2
    ) - d.sma(
        d.volume * ((d.close - d.low) - (d.high - d.close)) / (d.high - d.low), 4, 2
    )


def gtja191_112():
    """(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100"""
    return (
        (
            d.ts_sum(
                d.where(
                    d.gt(d.close - d.delay(d.close, 1), 0),
                    d.close - d.delay(d.close, 1),
                    0,
                ),
                12,
            )
            - d.ts_sum(
                d.where(
                    d.lt(d.close - d.delay(d.close, 1), 0),
                    d.abs(d.close - d.delay(d.close, 1)),
                    0,
                ),
                12,
            )
        )
        / (
            d.ts_sum(
                d.where(
                    d.gt(d.close - d.delay(d.close, 1), 0),
                    d.close - d.delay(d.close, 1),
                    0,
                ),
                12,
            )
            + d.ts_sum(
                d.where(
                    d.lt(d.close - d.delay(d.close, 1), 0),
                    d.abs(d.close - d.delay(d.close, 1)),
                    0,
                ),
                12,
            )
        )
        * 100
    )


def gtja191_113():
    """(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))"""
    return -1 * (
        (
            d.rank(d.ts_sum(d.delay(d.close, 5), 20) / 20)
            * d.correlation(d.close, d.volume, 2)
        )
        * d.rank(d.correlation(d.ts_sum(d.close, 5), d.ts_sum(d.close, 20), 2))
    )


def gtja191_114():
    """((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))"""
    return (
        d.rank(d.delay((d.high - d.low) / (d.ts_sum(d.close, 5) / 5), 2))
        * d.rank(d.rank(d.volume))
    ) / (((d.high - d.low) / (d.ts_sum(d.close, 5) / 5)) / (d.vwap - d.close))


def gtja191_115():
    """(RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) / 2), 4), TSRANK(VOLUME, 10), 7)))"""
    return d.power(
        d.rank(
            d.correlation(
                ((d.high * 0.9) + (d.close * 0.1)), d.ts_mean(d.volume, 30), 10
            )
        ),
        d.rank(
            d.correlation(
                d.ts_rank((d.high + d.low) / 2, 4),
                d.ts_rank(d.volume, 10),
                7,
            )
        ),
    )


def gtja191_116():
    """REGBETA(CLOSE,SEQUENCE,20)"""
    return d.ts_slope(d.close, 20)


def gtja191_117():
    """((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))"""
    return (
        d.ts_rank(d.volume, 32) * (1 - d.ts_rank(((d.close + d.high) - d.low), 16))
    ) * (1 - d.ts_rank(d.returns, 32))


def gtja191_118():
    """SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100"""
    return d.ts_sum(d.high - d.open, 20) / d.ts_sum(d.open - d.low, 20) * 100


def gtja191_119():
    """(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) -RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))"""
    return d.rank(
        d.decay_linear(
            d.correlation(d.vwap, d.ts_sum(d.ts_mean(d.volume, 5), 26), 5), 7
        )
    ) - d.rank(
        d.decay_linear(
            d.ts_rank(
                d.ts_min(
                    d.correlation(d.rank(d.open), d.rank(d.ts_mean(d.volume, 15)), 21),
                    9,
                ),
                7,
            ),
            8,
        )
    )


def gtja191_120():
    """(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))"""
    return d.rank(d.vwap - d.close) / d.rank(d.vwap + d.close)


def gtja191_121():
    """((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) * -1)"""
    return (
        d.power(
            d.rank(d.vwap - d.ts_min(d.vwap, 12)),
            d.ts_rank(
                d.correlation(
                    d.ts_rank(d.vwap, 20),
                    d.ts_rank(d.ts_mean(d.volume, 60), 2),
                    18,
                ),
                3,
            ),
        )
        * -1
    )


def gtja191_122():
    """(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)"""
    return (
        d.sma(d.sma(d.sma(d.log(d.close), 13, 2), 13, 2), 13, 2)
        - d.delay(d.sma(d.sma(d.sma(d.log(d.close), 13, 2), 13, 2), 13, 2), 1)
    ) / d.delay(d.sma(d.sma(d.sma(d.log(d.close), 13, 2), 13, 2), 13, 2), 1)


def gtja191_123():
    """((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)"""
    return (
        d.lt(
            d.rank(
                d.correlation(
                    d.ts_sum((d.high + d.low) / 2, 20),
                    d.ts_sum(d.ts_mean(d.volume, 60), 20),
                    9,
                )
            ),
            d.rank(d.correlation(d.low, d.volume, 6)),
        )
        * -1
    )


def gtja191_124():
    """(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)"""
    return (d.close - d.vwap) / d.decay_linear(d.rank(d.ts_max(d.close, 30)), 2)


def gtja191_125():
    """(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))"""
    return d.rank(
        d.decay_linear(d.correlation(d.vwap, d.ts_mean(d.volume, 80), 17), 20)
    ) / d.rank(d.decay_linear(d.delta((d.close * 0.5) + (d.vwap * 0.5), 3), 16))


def gtja191_126():
    """(CLOSE+HIGH+LOW)/3"""
    return (d.close + d.high + d.low) / 3


def gtja191_127():
    """(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)"""
    return d.power(
        d.ts_mean(
            d.power(100 * (d.close - d.ts_max(d.close, 12)) / d.ts_max(d.close, 12), 2),
            12,
        ),
        1 / 2,
    )


def gtja191_128():
    """100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))"""
    return 100 - (
        100
        / (
            1
            + d.ts_sum(
                d.where(
                    d.gt(
                        (d.high + d.low + d.close) / 3,
                        d.delay((d.high + d.low + d.close) / 3, 1),
                    ),
                    (d.high + d.low + d.close) / 3 * d.volume,
                    0,
                ),
                14,
            )
            / d.ts_sum(
                d.where(
                    d.lt(
                        (d.high + d.low + d.close) / 3,
                        d.delay((d.high + d.low + d.close) / 3, 1),
                    ),
                    (d.high + d.low + d.close) / 3 * d.volume,
                    0,
                ),
                14,
            )
        )
    )


def gtja191_129():
    """SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)"""
    return d.ts_sum(
        d.where(
            d.lt(d.close - d.delay(d.close, 1), 0),
            d.abs(d.close - d.delay(d.close, 1)),
            0,
        ),
        12,
    )


def gtja191_130():
    """(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))"""
    return d.rank(
        d.decay_linear(
            d.correlation((d.high + d.low) / 2, d.ts_mean(d.volume, 40), 9), 10
        )
    ) / d.rank(d.decay_linear(d.correlation(d.rank(d.vwap), d.rank(d.volume), 7), 3))


def gtja191_131():
    """(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))"""
    return d.power(
        d.rank(d.delta(d.vwap, 1)),
        d.ts_rank(d.correlation(d.close, d.ts_mean(d.volume, 50), 18), 18),
    )


def gtja191_132():
    """MEAN(AMOUNT,20)"""
    return d.ts_mean(d.amount, 20)


def gtja191_133():
    """((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100"""
    return ((20 - d.highday(d.high, 20)) / 20) * 100 - (
        (20 - d.lowday(d.low, 20)) / 20
    ) * 100


def gtja191_134():
    """(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME"""
    return (d.close - d.delay(d.close, 12)) / d.delay(d.close, 12) * d.volume


def gtja191_135():
    """SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)"""
    return d.sma(d.delay(d.close / d.delay(d.close, 20), 1), 20, 1)


def gtja191_136():
    """((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))"""
    return (-1 * d.rank(d.delta(d.returns, 3))) * d.correlation(d.open, d.volume, 10)


def gtja191_137():
    """16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))"""
    return (
        16
        * (
            d.close
            - d.delay(d.close, 1)
            + (d.close - d.open) / 2
            + d.delay(d.close, 1)
            - d.delay(d.open, 1)
        )
        / d.where(
            d.and_(
                d.gt(
                    d.abs(d.high - d.delay(d.close, 1)),
                    d.abs(d.low - d.delay(d.close, 1)),
                ),
                d.gt(
                    d.abs(d.high - d.delay(d.close, 1)),
                    d.abs(d.high - d.delay(d.low, 1)),
                ),
            ),
            d.abs(d.high - d.delay(d.close, 1))
            + d.abs(d.low - d.delay(d.close, 1)) / 2
            + d.abs(d.delay(d.close, 1) - d.delay(d.open, 1)) / 4,
            d.where(
                d.and_(
                    d.gt(
                        d.abs(d.low - d.delay(d.close, 1)),
                        d.abs(d.high - d.delay(d.low, 1)),
                    ),
                    d.gt(
                        d.abs(d.low - d.delay(d.close, 1)),
                        d.abs(d.high - d.delay(d.close, 1)),
                    ),
                ),
                d.abs(d.low - d.delay(d.close, 1))
                + d.abs(d.high - d.delay(d.close, 1)) / 2
                + d.abs(d.delay(d.close, 1) - d.delay(d.open, 1)) / 4,
                d.abs(d.high - d.delay(d.low, 1))
                + d.abs(d.delay(d.close, 1) - d.delay(d.open, 1)) / 4,
            ),
        )
        * d.maximum(
            d.abs(d.high - d.delay(d.close, 1)),
            d.abs(d.low - d.delay(d.close, 1)),
        )
    )


def gtja191_138():
    """((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) -TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)"""
    return (
        d.rank(
            d.decay_linear(
                d.delta(((d.low * 0.7) + (d.vwap * 0.3)), 3),
                20,
            )
        )
        - d.ts_rank(
            d.decay_linear(
                d.ts_rank(
                    d.correlation(
                        d.ts_rank(d.low, 8),
                        d.ts_rank(d.ts_mean(d.volume, 60), 17),
                        5,
                    ),
                    19,
                ),
                16,
            ),
            7,
        )
    ) * -1


def gtja191_139():
    """(-1 * CORR(OPEN, VOLUME, 10))"""
    return -1 * d.correlation(d.open, d.volume, 10)


def gtja191_140():
    """MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) -(RANK(HIGH) +RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))"""
    return d.minimum(
        d.rank(
            d.decay_linear(
                ((d.rank(d.open) + d.rank(d.low)) - (d.rank(d.high) + d.rank(d.close))),
                8,
            )
        ),
        d.ts_rank(
            d.decay_linear(
                d.correlation(
                    d.ts_rank(d.close, 8),
                    d.ts_rank(d.ts_mean(d.volume, 60), 20),
                    8,
                ),
                7,
            ),
            3,
        ),
    )


def gtja191_141():
    """(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)"""
    return (
        d.rank(d.correlation(d.rank(d.high), d.rank(d.ts_mean(d.volume, 15)), 9)) * -1
    )


def gtja191_142():
    """(((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME /MEAN(VOLUME,20)), 5)))"""
    return (
        (-1 * d.rank(d.ts_rank(d.close, 10))) * d.rank(d.delta(d.delta(d.close, 1), 1))
    ) * d.rank(d.ts_rank(d.volume / d.ts_mean(d.volume, 20), 5))


def gtja191_144():
    """SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)"""
    return d.ts_sumif(
        d.abs(d.close / d.delay(d.close, 1) - 1) / d.amount,
        d.lt(d.close, d.delay(d.close, 1)),
        20,
    ) / d.ts_count(d.lt(d.close, d.delay(d.close, 1)), 20)


def gtja191_145():
    """(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100"""
    return (
        (d.ts_mean(d.volume, 9) - d.ts_mean(d.volume, 26))
        / d.ts_mean(d.volume, 12)
        * 100
    )


def gtja191_146():
    """MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60);"""
    return (
        d.ts_mean(
            (d.close - d.delay(d.close, 1)) / d.delay(d.close, 1)
            - d.sma((d.close - d.delay(d.close, 1)) / d.delay(d.close, 1), 61, 2),
            20,
        )
        * (
            (d.close - d.delay(d.close, 1)) / d.delay(d.close, 1)
            - d.sma((d.close - d.delay(d.close, 1)) / d.delay(d.close, 1), 61, 2)
        )
        / d.sma(
            d.power(
                (d.close - d.delay(d.close, 1)) / d.delay(d.close, 1)
                - (
                    (d.close - d.delay(d.close, 1)) / d.delay(d.close, 1)
                    - d.sma(
                        (d.close - d.delay(d.close, 1)) / d.delay(d.close, 1), 61, 2
                    )
                ),
                2,
            ),
            60,
            2,
        )
    )


def gtja191_147():
    """REGBETA(MEAN(CLOSE,12),SEQUENCE(12))"""
    return d.ts_slope(d.ts_mean(d.close, 12), 12)


def gtja191_148():
    """((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)"""
    return (
        d.lt(
            d.rank(
                d.correlation(
                    d.open,
                    d.ts_sum(d.ts_mean(d.volume, 60), 9),
                    6,
                )
            ),
            d.rank(d.open - d.ts_min(d.open, 14)),
        )
        * -1
    )


def gtja191_149():
    """REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)"""
    return d.mask_missing(
        d.regbeta(
            d.filter_(
                d.close / d.delay(d.close, 1) - 1,
                d.lt(d.benchmark_close, d.delay(d.benchmark_close, 1)),
            ),
            d.filter_(
                d.benchmark_close / d.delay(d.benchmark_close, 1) - 1,
                d.lt(d.benchmark_close, d.delay(d.benchmark_close, 1)),
            ),
            252,
            min_pairs=20,
        ),
        d.close / d.delay(d.close, 1) - 1,
    )


def gtja191_150():
    """(CLOSE+HIGH+LOW)/3*VOLUME"""
    return (d.close + d.high + d.low) / 3 * d.volume


def gtja191_151():
    """SMA(CLOSE-DELAY(CLOSE,20),20,1)"""
    return d.sma(d.close - d.delay(d.close, 20), 20, 1)


def gtja191_152():
    """SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)"""
    return d.sma(
        d.ts_mean(
            d.delay(
                d.sma(
                    d.delay(d.close / d.delay(d.close, 9), 1),
                    9,
                    1,
                ),
                1,
            ),
            12,
        )
        - d.ts_mean(
            d.delay(
                d.sma(
                    d.delay(d.close / d.delay(d.close, 9), 1),
                    9,
                    1,
                ),
                1,
            ),
            26,
        ),
        9,
        1,
    )


def gtja191_153():
    """(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4"""
    return (
        d.ts_mean(d.close, 3)
        + d.ts_mean(d.close, 6)
        + d.ts_mean(d.close, 12)
        + d.ts_mean(d.close, 24)
    ) / 4


def gtja191_154():
    """(((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))"""
    return d.lt(
        d.vwap - d.ts_min(d.vwap, 16),
        d.correlation(d.vwap, d.ts_mean(d.volume, 180), 18),
    )


def gtja191_155():
    """SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)"""
    return (
        d.sma(d.volume, 13, 2)
        - d.sma(d.volume, 27, 2)
        - d.sma(d.sma(d.volume, 13, 2) - d.sma(d.volume, 27, 2), 10, 2)
    )


def gtja191_156():
    """(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)"""
    return (
        d.maximum(
            d.rank(d.decay_linear(d.delta(d.vwap, 5), 3)),
            d.rank(
                d.decay_linear(
                    (
                        d.delta((d.open * 0.15) + (d.low * 0.85), 2)
                        / ((d.open * 0.15) + (d.low * 0.85))
                    )
                    * -1,
                    3,
                )
            ),
        )
        * -1
    )


def gtja191_157():
    """(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))"""
    return d.ts_min(
        d.ts_product(
            d.rank(
                d.rank(
                    d.log(
                        d.ts_sum(
                            d.ts_min(
                                d.rank(d.rank((-1 * d.rank(d.delta(d.close - 1, 5))))),
                                2,
                            ),
                            1,
                        )
                    )
                )
            ),
            1,
        ),
        5,
    ) + d.ts_rank(d.delay((-1 * d.returns), 6), 5)


def gtja191_158():
    """((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE"""
    return (
        (d.high - d.sma(d.close, 15, 2)) - (d.low - d.sma(d.close, 15, 2))
    ) / d.close


def gtja191_159():
    """((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)"""
    return (
        (
            (d.close - d.ts_sum(d.minimum(d.low, d.delay(d.close, 1)), 6))
            / d.ts_sum(
                d.maximum(d.high, d.delay(d.close, 1))
                - d.minimum(d.low, d.delay(d.close, 1)),
                6,
            )
            * 12
            * 24
            + (d.close - d.ts_sum(d.minimum(d.low, d.delay(d.close, 1)), 12))
            / d.ts_sum(
                d.maximum(d.high, d.delay(d.close, 1))
                - d.minimum(d.low, d.delay(d.close, 1)),
                12,
            )
            * 6
            * 24
            + (d.close - d.ts_sum(d.minimum(d.low, d.delay(d.close, 1)), 24))
            / d.ts_sum(
                d.maximum(d.high, d.delay(d.close, 1))
                - d.minimum(d.low, d.delay(d.close, 1)),
                24,
            )
            * 6
            * 24
        )
        * 100
        / (6 * 12 + 6 * 24 + 12 * 24)
    )


def gtja191_160():
    """SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)"""
    return d.sma(
        d.where(
            d.le(d.close, d.delay(d.close, 1)),
            d.stddev(d.close, 20),
            0,
        ),
        20,
        1,
    )


def gtja191_161():
    """MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)"""
    return d.ts_mean(
        d.maximum(
            d.maximum(
                d.high - d.low,
                d.abs(d.delay(d.close, 1) - d.high),
            ),
            d.abs(d.delay(d.close, 1) - d.low),
        ),
        12,
    )


def gtja191_162():
    """(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))"""
    return (
        d.sma(d.maximum(d.close - d.delay(d.close, 1), 0), 12, 1)
        / d.sma(d.abs(d.close - d.delay(d.close, 1)), 12, 1)
        * 100
        - d.ts_min(
            d.sma(d.maximum(d.close - d.delay(d.close, 1), 0), 12, 1)
            / d.sma(d.abs(d.close - d.delay(d.close, 1)), 12, 1)
            * 100,
            12,
        )
    ) / (
        d.ts_max(
            d.sma(d.maximum(d.close - d.delay(d.close, 1), 0), 12, 1)
            / d.sma(d.abs(d.close - d.delay(d.close, 1)), 12, 1)
            * 100,
            12,
        )
        - d.ts_min(
            d.sma(d.maximum(d.close - d.delay(d.close, 1), 0), 12, 1)
            / d.sma(d.abs(d.close - d.delay(d.close, 1)), 12, 1)
            * 100,
            12,
        )
    )


def gtja191_163():
    """RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))"""
    return d.rank(
        (((-1 * d.returns) * d.ts_mean(d.volume, 20)) * d.vwap) * (d.high - d.close)
    )


def gtja191_164():
    """SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)"""
    return d.sma(
        (
            d.where(
                d.gt(d.close, d.delay(d.close, 1)),
                1 / (d.close - d.delay(d.close, 1)),
                1,
            )
            - d.ts_min(
                d.where(
                    d.gt(d.close, d.delay(d.close, 1)),
                    1 / (d.close - d.delay(d.close, 1)),
                    1,
                ),
                12,
            )
        )
        / (d.high - d.low)
        * 100,
        13,
        2,
    )


def gtja191_167():
    """SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)"""
    return d.ts_sum(
        d.where(
            d.gt(d.close - d.delay(d.close, 1), 0), d.close - d.delay(d.close, 1), 0
        ),
        12,
    )


def gtja191_168():
    """(-1*VOLUME/MEAN(VOLUME,20))"""
    return -1 * d.volume / d.ts_mean(d.volume, 20)


def gtja191_169():
    """SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)"""
    return d.sma(
        d.ts_mean(d.delay(d.sma(d.close - d.delay(d.close, 1), 9, 1), 1), 12)
        - d.ts_mean(d.delay(d.sma(d.close - d.delay(d.close, 1), 9, 1), 1), 26),
        10,
        1,
    )


def gtja191_170():
    """((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))"""
    return (d.rank(1 / d.close) * d.volume / d.ts_mean(d.volume, 20)) * (
        (d.high * d.rank(d.high - d.close)) / (d.ts_sum(d.high, 5) / 5)
    ) - d.rank(d.vwap - d.delay(d.vwap, 5))


def gtja191_171():
    """((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))"""
    return (-1 * ((d.low - d.close) * d.power(d.open, 5))) / (
        (d.close - d.high) * d.power(d.close, 5)
    )


def gtja191_172():
    """MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)"""
    return d.ts_mean(
        d.abs(
            d.ts_sum(d.where(d.and_(d.gt(d.ld, 0), d.gt(d.ld, d.hd)), d.ld, 0), 14)
            * 100
            / d.ts_sum(d.tr, 14)
            - d.ts_sum(d.where(d.and_(d.gt(d.hd, 0), d.gt(d.hd, d.ld)), d.hd, 0), 14)
            * 100
            / d.ts_sum(d.tr, 14)
        )
        / (
            d.ts_sum(d.where(d.and_(d.gt(d.ld, 0), d.gt(d.ld, d.hd)), d.ld, 0), 14)
            * 100
            / d.ts_sum(d.tr, 14)
            + d.ts_sum(d.where(d.and_(d.gt(d.hd, 0), d.gt(d.hd, d.ld)), d.hd, 0), 14)
            * 100
            / d.ts_sum(d.tr, 14)
        )
        * 100,
        6,
    )


def gtja191_173():
    """3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2);"""
    return (
        3 * d.sma(d.close, 13, 2)
        - 2 * d.sma(d.sma(d.close, 13, 2), 13, 2)
        + d.sma(d.sma(d.sma(d.log(d.close), 13, 2), 13, 2), 13, 2)
    )


def gtja191_174():
    """SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)"""
    return d.sma(
        d.where(d.gt(d.close, d.delay(d.close, 1)), d.stddev(d.close, 20), 0),
        20,
        1,
    )


def gtja191_175():
    """MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)"""
    return d.ts_mean(
        d.maximum(
            d.maximum(d.high - d.low, d.abs(d.delay(d.close, 1) - d.high)),
            d.abs(d.delay(d.close, 1) - d.low),
        ),
        6,
    )


def gtja191_176():
    """CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)"""
    return d.correlation(
        d.rank(
            (d.close - d.ts_min(d.low, 12))
            / (d.ts_max(d.high, 12) - d.ts_min(d.low, 12))
        ),
        d.rank(d.volume),
        6,
    )


def gtja191_177():
    """((20-HIGHDAY(HIGH,20))/20)*100"""
    return ((20 - d.highday(d.high, 20)) / 20) * 100


def gtja191_178():
    """(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME"""
    return (d.close - d.delay(d.close, 1)) / d.delay(d.close, 1) * d.volume


def gtja191_179():
    """(RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))"""
    return d.rank(d.correlation(d.vwap, d.volume, 4)) * d.rank(
        d.correlation(d.rank(d.low), d.rank(d.ts_mean(d.volume, 50)), 12)
    )


def gtja191_180():
    """((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 * VOLUME)))"""
    return d.where(
        d.lt(d.ts_mean(d.volume, 20), d.volume),
        (-1 * d.ts_rank(d.abs(d.delta(d.close, 7)), 60)) * d.sign(d.delta(d.close, 7)),
        (-1 * d.volume),
    )


def gtja191_181():
    """SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)"""
    return d.ts_sum(
        (d.close / d.delay(d.close, 1) - 1)
        - d.ts_mean(d.close / d.delay(d.close, 1) - 1, 20)
        - d.power(d.benchmark_close - d.ts_mean(d.benchmark_close, 20), 2),
        20,
    ) / d.ts_sum(
        d.power(d.benchmark_close - d.ts_mean(d.benchmark_close, 20), 3),
        20,
    )


def gtja191_182():
    """COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20"""
    return (
        d.ts_count(
            d.or_(
                d.and_(
                    d.gt(d.close, d.open), d.gt(d.benchmark_close, d.benchmark_open)
                ),
                d.and_(
                    d.lt(d.close, d.open), d.lt(d.benchmark_close, d.benchmark_open)
                ),
            ),
            20,
        )
        / 20
    )


def gtja191_184():
    """(RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))"""
    return d.rank(d.correlation(d.delay(d.open - d.close, 1), d.close, 200)) + d.rank(
        d.open - d.close
    )


def gtja191_185():
    """RANK((-1 * ((1 - (OPEN / CLOSE))^2)))"""
    return d.rank((-1 * d.power(1 - (d.open / d.close), 2)))


def gtja191_186():
    """(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2"""
    return (
        d.ts_mean(
            d.abs(
                d.ts_sum(d.where(d.and_(d.gt(d.ld, 0), d.gt(d.ld, d.hd)), d.ld, 0), 14)
                * 100
                / d.ts_sum(d.tr, 14)
                - d.ts_sum(
                    d.where(d.and_(d.gt(d.hd, 0), d.gt(d.hd, d.ld)), d.hd, 0), 14
                )
                * 100
                / d.ts_sum(d.tr, 14)
            )
            / (
                d.ts_sum(d.where(d.and_(d.gt(d.ld, 0), d.gt(d.ld, d.hd)), d.ld, 0), 14)
                * 100
                / d.ts_sum(d.tr, 14)
                + d.ts_sum(
                    d.where(d.and_(d.gt(d.hd, 0), d.gt(d.hd, d.ld)), d.hd, 0), 14
                )
                * 100
                / d.ts_sum(d.tr, 14)
            )
            * 100,
            6,
        )
        + d.delay(
            d.ts_mean(
                d.abs(
                    d.ts_sum(
                        d.where(d.and_(d.gt(d.ld, 0), d.gt(d.ld, d.hd)), d.ld, 0), 14
                    )
                    * 100
                    / d.ts_sum(d.tr, 14)
                    - d.ts_sum(
                        d.where(d.and_(d.gt(d.hd, 0), d.gt(d.hd, d.ld)), d.hd, 0), 14
                    )
                    * 100
                    / d.ts_sum(d.tr, 14)
                )
                / (
                    d.ts_sum(
                        d.where(d.and_(d.gt(d.ld, 0), d.gt(d.ld, d.hd)), d.ld, 0), 14
                    )
                    * 100
                    / d.ts_sum(d.tr, 14)
                    + d.ts_sum(
                        d.where(d.and_(d.gt(d.hd, 0), d.gt(d.hd, d.ld)), d.hd, 0), 14
                    )
                    * 100
                    / d.ts_sum(d.tr, 14)
                )
                * 100,
                6,
            ),
            6,
        )
    ) / 2


def gtja191_187():
    """SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)"""
    return d.ts_sum(
        d.where(
            d.le(d.open, d.delay(d.open, 1)),
            0,
            d.maximum(d.high - d.open, d.open - d.delay(d.open, 1)),
        ),
        20,
    )


def gtja191_188():
    """((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100"""
    return (
        (d.high - d.low - d.sma(d.high - d.low, 11, 2))
        / d.sma(d.high - d.low, 11, 2)
        * 100
    )


def gtja191_189():
    """MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)"""
    return d.ts_mean(d.abs(d.close - d.ts_mean(d.close, 6)), 6)


def gtja191_190():
    """LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))"""
    return d.log(
        (
            d.ts_count(
                d.gt(
                    d.close / d.delay(d.close, 1) - 1,
                    d.power(d.close / d.delay(d.close, 19), 1 / 20) - 1,
                ),
                20,
            )
            - 1
        )
        * d.ts_sumif(
            d.power(
                d.close / d.delay(d.close, 1)
                - 1
                - d.power(d.close / d.delay(d.close, 19), 1 / 20)
                - 1,
                2,
            ),
            d.lt(
                d.close / d.delay(d.close, 1) - 1,
                d.power(d.close / d.delay(d.close, 19), 1 / 20) - 1,
            ),
            20,
        )
        / (
            d.ts_count(
                d.lt(
                    d.close / d.delay(d.close, 1) - 1,
                    d.power(d.close / d.delay(d.close, 19), 1 / 20) - 1,
                ),
                20,
            )
            * d.ts_sumif(
                d.power(
                    d.close / d.delay(d.close, 1)
                    - 1
                    - (d.power(d.close / d.delay(d.close, 19), 1 / 20) - 1),
                    2,
                ),
                d.gt(
                    d.close / d.delay(d.close, 1) - 1,
                    d.power(d.close / d.delay(d.close, 19), 1 / 20) - 1,
                ),
                20,
            )
        )
    )


def gtja191_191():
    """((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)"""
    return (
        d.correlation(d.ts_mean(d.volume, 20), d.low, 5) + ((d.high + d.low) / 2)
    ) - d.close


NOTES = {}
