"""WorldQuant Alpha101 的翻译（第二步逐条填写，规范见 docs/DSL_REFERENCE.md）。

每条一个函数 ``alpha101_001`` … ``alpha101_101``，返回 ``Expr``，文档字符串写原文公式。
不收的条目不写函数，理由在 docs/CONVENTIONS.md 第 4 节。
"""

from ashare.factor.formula import dsl as d


def alpha101_001():
    """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
    return (
        d.rank(
            d.ts_argmax(
                d.signed_power(
                    d.where(d.lt(d.returns, 0), d.stddev(d.returns, 20), d.close), 2.0
                ),
                5,
            )
        )
        - 0.5
    )


def alpha101_002():
    """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    return -1 * d.correlation(
        d.rank(d.delta(d.log(d.volume), 2)),
        d.rank((d.close - d.open) / d.open),
        6,
    )


def alpha101_003():
    """(-1 * correlation(rank(open), rank(volume), 10))"""
    return -1 * d.correlation(d.rank(d.open), d.rank(d.volume), 10)


def alpha101_004():
    """(-1 * Ts_Rank(rank(low), 9))"""
    return -1 * d.ts_rank(d.rank(d.low), 9)


def alpha101_005():
    """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
    return d.rank(d.open - (d.ts_sum(d.vwap, 10) / 10)) * (
        -1 * d.abs(d.rank(d.close - d.vwap))
    )


def alpha101_006():
    """(-1 * correlation(open, volume, 10))"""
    return -1 * d.correlation(d.open, d.volume, 10)


def alpha101_007():
    """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))"""
    return d.where(
        d.lt(d.adv(20), d.volume),
        (-1 * d.ts_rank(d.abs(d.delta(d.close, 7)), 60)) * d.sign(d.delta(d.close, 7)),
        (-1 * 1),
    )


def alpha101_008():
    """(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
    return -1 * d.rank(
        (d.ts_sum(d.open, 5) * d.ts_sum(d.returns, 5))
        - d.delay((d.ts_sum(d.open, 5) * d.ts_sum(d.returns, 5)), 10)
    )


def alpha101_009():
    """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""
    return d.where(
        d.lt(0, d.ts_min(d.delta(d.close, 1), 5)),
        d.delta(d.close, 1),
        d.where(
            d.lt(d.ts_max(d.delta(d.close, 1), 5), 0),
            d.delta(d.close, 1),
            (-1 * d.delta(d.close, 1)),
        ),
    )


def alpha101_010():
    """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
    return d.rank(
        d.where(
            d.lt(0, d.ts_min(d.delta(d.close, 1), 4)),
            d.delta(d.close, 1),
            d.where(
                d.lt(d.ts_max(d.delta(d.close, 1), 4), 0),
                d.delta(d.close, 1),
                (-1 * d.delta(d.close, 1)),
            ),
        )
    )


def alpha101_011():
    """((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
    return (
        d.rank(d.ts_max((d.vwap - d.close), 3))
        + d.rank(d.ts_min((d.vwap - d.close), 3))
    ) * d.rank(d.delta(d.volume, 3))


def alpha101_012():
    """(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
    return d.sign(d.delta(d.volume, 1)) * (-1 * d.delta(d.close, 1))


def alpha101_013():
    """(-1 * rank(covariance(rank(close), rank(volume), 5)))"""
    return -1 * d.rank(d.covariance(d.rank(d.close), d.rank(d.volume), 5))


def alpha101_014():
    """((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
    return (-1 * d.rank(d.delta(d.returns, 3))) * d.correlation(d.open, d.volume, 10)


def alpha101_015():
    """(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    return -1 * d.ts_sum(d.rank(d.correlation(d.rank(d.high), d.rank(d.volume), 3)), 3)


def alpha101_016():
    """(-1 * rank(covariance(rank(high), rank(volume), 5)))"""
    return -1 * d.rank(d.covariance(d.rank(d.high), d.rank(d.volume), 5))


def alpha101_017():
    """(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
    return (
        (-1 * d.rank(d.ts_rank(d.close, 10))) * d.rank(d.delta(d.delta(d.close, 1), 1))
    ) * d.rank(d.ts_rank((d.volume / d.adv(20)), 5))


def alpha101_018():
    """(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
    return -1 * d.rank(
        (d.stddev(d.abs((d.close - d.open)), 5) + (d.close - d.open))
        + d.correlation(d.close, d.open, 10)
    )


def alpha101_019():
    """((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    return (-1 * d.sign(((d.close - d.delay(d.close, 7)) + d.delta(d.close, 7)))) * (
        1 + d.rank((1 + d.ts_sum(d.returns, 250)))
    )


def alpha101_020():
    """(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))"""
    return (
        (-1 * d.rank((d.open - d.delay(d.high, 1))))
        * d.rank((d.open - d.delay(d.close, 1)))
    ) * d.rank((d.open - d.delay(d.low, 1)))


def alpha101_021():
    """((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))"""
    return d.where(
        d.lt(
            (d.ts_sum(d.close, 8) / 8) + d.stddev(d.close, 8), d.ts_sum(d.close, 2) / 2
        ),
        (-1 * 1),
        d.where(
            d.lt(
                d.ts_sum(d.close, 2) / 2,
                (d.ts_sum(d.close, 8) / 8) - d.stddev(d.close, 8),
            ),
            1,
            d.where(
                d.or_(d.lt(1, (d.volume / d.adv(20))), d.eq((d.volume / d.adv(20)), 1)),
                1,
                (-1 * 1),
            ),
        ),
    )


def alpha101_022():
    """(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
    return -1 * (
        d.delta(d.correlation(d.high, d.volume, 5), 5) * d.rank(d.stddev(d.close, 20))
    )


def alpha101_023():
    """(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"""
    return d.where(
        d.lt(d.ts_sum(d.high, 20) / 20, d.high), (-1 * d.delta(d.high, 2)), 0
    )


def alpha101_024():
    """((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))"""
    return d.where(
        d.or_(
            d.lt(
                d.delta((d.ts_sum(d.close, 100) / 100), 100) / d.delay(d.close, 100),
                0.05,
            ),
            d.eq(
                d.delta((d.ts_sum(d.close, 100) / 100), 100) / d.delay(d.close, 100),
                0.05,
            ),
        ),
        (-1 * (d.close - d.ts_min(d.close, 100))),
        (-1 * d.delta(d.close, 3)),
    )


def alpha101_025():
    """rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"""
    return d.rank(((((-1 * d.returns) * d.adv(20)) * d.vwap) * (d.high - d.close)))


def alpha101_026():
    """(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
    return -1 * d.ts_max(
        d.correlation(d.ts_rank(d.volume, 5), d.ts_rank(d.high, 5), 5), 3
    )


def alpha101_027():
    """((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)"""
    return d.where(
        d.lt(
            0.5,
            d.rank(
                d.ts_sum(d.correlation(d.rank(d.volume), d.rank(d.vwap), 6), 2) / 2.0
            ),
        ),
        (-1 * 1),
        1,
    )


def alpha101_028():
    """scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    return d.scale(
        (d.correlation(d.adv(20), d.low, 5) + ((d.high + d.low) / 2)) - d.close
    )


def alpha101_029():
    """(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))"""
    return d.ts_min(
        d.ts_product(
            d.rank(
                d.rank(
                    d.scale(
                        d.log(
                            d.ts_sum(
                                d.ts_min(
                                    d.rank(
                                        d.rank((-1 * d.rank(d.delta((d.close - 1), 5))))
                                    ),
                                    2,
                                ),
                                1,
                            )
                        )
                    )
                )
            ),
            1,
        ),
        5,
    ) + d.ts_rank(d.delay((-1 * d.returns), 6), 5)


def alpha101_030():
    """(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))"""
    return (
        (
            1.0
            - d.rank(
                (
                    (
                        d.sign((d.close - d.delay(d.close, 1)))
                        + d.sign((d.delay(d.close, 1) - d.delay(d.close, 2)))
                    )
                    + d.sign((d.delay(d.close, 2) - d.delay(d.close, 3)))
                )
            )
        )
        * d.ts_sum(d.volume, 5)
    ) / d.ts_sum(d.volume, 20)


def alpha101_031():
    """((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))"""
    return (
        d.rank(
            d.rank(
                d.rank(
                    d.decay_linear(
                        (-1 * d.rank(d.rank(d.delta(d.close, 10)))),
                        10,
                    )
                )
            )
        )
        + d.rank((-1 * d.delta(d.close, 3)))
    ) + d.sign(d.scale(d.correlation(d.adv(20), d.low, 12)))


def alpha101_032():
    """(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))"""
    return d.scale(((d.ts_sum(d.close, 7) / 7) - d.close)) + (
        20 * d.scale(d.correlation(d.vwap, d.delay(d.close, 5), 230))
    )


def alpha101_033():
    """rank((-1 * ((1 - (open / close))^1)))"""
    return d.rank((-1 * d.power((1 - (d.open / d.close)), 1)))


def alpha101_034():
    """rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
    return d.rank(
        (1 - d.rank(d.stddev(d.returns, 2) / d.stddev(d.returns, 5)))
        + (1 - d.rank(d.delta(d.close, 1)))
    )


def alpha101_035():
    """((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
    return (
        d.ts_rank(d.volume, 32) * (1 - d.ts_rank((d.close + d.high) - d.low, 16))
    ) * (1 - d.ts_rank(d.returns, 32))


def alpha101_036():
    """(((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))"""
    return (
        (
            (
                (
                    2.21
                    * d.rank(
                        d.correlation((d.close - d.open), d.delay(d.volume, 1), 15)
                    )
                )
                + (0.7 * d.rank((d.open - d.close)))
            )
            + (0.73 * d.rank(d.ts_rank(d.delay((-1 * d.returns), 6), 5)))
        )
        + d.rank(d.abs(d.correlation(d.vwap, d.adv(20), 6)))
    ) + (0.6 * d.rank((((d.ts_sum(d.close, 200) / 200) - d.open) * (d.close - d.open))))


def alpha101_037():
    """(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
    return d.rank(d.correlation(d.delay((d.open - d.close), 1), d.close, 200)) + d.rank(
        (d.open - d.close)
    )


def alpha101_038():
    """((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))"""
    return (-1 * d.rank(d.ts_rank(d.close, 10))) * d.rank((d.close / d.open))


def alpha101_039():
    """((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
    return (
        -1
        * d.rank(
            d.delta(d.close, 7)
            * (1 - d.rank(d.decay_linear((d.volume / d.adv(20)), 9)))
        )
    ) * (1 + d.rank(d.ts_sum(d.returns, 250)))


def alpha101_040():
    """((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))"""
    return (-1 * d.rank(d.stddev(d.high, 10))) * d.correlation(d.high, d.volume, 10)


def alpha101_041():
    """(((high * low)^0.5) - vwap)"""
    return d.power(d.high * d.low, 0.5) - d.vwap


def alpha101_042():
    """(rank((vwap - close)) / rank((vwap + close)))"""
    return d.rank(d.vwap - d.close) / d.rank(d.vwap + d.close)


def alpha101_043():
    """(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))"""
    return d.ts_rank(d.volume / d.adv(20), 20) * d.ts_rank(-1 * d.delta(d.close, 7), 8)


def alpha101_044():
    """(-1 * correlation(high, rank(volume), 5))"""
    return -1 * d.correlation(d.high, d.rank(d.volume), 5)


def alpha101_045():
    """(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
    return -1 * (
        (
            d.rank(d.ts_sum(d.delay(d.close, 5), 20) / 20)
            * d.correlation(d.close, d.volume, 2)
        )
        * d.rank(d.correlation(d.ts_sum(d.close, 5), d.ts_sum(d.close, 20), 2))
    )


def alpha101_046():
    """((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))"""
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


def alpha101_047():
    """((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))"""
    return (
        ((d.rank(1 / d.close) * d.volume) / d.adv(20))
        * ((d.high * d.rank(d.high - d.close)) / (d.ts_sum(d.high, 5) / 5))
    ) - d.rank(d.vwap - d.delay(d.vwap, 5))


def alpha101_048():
    """(indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))"""
    return d.indneutralize(
        (
            d.correlation(d.delta(d.close, 1), d.delta(d.delay(d.close, 1), 1), 250)
            * d.delta(d.close, 1)
        )
        / d.close,
        "subindustry",
    ) / d.ts_sum(d.power(d.delta(d.close, 1) / d.delay(d.close, 1), 2), 250)


def alpha101_049():
    """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
    return d.where(
        d.lt(
            (
                ((d.delay(d.close, 20) - d.delay(d.close, 10)) / 10)
                - ((d.delay(d.close, 10) - d.close) / 10)
            ),
            (-1 * 0.1),
        ),
        1,
        ((-1 * 1) * (d.close - d.delay(d.close, 1))),
    )


def alpha101_050():
    """(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    return -1 * d.ts_max(d.rank(d.correlation(d.rank(d.volume), d.rank(d.vwap), 5)), 5)


def alpha101_051():
    """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
    return d.where(
        d.lt(
            (
                ((d.delay(d.close, 20) - d.delay(d.close, 10)) / 10)
                - ((d.delay(d.close, 10) - d.close) / 10)
            ),
            (-1 * 0.05),
        ),
        1,
        ((-1 * 1) * (d.close - d.delay(d.close, 1))),
    )


def alpha101_052():
    """((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
    return (
        ((-1 * d.ts_min(d.low, 5)) + d.delay(d.ts_min(d.low, 5), 5))
        * d.rank((d.ts_sum(d.returns, 240) - d.ts_sum(d.returns, 20)) / 220)
    ) * d.ts_rank(d.volume, 5)


def alpha101_053():
    """(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
    return -1 * d.delta(
        (((d.close - d.low) - (d.high - d.close)) / (d.close - d.low)), 9
    )


def alpha101_054():
    """((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""
    return (-1 * ((d.low - d.close) * d.power(d.open, 5))) / (
        (d.low - d.high) * d.power(d.close, 5)
    )


def alpha101_055():
    """(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))"""
    return -1 * d.correlation(
        d.rank(
            (
                (d.close - d.ts_min(d.low, 12))
                / (d.ts_max(d.high, 12) - d.ts_min(d.low, 12))
            )
        ),
        d.rank(d.volume),
        6,
    )


def alpha101_056():
    """(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"""
    return 0 - (
        1
        * (
            d.rank((d.ts_sum(d.returns, 10) / d.ts_sum(d.ts_sum(d.returns, 2), 3)))
            * d.rank((d.returns * d.cap))
        )
    )


def alpha101_057():
    """(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""
    return 0 - (
        1 * ((d.close - d.vwap) / d.decay_linear(d.rank(d.ts_argmax(d.close, 30)), 2))
    )


def alpha101_058():
    """(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))"""
    return -1 * d.ts_rank(
        d.decay_linear(
            d.correlation(d.indneutralize(d.vwap, "sector"), d.volume, 3.92795),
            7.89291,
        ),
        5.50322,
    )


def alpha101_059():
    """(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))"""
    return -1 * d.ts_rank(
        d.decay_linear(
            d.correlation(
                d.indneutralize(
                    ((d.vwap * 0.728317) + (d.vwap * (1 - 0.728317))), "industry"
                ),
                d.volume,
                4.25197,
            ),
            16.2289,
        ),
        8.19648,
    )


def alpha101_060():
    """(0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))"""
    return 0 - (
        1
        * (
            (
                2
                * d.scale(
                    d.rank(
                        (
                            (
                                ((d.close - d.low) - (d.high - d.close))
                                / (d.high - d.low)
                            )
                            * d.volume
                        )
                    )
                )
            )
            - d.scale(d.rank(d.ts_argmax(d.close, 10)))
        )
    )


def alpha101_061():
    """(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))"""
    return d.lt(
        d.rank((d.vwap - d.ts_min(d.vwap, 16.1219))),
        d.rank(d.correlation(d.vwap, d.adv(180), 17.9282)),
    )


def alpha101_062():
    """((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)"""
    return (
        d.lt(
            d.rank(d.correlation(d.vwap, d.ts_sum(d.adv(20), 22.4101), 9.91009)),
            d.rank(
                d.lt(
                    (d.rank(d.open) + d.rank(d.open)),
                    (d.rank(((d.high + d.low) / 2)) + d.rank(d.high)),
                )
            ),
        )
        * -1
    )


def alpha101_063():
    """((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)"""
    return (
        d.rank(
            d.decay_linear(
                d.delta(d.indneutralize(d.close, "industry"), 2.25164), 8.22237
            )
        )
        - d.rank(
            d.decay_linear(
                d.correlation(
                    ((d.vwap * 0.318108) + (d.open * (1 - 0.318108))),
                    d.ts_sum(d.adv(180), 37.2467),
                    13.557,
                ),
                12.2883,
            )
        )
    ) * -1


def alpha101_064():
    """((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)"""
    return (
        d.lt(
            d.rank(
                d.correlation(
                    d.ts_sum(((d.open * 0.178404) + (d.low * (1 - 0.178404))), 12.7054),
                    d.ts_sum(d.adv(120), 12.7054),
                    16.6208,
                )
            ),
            d.rank(
                d.delta(
                    ((((d.high + d.low) / 2) * 0.178404) + (d.vwap * (1 - 0.178404))),
                    3.69741,
                )
            ),
        )
        * -1
    )


def alpha101_065():
    """((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)"""
    return (
        d.lt(
            d.rank(
                d.correlation(
                    ((d.open * 0.00817205) + (d.vwap * (1 - 0.00817205))),
                    d.ts_sum(d.adv(60), 8.6911),
                    6.40374,
                )
            ),
            d.rank((d.open - d.ts_min(d.open, 13.635))),
        )
        * -1
    )


def alpha101_066():
    """((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)"""
    return (
        d.rank(d.decay_linear(d.delta(d.vwap, 3.51013), 7.23052))
        + d.ts_rank(
            d.decay_linear(
                (
                    (((d.low * 0.96633) + (d.low * (1 - 0.96633))) - d.vwap)
                    / (d.open - ((d.high + d.low) / 2))
                ),
                11.4157,
            ),
            6.72611,
        )
    ) * -1


def alpha101_067():
    """((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)"""
    return (
        d.power(
            d.rank((d.high - d.ts_min(d.high, 2.14593))),
            d.rank(
                d.correlation(
                    d.indneutralize(d.vwap, "sector"),
                    d.indneutralize(d.adv(20), "subindustry"),
                    6.02936,
                )
            ),
        )
        * -1
    )


def alpha101_068():
    """((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)"""
    return (
        d.lt(
            d.ts_rank(
                d.correlation(d.rank(d.high), d.rank(d.adv(15)), 8.91644),
                13.9333,
            ),
            d.rank(d.delta(((d.close * 0.518371) + (d.low * (1 - 0.518371))), 1.06157)),
        )
        * -1
    )


def alpha101_069():
    """((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)"""
    return (
        d.power(
            d.rank(
                d.ts_max(d.delta(d.indneutralize(d.vwap, "industry"), 2.72412), 4.79344)
            ),
            d.ts_rank(
                d.correlation(
                    ((d.close * 0.490655) + (d.vwap * (1 - 0.490655))),
                    d.adv(20),
                    4.92416,
                ),
                9.0615,
            ),
        )
        * -1
    )


def alpha101_070():
    """((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)"""
    return (
        d.power(
            d.rank(d.delta(d.vwap, 1.29456)),
            d.ts_rank(
                d.correlation(d.indneutralize(d.close, "industry"), d.adv(50), 17.8256),
                17.9171,
            ),
        )
        * -1
    )


def alpha101_071():
    """max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))"""
    return d.maximum(
        d.ts_rank(
            d.decay_linear(
                d.correlation(
                    d.ts_rank(d.close, 3.43976),
                    d.ts_rank(d.adv(180), 12.0647),
                    18.0175,
                ),
                4.20501,
            ),
            15.6948,
        ),
        d.ts_rank(
            d.decay_linear(
                d.power(d.rank(((d.low + d.open) - (d.vwap + d.vwap))), 2),
                16.4662,
            ),
            4.4388,
        ),
    )


def alpha101_072():
    """(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))"""
    return d.rank(
        d.decay_linear(d.correlation((d.high + d.low) / 2, d.adv(40), 8.93345), 10.1519)
    ) / d.rank(
        d.decay_linear(
            d.correlation(
                d.ts_rank(d.vwap, 3.72469), d.ts_rank(d.volume, 18.5188), 6.86671
            ),
            2.95011,
        )
    )


def alpha101_073():
    """(max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)"""
    return (
        d.maximum(
            d.rank(d.decay_linear(d.delta(d.vwap, 4.72775), 2.91864)),
            d.ts_rank(
                d.decay_linear(
                    (
                        d.delta((d.open * 0.147155) + (d.low * (1 - 0.147155)), 2.03608)
                        / ((d.open * 0.147155) + (d.low * (1 - 0.147155)))
                    )
                    * -1,
                    3.33829,
                ),
                16.7411,
            ),
        )
        * -1
    )


def alpha101_074():
    """((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)"""
    return (
        d.lt(
            d.rank(d.correlation(d.close, d.ts_sum(d.adv(30), 37.4843), 15.1365)),
            d.rank(
                d.correlation(
                    d.rank((d.high * 0.0261661) + (d.vwap * (1 - 0.0261661))),
                    d.rank(d.volume),
                    11.4791,
                )
            ),
        )
        * -1
    )


def alpha101_075():
    """(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))"""
    return d.lt(
        d.rank(d.correlation(d.vwap, d.volume, 4.24304)),
        d.rank(d.correlation(d.rank(d.low), d.rank(d.adv(50)), 12.4413)),
    )


def alpha101_076():
    """(max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)"""
    return (
        d.maximum(
            d.rank(d.decay_linear(d.delta(d.vwap, 1.24383), 11.8259)),
            d.ts_rank(
                d.decay_linear(
                    d.ts_rank(
                        d.correlation(
                            d.indneutralize(d.low, "sector"), d.adv(81), 8.14941
                        ),
                        19.569,
                    ),
                    17.1543,
                ),
                19.383,
            ),
        )
        * -1
    )


def alpha101_077():
    """min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))"""
    return d.minimum(
        d.rank(
            d.decay_linear(
                ((((d.high + d.low) / 2) + d.high) - (d.vwap + d.high)), 20.0451
            )
        ),
        d.rank(
            d.decay_linear(
                d.correlation((d.high + d.low) / 2, d.adv(40), 3.1614), 5.64125
            )
        ),
    )


def alpha101_078():
    """(rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))"""
    return d.power(
        d.rank(
            d.correlation(
                d.ts_sum((d.low * 0.352233) + (d.vwap * (1 - 0.352233)), 19.7428),
                d.ts_sum(d.adv(40), 19.7428),
                6.83313,
            )
        ),
        d.rank(d.correlation(d.rank(d.vwap), d.rank(d.volume), 5.77492)),
    )


def alpha101_079():
    """(rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))"""
    return d.lt(
        d.rank(
            d.delta(
                d.indneutralize(
                    (d.close * 0.60733) + (d.open * (1 - 0.60733)), "sector"
                ),
                1.23438,
            )
        ),
        d.rank(
            d.correlation(
                d.ts_rank(d.vwap, 3.60973),
                d.ts_rank(d.adv(150), 9.18637),
                14.6644,
            )
        ),
    )


def alpha101_080():
    """((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)"""
    return (
        d.power(
            d.rank(
                d.sign(
                    d.delta(
                        d.indneutralize(
                            (d.open * 0.868128) + (d.high * (1 - 0.868128)), "industry"
                        ),
                        4.04545,
                    )
                )
            ),
            d.ts_rank(d.correlation(d.high, d.adv(10), 5.11456), 5.53756),
        )
        * -1
    )


def alpha101_081():
    """((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)"""
    return (
        d.lt(
            d.rank(
                d.log(
                    d.ts_product(
                        d.rank(
                            d.power(
                                d.rank(
                                    d.correlation(
                                        d.vwap, d.ts_sum(d.adv(10), 49.6054), 8.47743
                                    )
                                ),
                                4,
                            )
                        ),
                        14.9655,
                    )
                )
            ),
            d.rank(d.correlation(d.rank(d.vwap), d.rank(d.volume), 5.07914)),
        )
        * -1
    )


def alpha101_082():
    """(min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)"""
    return (
        d.minimum(
            d.rank(d.decay_linear(d.delta(d.open, 1.46063), 14.8717)),
            d.ts_rank(
                d.decay_linear(
                    d.correlation(
                        d.indneutralize(d.volume, "sector"),
                        (d.open * 0.634196) + (d.open * (1 - 0.634196)),
                        17.4842,
                    ),
                    6.92131,
                ),
                13.4283,
            ),
        )
        * -1
    )


def alpha101_083():
    """((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))"""
    return (
        d.rank(d.delay((d.high - d.low) / (d.ts_sum(d.close, 5) / 5), 2))
        * d.rank(d.rank(d.volume))
    ) / (((d.high - d.low) / (d.ts_sum(d.close, 5) / 5)) / (d.vwap - d.close))


def alpha101_084():
    """SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))"""
    return d.signed_power(
        d.ts_rank(d.vwap - d.ts_max(d.vwap, 15.3217), 20.7127),
        d.delta(d.close, 4.96796),
    )


def alpha101_085():
    """(rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))"""
    return d.power(
        d.rank(
            d.correlation(
                (d.high * 0.876703) + (d.close * (1 - 0.876703)),
                d.adv(30),
                9.61331,
            )
        ),
        d.rank(
            d.correlation(
                d.ts_rank((d.high + d.low) / 2, 3.70596),
                d.ts_rank(d.volume, 10.1595),
                7.11408,
            )
        ),
    )


def alpha101_086():
    """((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)"""
    return (
        d.lt(
            d.ts_rank(
                d.correlation(d.close, d.ts_sum(d.adv(20), 14.7444), 6.00049),
                20.4195,
            ),
            d.rank((d.open + d.close) - (d.vwap + d.open)),
        )
        * -1
    )


def alpha101_087():
    """(max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)"""
    return (
        d.maximum(
            d.rank(
                d.decay_linear(
                    d.delta((d.close * 0.369701) + (d.vwap * (1 - 0.369701)), 1.91233),
                    2.65461,
                )
            ),
            d.ts_rank(
                d.decay_linear(
                    d.abs(
                        d.correlation(
                            d.indneutralize(d.adv(81), "industry"),
                            d.close,
                            13.4132,
                        )
                    ),
                    4.89768,
                ),
                14.4535,
            ),
        )
        * -1
    )


def alpha101_088():
    """min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))"""
    return d.minimum(
        d.rank(
            d.decay_linear(
                (d.rank(d.open) + d.rank(d.low)) - (d.rank(d.high) + d.rank(d.close)),
                8.06882,
            )
        ),
        d.ts_rank(
            d.decay_linear(
                d.correlation(
                    d.ts_rank(d.close, 8.44728),
                    d.ts_rank(d.adv(60), 20.6966),
                    8.01266,
                ),
                6.65053,
            ),
            2.61957,
        ),
    )


def alpha101_089():
    """(Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))"""
    return d.ts_rank(
        d.decay_linear(
            d.correlation(
                ((d.low * 0.967285) + (d.low * (1 - 0.967285))),
                d.adv(10),
                6.94279,
            ),
            5.51607,
        ),
        3.79744,
    ) - d.ts_rank(
        d.decay_linear(d.delta(d.indneutralize(d.vwap, "industry"), 3.48158), 10.1466),
        15.3012,
    )


def alpha101_090():
    """((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)"""
    return (
        d.power(
            d.rank(d.close - d.ts_max(d.close, 4.66719)),
            d.ts_rank(
                d.correlation(
                    d.indneutralize(d.adv(40), "subindustry"), d.low, 5.38375
                ),
                3.21856,
            ),
        )
        * -1
    )


def alpha101_091():
    """((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)"""
    return (
        d.ts_rank(
            d.decay_linear(
                d.decay_linear(
                    d.correlation(
                        d.indneutralize(d.close, "industry"), d.volume, 9.74928
                    ),
                    16.398,
                ),
                3.83219,
            ),
            4.8667,
        )
        - d.rank(d.decay_linear(d.correlation(d.vwap, d.adv(30), 4.01303), 2.6809))
    ) * -1


def alpha101_092():
    """min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))"""
    return d.minimum(
        d.ts_rank(
            d.decay_linear(
                d.lt(((d.high + d.low) / 2) + d.close, d.low + d.open),
                14.7221,
            ),
            18.8683,
        ),
        d.ts_rank(
            d.decay_linear(
                d.correlation(d.rank(d.low), d.rank(d.adv(30)), 7.58555),
                6.94024,
            ),
            6.80584,
        ),
    )


def alpha101_093():
    """(Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))"""
    return d.ts_rank(
        d.decay_linear(
            d.correlation(d.indneutralize(d.vwap, "industry"), d.adv(81), 17.4193),
            19.848,
        ),
        7.54455,
    ) / d.rank(
        d.decay_linear(
            d.delta(
                ((d.close * 0.524434) + (d.vwap * (1 - 0.524434))),
                2.77377,
            ),
            16.2664,
        )
    )


def alpha101_094():
    """((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)"""
    return (
        d.power(
            d.rank(d.vwap - d.ts_min(d.vwap, 11.5783)),
            d.ts_rank(
                d.correlation(
                    d.ts_rank(d.vwap, 19.6462),
                    d.ts_rank(d.adv(60), 4.02992),
                    18.0926,
                ),
                2.70756,
            ),
        )
        * -1
    )


def alpha101_095():
    """(rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))"""
    return d.lt(
        d.rank(d.open - d.ts_min(d.open, 12.4105)),
        d.ts_rank(
            d.power(
                d.rank(
                    d.correlation(
                        d.ts_sum((d.high + d.low) / 2, 19.1351),
                        d.ts_sum(d.adv(40), 19.1351),
                        12.8742,
                    )
                ),
                5,
            ),
            11.7584,
        ),
    )


def alpha101_096():
    """(max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)"""
    return (
        d.maximum(
            d.ts_rank(
                d.decay_linear(
                    d.correlation(d.rank(d.vwap), d.rank(d.volume), 3.83878),
                    4.16783,
                ),
                8.38151,
            ),
            d.ts_rank(
                d.decay_linear(
                    d.ts_argmax(
                        d.correlation(
                            d.ts_rank(d.close, 7.45404),
                            d.ts_rank(d.adv(60), 4.13242),
                            3.65459,
                        ),
                        12.6556,
                    ),
                    14.0365,
                ),
                13.4143,
            ),
        )
        * -1
    )


def alpha101_097():
    """((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)"""
    return (
        d.rank(
            d.decay_linear(
                d.delta(
                    d.indneutralize(
                        ((d.low * 0.721001) + (d.vwap * (1 - 0.721001))),
                        "industry",
                    ),
                    3.3705,
                ),
                20.4523,
            )
        )
        - d.ts_rank(
            d.decay_linear(
                d.ts_rank(
                    d.correlation(
                        d.ts_rank(d.low, 7.87871),
                        d.ts_rank(d.adv(60), 17.255),
                        4.97547,
                    ),
                    18.5925,
                ),
                15.7152,
            ),
            6.71659,
        )
    ) * -1


def alpha101_098():
    """(rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))"""
    return d.rank(
        d.decay_linear(
            d.correlation(
                d.vwap,
                d.ts_sum(d.adv(5), 26.4719),
                4.58418,
            ),
            7.18088,
        )
    ) - d.rank(
        d.decay_linear(
            d.ts_rank(
                d.ts_argmin(
                    d.correlation(
                        d.rank(d.open),
                        d.rank(d.adv(15)),
                        20.8187,
                    ),
                    8.62571,
                ),
                6.95668,
            ),
            8.07206,
        )
    )


def alpha101_099():
    """((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)"""
    return (
        d.lt(
            d.rank(
                d.correlation(
                    d.ts_sum((d.high + d.low) / 2, 19.8975),
                    d.ts_sum(d.adv(60), 19.8975),
                    8.8136,
                )
            ),
            d.rank(d.correlation(d.low, d.volume, 6.28259)),
        )
        * -1
    )


def alpha101_100():
    """(0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))"""
    return 0 - (
        1
        * (
            (
                (
                    1.5
                    * d.scale(
                        d.indneutralize(
                            d.indneutralize(
                                d.rank(
                                    (
                                        (
                                            ((d.close - d.low) - (d.high - d.close))
                                            / (d.high - d.low)
                                        )
                                        * d.volume
                                    )
                                ),
                                "subindustry",
                            ),
                            "subindustry",
                        )
                    )
                )
                - d.scale(
                    d.indneutralize(
                        (
                            d.correlation(d.close, d.rank(d.adv(20)), 5)
                            - d.rank(d.ts_argmin(d.close, 30))
                        ),
                        "subindustry",
                    )
                )
            )
            * (d.volume / d.adv(20))
        )
    )


def alpha101_101():
    """((close - open) / ((high - low) + .001))"""
    return (d.close - d.open) / ((d.high - d.low) + 0.001)


NOTES = {}
