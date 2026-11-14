"""DSL_REFERENCE.md 里的 5 个例子：能求值，前三个与 NumPy 直算逐格一致。"""

import numpy as np
import pandas as pd

from ashare.factor import Evaluator
from ashare.factor.formula import dsl as d
from ashare.factor.formula.testing import synthetic_source

nan = np.nan


def gtja191_014():
    """CLOSE-DELAY(CLOSE,5)"""
    return d.close - d.delay(d.close, 5)


def alpha101_012():
    """(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
    return d.sign(d.delta(d.volume, 1)) * (-1 * d.delta(d.close, 1))


def gtja191_053():
    """COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100"""
    return d.ts_count(d.gt(d.close, d.delay(d.close, 1)), 12) / 12 * 100


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


def alpha101_072():
    """(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /
    rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))"""
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


def test_examples_evaluate_and_simple_ones_match_numpy():
    source = synthetic_source(t=200, seed=5)
    e = Evaluator()
    a = {k: np.asarray(v, float) for k, v in source.arrays.items()}
    close = a["close"] * a["adj_factor"]
    with np.errstate(invalid="ignore"):
        lag = np.vstack([np.full((5, close.shape[1]), nan), close[:-5]])
        np.testing.assert_array_equal(e.evaluate(gtja191_014(), source), close - lag)
        up = pd.DataFrame(
            np.where(
                np.isnan(close[1:]) | np.isnan(close[:-1]), nan, close[1:] > close[:-1]
            )
        )
        count = up.rolling(12).sum().to_numpy() / 12 * 100
    np.testing.assert_allclose(e.evaluate(gtja191_053(), source)[1:], count, rtol=1e-15)
    for build in (alpha101_012, alpha101_001, alpha101_072):
        values = e.evaluate(build(), source)
        assert values.shape == source.shape and np.isfinite(values).any()
    ranks = e.evaluate(alpha101_001(), source)
    finite = ranks[np.isfinite(ranks)]
    assert finite.min() > -0.5 and finite.max() <= 0.5
