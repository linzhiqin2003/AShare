"""R1 development regressions: missing marks combined with market blocking."""

from functools import partial

import numpy as np
import panelbt as bt
import pytest
from panelbt.reference_crypto import run as reference_run

from .test_crypto import scenario
from .test_crypto_review import step_run


@pytest.fixture(
    params=[bt.run, reference_run, step_run, partial(step_run, restore=True)],
    ids=["native", "reference", "step", "snapshot"],
)
def engine(request):
    return request.param


@pytest.mark.parametrize("obstacle", ["mask", "price"])
@pytest.mark.parametrize("carry", [True, False])
@pytest.mark.parametrize("kind", ["weights", "shares"])
def test_review_first_row_inactive_nan_mark_carries_intent(
    engine, obstacle, carry, kind
):
    # Found while checking P1's exception: allowing inactive NaN must not hide
    # a price/mask block just because the first sizing denominator is missing.
    m, _, costs, cfg = scenario(
        [[100 if obstacle == "mask" else np.nan], [100]],
        [[1], [np.nan]],
        marks=[[np.nan], [100]],
        settlement=[[100], [100]],
        buy=[[obstacle != "mask"], [True]],
        config=bt.Config(initial_cash=100, carry_blocked=carry, record_trades=True),
    )
    r = engine(m, bt.Targets.from_dense([[1], [np.nan]], kind=kind), costs, cfg)
    np.testing.assert_array_equal(r.positions[:, 0], [0, int(carry)])
    assert r.diagnostics["blocked_buy"] == 1
    assert r.diagnostics["carry_trade"] == int(carry)


@pytest.mark.parametrize("side", [1, -1])
def test_review_tiny_weight_intent_cannot_hide_missing_marks(engine, side):
    # Intent validation must precede rounding even if 0.001 would size to zero
    # using the finite settlement placeholder. No fabricated mark may size it.
    m, _, costs, cfg = scenario([[100]], [[0]], marks=[[np.nan]], settlement=[[100]])
    with pytest.raises(ValueError, match="mark"):
        engine(m, bt.Targets.from_dense([[side * 0.001]]), costs, cfg)
