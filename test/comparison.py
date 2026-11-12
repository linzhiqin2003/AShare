"""Full-result comparison shared by deterministic tests and corpus validation."""

from dataclasses import astuple

import numpy as np
from panelbt.models import METRICS


def assert_results_equal(actual, expected):
    for name in (*METRICS, "positions", "contribution"):
        a, b = getattr(actual, name), getattr(expected, name)
        if a is None or b is None:
            assert a is b, name
        else:
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, err_msg=name)
    assert actual.diagnostics == expected.diagnostics
    assert actual.coverage == expected.coverage
    assert actual.bankruptcy_events == expected.bankruptcy_events
    for name in ("trades", "delist_events", "liquidation_events", "expiry_events"):
        a, b = getattr(actual, name), getattr(expected, name)
        if a is None or b is None:
            assert a is b
            continue
        assert len(a) == len(b), (name, len(a), len(b))
        for left, right in zip(a, b, strict=True):
            assert (left.day, left.col) == (right.day, right.col)
            np.testing.assert_allclose(
                astuple(left), astuple(right), rtol=1e-9, atol=1e-9
            )
