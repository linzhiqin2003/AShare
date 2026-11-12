"""Bitwise digest of a Result (used by the frozen no-rules fixture).

The fixture `fixtures/no_rules_digest.json` was produced by the engine build
before the stepping interface and declarative rules were added; matching it
proves that runs without rules are unchanged bit for bit.
"""

import hashlib
from dataclasses import astuple

import numpy as np
from panelbt.models import METRICS

from . import generators, generators_futures


def digest(result) -> str:
    h = hashlib.sha256()
    for name in (*METRICS, "positions", "contribution"):
        a = getattr(result, name)
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(sorted(result.diagnostics.items())).encode())
    h.update(repr(sorted(result.coverage.items())).encode())
    h.update(repr(result.bankruptcy_events).encode())
    for name in ("trades", "delist_events", "liquidation_events", "expiry_events"):
        items = getattr(result, name) or []
        for x in items:
            h.update(np.asarray(astuple(x)[:9], dtype=float).tobytes())
    return h.hexdigest()


def corpora(count):
    from .test_extended import extended_case

    for seed in range(count):
        yield f"spot-{seed}", generators.case(seed)
        yield f"margin-{seed}", extended_case(seed)
        yield f"futures-{seed}", generators_futures.case(seed)
        yield f"futures-rich-{seed}", generators_futures.rich_case(seed)
