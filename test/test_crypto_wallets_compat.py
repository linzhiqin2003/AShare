"""C4 audit: the bitwise digest must cover all pre-existing crypto counters."""

import copy

import pytest

from bench.crypto_wallets_compat import digest
from test.test_crypto_wallets import simulator


@pytest.mark.parametrize(
    "key",
    ["funding_paid", "liquidation_open", "bankruptcy_deficit", "tier_cap_exceeded"],
)
def test_bitwise_digest_includes_old_crypto_coverage(key):
    # Source: final C4 compatibility audit; the initial digest omitted CRYPTO_COUNTERS.
    sim, _ = simulator([[100]])
    sim.step()
    original = sim.result()
    changed = copy.deepcopy(original)
    changed.coverage[key] += 1
    assert digest(original) != digest(changed)
