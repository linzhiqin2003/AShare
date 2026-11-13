"""平台对照工具须检查每个持仓数组和摘要的原始字节。"""

import numpy as np
import pytest

from bench.platform.bitwise import compare_snapshots


@pytest.mark.parametrize("mutation", ["summary", "negative-zero", "nan", "missing"])
def test_snapshot_comparison_detects_mutation(tmp_path, mutation):
    before, after = tmp_path / "before", tmp_path / "after"
    for folder in (before, after):
        folder.mkdir()
        (folder / "0.json").write_text('{"sharpe":null}')
        np.savez(folder / "0.holdings.npz", weights=np.array([0.0, np.nan]))
    assert compare_snapshots(before, after) == 1
    if mutation == "summary":
        (after / "0.json").write_text('{"sharpe":0}')
    elif mutation == "missing":
        (after / "0.holdings.npz").unlink()
    else:
        weights = np.array([0.0, np.nan])
        if mutation == "negative-zero":
            weights[0] = -0.0
        else:
            weights.view(np.uint64)[1] ^= np.uint64(1)
        np.savez(after / "0.holdings.npz", weights=weights)
    with pytest.raises(AssertionError):
        compare_snapshots(before, after)
