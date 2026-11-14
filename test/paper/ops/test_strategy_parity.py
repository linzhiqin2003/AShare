"""目标持仓与回测页逐位一致（2026-11-14 新增，模拟盘每日运行）。

回测页 ``build_targets`` 把信号日 d 的目标放到第 d+1 行；模拟盘对最新一行没有下一行，所以在
面板中间的日子上对照：同一份信号、同一个股票池，模拟盘的「证券 → 权重」必须与回测页在 d+1 行
给出的权重逐位相同。数据全部来自合成数据目录。
"""

from decimal import Decimal

import numpy as np
import pytest

from ashare.paper.ops.strategy import (
    LivePanel,
    Strategy,
    parse_source,
    pool_fields,
    pool_row,
    runner_safe,
    signal_alpha,
    target_weights,
)
from ashare.paper.runner import load_targets
from ashare.platform.backtest import settings as contract
from ashare.platform.backtest.engine import build_targets, universe_mask
from test.fixtures.synthetic_datadir import cached_datadir

UNIVERSES = ("csi300", "csi500", "csi1000", "TOP800", "TOP1800", "ALL")


@pytest.fixture(scope="module")
def panel():
    live = LivePanel(cached_datadir().root)
    for universe in UNIVERSES:
        live.ensure(pool_fields(universe))
    return live


@pytest.fixture(scope="module")
def alpha(panel):
    expression = parse_source("rank(-ts_std(close, 20)) + 0.1 * rank(volume)")[
        0
    ].expression
    return signal_alpha(panel, expression)


@pytest.mark.parametrize("universe", UNIVERSES)
def test_pool_row_matches_backtest_universe_mask_parity(panel, universe):
    full = universe_mask(panel, universe)
    for row in range(0, len(panel.dates), 17):
        np.testing.assert_array_equal(pool_row(panel, universe, row), full[row])


@pytest.mark.parametrize(
    "universe,top_n", [("ALL", 5), ("ALL", 11), ("TOP800", 3), ("csi300", 30)]
)
def test_target_weights_bitwise_equal_backtest_build_targets(
    panel, alpha, universe, top_n
):
    full = universe_mask(panel, universe)
    rows = [r for r in range(30, len(panel.dates) - 1, 23) if full[r].any()]
    assert rows
    for row in rows:
        targets = build_targets(alpha, full, np.array([row]), top_n, len(panel.dates))
        assert list(targets.days) == [row + 1]
        expected = {
            panel.symbols[int(c)]: float(w)
            for c, w in zip(targets.cols, targets.weights, strict=True)
        }
        mine = target_weights(
            alpha[row], pool_row(panel, universe, row), top_n, panel.symbols
        )
        assert list(mine) == list(expected)  # 同样按面板列序
        assert all(
            np.float64(mine[k]).view(np.uint64) == np.float64(v).view(np.uint64)
            for k, v in expected.items()
        )


def test_runner_safe_only_moves_when_decimal_sum_exceeds_one(tmp_path):
    """1/11 的最短十进制表示合计超过 1（运行器会拒收）：最大权重往下挪末位后才合格。"""
    for n in (3, 7, 11, 30):
        weights = {f"S{i:02d}": 1 / n for i in range(n)}
        safe, moved = runner_safe(weights)
        over = sum((Decimal(str(v)) for v in weights.values()), Decimal(0)) > 1
        assert (moved > 0) == over
        assert sum((Decimal(str(v)) for v in safe.values()), Decimal(0)) <= 1
        changed = [k for k in safe if safe[k] != weights[k]]
        assert len(changed) <= moved
        assert all(abs(safe[k] - weights[k]) < 1e-15 for k in safe)
        path = tmp_path / f"{n}.json"
        import json

        path.write_text(json.dumps(dict(version=1, at="09:15:00", weights=safe)))
        load_targets(path)
    assert runner_safe({f"S{i:02d}": 1 / 11 for i in range(11)})[1] > 0


def test_strategy_file_round_trip_and_rejections():
    raw = Strategy(
        name="演示",
        source="rank(close)",
        settings=contract.normalize({"top_n": 10}),
        initial_cash=1e7,
        enabled=True,
        created_at="2026-11-14T10:00:00+08:00",
        origin={"kind": "expression"},
    ).to_dict()
    assert Strategy.from_dict(raw).to_dict() == raw
    for change in (
        dict(version=2),
        dict(extra=1),
        dict(enabled="yes"),
        dict(initial_cash=0),
        dict(initial_cash=True),
        dict(settings={"window": "test"}),  # 回测页白名单不收窗口
        dict(settings={"top_n": 0}),
    ):
        with pytest.raises(ValueError):
            Strategy.from_dict(raw | change)
    strategy = Strategy.from_dict(raw)
    assert strategy.execution["at"] == "09:15:00"
    close = Strategy.from_dict(
        raw | dict(settings=raw["settings"] | {"trade_at": "close"})
    )
    assert close.execution["at"] == "14:50:00"
    with pytest.raises(ValueError):
        parse_source("rank(ts_mean(close, [5, 10]))")  # 两个候选
    with pytest.raises(ValueError):
        parse_source("   ")
