"""共享任务输入与合并记账的逐位契约。"""

import numpy as np
import pytest

from ashare.factor import groupbt
from ashare.factor.groupbt.api import _factor_window
from bench.groupbt.common import compare, snapshot, synthetic


@pytest.mark.parametrize("mode", ["r", "r+", "c"])
def test_factor_mmap_ownership(tmp_path, mode):
    panel, factor = synthetic()
    path = tmp_path / "signal.npy"
    np.save(path, factor)
    mapped = np.load(path, mmap_mode=mode)
    mapped.flags.writeable = False
    window = np.asarray(mapped)[250:]
    owned = _factor_window(window)
    assert np.shares_memory(owned, mapped) == (mode == "r")
    options = dict(universe="ALL", record_holdings=True)
    a = groupbt.run(factor, panel, **options)
    b = groupbt.run(mapped, panel, **options)
    compare(snapshot(a), snapshot(b))
    compare(a.holdings, b.holdings)


def test_readonly_flag_does_not_make_writable_alias_safe():
    original = np.arange(50.0).reshape(5, 10).copy()
    alias = original.view()
    original.flags.writeable = False
    safe = _factor_window(original[1:])
    before = safe.tobytes()
    alias[:] = -0.0
    assert not np.shares_memory(safe, original)
    assert safe.tobytes() == before


@pytest.mark.parametrize("owner_readonly", [False, True])
def test_readonly_memoryview_does_not_hide_writable_owner(owner_readonly):
    owner = np.arange(50.0)
    alias = owner.view()
    owner.flags.writeable = not owner_readonly
    x = np.frombuffer(memoryview(owner).toreadonly(), dtype=np.float64).reshape(5, 10)
    assert not x.flags.writeable
    protected = _factor_window(x)
    before = protected.tobytes()
    alias[:] = -0.0
    assert not np.shares_memory(protected, owner)
    assert protected.tobytes() == before


def test_noncontiguous_immutable_input_still_copies():
    x = np.frombuffer(np.arange(50.0).tobytes(), dtype=np.float64).reshape(5, 10)
    assert _factor_window(x) is x
    assert not np.shares_memory(_factor_window(x[:, ::2]), x)


@pytest.mark.parametrize("workers", [1, 4])
def test_shared_pools_windows_and_ledgers_are_bitwise(workers):
    panel, factor = synthetic()
    factors = [factor, -factor]
    configurations = [
        dict(
            universe=pool,
            window=window,
            group_neutralize=mode,
            rebalance=rebalance,
            weights_portfolio=weighted,
            decay=3,
            record_holdings=True,
        )
        for pool in ("TOP800", "ALL")
        for window in (("20150101", "20150713"), groupbt.IS, groupbt.VALIDATION)
        for mode in (None, "cap+industry")
        for rebalance, weighted in ((1, False), (3, False), (3, True), (5, True))
    ]
    inputs = [factors[(i // 4) % 2] for i in range(len(configurations))]
    expected = [
        groupbt.run(x, panel, **c) for x, c in zip(inputs, configurations, strict=True)
    ]
    actual = groupbt.run_many(inputs, panel, configurations, workers=workers)
    for a, b in zip(expected, actual, strict=True):
        compare(snapshot(a), snapshot(b))
        compare(a.holdings, b.holdings)


def test_one_native_batch_and_ordered_callbacks(monkeypatch):
    import panelbt

    panel, factor = synthetic()
    configs = [
        dict(
            universe="ALL",
            rebalance=n,
            weights_portfolio=w,
            record_holdings=h,
            window=("20150101", "20260101"),
        )
        for n, w, h in ((3, True, False), (1, False, True), (3, False, True))
    ]
    expected = [groupbt.run(factor, panel, **c) for c in configs]
    calls, emitted = [], []
    native = panelbt.run_group_weights_batch

    def wrapped(*args, **kwargs):
        calls.append(kwargs["variants"])
        return native(*args, **kwargs)

    monkeypatch.setattr(panelbt, "run_group_weights_batch", wrapped)
    actual = groupbt.run_many(
        [factor] * 3,
        panel,
        configs,
        workers=2,
        on_result=lambda i, r: emitted.append((i, r)),
    )
    assert len(calls) == 1 and len(calls[0]) == 3
    assert {i for i, _ in emitted} == {0, 1, 2}
    for a, b in zip(expected, actual, strict=True):
        compare(snapshot(a), snapshot(b))
        if a.holdings is None:
            assert b.holdings is None
        else:
            compare(a.holdings, b.holdings)
    # 同调仓周期共享账本，但对外结果可独立修改。
    before = actual[2].pnl["long_short"].tobytes()
    actual[0].pnl["long_short"][:] = 0
    assert actual[2].pnl["long_short"].tobytes() == before


@pytest.mark.parametrize("changed", ["pool", "window", "close", "adjustment", "logcap"])
def test_exposure_cache_rejects_a_different_market_identity(changed):
    import panelbt

    from ashare.factor.groupbt.api import _prepare

    panel, factor = synthetic()
    prepared = _prepare(factor, panel, "ALL", _shared={})
    args, options = list(prepared.args), dict(prepared.options)
    if changed == "pool":
        args[5] = list(reversed(args[5]))
    elif changed == "window":
        options["signal_start"] -= 1
    else:
        i = {"close": 2, "adjustment": 3, "logcap": 4}[changed]
        args[i] = args[i].copy()
    with pytest.raises(ValueError, match="identical risk buffers"):
        panelbt.run_group_weights(*args, **options)


def test_vector_identity_and_parameters_control_target_sharing():
    panel, factor = synthetic()
    vector = np.random.default_rng(405).normal(size=factor.shape)
    configs = [
        dict(
            universe="ALL",
            vector_neut=v,
            group_neutralize=mode,
            n_groups=groups,
            trade_at=trade,
            trade_limitation=limited,
            window=("20150101", "20260101"),
            rebalance=n,
            weights_portfolio=True,
        )
        for v, mode, groups, trade, limited, n in (
            (vector, None, 5, "open", True, 1),
            (vector, None, 5, "open", True, 3),
            (-vector, "industry", 10, "close", False, 1),
        )
    ]
    expected = [groupbt.run(factor, panel, **c) for c in configs]
    for a, b in zip(
        expected, groupbt.run_many([factor] * 3, panel, configs), strict=True
    ):
        compare(snapshot(a), snapshot(b))


def test_external_executor_remains_owned_by_caller():
    from concurrent.futures import ThreadPoolExecutor
    from threading import get_ident

    panel, factor = synthetic()
    configs = [dict(universe="ALL", window=w) for w in (groupbt.IS, groupbt.VALIDATION)]
    expected = [groupbt.run(factor, panel, **c) for c in configs]
    caller = get_ident()
    jobs = []
    with ThreadPoolExecutor(max_workers=1) as executor:

        def completed(i, result):
            assert get_ident() == caller
            jobs.append(executor.submit(lambda: result.pnl["long_short"].tobytes()))

        actual = groupbt.run_many(
            [factor] * 2,
            panel,
            configs,
            workers=1,
            executor=executor,
            on_result=completed,
        )
        assert len([f.result(timeout=10) for f in jobs]) == 2
        assert executor.submit(lambda: 42).result(timeout=10) == 42
    for a, b in zip(expected, actual, strict=True):
        compare(snapshot(a), snapshot(b))


def test_zero_compaction_is_not_an_exact_exposure_reduction():
    # 同位置的零决定八路求和树。即使全是有限数，也不能压缩后再归约。
    products = np.array([1e16, 1.0, -1e16, 1.0, 0.0, -0.0, 0.0, -0.0])
    assert np.add.reduce(products) == 0.0
    assert np.add.reduce(products[products != 0.0]) == 1.0
    # 负零必须保留原 NumPy 身份元；NaN 不能因持仓为零而被略过。
    for values in (np.zeros(8), np.full(8, -0.0), np.array([np.nan, -0.0])):
        from panelbt import _core

        actual = _core._numpy_sums([values])
        assert actual.tobytes() == np.array([np.add.reduce(values)]).tobytes()


@pytest.mark.parametrize("pool", ["TOP800", "ALL"])
def test_same_factor_different_neutralization_never_share_targets(pool):
    """同一因子、只有中性化不同的变体不能合进同一批（2026-11-14 审查：批量键漏掉中性化时原有测试全过）。"""
    panel, factor = synthetic()
    configurations = [
        dict(universe=pool, group_neutralize=mode, rebalance=n, record_holdings=True)
        for mode in (None, "industry", "cap", "cap+industry")
        for n in (1, 3)
    ]
    expected = [groupbt.run(factor, panel, **c) for c in configurations]
    actual = groupbt.run_many([factor] * len(configurations), panel, configurations)
    for a, b in zip(expected, actual, strict=True):
        compare(snapshot(a), snapshot(b))
        compare(a.holdings, b.holdings)


def test_each_target_parameter_alone_prevents_sharing():
    """同一因子，每次只改一个影响每日目标的参数，结果都必须和单独运行逐位相同。

    2026-11-14 审查：从批量键里单独删掉成交时点、涨跌停限制、分组数、衰减或中性化向量，
    原有测试全过——原测试一次同时改好几个参数，删掉任何一个仍会被其他参数分开。
    """
    panel, factor = synthetic()
    rng = np.random.default_rng(406)
    vector, other = rng.normal(size=factor.shape), rng.normal(size=factor.shape)
    # 衰减只作用于全池权重组合（docs/GROUP_BACKTEST.md），所以基准配置打开它。
    base = dict(
        universe="ALL",
        rebalance=1,
        record_holdings=True,
        vector_neut=vector,
        weights_portfolio=True,
    )
    changes = [
        {},
        {"trade_at": "close"},
        {"trade_limitation": False},
        {"n_groups": 5},
        {"decay": 3},
        {"vector_neut": other},
        {"group_neutralize": "industry"},
    ]
    configurations = [dict(base, **change) for change in changes]
    # 每个配置再配一个只有调仓周期不同的兄弟，确保真的会成批。
    configurations += [dict(c, rebalance=3) for c in configurations]
    expected = [groupbt.run(factor, panel, **c) for c in configurations]
    actual = groupbt.run_many([factor] * len(configurations), panel, configurations)
    for a, b in zip(expected, actual, strict=True):
        compare(snapshot(a), snapshot(b))
        compare(a.holdings, b.holdings)
