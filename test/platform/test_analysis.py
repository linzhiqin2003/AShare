"""逐位对拍、窗口隔离和检验口径。"""

import base64
import math
from statistics import NormalDist

import numpy as np
import pytest

from ashare.factor import Evaluator, groupbt
from ashare.platform.analysis import checks, plot_payload, run_variant
from ashare.platform.config import DEFAULTS
from ashare.platform.language import parse
from bench.platform.synthetic import panel


@pytest.mark.parametrize("neutralize", ["industry", "cap+industry"])
@pytest.mark.parametrize("window", [20, 60])
def test_bitwise_equal_to_direct_groupbt(neutralize, window):
    source = panel()
    expression = parse(f"-ts_std(returns, {window})", factors={})[0].expression
    settings = DEFAULTS | {"group_neutralize": neutralize, "weights_portfolio": True}
    summary, arrays = run_variant(expression, source, settings)
    alpha = Evaluator().evaluate(expression, source)
    for part, dates in (("is", groupbt.IS), ("validation", groupbt.VALIDATION)):
        options = settings | ({"sub_universes": False} if part == "validation" else {})
        direct = groupbt.run(alpha, source, **options, window=dates)
        mask = direct.dates >= dates[0]
        assert summary[part]["sharpe"] == direct.table().to_dicts()[-1]["sharpe"]
        np.testing.assert_array_equal(
            arrays[f"{part}.groups"].view(np.uint64),
            direct.group_pnl[:, mask].copy().view(np.uint64),
        )
        for name, values in direct.pnl.items():
            np.testing.assert_array_equal(
                arrays[f"{part}.pnl.{name}"].view(np.uint64),
                values[mask].view(np.uint64),
            )
        for name, values in direct.turnover.items():
            np.testing.assert_array_equal(
                arrays[f"{part}.turnover.{name}"].view(np.uint64),
                values[mask].view(np.uint64),
            )


def test_long_view_only_packs_first_group_without_mutating_results():
    arrays = {
        f"{part}.{key}": value
        for part in ("is", "validation")
        for key, value in {
            "dates": np.array(["20240102", "20240103", "20240104"]),
            "groups": np.array([[0.03, -0.02, np.nan], [-0.01, 0.04, 0.01]]),
        }.items()
    }
    before = {key: value.copy() for key, value in arrays.items()}
    for part in plot_payload(arrays, "long"):
        assert set(part["series"]) == {"pnl.long"}
        packed = part["series"]["pnl.long"]
        assert packed["shape"] == [3]
        result = np.frombuffer(base64.b64decode(packed["data"]), dtype="<f4")
        np.testing.assert_array_equal(
            result, arrays[f"{part['part']}.groups"][0].astype("<f4")
        )
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, before[key])


def test_test_window_never_reaches_evaluation_or_backtest(monkeypatch):
    source = panel()
    calls, evaluated = [], []
    actual_run, actual_evaluate = groupbt.run, Evaluator.evaluate

    def recording_run(alpha, source, **options):
        assert options["window"] in (groupbt.IS, groupbt.VALIDATION)
        assert source.dates[-1] < options["window"][1]
        calls.append(options)
        return actual_run(alpha, source, **options)

    def recording_evaluate(self, expression, source, **options):
        evaluated.append(source.dates[-1])
        assert source.dates[-1] < groupbt.VALIDATION[1]
        return actual_evaluate(self, expression, source, **options)

    monkeypatch.setattr(groupbt, "run", recording_run)
    monkeypatch.setattr(Evaluator, "evaluate", recording_evaluate)
    run_variant(
        parse("-ts_std(returns,20)", factors={})[0].expression, source, DEFAULTS
    )
    assert len(calls) == 2
    assert len(evaluated) == 1
    assert calls[1]["sub_universes"] is False


def test_check_algorithms_and_missing_evidence():
    summary, _ = run_variant(
        parse("-ts_std(returns,20)", factors={})[0].expression, panel(), DEFAULTS
    )
    rows = {
        r["key"]: r
        for r in checks(summary, 100, [{"name": "synthetic", "value": -0.9}])
    }
    a = summary["is"]
    assert a["fitness"] == a["sharpe"] * math.sqrt(
        abs(a["return"]) / max(a["turnover"], 0.125)
    )
    assert a["t"] == a["sharpe"] * math.sqrt(a["n"] / 240)
    assert rows["t"]["threshold"] == f"≥ {NormalDist().inv_cdf(1 - 0.025 / 100):.2f}"
    assert rows["correlation"]["status"] == "fail"
    assert rows["exposure"]["status"] == "hint"
    summary["is"]["subs"] = {}
    rows = {r["key"]: r for r in checks(summary, 1)}
    assert rows["sub_sharpe"]["status"] == rows["correlation"]["status"] == "hint"


@pytest.mark.parametrize(
    "text",
    [
        "ts_std(returns,20)",
        "ts_mean(close,60)",
        "rank(close)",
        "ts_delay(returns,5)",
        "rank(ts_mean(ts_delay(returns,2),20))",
    ],
)
def test_causal_validation_prefix_is_bitwise_equal(text):
    from ashare.platform.analysis import CutPanel, evaluate_signal

    source = panel(full=True)
    expression = parse(text, factors={})[0].expression
    shared = evaluate_signal(expression, source)
    for end in (groupbt.IS[1], groupbt.VALIDATION[1]):
        cut = CutPanel(source, end)
        separate = Evaluator().evaluate(expression, cut)
        np.testing.assert_array_equal(
            shared[: cut.stop].view(np.uint64), separate.view(np.uint64)
        )


def test_config_variants_share_one_evaluation_per_expression(tmp_path, monkeypatch):
    from ashare.platform import runner
    from bench.platform.synthetic import write_panel

    write_panel(tmp_path / "data")
    calls, progress = [], []
    actual = runner.evaluate_signal

    def recording(expression, source):
        calls.append(expression.digest)
        return actual(expression, source)

    monkeypatch.setattr(runner, "evaluate_signal", recording)
    runs = [{"source": f"-ts_std(returns,{w})"} for _ in range(2) for w in (20, 60)]
    prepared = runner.prepare_signals(
        runs,
        runner.PanelCache(tmp_path / "data"),
        tmp_path / "signals",
        lambda i, phase: progress.append((i, phase)),
    )
    assert len(calls) == len(set(calls)) == 2
    assert prepared[0]["signal"] == prepared[2]["signal"]
    assert prepared[1]["signal"] == prepared[3]["signal"]
    assert all(
        not np.load(run["signal"], mmap_mode="r").flags.writeable for run in prepared
    )
    assert progress == [(i, p) for i in range(4) for p in ("evaluate", "prepared")]


def test_benchmark_does_not_change_backtest_bits():
    expression = parse("rank(close)", factors={})[0].expression
    source = panel()
    summary, original = run_variant(expression, source, DEFAULTS)
    changed, actual = run_variant(
        expression, source, DEFAULTS | {"benchmark": "000001.SH"}
    )
    assert summary == changed
    for key, value in original.items():
        np.testing.assert_array_equal(
            value.view(np.uint64) if value.dtype.kind == "f" else value,
            actual[key].view(np.uint64) if value.dtype.kind == "f" else actual[key],
        )


def test_shared_diagnostics_do_not_change_any_result_bits():
    source = panel()
    expression = parse("rank(close)", factors={})[0].expression
    cache = {}
    for neutralize in ("industry", "cap+industry"):
        settings = DEFAULTS | {"group_neutralize": neutralize}
        expected_summary, expected = run_variant(expression, source, settings)
        actual_summary, actual = run_variant(
            expression, source, settings, diagnostics=cache
        )
        assert actual_summary == expected_summary
        for key, value in expected.items():
            np.testing.assert_array_equal(
                actual[key].view(np.uint64)
                if value.dtype == np.float64
                else actual[key],
                value.view(np.uint64) if value.dtype == np.float64 else value,
            )
    assert set(cache) == {"is", "validation"}


def test_shared_ic_inputs_are_bitwise_identical(tmp_path):
    """任务内缓存的股票池和未来收益：多个信号、两种成交时点、两个股票池复用后，输出与不缓存逐位相同。"""
    from ashare.platform.analysis import evaluate_signal, run_variant
    from ashare.platform.config import DEFAULTS
    from ashare.platform.language import parse
    from bench.platform.synthetic import write_panel

    source = write_panel(tmp_path / "data")

    class Cached:
        diagnostics = {}

        def __getattr__(self, name):
            return getattr(source, name)

        def __getitem__(self, name):
            return source[name]

    cached = Cached()
    for text in ("rank(close)", "-ts_std(returns, 5)"):
        expression = parse(text, factors={})[0].expression
        for trade_at in ("open", "close"):
            for universe in ("TOP1800", "ALL"):
                settings = dict(DEFAULTS, trade_at=trade_at, universe=universe)
                plain = run_variant(expression, source, settings)
                shared = run_variant(
                    expression,
                    cached,
                    settings,
                    signal=evaluate_signal(expression, cached),
                )
                assert plain[0] == shared[0]
                assert plain[1].keys() == shared[1].keys()
                for key, value in plain[1].items():
                    assert value.dtype == shared[1][key].dtype, key
                    assert value.tobytes() == shared[1][key].tobytes(), key
    assert sum(1 for k in Cached.diagnostics if k[0] == "ic-inputs") == 8
