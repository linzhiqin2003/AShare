"""来源：C8 数据防火墙、隐性前视、阶段闸门和计数验收；只用合成行情。"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ashare.data.crypto.storage import digest
from ashare.data.crypto.timeutil import DAY, HOUR, millis
from ashare.factor.stats import adjust
from ashare.factor.timing import Factor, Settings, Study, View, lookahead_check
from ashare.factor.timing.backtest import combine, run, trend_targets
from ashare.factor.timing.inference import calibration, labels, stage_mask
from ashare.factor.timing.ledger import StageLedger
from test.crypto.test_backtest import sample

T = millis("2024-01-01")


def causal(view, window):
    frame = view.hourly()
    px = frame["close"].to_numpy()
    out = np.full(len(px), np.nan)
    out[window:] = px[window:] / px[:-window] - 1
    return frame["available_at"].to_numpy(), out


def causal_renamed(other, lag):
    """函数名、形参名、局部变量名和文档不改变定义。"""
    data = other.hourly()
    close = data["close"].to_numpy()
    result = np.full(len(close), np.nan)
    result[lag:] = close[lag:] / close[:-lag] - 1
    return data["available_at"].to_numpy(), result


def tomorrow(view):
    frame = view.hourly()
    px = frame["close"].to_numpy()
    out = np.full(len(px), np.nan)
    out[:-24] = px[24:]
    return frame["available_at"].to_numpy(), out


def sealed(view):
    frame = view.hourly(start="2025-01-01", end="2025-02-01")
    return frame["available_at"].to_numpy(), frame["close"].to_numpy()


def full_mean(view):
    frame = view.hourly()
    px = frame["close"].to_numpy()
    return frame["available_at"].to_numpy(), (px - px.mean()) / px.std()


def wave(view):
    frame = view.hourly()
    return frame["available_at"].to_numpy(), np.sin(frame["time"].to_numpy() / 86400000)


def global_data(view):
    return T, np.array([1.0])


def settings(**changes):
    value = dict(
        ledger=str(Path("tmp/synthetic-ledger.jsonl").resolve()),
        symbol="BTCUSDT",
        trade_hour=0,
        horizon=7,
        stages=dict(
            mining=[T, T + 40 * DAY],
            validation=[T + 40 * DAY, T + 60 * DAY],
            confirmation=[T + 60 * DAY, T + 80 * DAY],
        ),
        validation_hash=None,
        confirmation_hash=None,
        repetitions=16,
    )
    value.update(changes)
    return Settings.from_dict(value)


class Synthetic:
    version = "synthetic-v1"

    def __init__(self, days=100):
        t = np.arange(T - 10 * DAY, T + days * DAY + HOUR, HOUR)
        px = 100 * np.exp(
            0.03 * np.sin(np.arange(len(t)) * 0.13) + np.arange(len(t)) * 0.00001
        )
        self.frame = pl.DataFrame(
            dict(
                time=t,
                open=px,
                close=px + 0.02,
                tradable=True,
                quality=0,
                funding_rate=np.where(np.arange(len(t)) % 8 == 7, 0.0001, np.nan),
                funding_price=px,
            )
        )

    def read(self, start=None, end=None, instruments=None, fields=None):
        f = self.frame
        if start is not None:
            f = f.filter(pl.col("time") >= millis(start))
        if end is not None:
            f = f.filter(pl.col("time") < millis(end))
        return f.clone()

    def arrays(self, fields, *, instruments, end=None):
        f = self.read(end=end)
        return dict(
            time=f["time"].to_numpy(),
            **{
                k: np.repeat(f[k].to_numpy()[:, None], len(instruments), axis=1)
                for k in fields
            },
        )


def test_settings_load_hash_dates_and_invalid(tmp_path):
    cfg = settings()
    path = tmp_path / "study.json"
    path.write_text(json.dumps(cfg.value, indent=4))
    assert Settings.load(path).hash == cfg.hash
    with pytest.raises(ValueError, match="调仓时点或持有期非法"):
        settings(trade_hour=24)
    stages = cfg.value["stages"]
    stages["validation"][0] = T
    with pytest.raises(ValueError, match="重叠"):
        settings(stages=stages)


def test_definition_normalization_and_parameter_identity():
    a = Factor(causal, params=dict(window=24))
    b = Factor(causal_renamed, params=dict(lag=24))
    assert a.definition_hash == b.definition_hash
    assert Factor(causal, params=dict(window=48)).definition_hash != a.definition_hash
    with pytest.raises(ValueError, match="全局"):
        Factor(global_data)


def test_parameters_cannot_change_after_hash():
    # 来源：开发复核，公开可变 params 会让同哈希对应不同因子。
    f = Factor(causal, params=dict(window=24))
    f.params["window"] = 48
    assert f.params["window"] == 24


def test_firewall_and_lookahead_reject_next_day_and_sealed():
    view = View(Synthetic().frame, symbol="BTCUSDT", cutoff=T + 40 * DAY)
    assert view.hourly()["available_at"].max() < T + 40 * DAY
    for fn in (tomorrow, sealed, full_mean):
        with pytest.raises(ValueError, match="前视|防火墙"):
            lookahead_check(Factor(fn), view)
    a, cuts = lookahead_check(Factor(causal, params=dict(window=24)), view)
    b, cuts2 = lookahead_check(Factor(causal, params=dict(window=24)), view)
    assert cuts == cuts2 and len(cuts) >= 3
    np.testing.assert_array_equal(a[1], b[1])


def test_firewall_all_readers_at_end_excluded(monkeypatch):
    from ashare.data.crypto import metrics, minute

    end = T + DAY
    view = View(
        Synthetic().frame,
        symbol="BTCUSDT",
        cutoff=end,
        minute_source=("synthetic", "v"),
        metrics_source=("synthetic", "v"),
    )
    monkeypatch.setattr(
        minute,
        "bars",
        lambda *a, **k: pl.DataFrame(dict(open_time=[end - 120000, end - 60000, end])),
    )
    monkeypatch.setattr(
        metrics,
        "read",
        lambda *a, **k: pl.DataFrame(
            dict(
                time=[end - 600000, end - 300000, end],
                available_at=[end - 300000, end, end + 300000],
            )
        ),
    )
    assert view.minutes(T)["available_at"].to_list() == [end - 60000]
    assert view.metrics(T)["available_at"].to_list() == [end - 300000]
    for read in (view.hourly, view.minutes, view.metrics):
        with pytest.raises(ValueError, match="防火墙"):
            read(T, end + HOUR)
    with pytest.raises(ValueError, match="防火墙不能扩大"):
        view.truncate(end + HOUR)


def test_cross_boundary_labels_hand_calculation():
    # 入场 Jan 1..5，持有2天，阶段 Jan 2..5：仅 Jan2/Jan3；Jan3退出恰等于末端。
    entries = T + np.arange(5) * DAY
    assert stage_mask(entries, 2, T + DAY, T + 4 * DAY).tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]
    ds = Synthetic()
    # 固定100→110→121…，单日 ln(1.1)，2日可加=2ln(1.1)；退出边界开盘可供标签使用。
    t = ds.frame["time"].to_numpy()
    px = 100 * 1.1 ** ((t - T) // DAY)
    ds.frame = ds.frame.with_columns(
        pl.Series("open", px), pl.lit(np.nan).alias("funding_rate")
    )
    cfg = settings(
        horizon=2,
        stages=dict(
            mining=[T + DAY, T + 4 * DAY],
            validation=[T + 4 * DAY, T + 6 * DAY],
            confirmation=[T + 6 * DAY, T + 8 * DAY],
        ),
    )
    values = (entries, np.arange(5, dtype=float))
    result = labels(ds, values, cfg, "mining")
    assert result["entries"][result["mask"]].tolist() == [T + DAY, T + 2 * DAY]
    assert result["y"][result["mask"]].tolist() == pytest.approx([0.21, 0.21])
    assert result["g"][1:4].tolist() == pytest.approx([np.log(1.1)] * 3)
    assert np.isnan(result["g"][0])


def test_calibration_excludes_warmup_and_heldout():
    # 来源：开发复核，C6 的 sd_x 也必须排除挖掘开始之前的预热。
    cfg = settings()
    rng = np.random.default_rng(8)
    entries = T + np.arange(-40, 80) * DAY
    sample = dict(entries=entries, x=rng.normal(size=120), g=rng.normal(size=120))
    a = calibration(sample, cfg)
    sample["x"][:40] = 1e9
    sample["g"][:40] = -1e9
    sample["x"][80:] = 1e12
    sample["g"][80:] = -1e12
    b = calibration(sample, cfg)
    for key in a:
        np.testing.assert_equal(a[key], b[key])


def study_with_lists(tmp_path):
    f = Factor(wave)
    path = tmp_path / "list.json"
    path.write_text(json.dumps([settings().test_hash(f.definition_hash)]))
    sha = digest(path.read_bytes())
    cfg = settings(
        ledger=str(tmp_path / "ledger.jsonl"),
        validation_hash=sha,
        confirmation_hash=sha,
    )
    study = Study(cfg, Synthetic(), ledger=tmp_path / "ledger.jsonl")
    return study, f, path


def test_mining_repeat_definition_does_not_increment(tmp_path):
    study, f, path = study_with_lists(tmp_path)
    a = study.evaluate(f)
    n = len(study.ledger.records())
    b = study.evaluate(Factor(wave))
    assert len(study.ledger.records()) == n
    assert a["n"] == b["n"]
    assert study.summary()["attempts"] == 1
    assert all(
        r["settings_hash"] == study.settings.hash for r in study.ledger.records()
    )


def test_validation_once_list_hash_and_frozen_dates(tmp_path):
    study, f, path = study_with_lists(tmp_path)
    study.evaluate(f)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="哈希"):
        study.evaluate(f, stage="validation", list_path=path)
    path.write_text(path.read_text()[:-1])
    study.evaluate(f, stage="validation", list_path=path)
    with pytest.raises(ValueError, match="一次"):
        study.evaluate(f, stage="validation", list_path=path)
    cfg = study.settings.value
    cfg["stages"]["mining"][0] += DAY
    with pytest.raises(ValueError, match="阶段日期"):
        Study(Settings.from_dict(cfg), Synthetic(), ledger=tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError, match="预约"):
        study.view("confirmation")


def test_confirmation_whole_list_once_and_hash(tmp_path, monkeypatch):
    study, f, path = study_with_lists(tmp_path)
    study.evaluate(f)
    study.evaluate(f, stage="validation", list_path=path)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="哈希"):
        study.confirm([f], list_path=path)
    path.write_text(path.read_text()[:-1])
    # 此测试只检验闸门，真实 panelbt 路径另有合成集成测试。
    monkeypatch.setattr(
        "ashare.factor.timing.backtest.run", lambda *a: dict(combined={})
    )
    result = study.confirm([f], list_path=path)
    assert result["family_size"] == 1
    with pytest.raises(ValueError, match="一次"):
        study.confirm([f], list_path=path)


def test_failed_reserved_stage_cannot_retry(tmp_path, monkeypatch):
    study, f, path = study_with_lists(tmp_path)
    study.evaluate(f)

    def fail(*args):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(study, "_evaluate", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        study.evaluate(f, stage="validation", list_path=path)
    with pytest.raises(ValueError, match="一次"):
        study.evaluate(f, stage="validation", list_path=path)


def test_bh_matches_existing_and_duplicate_definition_count(tmp_path):
    study = Study(settings(ledger=str(tmp_path / "ledger")), Synthetic())
    factors = [Factor(causal, params=dict(window=k)) for k in (12, 24, 48)]
    for f in factors:
        study.evaluate(f)
    rows = study.ledger.records()
    completed = [r for r in rows if r["event"] == "completed"]
    for r, p in zip(completed, (0.01, 0.3, np.nan), strict=True):
        r["result"]["p"] = p
    import copy

    repeat = copy.deepcopy(completed[0])
    repeat["result"]["p"] = 0.02
    rows.append(repeat)
    study.ledger.ledger.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = study.summary()
    assert summary["attempts"] == 3
    np.testing.assert_equal(
        [summary["q"][r["test_hash"]] for r in completed],
        adjust([0.02, 0.3, np.nan], "bh"),
    )


def test_corrupt_ledger_fails_closed(tmp_path):
    path = tmp_path / "ledger"
    path.write_text('{"stage":"validation"')
    with pytest.raises(ValueError, match="损坏"):
        Study(settings(ledger=str(path)), Synthetic(), ledger=path)


def test_combination_hand_normalization_equal_weight_and_clipping():
    at = T + np.arange(3) * DAY
    days, w = combine(
        [(at, np.array([1.0, 5.0, np.nan])), (at, np.array([0.0, -2.0, 2.0]))],
        [dict(mean=1.0, std=2.0), dict(mean=0.0, std=1.0)],
        0,
    )
    np.testing.assert_array_equal(days, at - DAY)
    np.testing.assert_array_equal(w[:, 0], [0, 0, 1])
    with pytest.raises(ValueError, match="标准化参数退化"):
        combine([(at, np.ones(3))], [dict(mean=0, std=0)], 0)


def test_confirmation_backtest_runs_real_panelbt_on_synthetic(tmp_path):
    ds = sample(tmp_path, kind="usdm", hours=80 * 24 + 1)
    cfg = settings(assume="conservative")
    at = T + np.arange(80) * DAY
    out = run(ds, [(at, np.sin(np.arange(80)))], [dict(mean=0.0, std=1.0)], cfg)
    assert set(out) == {"combined", "hold", "trend"}
    assert all(v["valid"] and np.isfinite(v["final_nav"]) for v in out.values())
    assert out["hold"]["funding_paid"] > 0
    assert out["hold"]["fees"] > 0
    assert out["trend"]["final_nav"] == 1
    # C3 规则使用相同日线、投票和波动，仅以明确的一倍名义上限比较。
    from bench.crypto_examples import signals

    expected = signals(ds, "usdm", symbols=("BTCUSDT",), gross_cap=1.0)
    actual = trend_targets(ds, "BTCUSDT", 0, T + 80 * DAY)
    np.testing.assert_equal(actual[1], expected[1][: len(actual[1])])


def test_trend_benchmark_matches_c3_after_full_warmup():
    # 来源：开发复核，80日回测的趋势全处于预热；另测真实触发仓位后的 C3 对齐。
    from bench.crypto_examples import signals

    class DailySource:
        def daily(self, **kwargs):
            n = np.arange(180)
            frame = pl.DataFrame(
                dict(
                    day=T + n * DAY,
                    instrument="usdm:BTCUSDT",
                    close=100 * np.exp(0.002 * n + 0.08 * np.sin(n / 7)),
                )
            )
            if "end" in kwargs:
                frame = frame.filter(pl.col("day") < kwargs["end"])
            return frame

    source = DailySource()
    expected = signals(source, "usdm", symbols=("BTCUSDT",), gross_cap=1.0)
    actual = trend_targets(source, "BTCUSDT", 0, T + 170 * DAY)
    assert np.count_nonzero(actual[1][120:]) > 30
    assert np.max(np.abs(actual[1])) <= 1
    np.testing.assert_equal(actual[0], expected[0][:170])
    np.testing.assert_equal(actual[1], expected[1][:170])


def test_list_limits_and_nonmember_rejected(tmp_path):
    from ashare.factor.timing.settings import read_list

    path = tmp_path / "list.json"
    for limit in (5, 10):
        path.write_text(json.dumps([f"{i:064x}" for i in range(limit + 1)]))
        with pytest.raises(ValueError, match="名额"):
            read_list(path, digest(path.read_bytes()), limit)
    study, f, path = study_with_lists(tmp_path)
    outsider = Factor(causal, params=dict(window=24))
    study.evaluate(outsider)
    with pytest.raises(ValueError, match="不在验证名单"):
        study.evaluate(outsider, stage="validation", list_path=path)


def test_confirmation_keeps_unestimable_slot_in_holm(tmp_path, monkeypatch):
    factors = [Factor(wave), Factor(causal, params=dict(window=24))]
    path = tmp_path / "list.json"
    path.write_text(
        json.dumps([settings().test_hash(f.definition_hash) for f in factors])
    )
    sha = digest(path.read_bytes())
    study = Study(
        settings(
            ledger=str(tmp_path / "ledger"), validation_hash=sha, confirmation_hash=sha
        ),
        Synthetic(),
        ledger=tmp_path / "ledger",
    )
    for f in factors:
        study.evaluate(f)
        study.ledger.append(
            "validation",
            "completed",
            definition_hash=f.definition_hash,
            test_hash=study.test_hash(f),
            result=dict(p=0.5),
        )

    def fake(f, stage):
        return dict(p=0.01 if f is factors[0] else np.nan), (
            T + np.arange(10) * DAY,
            np.arange(10, dtype=float),
        )

    monkeypatch.setattr(study, "_evaluate", fake)
    monkeypatch.setattr("ashare.factor.timing.backtest.run", lambda *a: {})
    result = study.confirm(factors, list_path=path)
    assert result["family_size"] == 2
    assert result["candidates"][0]["holm_p"] == 0.02
    assert np.isnan(result["candidates"][1]["holm_p"])


def test_gate_lock_serializes_concurrent_reservations(tmp_path):
    # 来源：只跑一次必须由锁保护 read-check-append，而不只是 append 有锁。
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "ledger"

    def reserve(_):
        ledger = StageLedger(path, settings(ledger=str(path)), "v1")
        try:
            with ledger.locked():
                ledger.reserve("validation", ["a"])
            return True
        except ValueError as exc:
            assert "验证候选只能运行一次" in str(exc)
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve, range(2))) == [False, True]


def test_public_view_never_reads_after_mining_bound(tmp_path):
    class Recording(Synthetic):
        def read(self, *a, **kw):
            assert kw["end"] <= T + 40 * DAY
            return super().read(*a, **kw)

    study = Study(
        settings(ledger=str(tmp_path / "ledger")),
        Recording(),
        ledger=tmp_path / "ledger",
    )
    assert study.view().hourly()["available_at"].max() < T + 40 * DAY


def test_c6_boundary_is_parameter_default_is_unchanged():
    from ashare.factor.crypto_eval import BOUNDARY, calibrate, sample_mask

    entries = BOUNDARY + np.arange(-100, 100) * DAY
    rng = np.random.default_rng(2)
    x, g = rng.normal(size=(2, 200))
    np.testing.assert_equal(
        sample_mask(entries, 7, "in"), sample_mask(entries, 7, "in", boundary=BOUNDARY)
    )
    assert sample_mask(entries, 7, "in", boundary=BOUNDARY + 20 * DAY).sum() == 113
    a = calibrate(x, g, entries)
    b = calibrate(x, g, entries, boundary=BOUNDARY)
    for k in a:
        np.testing.assert_equal(a[k], b[k])
    assert calibrate(x, g, entries, boundary=BOUNDARY + 20 * DAY)["n"] > a["n"]


def test_firewall_cutoff_and_symbol_cannot_be_reassigned():
    # 来源：开发复核，因子不得修改公开截止时刻再调用分钟/metrics 读取器。
    view = View(Synthetic().frame, symbol="BTCUSDT", cutoff=T + 40 * DAY)
    with pytest.raises(AttributeError, match="has no setter"):
        view.cutoff = T + 1000 * DAY
    with pytest.raises(AttributeError, match="has no setter"):
        view.symbol = "ETHUSDT"


def test_c6_benchmark_checkpoint_at_horizon_boundary():
    # 来源：共享机器的 C6 长组接近 840 秒；检查点只控制保存和续跑，不改统计核。
    from ashare.factor.crypto import FACTOR_NAMES
    from ashare.factor.crypto_eval import BOUNDARY
    from bench.crypto_factors_report import evaluate_group

    rng = np.random.default_rng(91)
    n = 40
    task = (
        "BTCUSDT",
        0,
        "usdm",
        "in",
        rng.normal(size=n),
        {name: rng.normal(size=n) for name in FACTOR_NAMES},
        BOUNDARY + np.arange(-n - 2, -2) * DAY,
        {h: rng.normal(size=n) for h in (1, 7, 30)},
        4000,
        False,
    )
    saved = []
    result = evaluate_group(
        task, horizons=(1,), checkpoint=lambda rows: saved.append(list(rows))
    )
    assert len(saved) == 1 and len(saved[0]) == len(result) == 12
    assert all(r["horizon"] == 1 for r in result)


def test_view_real_reader_adapters_on_synthetic_versions(tmp_path):
    # 来源：防火墙不仅检验替身，还通过真实分钟/metrics 版本读取器验证结束边界。
    from ashare.data.crypto import metrics
    from test.crypto.test_metrics_c8 import T as begin
    from test.crypto.test_metrics_c8 import fixture

    mr, kr = fixture(tmp_path)
    metrics.build(root=mr, minute_root=kr)
    cfg = settings(
        ledger=str(tmp_path / "ledger"),
        stages=dict(
            mining=[begin, begin + 2 * DAY],
            validation=[begin + 2 * DAY, begin + 3 * DAY],
            confirmation=[begin + 3 * DAY, begin + 4 * DAY],
        ),
    )
    study = Study(
        cfg, Synthetic(), ledger=tmp_path / "ledger", minute_root=kr, metrics_root=mr
    )
    view = study.view()
    mf = view.metrics(begin - 300000)
    kf = view.minutes(begin)
    assert mf.height == 575 and mf["available_at"].max() < begin + 2 * DAY
    assert kf.height == 2879 and kf["available_at"].max() == begin + 2 * DAY - 60000
    assert study.data_versions["metrics"] == metrics.metadata(root=mr)["version"]


def test_failed_mining_revision_does_not_reuse_old_p_for_bh(tmp_path):
    # 来源：开发复核，同定义新版本失败不能在 BH 中继续展示旧成功结果。
    ledger = StageLedger(
        tmp_path / "ledger", settings(ledger=str(tmp_path / "ledger")), "v1"
    )
    ledger.append("mining", "completed", test_hash="a", result=dict(p=0.01))
    ledger.append("mining", "started", test_hash="a")
    ledger.append("mining", "failed", test_hash="a")
    assert ledger.mining_summary()["attempts"] == 1
    assert np.isnan(ledger.mining_summary()["q"]["a"])


def test_mutation_runner_preserves_native_build_guard(tmp_path, monkeypatch):
    # 来源：C8 变异初跑，复制测试树会让 build_guard 误判本 worktree 的扩展。
    from bench import mutations_c8

    calls = []
    monkeypatch.setattr(
        mutations_c8.subprocess, "run", lambda *a, **kw: calls.append(kw)
    )
    mutations_c8.run(tmp_path, ["test/crypto/test_metrics_c8.py"])
    assert calls[0]["cwd"] == mutations_c8.ROOT
    assert calls[0]["env"]["PYTHONPATH"] == str(tmp_path / "src")
    assert calls[0]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_factor_definition_identity_is_read_only():
    # 来源：最终接口复核，公开定义哈希不能赋值来冒用名单中的其他因子。
    factor = Factor(causal, params=dict(window=24))
    with pytest.raises(AttributeError, match="只读"):
        factor.definition_hash = "0" * 64


def test_study_settings_reference_cannot_change_after_validation(tmp_path):
    # 来源：最终接口复核，Settings 内容冻结之外，Study 的公开引用也不能替换。
    study, factor, path = study_with_lists(tmp_path)
    study.evaluate(factor)
    study.evaluate(factor, stage="validation", list_path=path)
    value = study.settings.value
    value["stages"]["mining"][0] -= DAY
    with pytest.raises(AttributeError, match="has no setter"):
        study.settings = Settings.from_dict(value)
