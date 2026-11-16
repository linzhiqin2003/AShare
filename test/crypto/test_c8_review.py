"""来源：独立审查 review_c8.md 的 P0/P1 最小复现；所有数据均为合成。"""

import json

import numpy as np
import polars as pl
import pytest

from ashare.data.crypto import metrics as m
from ashare.data.crypto.timeutil import DAY, HOUR
from ashare.data.crypto.verify_metrics import _alignment
from ashare.factor.stats import adjust
from ashare.factor.timing import Factor, Study
from test.crypto.test_timing_c8 import Synthetic, T, settings, wave


@pytest.mark.parametrize("entry", ["mining", "validation", "confirmation"])
@pytest.mark.parametrize("kind", ["plain", "forged", "subclass"])
def test_uncontrolled_callable_rejected_before_cache_and_lists(tmp_path, entry, kind):
    ds = Synthetic()
    raw = ds.frame

    def future(view):
        at = view.hourly()["available_at"].to_numpy()
        return at, raw["close"].to_numpy()[
            np.searchsorted(raw["time"].to_numpy(), at + DAY)
        ]

    class Override(Factor):
        def __call__(self, view):
            return future(view)

    candidate = Override(wave) if kind == "subclass" else future
    if kind == "forged":
        candidate.definition_hash = Factor(wave).definition_hash
        candidate.family = "external_future"
    study = Study(
        settings(ledger=str(tmp_path / "ledger")), ds, ledger=tmp_path / "ledger"
    )
    with pytest.raises(ValueError, match="FACTOR_TYPE"):
        if entry == "confirmation":
            study.confirm([candidate], list_path=tmp_path / "missing-list")
        else:
            study.evaluate(candidate, stage=entry, list_path=tmp_path / "missing-list")
    assert study.ledger.records() == []


def test_forged_hash_cannot_poison_legitimate_cache(tmp_path):
    ds = Synthetic()
    raw = ds.frame
    legit = Factor(wave)

    def future(view):
        at = view.hourly()["available_at"].to_numpy()
        return at, raw["close"].to_numpy()[
            np.searchsorted(raw["time"].to_numpy(), at + DAY)
        ]

    future.definition_hash = legit.definition_hash
    future.family = "external_future"
    study = Study(
        settings(ledger=str(tmp_path / "ledger")), ds, ledger=tmp_path / "ledger"
    )
    with pytest.raises(ValueError, match="FACTOR_TYPE"):
        study.evaluate(future)
    result = study.evaluate(legit)
    assert result["family"] == legit.family
    assert len(study.ledger.records()) == 2
    np.testing.assert_equal(study.evaluate(legit), result)
    with pytest.raises(ValueError, match="FACTOR_TYPE"):
        study.evaluate(future)
    assert len(study.ledger.records()) == 2


@pytest.mark.parametrize("field", ["_code", "_params"])
def test_execution_and_parameters_cannot_be_replaced(tmp_path, field):
    factor = Factor(wave)
    replacement = (
        compile("def _factor(v): return None", "fake", "exec")
        if field == "_code"
        else b"{}"
    )
    with pytest.raises(AttributeError, match="只读"):
        setattr(factor, field, replacement)


@pytest.mark.parametrize("hour", [1, 11, 13, 23])
def test_non_midnight_switch_rejected_from_minute_volumes(hour):
    from ashare.data.crypto.timeutil import millis

    start = millis("2024-03-03")
    stamps = start + np.arange(3 * 288) * m.STEP
    mt = np.arange(start - m.STEP, start + 3 * DAY, 60000)
    buy = np.where((mt // m.STEP) % 2, 1.0, 3.0)
    windows = m.minute_windows(
        pl.DataFrame(dict(open_time=mt, volume=buy + 1, taker_buy_volume=buy))
    )
    truth = (stamps >= start + DAY + hour * HOUR).astype(int)
    raw = pl.DataFrame(
        {"create_time": stamps, **{f: np.ones(len(stamps)) for f in m.FIELDS.values()}}
    ).with_columns(
        pl.Series(
            "taker_buy_sell_ratio",
            [
                windows[int(t) - (1 - int(a)) * m.STEP]
                for t, a in zip(stamps, truth, strict=True)
            ],
        )
    )
    errors = m.errors(raw, windows)
    with pytest.raises(
        ValueError, match="SWITCH_WINDOW" if hour < 12 else "REGISTERED_ALIGNMENT"
    ):
        m.normalize(raw, errors)
    with pytest.raises(
        ValueError, match="SWITCH_WINDOW" if hour < 12 else "REGISTERED_ALIGNMENT"
    ):
        _alignment(stamps, errors, 0.01, 0.1)


@pytest.mark.parametrize("length", [1, 2])
def test_real_longest_isolated_reverse_and_k_boundary(length):
    # BTC 30 段、ETH 25 段反向证据均长 1；夹具仅复现长度，不使用真实行。
    stamps = T + np.arange(288) * m.STEP
    error = np.tile([0.001, 0.4], (288, 1))
    error[100 : 100 + length] = [0.4, 0.001]
    if length == 1:
        assert m.alignment_days(stamps, error)[0]["alignment"] == 0
        assert _alignment(stamps, error, 0.01, 0.1)[0] == [0] * 288
    else:
        with pytest.raises(ValueError, match="SUSTAINED_OPPOSITE"):
            m.alignment_days(stamps, error)
        with pytest.raises(ValueError, match="SUSTAINED_OPPOSITE"):
            _alignment(stamps, error, 0.01, 0.1)


@pytest.mark.parametrize("samples", [119, 120])
def test_minimum_paired_daily_samples(samples):
    stamps = T + np.arange(samples) * m.STEP
    error = np.tile([0.001, 0.4], (samples, 1))
    assert m.alignment_days(stamps, error)[0]["inherited"] == int(samples < 120)
    assert _alignment(stamps, error, 0.01, 0.1)[1] == [samples < 120] * samples
    assert m.alignment_days(stamps, error)[0]["alignment"] == 0
    assert _alignment(stamps, error, 0.01, 0.1)[0] == [0] * samples


def test_ledger_path_cannot_reset_gates(tmp_path):
    cfg = settings(ledger=str(tmp_path / "ledger"))
    Study(cfg, Synthetic(), ledger=tmp_path / "ledger")
    with pytest.raises(ValueError, match="LEDGER_PATH"):
        Study(cfg, Synthetic(), ledger=tmp_path / "other")


def test_label_horizons_are_three_tests_and_bh(tmp_path):
    f = Factor(wave)
    studies = []
    for h in [1, 2, 7]:
        s = Study(
            settings(ledger=str(tmp_path / "ledger"), horizon=h),
            Synthetic(),
            ledger=tmp_path / "ledger",
        )
        s.evaluate(f)
        studies.append(s)
    assert s.summary()["attempts"] == 3
    rows = s.ledger.records()
    completed = [r for r in rows if r["event"] == "completed"]
    assert [r["result"]["horizon"] for r in completed] == [1, 2, 7]
    for r, p in zip(completed, [0.8, 0.3, 0.01], strict=True):
        r["result"]["p"] = p
    s.ledger.ledger.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    expected = adjust([0.8, 0.3, 0.01], "bh")
    np.testing.assert_equal(
        [s.summary()["q"][r["test_hash"]] for r in completed], expected
    )


def test_hash_follows_actual_executable_even_after_private_replacement():
    # 来源：P0-1，哈希必须由实际执行代码求出，不能使用缓存的身份属性。
    from test.crypto.test_timing_c8 import tomorrow

    f, other = Factor(wave), Factor(tomorrow)
    old = f.definition_hash
    object.__setattr__(f, "_code", other._code)
    assert f.definition_hash == other.definition_hash != old


def test_editing_configured_ledger_changes_settings_hash(tmp_path):
    a = settings(ledger=str(tmp_path / "a"))
    b = settings(**dict(a.value, ledger=str(tmp_path / "b")))
    assert a.hash != b.hash
    assert Study(a, Synthetic()).ledger.ledger.path == tmp_path / "a"
    assert Study(b, Synthetic()).ledger.ledger.path == tmp_path / "b"


def test_relative_ledger_resolves_against_settings_file(tmp_path):
    from ashare.factor.timing import Settings

    cfg = settings().value
    cfg["ledger"] = "records/ledger.jsonl"
    path = tmp_path / "study.json"
    path.write_text(json.dumps(cfg))
    loaded = Settings.load(path)
    assert loaded.value["ledger"] == str(tmp_path / "records/ledger.jsonl")
    assert (
        loaded.hash
        == settings(**dict(cfg, ledger=str(tmp_path / "records/ledger.jsonl"))).hash
    )
    with pytest.raises(ValueError, match="LEDGER_PATH"):
        Settings.from_dict(cfg)


@pytest.mark.parametrize("change", ["symbol", "trade_hour", "horizon", "label"])
def test_test_identity_includes_entire_label_definition(change, monkeypatch):
    cfg = settings()
    f = Factor(wave)
    before = cfg.test_hash(f.definition_hash)
    if change == "label":
        monkeypatch.setattr(
            "ashare.factor.timing.inference.LABEL_DEFINITION", "synthetic-other-label"
        )
        after = cfg.test_hash(f.definition_hash)
    else:
        after = settings(
            **{change: {"symbol": "ETHUSDT", "trade_hour": 1, "horizon": 2}[change]}
        ).test_hash(f.definition_hash)
    assert after != before


def test_data_revision_recomputes_same_test_without_increment(tmp_path):
    cfg = settings(ledger=str(tmp_path / "ledger"))
    ds = Synthetic()
    f = Factor(wave)
    first = Study(cfg, ds)
    a = first.evaluate(f)
    assert a["recomputation"] is False
    ds.version = "synthetic-revised"
    revised = Study(cfg, ds)
    b = revised.evaluate(f)
    assert b["recomputation"] is True
    assert a["test_hash"] == b["test_hash"]
    assert revised.summary()["attempts"] == 1
    n = len(revised.ledger.records())
    revised.evaluate(f)
    assert len(revised.ledger.records()) == n


def test_validation_list_binds_label_horizon(tmp_path):
    from ashare.data.crypto.storage import digest

    ledger = str(tmp_path / "ledger")
    cfg = settings(ledger=ledger, horizon=1)
    f = Factor(wave)
    path = tmp_path / "list.json"
    path.write_text(json.dumps([cfg.test_hash(f.definition_hash)]))
    other = Study(
        settings(ledger=ledger, horizon=2, validation_hash=digest(path.read_bytes())),
        Synthetic(),
    )
    other.evaluate(f)
    with pytest.raises(ValueError, match="不在验证名单"):
        other.evaluate(f, stage="validation", list_path=path)


def test_old_formula_only_ledger_is_rejected_not_reset(tmp_path):
    cfg = settings(ledger=str(tmp_path / "ledger"))
    s = Study(cfg, Synthetic())
    s.ledger.append(
        "mining",
        "completed",
        definition_hash=Factor(wave).definition_hash,
        result=dict(p=0.1),
    )
    with pytest.raises(ValueError, match="LEDGER_IDENTITY"):
        Study(cfg, Synthetic())


def test_definition_hash_stable_after_execution_and_recreation():
    # 来源：修复 P0 时发现 marshal 的引用标志会随 code object 被执行而变化。
    from ashare.factor.timing import View

    factor = Factor(wave)
    before = factor.definition_hash
    factor(View(Synthetic().frame, symbol="BTCUSDT", cutoff=T + 40 * DAY))
    assert factor.definition_hash == before
    assert Factor(wave).definition_hash == before


def test_nonfinite_error_breaks_opposite_evidence():
    # 来源：持续反向检查实现复核；无穷误差不是双边可比证据。
    stamps = T + np.arange(288) * m.STEP
    error = np.tile([0.001, 0.4], (288, 1))
    error[100:102] = [np.inf, 0.001]
    assert m.alignment_days(stamps, error)[0]["opposite_max_run"] == 0
    assert _alignment(stamps, error, 0.01, 0.1)[2][0]["opposite_max_run"] == 0


def test_build_and_independent_verify_reject_sustained_reverse(tmp_path, monkeypatch):
    # 来源：P0-2，完整合成归档在 13:00 切换；独立 verify 也拒绝重签版本。
    import csv
    import io
    import zipfile

    from ashare.data.crypto.storage import RawStore, digest, encoded
    from ashare.data.crypto.verify_metrics import verify
    from ashare.data.store import atomic_json
    from test.crypto.test_metrics_c8 import fixture, zipped

    mr, kr = fixture(tmp_path)
    plan = json.loads((mr / "metrics_plan.json").read_text())
    key = sorted(plan["archives"])[0]
    blob = m._raw(mr, key, plan["archives"][key])
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        rows = list(csv.reader(io.StringIO(z.read(z.namelist()[0]).decode())))[1:]
    for row in rows[13 * 12 :]:
        row[-1] = str(1 / float(row[-1]))
    blob = zipped(rows)
    store = RawStore(mr)
    ref = store.put(
        key,
        blob,
        checksum=digest(blob),
        checksum_text=f"{digest(blob)}  {key.rsplit('/', 1)[-1]}\n",
    )
    store.close()
    plan["archives"][key] = {
        k: ref[k] for k in ("sha256", "bytes", "checksum", "checksum_text")
    }
    atomic_json(mr / "metrics_plan.json", plan)
    with pytest.raises(ValueError, match="SWITCH_WINDOW"):
        m.build(root=mr, minute_root=kr)
    assert not (mr / "latest.json").exists()
    with monkeypatch.context() as patch:
        patch.setattr(m, "OPPOSITE_RUN_LIMIT", 289)
        patch.setattr(m, "SWITCH_WINDOW", -1)
        m.build(root=mr, minute_root=kr)
    body = m.metadata(root=mr)
    body.pop("version")
    body["thresholds"]["opposite_run"] = 2
    body["thresholds"]["switch_window"] = DAY
    version = digest(encoded(body))
    atomic_json(mr / "versions" / f"{version}.json", dict(body, version=version))
    with pytest.raises(ValueError, match="SWITCH_WINDOW"):
        verify(root=mr, minute_root=kr, version=version)


def test_explicit_empty_ledger_is_not_default(tmp_path):
    # 来源：路径绑定复核，显式空字符串不能用 truthiness 退回设定路径。
    cfg = settings(ledger=str(tmp_path / "ledger"))
    with pytest.raises(ValueError, match="LEDGER_PATH"):
        Study(cfg, Synthetic(), ledger="")
