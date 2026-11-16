"""来源：第二次独立复核 recheck_c8.md；只用合成输入及私有临时台账。"""

import copy
import json

import numpy as np
import pytest

from ashare.data.crypto import metrics as m
from ashare.data.crypto.timeutil import DAY, HOUR, millis
from ashare.data.crypto.verify_metrics import _alignment
from ashare.factor.stats import adjust
from ashare.factor.timing import Factor, Study
from test.crypto.test_timing_c8 import Synthetic, settings, wave

SWITCH = millis("2024-03-04")


def evidence(start, days=3, switch=SWITCH):
    stamps = start + np.arange(days * 288) * m.STEP
    error = np.where((stamps < switch)[:, None], [0.001, 0.4], [0.4, 0.001])
    return stamps, error


def reject_both(stamps, error, reason):
    with pytest.raises(ValueError, match=reason):
        m.alignment_days(stamps, error)
    with pytest.raises(ValueError, match=reason):
        _alignment(stamps, error, 0.01, 0.1)


@pytest.mark.parametrize("offset", [-5, 5])
def test_registered_boundary_rejects_2355_and_0005(offset):
    stamps, error = evidence(SWITCH - DAY, switch=SWITCH + offset * 60000)
    reject_both(stamps, error, "SWITCH_WINDOW")


def test_reverse_run_crosses_midnight():
    stamps, error = evidence(SWITCH - 10 * DAY)
    error[287:289] = [0.4, 0.001]
    reject_both(stamps, error, "SUSTAINED_OPPOSITE")


@pytest.mark.parametrize("offset", [-DAY, -HOUR, 0, HOUR, DAY])
def test_switch_window_rejects_single_reverse(offset):
    stamps, error = evidence(SWITCH - DAY)
    i = int((DAY + offset) // m.STEP)
    error[i] = error[i, ::-1]
    reject_both(stamps, error, "SWITCH_WINDOW")


def test_registered_constant_checked_by_daily_median():
    stamps, error = evidence(SWITCH - 10 * DAY, days=1)
    # 来源：复核的每日核对；日中位数矛盾须给出明确的登记矛盾原因。
    error[:] = np.tile([[0.2, 0.005], [0.2, 0.005], [0.005, 0.4]], (96, 1))
    reject_both(stamps, error, "REGISTERED_ALIGNMENT")


def test_unknown_contract_requires_registration():
    stamps, error = evidence(SWITCH - DAY, days=1)
    with pytest.raises(ValueError, match="UNREGISTERED_SWITCH"):
        m.alignment_days(stamps, error, symbol="NEWUSDT")
    with pytest.raises(ValueError, match="UNREGISTERED_SWITCH"):
        _alignment(stamps, error, 0.01, 0.1, symbol="NEWUSDT")


def test_ambiguous_evidence_uses_registered_rows_even_across_switch():
    stamps, error = evidence(SWITCH - DAY)
    error[:] = [0.02, 0.03]
    expected = (stamps >= SWITCH).astype(int).tolist()
    a, inherited = m.resolve_alignment(stamps, error)
    assert a.tolist() == expected and inherited.all()
    assert _alignment(stamps, error, 0.01, 0.1)[:2] == (expected, [True] * len(stamps))


@pytest.mark.parametrize("side", ["outer", "inner"])
@pytest.mark.parametrize(
    "field", ["definition_hash", "test_hash", "data_version", "settings_hash"]
)
def test_cache_identity_mismatch_recomputed_with_warning(tmp_path, side, field):
    s = Study(settings(ledger=str(tmp_path / "ledger")), Synthetic())
    f = Factor(wave)
    original = s.evaluate(f)
    rows = s.ledger.records()
    record = rows[-1]
    target = record if side == "outer" else record["result"]
    target[field] = "0" * 64
    record["result"]["n"] = 999999
    s.ledger.ledger.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = s.evaluate(f)
    assert result["n"] == original["n"]
    assert result["definition_hash"] == f.definition_hash
    assert not result["recomputation"]
    warnings = [r for r in s.ledger.records() if r["event"] == "cache_warning"]
    assert len(warnings) == 1 and warnings[0]["code"] == "CACHE_INCONSISTENT"
    assert s.summary()["attempts"] == 1
    before = len(s.ledger.records())
    np.testing.assert_equal(s.evaluate(f), result)
    assert len(s.ledger.records()) == before


@pytest.mark.parametrize(
    "damage",
    [
        "result_type",
        "missing_p",
        "missing_normalization",
        "wrong_versions",
        "missing_outer_settings",
    ],
)
def test_incomplete_cache_is_miss_with_warning(tmp_path, damage):
    s = Study(settings(ledger=str(tmp_path / "ledger")), Synthetic())
    f = Factor(wave)
    result = s.evaluate(f)
    rows = s.ledger.records()
    record = rows[-1]
    if damage == "result_type":
        record["result"] = []
    elif damage == "missing_outer_settings":
        del record["settings_hash"]
    elif damage == "wrong_versions":
        record["result"]["data_versions"]["hourly"] = "forged"
    else:
        del record["result"][damage.removeprefix("missing_")]
    s.ledger.ledger.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.testing.assert_equal(s.evaluate(f), result)
    assert any(r.get("code") == "CACHE_INCONSISTENT" for r in s.ledger.records())


@pytest.mark.parametrize("stage", ["mining", "validation", "confirmation"])
def test_each_stage_dates_part_of_test_identity(stage):
    cfg = settings()
    values = cfg.value
    values["stages"][stage][0] += DAY
    assert cfg.test_hash(Factor(wave).definition_hash) != settings(**values).test_hash(
        Factor(wave).definition_hash
    )


def test_three_mining_starts_are_three_attempts_and_bh(tmp_path):
    cfg = settings(ledger=str(tmp_path / "ledger"))
    f = Factor(wave)
    for offset in range(3):
        values = copy.deepcopy(cfg.value)
        values["stages"]["mining"][0] += offset * DAY
        s = Study(settings(**values), Synthetic())
        assert not s.evaluate(f)["recomputation"]
    rows = s.ledger.records()
    completed = [r for r in rows if r["event"] == "completed"]
    assert len({r["test_hash"] for r in completed}) == 3
    assert len({r["data_version"] for r in completed}) == 1
    for r, p in zip(completed, [0.8, 0.3, 0.01], strict=True):
        r["result"]["p"] = p
    s.ledger.ledger.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert s.summary()["attempts"] == 3
    np.testing.assert_equal(
        [s.summary()["q"][r["test_hash"]] for r in completed],
        adjust([0.8, 0.3, 0.01], "bh"),
    )


def test_same_data_failed_retry_is_not_revision(tmp_path, monkeypatch):
    s = Study(settings(ledger=str(tmp_path / "ledger")), Synthetic())
    f = Factor(wave)
    s.ledger.append(
        "mining",
        "failed",
        definition_hash=f.definition_hash,
        test_hash=s.test_hash(f),
        error="SyntheticFailure",
    )
    result = s.evaluate(f)
    assert not result["recomputation"]
    count = len(s.ledger.records())
    np.testing.assert_equal(s.evaluate(f), result)
    assert len(s.ledger.records()) == count and s.summary()["attempts"] == 1


def test_cache_warning_accepts_unestimable_nan_result(tmp_path):
    # 来源：开发首轮回归，内容版本的严格 JSON 编码不能用于含 NaN 的统计缓存。
    s = Study(settings(ledger=str(tmp_path / "ledger")), Synthetic())
    f = Factor(wave)
    s.evaluate(f)
    rows = s.ledger.records()
    assert np.isnan(rows[-1]["result"]["p"])
    rows[-1]["result"]["definition_hash"] = "0" * 64
    s.ledger.ledger.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert np.isnan(s.evaluate(f)["p"])
    assert any(r.get("code") == "CACHE_INCONSISTENT" for r in s.ledger.records())


def test_same_test_and_data_with_changed_seed_is_not_revision(tmp_path):
    # 来源：P1-1，计算设定改变不代表数据版本修订。
    cfg = settings(ledger=str(tmp_path / "ledger"))
    f = Factor(wave)
    Study(cfg, Synthetic()).evaluate(f)
    s = Study(settings(**dict(cfg.value, seed=cfg.value["seed"] + 1)), Synthetic())
    assert not s.evaluate(f)["recomputation"]
    assert s.summary()["attempts"] == 1


def test_latest_invalid_cache_recomputes_instead_of_using_older_record(tmp_path):
    s = Study(settings(ledger=str(tmp_path / "ledger")), Synthetic())
    f = Factor(wave)
    s.evaluate(f)
    bad = copy.deepcopy(s.ledger.records()[-1])
    del bad["result"]["p"]
    s.ledger.ledger.append(bad)
    s.evaluate(f)
    assert [r["event"] for r in s.ledger.records()][-3:] == [
        "cache_warning",
        "started",
        "completed",
    ]


def test_build_unregistered_contract_rejected_before_raw_access(tmp_path):
    from test.crypto.test_metrics_c8 import fixture

    root, minute_root = fixture(tmp_path)
    plan = json.loads((root / "metrics_plan.json").read_text())
    plan["archives"] = {
        key.replace("BTCUSDT", "NEWUSDT"): value
        for key, value in plan["archives"].items()
    }
    (root / "metrics_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="UNREGISTERED_SWITCH"):
        m.build(root=root, minute_root=minute_root)
    assert not (root / "latest.json").exists()


def test_verify_rejects_resigned_switch_registry(tmp_path):
    from ashare.data.crypto.storage import digest, encoded
    from ashare.data.crypto.verify_metrics import verify
    from ashare.data.store import atomic_json
    from test.crypto.test_metrics_c8 import fixture

    root, minute_root = fixture(tmp_path)
    m.build(root=root, minute_root=minute_root)
    body = m.metadata(root=root)
    body.pop("version")
    body["registered_switches"]["BTCUSDT"] += DAY
    version = digest(encoded(body))
    atomic_json(root / "versions" / f"{version}.json", dict(body, version=version))
    with pytest.raises(ValueError, match="REGISTERED_METADATA"):
        verify(root=root, minute_root=minute_root, version=version)


def test_cache_consistent_inner_outer_still_checks_current_settings(tmp_path):
    # 来源：缓存完整性复核，内外彼此一致仍不能冒用本次设定身份。
    s = Study(settings(ledger=str(tmp_path / "ledger")), Synthetic())
    f = Factor(wave)
    s.evaluate(f)
    rows = s.ledger.records()
    rows[-1]["settings_hash"] = rows[-1]["result"]["settings_hash"] = "0" * 64
    s.ledger.ledger.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s.evaluate(f)
    assert any(r.get("code") == "CACHE_INCONSISTENT" for r in s.ledger.records())


def test_review_minimal_forged_cache_not_returned_or_used_by_bh(tmp_path):
    s = Study(settings(ledger=str(tmp_path / "ledger")), Synthetic())
    f = Factor(wave)
    s.ledger.append(
        "mining",
        "completed",
        definition_hash="0" * 64,
        test_hash=s.test_hash(f),
        result=dict(
            definition_hash="0" * 64,
            test_hash="0" * 64,
            family="external_future",
            n=999999,
            t=12345.0,
            p=0.0,
        ),
    )
    assert s.summary()["attempts"] == 0
    result = s.evaluate(f)
    assert result["family"] == f.family and result["n"] != 999999
    assert result["definition_hash"] == f.definition_hash
    assert s.summary()["attempts"] == 1
    assert all(np.isnan(q) for q in s.summary()["q"].values())


def test_registered_constant_itself_must_be_day_boundary(monkeypatch):
    stamps, error = evidence(SWITCH - 10 * DAY, days=1)
    monkeypatch.setitem(m.REGISTERED_SWITCHES, "BTCUSDT", SWITCH + m.STEP)
    with pytest.raises(ValueError, match="REGISTERED_BOUNDARY"):
        m.alignment_days(stamps, error)


def test_mutation_resume_refuses_changed_source_and_nonprefix_results(tmp_path):
    # 来源：共享负载下变异总运行达到 840 秒；分批续跑不能混用不同代码的结果。
    from bench import mutations_c8

    path = tmp_path / "progress.json"
    state = dict(fingerprint="old", mutations=[])
    path.write_text(json.dumps(state))
    assert mutations_c8.load_progress(path, "old") == state
    with pytest.raises(ValueError, match="MUTATION_STATE: source/test changed"):
        mutations_c8.load_progress(path, "new")
    state["mutations"] = [dict(name="unrelated")]
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="MUTATION_STATE: non-prefix"):
        mutations_c8.load_progress(path, "old")


@pytest.mark.parametrize("damage", ["cache", "failed_attempt"])
def test_revision_compares_latest_attempt_not_any_old_version(
    tmp_path, monkeypatch, damage
):
    # 来源：收尾复核 P1-1；v1 成功后 v2 再次计算，旧 v1 不能永久触发复算标记。
    cfg = settings(ledger=str(tmp_path / "ledger"))
    f = Factor(wave)
    Study(cfg, Synthetic()).evaluate(f)
    dataset = Synthetic()
    dataset.version = "synthetic-v2"
    revised = Study(cfg, dataset)
    if damage == "cache":
        assert revised.evaluate(f)["recomputation"]
        rows = revised.ledger.records()
        del rows[-1]["result"]["p"]
        revised.ledger.ledger.path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    else:

        def fail(*args):
            raise ValueError("SYNTHETIC_REVISION_FAILURE")

        with monkeypatch.context() as patch:
            patch.setattr(revised, "_evaluate", fail)
            with pytest.raises(ValueError, match="SYNTHETIC_REVISION_FAILURE"):
                revised.evaluate(f)
    assert not revised.evaluate(f)["recomputation"]
    assert revised.summary()["attempts"] == 1


def test_mutation_anchors_are_unique_and_compile():
    # 来源：复算逻辑重构后旧变异锚点失效；必须在变异 0 就发现，不能跑到末批才失败。
    from bench.mutations_c8 import MUTATIONS, ROOT

    for name, file, before, after, _ in MUTATIONS:
        source = (ROOT / file).read_text()
        assert source.count(before) == 1, name
        compile(source.replace(before, after), file, "exec")


def test_malformed_numeric_cache_is_miss_not_validator_exception(tmp_path):
    # 来源：完整缓存结构复核；JSON 的大整数会令 numpy.isinf 抛 TypeError。
    s = Study(settings(ledger=str(tmp_path / "ledger")), Synthetic())
    f = Factor(wave)
    expected = s.evaluate(f)
    rows = s.ledger.records()
    rows[-1]["result"]["p"] = 10**100
    s.ledger.ledger.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.testing.assert_equal(s.evaluate(f), expected)
    assert any(r.get("code") == "CACHE_INCONSISTENT" for r in s.ledger.records())
