"""来源：C6 独立审查 P1-1..3、P2-4..5；所有夹具均为合成数据。"""

import csv
import io
import zipfile

import numpy as np
import pytest

from ashare.data.crypto.archive import archive_key
from ashare.data.crypto.signals import sync_signals
from ashare.data.crypto.storage import digest
from ashare.data.crypto.timeutil import DAY, HOUR
from ashare.data.crypto.verify_signals import verify_signals
from ashare.factor.crypto import crypto_factors
from ashare.factor.crypto_eval import (
    SimulationBank,
    conservative_test,
    crypto_labels,
    hodrick_1b,
    power_curve,
)
from test.crypto.test_eval_c6 import independent_1b
from test.crypto.test_factors_c6 import START, Synthetic
from test.crypto.test_signals_c6 import SyntheticClient


def signal_dates(offsets):
    at = START + np.asarray(offsets) * DAY
    return at, np.zeros((len(at), 1))


def test_review_p1_labels_crop_commutes():
    ds = Synthetic(50 * 24 + 1)
    full = crypto_labels(ds, ["BTCUSDT"], signal_dates(np.arange(1, 40)))
    cropped = crypto_labels(ds, ["BTCUSDT"], signal_dates([10]))
    for family in ("descriptive", "inference"):
        for h in (1, 7, 30):
            for kind in ("spot", "usdm"):
                expected = full[family][h][kind][9, 0]
                assert np.isfinite(expected)
                assert cropped[family][h][kind][0, 0] == pytest.approx(
                    expected, abs=1e-13
                )


def test_review_p1_missing_signal_day_keeps_calendar_return():
    ds = Synthetic(50 * 24 + 1)
    days = np.arange(1, 40)
    full = crypto_labels(ds, ["BTCUSDT"], signal_dates(days))
    sparse = crypto_labels(ds, ["BTCUSDT"], signal_dates(days[days != 12]))
    for kind in ("spot", "usdm"):
        for h in (7, 30):
            assert sparse["inference"][h][kind][9, 0] == pytest.approx(
                full["inference"][h][kind][9, 0], abs=1e-13
            )
    # 信号缺日不删收益；该日收益缺失则必须使覆盖它的 h 日和缺失。
    ds.values["tradable"][12 * 24, 1] = False
    missing = crypto_labels(ds, ["BTCUSDT"], signal_dates([10]))
    assert np.isnan(missing["inference"][7]["usdm"][0, 0])


def test_review_p1_uncovered_entry_and_exit_are_nan():
    ds = Synthetic(50 * 24 + 1)
    r = crypto_labels(ds, ["BTCUSDT"], signal_dates([-1, 0, 10, 49, 51]))
    for family in ("descriptive", "inference"):
        for h in (1, 7, 30):
            for kind in ("spot", "usdm"):
                v = r[family][h][kind][:, 0]
                assert np.isnan(v[[0, 4]]).all()
                assert np.isfinite(v[[1, 2]]).all()
                assert np.isfinite(v[3]) if h == 1 else np.isnan(v[3])


def test_review_p1_irregular_panel_is_distinct_from_uncovered_request():
    ds = Synthetic(50 * 24 + 1)
    # 删除一个不是入出场端点的小时，不能被 searchsorted 的端点匹配掩盖。
    keep = np.arange(len(ds.time)) != 36
    ds.time = ds.time[keep]
    ds.values = {k: v[keep] for k, v in ds.values.items()}
    with pytest.raises(ValueError, match="小时轴"):
        crypto_labels(ds, ["BTCUSDT"], signal_dates([1]))


@pytest.mark.parametrize("n,offset,step", [(10, 0.01, 1e-12), (100, 0.1, 1e-10)])
def test_review_p1_near_constant_1b(n, offset, step):
    # 两个审查反例：旧实现分别抛 Singular matrix、把正斜率变成负斜率。
    x = offset + step * np.arange(n)
    g = np.sin(np.arange(n)) + (0 if n == 10 else 0.01 * np.arange(n))
    centered = x - x[0]
    centered -= centered.mean()
    scale = centered.std()
    beta, se = independent_1b(centered / scale, g, 1)
    observed = hodrick_1b(x, g, 1)
    assert observed["n"] == n
    assert observed["beta"] > 0
    np.testing.assert_allclose(
        [observed["beta"], observed["se"], observed["t"]],
        [beta / scale, se / scale, beta / se],
        rtol=1e-10,
    )


@pytest.mark.parametrize("h", [1, 7])
def test_review_p1_1b_affine_invariance_with_missing_samples(h):
    x = np.arange(100, dtype=float) / 16
    g = np.sin(np.arange(100)) + 0.01 * np.arange(100)
    x[[20, 45]] = np.nan
    g[55] = np.nan
    sample = np.arange(100) < 95
    base = hodrick_1b(x, g, h, sample=sample)
    for multiplier, shift in [(1, 2**20), (2**-20, 1), (2**20, -16)]:
        r = hodrick_1b(multiplier * x + shift, g, h, sample=sample)
        assert (r["n"], r["meat_terms"]) == (base["n"], base["meat_terms"])
        np.testing.assert_allclose(
            [r["beta"] * multiplier, r["se"] * multiplier, r["t"]],
            [base["beta"], base["se"], base["t"]],
            rtol=1e-10,
        )


def test_review_p2_mde_bracket_is_part_of_json_contract():
    bank = SimulationBank(
        np.linspace(-0.02, 0.02, 101), 64, repetitions=400, phi_grid=[0]
    )
    mask = np.ones(64, bool)
    table, _ = conservative_test(bank, mask, mask, 7, [0], 1, refine=False)
    r = power_curve(
        bank,
        table,
        mask,
        mask,
        7,
        {"sd_eta": 0.2, "sd_x": 0.5},
        effects=[0, 0.5, 1, 2, 10],
    )
    assert r["effect_unit_label"] == "持续七日增加 1 sd 的等价效应（对数收益）"
    assert r["mde_is_discrete"] is True
    assert r["search_upper_bound"] == 10
    for direction in ("positive", "negative"):
        bracket = r[direction]["search_bracket"]
        assert set(bracket) == {"previous", "reached"}
        if bracket["previous"] is not None:
            assert bracket["previous"]["worst_power"] < 0.8
        if bracket["reached"] is not None:
            assert bracket["reached"]["worst_power"] >= 0.8


def test_review_p2_mde_table_names_sustained_effect_and_adjacent_power():
    from bench.crypto_factors_report import markdown

    direction = dict(
        minimum_detectable=1,
        search_bracket=dict(
            previous=dict(effect=0.5, worst_power=0.75725),
            reached=dict(effect=1, worst_power=0.999),
        ),
    )
    row = dict(
        symbol="SYNTHETIC",
        factor="synthetic",
        family="confirmation",
        n=100,
        beta=0,
        t=0,
        p=1,
        holm_p=1,
        significant=False,
        power=dict(positive=direction, negative=direction),
        split="out",
        kind="usdm",
        horizon=1,
    )
    report = dict(
        data_version="synthetic",
        signal_version="synthetic",
        definition="synthetic",
        start=0,
        end=DAY,
        simulation_repetitions=4000,
        simulation_workers=1,
        tests=[row],
        self_check={},
        runtime_seconds=0,
        limitations=[],
    )
    text = markdown(report)
    assert "持续七日增加 1 sd 的等价效应（对数收益）" in text
    assert "相邻搜索点" in text
    assert "75.725%" in text and "99.900%" in text


@pytest.mark.parametrize("hour", [0, 8])
def test_review_p2_z_requires_60_prior_days(hour):
    ds = Synthetic(69 * 24)
    r = crypto_factors(ds, None, ["BTCUSDT"], START, ds.meta["end"])
    at, z = r["funding_z_90d"]
    mean7 = r["funding_mean_7d"][1][:, 0]
    for day, count in [(66, 59), (67, 60)]:
        now = START + day * DAY + hour * HOUR
        i = np.flatnonzero(at == now)[0]
        past = mean7[(at < now) & (at >= now - 90 * DAY) & (at % DAY == hour * HOUR)]
        past = past[np.isfinite(past)]
        assert len(past) == count
        if count == 59:
            assert np.isnan(z[i, 0])
        else:
            assert z[i, 0] == pytest.approx((mean7[i] - past.mean()) / past.std(ddof=1))


class QuarterArchiveClient(SyntheticClient):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.keys = [archive_key("usdm", "klines", self.symbol, "2023-10")]

    def archive(self, store, key, refresh=False):
        text = io.StringIO()
        csv.writer(text).writerows(self.rows)
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr("synthetic.csv", text.getvalue())
        payload = data.getvalue()
        sha = digest(payload)
        store.put(
            key, payload, checksum=sha, checksum_text=sha + " " + key.rsplit("/", 1)[1]
        )
        return payload


def test_review_p2_quarter_archive_truncation_rebuilds_production_quality(tmp_path):
    # 审查 P2：夹具要长到截断点之前 7 天也已上市满 7 天，basis_q_chg_7d 才有有限值可比。
    n, b = 600, 500
    panel = Synthetic(n)
    raw = []
    for i, t in enumerate(panel.time):
        close = 110.0 if i in (b, b + 1) else 105 + i * 0.01
        volume = 0 if i == b + 2 else 1
        close_time = int(t + (HOUR // 2 if i == b + 3 else HOUR) - 1)
        raw.append(
            [
                int(t),
                close,
                close,
                close,
                close,
                volume,
                close_time,
                volume * close,
                int(volume),
                0,
                0,
                0,
            ]
        )

    def rebuild(folder, cutoff):
        # 从原始归档行按 available_at 截断，再走 ZIP、生产解析、版本构建和质量生成。
        client = QuarterArchiveClient([r for r in raw if r[0] + HOUR <= cutoff])
        result = sync_signals(folder, start=START, end=cutoff, client=client)
        assert verify_signals(folder)["verdict"] == "pass"
        return result

    end = START + n * HOUR
    full = rebuild(tmp_path / "full", end)
    quality = dict(zip(full.read()["time"], full.read()["quality"], strict=True))
    assert [quality[START + i * HOUR] for i in (b, b + 1, b + 2, b + 3)] == [
        0,
        4,
        2,
        512,
    ]
    finite_chg = 0
    for hours in (b + 1, b + 2, b + 3, b + 4):
        cutoff = START + hours * HOUR
        truncated = rebuild(tmp_path / str(hours), cutoff)
        assert truncated.read().equals(full.read(end=cutoff))
        a = crypto_factors(panel, full, ["BTCUSDT"], cutoff - HOUR, cutoff + HOUR)
        c = crypto_factors(
            panel.cut(cutoff), truncated, ["BTCUSDT"], cutoff - HOUR, cutoff + HOUR
        )
        np.testing.assert_array_equal(a.selected_contract[1], c.selected_contract[1])
        for name in ("basis_q", "basis_q_chg_7d", "carry_spread"):
            np.testing.assert_allclose(a[name][1], c[name][1], equal_nan=True)
        finite_chg += int(np.isfinite(a["basis_q_chg_7d"][1]).sum())
    assert finite_chg > 0
