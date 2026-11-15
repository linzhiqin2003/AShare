"""币安历史风险规则（2026-11-15 调研补齐，证据见 docs/markets/BINANCE.md 5.2、5.3）。

来源：第一批遗留「2021-06 以前的保证金参数、清算费历史没查到」；调研后按公告、FAQ 存档、
exchangeInfo 与档位接口存档写进档案。每条期望值都直接来自公告或存档原文，不从实现取。
"""

import numpy as np
import pytest
from panelbt import markets
from panelbt.markets import binance

P = markets.get("binance_usdm")


def rules(symbols, times, **kw):
    return P.rules(symbols, np.array(times, dtype="datetime64[h]"), **kw)


def test_2021_05_19_uses_the_announced_clearance_fee_without_assumption():
    # 公告 aeeccb6d…：2021-02-07 04:00 起 BTCUSDT 2.00%、ETHUSDT 3.00%；
    # 公告 839a865d…：2021-05-20 09:00 起 0.30% / 0.50%。旧档案在这里外推 1.25%。
    # ETH 第一档上限在 2021 年没有证据，这里只看清算费：用保守假设放行其他字段。
    r, notes = rules(
        ["BTCUSDT", "ETHUSDT"],
        ["2021-05-19T00", "2021-05-20T08", "2021-05-20T09"],
        assume="conservative",
    )
    np.testing.assert_array_equal(r["clearance_fee_rate"][:, 0], [0.02, 0.02, 0.003])
    np.testing.assert_array_equal(r["clearance_fee_rate"][:, 1], [0.03, 0.03, 0.005])
    clearance = [n for n in notes if n["parameter"] == "clearance_fee_rate"]
    assert clearance and all(n["confidence"] == "official" for n in clearance)


def test_btc_first_tier_history():
    # 2019-11-14 FAQ 存档起第一档 0–50,000、MMR 0.4%；公告 e778988e…：
    # 2025-06-17 06:30 第一档上限改为 300,000（下一个整点起生效）；2025-08-19 06:30 杠杆 150x。
    times = [
        "2020-01-01T00",
        "2025-06-17T06",
        "2025-06-17T07",
        "2025-08-19T06",
        "2025-08-19T07",
        "2026-11-15T00",
    ]
    r, _ = rules(["BTCUSDT"], times, assume="conservative")
    np.testing.assert_array_equal(r["maint_rate"][:, 0], [0.004] * 6)
    np.testing.assert_array_equal(
        r["tier_cap"][:, 0], [50000, 50000, 300000, 300000, 300000, 300000]
    )
    np.testing.assert_array_equal(
        r["max_leverage"][:, 0], [125, 125, 125, 125, 150, 150]
    )


def test_eth_first_tier_history_and_conservative_gaps():
    # 2020-05/09 FAQ 表：首 10,000、MMR 0.65%；2021-01-24 FAQ：首 10,000、0.5%；
    # 两次观测之间（2020-10）没有证据，保守取较高的 0.65% 和较低的上限 10,000。
    times = ["2020-06-01T00", "2020-11-01T00", "2021-02-01T00", "2023-12-24T10"]
    with pytest.raises(ValueError, match="ETHUSDT maint_rate"):
        rules(["ETHUSDT"], times)
    r, notes = rules(["ETHUSDT"], times, assume="conservative")
    np.testing.assert_array_equal(r["maint_rate"][:, 0], [0.0065, 0.0065, 0.005, 0.004])
    # 2021-02 的上限介于 2021-01-24 的 10,000 与 2022-11-17 的 100,000 之间，保守取 10,000。
    np.testing.assert_array_equal(r["tier_cap"][:, 0], [10000, 10000, 10000, 50000])
    gap = [
        n
        for n in notes
        if n["parameter"] == "maint_rate" and n["start"].startswith("2020-11-01")
    ]
    assert gap and gap[0]["confidence"] == "assumption"
    assert "conservative" in gap[0]["reason"]


def test_conservative_picks_the_adverse_neighbour_for_each_field():
    # exchangeInfo 存档：2021-09-14 BTC 0.6%、2021-10-14 BTC 2.0%；之间取较高者。
    r, _ = rules(["BTCUSDT"], ["2021-10-01T00"], assume="conservative")
    assert r["clearance_fee_rate"][0, 0] == 0.02
    # 同一缺口按「沿用上一个观测」得到 0.6%：两种策略确实不同。
    r, _ = rules(["BTCUSDT"], ["2021-10-01T00"], assume="previous")
    assert r["clearance_fee_rate"][0, 0] == 0.006
    assert binance.ADVERSE == dict(
        maint_rate=max, clearance_fee_rate=max, tier_cap=min, max_leverage=min
    )


def test_clearance_gaps_still_require_explicit_assumption():
    # 2021-06-10 公告之后到 2021-07-16 存档之间没有证据：默认报错。
    with pytest.raises(ValueError, match="BTCUSDT clearance_fee_rate"):
        rules(["BTCUSDT"], ["2021-07-01T00"])
    with pytest.raises(ValueError, match="conservative"):
        rules(["BTCUSDT"], ["2021-07-01T00"], assume="later")


def test_non_hourly_effective_times_apply_from_the_next_hour():
    _, notes = rules(["BTCUSDT"], ["2025-06-17T07"], assume="conservative")
    cap = [n for n in notes if n["parameter"] == "tier_cap"][0]
    assert cap["effective_at"].startswith("2025-06-17T06:30")
    assert cap["applied_at"].startswith("2025-06-17T07:00")
    assert cap["approximation"] is True


def test_evidence_records_are_sorted_and_disjoint():
    for symbol in ("BTCUSDT", "ETHUSDT", "BTCUSDT_261225"):
        for field, records in binance._observations(symbol).items():
            # 同一类（证据 / 推断）之间不重叠；证据点可以落在推断区间里面。
            for group in (binance.EVIDENCE, binance.INFERRED):
                same = [r for r in records if r[4] in group]
                for a, b in zip(same, same[1:], strict=False):
                    assert a[0] < a[1] <= b[0], (symbol, field)
            for _lo, _hi, value, source, confidence, _timing in records:
                assert source.startswith("https://") and confidence
                assert np.isfinite(value) and value > 0


def test_conservative_fill_is_linear_in_rows():
    # 回归（自查）：逐小时生成说明会让 7 万行的缺口变成 O(T²)。
    times = np.arange(
        np.datetime64("2020-01-01T00"),
        np.datetime64("2026-11-15T00"),
        np.timedelta64(1, "h"),
    )
    _, notes = P.rules(["BTCUSDT", "ETHUSDT"], times, assume="conservative")
    # 说明条数只随记录数增长（每条记录及其前后缺口各至多一段），与行数无关。
    records = sum(
        len(r)
        for symbol in ("BTCUSDT", "ETHUSDT")
        for r in binance._observations(symbol).values()
    )
    assert len(notes) <= 3 * records


# 以下来自 2026-11-15 Codex 对规则分支的独立审查（.review/review_rules.out）。


def test_inferred_spans_are_not_used_by_default_review_p1():
    # P1：未定义的 updateTime 回填区间、两端观测一致的区间，都不能默认当证据。
    with pytest.raises(ValueError, match="ETHUSDT tier_cap"):
        rules(["ETHUSDT"], ["2022-11-17T07"])
    with pytest.raises(ValueError, match="BTCUSDT maint_rate"):
        rules(["BTCUSDT"], ["2020-06-01T00"])
    r, notes = rules(["ETHUSDT"], ["2022-11-17T07"], assume="conservative")
    assert r["tier_cap"][0, 0] == 100000
    cap = [n for n in notes if n["parameter"] == "tier_cap"]
    assert cap[0]["confidence"] == "assumption"
    assert "bracket_update_time" in cap[0]["reason"]
    # 显式映射优先于推断区间。
    mapped = dict(
        rules={
            "ETHUSDT": dict(
                tier_cap=10000, reason="test", source_url="https://example.test"
            )
        },
        policy="conservative",
    )
    r, _ = rules(["ETHUSDT"], ["2022-11-17T07"], assume=mapped)
    assert r["tier_cap"][0, 0] == 10000


def test_btc_launch_tier_is_recorded_review_p2():
    # P2：2019-09-18 FAQ 存档：最高 20x，第一档 0–500,000、维持 2.5%（18:42:07 观测，19 点起）。
    r, notes = rules(["BTCUSDT"], ["2019-09-18T19"], assume="conservative")
    assert r["maint_rate"][0, 0] == 0.025
    assert r["tier_cap"][0, 0] == 500000
    assert r["max_leverage"][0, 0] == 20
    assert all(
        n["confidence"] == "dated_snapshot"
        for n in notes
        if n["parameter"] != "clearance_fee_rate"
    )


def test_observation_timestamps_keep_seconds_review_p2():
    # P2：ETH 75x 公告页时间 2019-12-19 10:45:42 → 11 点起；2020-09-27 19:54:52 的观测覆盖 20 点。
    r, notes = rules(
        ["ETHUSDT"], ["2019-12-19T11", "2020-09-27T20"], assume="conservative"
    )
    np.testing.assert_array_equal(r["max_leverage"][:, 0], [75, 75])
    lev = [n for n in notes if n["parameter"] == "max_leverage"]
    assert lev[0]["confidence"] == "official"
    assert lev[0]["effective_at"].startswith("2019-12-19T10:45:42")
    _, notes = rules(["BTCUSDT"], ["2021-07-16T22"], assume="conservative")
    snap = [n for n in notes if n["parameter"] == "clearance_fee_rate"][0]
    assert snap["confidence"] == "dated_snapshot"
    assert snap["effective_at"].startswith("2021-07-16T21:43:52")


def test_conservative_notes_list_candidates_review_p2():
    # P2：说明要写出两个候选、选中哪个、在哪一侧。
    _, notes = rules(["BTCUSDT"], ["2021-10-01T00"], assume="conservative")
    gap = [n for n in notes if n["parameter"] == "clearance_fee_rate"][0]
    assert gap["side"] == "between"
    assert sorted(c["value"] for c in gap["candidates"]) == [0.006, 0.02]
    assert gap["chosen"]["value"] == 0.02
    _, notes = rules(["BTCUSDT"], ["2027-01-01T00"], assume="conservative")
    late = [n for n in notes if n["parameter"] == "clearance_fee_rate"][0]
    assert late["side"] == "after_last" and len(late["candidates"]) == 1


# 以下来自规则分支第二轮复核（.review/review_rules_r2.out）。


def test_evidence_points_inside_inferred_spans_win_review_r2_p1():
    # P1：2025-08-19 公告写明 BTC 第一档维持 0.40%；「只补缺口」的映射不能改掉它。
    mapped = dict(
        policy="conservative",
        rules={
            "BTCUSDT": dict(
                maint_rate=0.001,
                reason="fill gaps only",
                source_url="https://example.test",
            )
        },
    )
    r, notes = rules(["BTCUSDT"], ["2025-08-19T07"], assume=mapped, leverage=100)
    assert r["maint_rate"][0, 0] == 0.004
    mmr = [n for n in notes if n["parameter"] == "maint_rate"][0]
    assert mmr["confidence"] == "official"
    # ETH 档位接口快照 2023-12-12 14:27:39 是证据点（15 点起），不是推断。
    r, notes = rules(["ETHUSDT"], ["2023-12-12T15"], assume="conservative")
    assert r["tier_cap"][0, 0] == 200000
    cap = [n for n in notes if n["parameter"] == "tier_cap"][0]
    assert cap["confidence"] == "dated_snapshot"


def test_gap_notes_carry_rounding_fields_review_r2_p2():
    # P2：缺口说明要把整点近似字段带到顶层，下游诊断只看顶层。
    for policy in ("previous", "conservative"):
        _, notes = rules(["BTCUSDT"], ["2021-10-01T00"], assume=policy)
        gap = [n for n in notes if n["parameter"] == "clearance_fee_rate"][0]
        assert gap["approximation"] is True and gap["rounding"] == "ceil_to_hour"


def test_sources_and_effective_times_review_r2_p2():
    # P2：2020-05 观测引用五月的存档；ETH 2025-08-19 上限保留 06:30 原始生效时刻。
    _, notes = rules(["ETHUSDT"], ["2020-05-19T17"], assume="conservative")
    for n in notes:
        if n["parameter"] in ("maint_rate", "tier_cap"):
            assert "20200519164655" in n["source_url"]
    _, notes = rules(["ETHUSDT"], ["2025-08-19T07"], assume="conservative")
    cap = [n for n in notes if n["parameter"] == "tier_cap"][0]
    assert cap["effective_at"].startswith("2025-08-19T06:30")
    assert cap["approximation"] is True


def test_every_researched_observation_is_evidence_review_r3_p1():
    # 第三轮复核 P1：笔记里每个观测点都是证据，缺口映射不能改掉。
    mapped = dict(
        policy="conservative",
        rules={
            name: dict(
                maint_rate=0.001,
                tier_cap=1.0,
                max_leverage=1.0,
                clearance_fee_rate=0.0,
                reason="fill gaps only",
                source_url="https://example.test",
            )
            for name in ("BTCUSDT", "ETHUSDT")
        },
    )
    points = [
        ("BTCUSDT", "2020-05-19T17", "maint_rate", 0.004),
        ("BTCUSDT", "2025-08-19T06", "maint_rate", 0.004),
        ("BTCUSDT", "2023-01-09T21", "tier_cap", 50000),
        ("ETHUSDT", "2023-10-02T14", "tier_cap", 100000),
        ("BTCUSDT", "2026-08-26T19", "max_leverage", 150),
        ("ETHUSDT", "2026-08-26T19", "maint_rate", 0.004),
        ("BTCUSDT", "2020-09-27T22", "clearance_fee_rate", 0.003),
        ("ETHUSDT", "2020-05-19T17", "max_leverage", 75),
        # 第四轮复核：2020-11-23 公告的调整前值、2019 年两处杠杆观测。
        ("BTCUSDT", "2020-11-23T06", "clearance_fee_rate", 0.003),
        ("ETHUSDT", "2020-11-23T06", "clearance_fee_rate", 0.005),
        ("BTCUSDT", "2019-11-14T23", "max_leverage", 125),
        ("ETHUSDT", "2019-11-29T13", "max_leverage", 50),
    ]
    for symbol, at, field, value in points:
        r, notes = rules([symbol], [at], assume=mapped)
        assert r[field][0, 0] == value, (symbol, at, field)
        note = [n for n in notes if n["parameter"] == field][0]
        assert note["confidence"] != "assumption", (symbol, at, field)
