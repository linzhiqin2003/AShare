"""来源 C5 第二轮 P1：覆盖豁免不是归档之后没有结算的证据。"""

import copy

import polars as pl
import pytest

from ashare.data import crypto
from ashare.data.crypto.events import normalize_event
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.timeutil import HOUR
from ashare.data.crypto.verify import verify_frame
from test.crypto.test_c5_review_r1 import ended_case
from test.crypto.test_recent import NAME, START
from test.crypto.test_recent import recent as recent


def short_horizon(recent, case, info):
    plan, data = recent
    name = NAME
    if case == "request_end":
        plan["end"] = START + 21 * HOUR
    else:
        ended_case(plan, data)
        name = "usdm:ENDEDUSDT"
        event = dict(plan["events"][0], effective_at=START + 21 * HOUR)
        event.pop("version_id")
        plan["events"] = [normalize_event(event)]
        for dataset in ("klines", "markPriceKlines", "indexPriceKlines"):
            data[name][dataset] = copy.deepcopy(data[NAME][dataset][:21])
    if info == "missing":
        plan.pop("funding_info")
    elif info == "changed":
        plan["funding_info"]["records"] = [
            dict(symbol=name.split(":")[1], fundingIntervalHours=4)
        ]
    return plan, data, name


@pytest.mark.parametrize("case", ["request_end", "known_exit"])
@pytest.mark.parametrize("info", ["confirmed", "missing", "changed"])
def test_rest_inside_short_horizon_stops_extension(tmp_path, recent, case, info):
    """审查两个反例：归档 16h/8h、REST 20h、有效终点 21h，必须退到 16h。"""
    _, data, name = short_horizon(recent, case, info)
    data[name]["funding_prices"].insert(
        3, dict(fundingTime=START + 20 * HOUR, fundingRate="0.0001", markPrice="100")
    )
    ds = crypto.build(tmp_path, allow_inferred_funding=True)
    assert ds.meta["end"] == START + 16 * HOUR
    assert ds.meta["recent"]["instruments"][name]["reason"] == "funding_spacing_or_grid"
    assert crypto.verify(tmp_path, ds.version)["errors"] == []


@pytest.mark.parametrize("case", ["request_end", "known_exit"])
@pytest.mark.parametrize("info", ["confirmed", "missing", "changed"])
def test_verifier_rejects_unchecked_rest_inside_short_horizon(recent, case, info):
    """独立构造漏费面板，不能用修复后的 build 产物作为 verifier 反例。"""
    plan, data, name = short_horizon(recent, case, info)
    data[name]["funding_prices"].insert(
        3, dict(fundingTime=START + 20 * HOUR, fundingRate="0.0001", markPrice="100")
    )
    frame, notes = build_frame(
        plan["profiles"], data, plan["events"], START, plan["end"]
    )
    row = (pl.col("instrument") == name) & (pl.col("time") == START + 19 * HOUR)
    assert frame.filter(row)["funding"].item() == 0
    archived = copy.deepcopy(data)
    archived[name]["fundingRate"].append(
        dict(time=START + 20 * HOUR, interval=4, rate=0.0001, source="archive")
    )
    economic, _ = build_frame(
        plan["profiles"], archived, plan["events"], START, plan["end"]
    )
    assert economic.filter(row)["funding"].item() == 0.0001
    inputs = dict(
        profiles=plan["profiles"],
        data=data,
        events=plan["events"],
        notes=notes,
        recent=dict(
            allow_inferred_funding=True,
            requested_end=plan["end"],
            funding_info=plan.get("funding_info"),
        ),
    )
    errors = verify_frame(frame, inputs)["errors"]
    assert any(
        e["check"] == "inferred_terminal" and e["expected"] == START + 16 * HOUR
        for e in errors
    )


@pytest.mark.parametrize("case", ["request_end", "known_exit"])
@pytest.mark.parametrize("info", ["confirmed", "missing", "changed"])
def test_rest_after_effective_end_preserves_exemption(tmp_path, recent, case, info):
    """配套边界：21h 之后的坏间距不应影响请求窗口或已退出列。"""
    plan, data, name = short_horizon(recent, case, info)
    data[name]["funding_prices"].insert(
        3, dict(fundingTime=START + 22 * HOUR, fundingRate="0.0001", markPrice="100")
    )
    ds = crypto.build(tmp_path, allow_inferred_funding=True)
    assert ds.meta["end"] == plan["end"]
    assert crypto.verify(tmp_path, ds.version)["errors"] == []
