"""因子对比接口：只读已存结果，默认不返回验证期（2026-11-14 用户要求默认隐藏验证期）。"""

import base64
import json

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ashare.platform.app import create_app
from ashare.platform.compare import _parse_ids, compare
from ashare.platform.config import DEFAULTS
from ashare.platform.language import parse
from ashare.platform.storage import Store
from bench.platform.synthetic import write_panel

IS_DATES = np.array(["20241227", "20241230", "20241231"])
VALIDATION_DATES = np.array(["20250102", "20250103"])


def _stats(sharpe, period):
    return {
        "sharpe": sharpe,
        "return": 0.1,
        "turnover": 0.2,
        "maxdrawdown": 0.05,
        "margin": 3.0,
        "fitness": 1.0,
        "t": 2.0,
        "ic": 0.02,
        "coverage": 0.9,
        "holding_days": 4.0,
        "n": 3,
        "periods": [{"period": period, "sharpe": sharpe}],
        "years": [
            {"period": period, "sharpe": sharpe, "return": 0.1, "maxdrawdown": 0.05}
        ],
        "exposure": {"size": 0.1},
        "subs": {"TOP800": {"sharpe": 0.4, "n": 3}},
    }


def _store(tmp_path, count=3, with_result=(0, 1)):
    store = Store(tmp_path)
    expression = parse("rank(close)", factors={})[0]
    _, runs = store.begin(
        "tab",
        "alpha",
        [(expression, DEFAULTS | {"decay": i}) for i in range(count)],
        {},
    )
    for i, run in enumerate(runs):
        if i not in with_result:
            continue
        sign = 1 if i == 0 else -1
        np.savez_compressed(
            tmp_path / "series" / f"{run['id']}.npz",
            **{
                "is.dates": IS_DATES,
                "is.pnl.long_short": sign * np.array([0.01, 0.02, -0.01]),
                "validation.dates": VALIDATION_DATES,
                "validation.pnl.long_short": np.array([0.5, 0.6]),
            },
        )
        summary = {
            "is": _stats(1.5 + i, "2024"),
            "validation": _stats(-0.77, "2025"),
            "checks": [],
            "correlations": [{"entry": "x", "name": "other", "value": -0.4}],
        }
        with store.connect() as db:
            db.execute(
                "UPDATE runs SET status='complete',summary=? WHERE id=?",
                (json.dumps(summary), run["id"]),
            )
            db.execute(
                "UPDATE entries SET latest=? WHERE id=?", (run["id"], run["entry"])
            )
    return store, [r["entry"] for r in runs]


def _unpack(packed):
    return np.frombuffer(base64.b64decode(packed["data"]), dtype="<f4")


def test_default_response_has_no_validation_numbers(tmp_path):
    """来源：2026-11-14 用户要求「验证期默认隐藏」；关时响应里不能出现任何验证期数字。"""
    store, ids = _store(tmp_path)
    result = compare(store, ids[:2])
    assert [item["id"] for item in result["items"]] == ids[:2]
    assert all("validation" not in item for item in result["items"])
    assert result["dates"] == IS_DATES.tolist()
    text = json.dumps(result)
    assert "-0.77" not in text and "2025" not in text
    for entry in ids[:2]:
        assert len(_unpack(result["series"][entry])) == len(IS_DATES)
    np.testing.assert_allclose(
        _unpack(result["series"][ids[1]]), [-0.01, -0.02, 0.01], rtol=1e-6
    )
    # 只传页面画得到的字段：分季明细和持仓不出现。
    assert "periods" not in result["items"][0]["is"]
    assert result["items"][0]["is"]["subs"] == {"TOP800": {"sharpe": 0.4}}
    assert result["items"][0]["correlation"] == {"name": "other", "value": -0.4}
    # 日盈亏相关只用样本内序列。
    assert result["correlation"][0][1] == pytest.approx(-1)


def test_validation_is_returned_only_on_request(tmp_path):
    store, ids = _store(tmp_path)
    result = compare(store, ids[:2], validation=True)
    assert result["dates"] == IS_DATES.tolist() + VALIDATION_DATES.tolist()
    assert result["items"][0]["validation"]["sharpe"] == -0.77
    assert result["items"][0]["validation"]["years"][0]["period"] == "2025"
    np.testing.assert_allclose(
        _unpack(result["series"][ids[0]])[-2:], [0.5, 0.6], rtol=1e-6
    )


def test_entries_without_results_and_misaligned_dates(tmp_path):
    store, ids = _store(tmp_path, with_result=(0, 1))
    result = compare(store, ids)
    pending = result["items"][2]
    assert pending["is"] is None and pending["latest"] is None
    assert ids[2] not in result["series"]
    assert result["correlation"][2] == [None, None, None]


def test_unknown_and_invalid_ids(tmp_path):
    store, ids = _store(tmp_path)
    with pytest.raises(HTTPException) as error:
        compare(store, [ids[0], "missing"])
    assert error.value.status_code == 404
    for bad in ("", ",", "a,a", ",".join(map(str, range(9)))):
        with pytest.raises(ValueError):
            _parse_ids(bad)
    assert _parse_ids("a,b") == ["a", "b"]


def test_http_route_registered_before_static(tmp_path):
    data = tmp_path / "data"
    write_panel(data)
    app = create_app(tmp_path / "platform", data_root=str(data))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        store = client.app.state.store
        _, ids = _store(store.root)
        response = client.get("/api/compare", params={"ids": ",".join(ids[:2])})
        assert response.status_code == 200, response.text
        assert "validation" not in response.json()["items"][0]
        assert (
            client.get(
                "/api/compare", params={"ids": ids[0] + ",x", "validation": "true"}
            ).status_code
            == 404
        )
        assert client.get("/api/compare", params={"ids": "a,a"}).status_code == 422
        assert client.get("/api/compare").status_code == 422
        assert str(store.root) not in response.text
