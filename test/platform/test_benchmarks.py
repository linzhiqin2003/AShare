"""新名单接口、旧版数据兼容与相对指标均使用合成行情。"""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from ashare.platform.app import create_app
from ashare.platform.benchmarks import Benchmarks
from bench.platform.synthetic import write_indexes, write_panel


class Data:
    version = "synthetic"

    def __init__(self, count=22, default=True):
        self.codes = [f"{i + 1:06d}.SH" for i in range(count)]
        if not default:
            self.codes[0] = "999999.SH"

    def table(self, name, **options):
        assert name == "index_daily"
        assert options.get("end") == "20251231"
        return pl.DataFrame({"symbol": self.codes}).lazy()


class NamedData(Data):
    def indexes(self, query=None):
        assert query is None
        return pl.DataFrame(
            {
                "symbol": self.codes,
                "name": [f"合成指数{i:02d}" for i in range(len(self.codes))],
                "first_date": ["20160101"] * len(self.codes),
                "last_date": ["20251231"] * len(self.codes),
            }
        )

    def table(self, name, **options):
        raise AssertionError("新名单接口可用时不能扫描日线猜测名单")


def test_curated_indexes_search_and_default():
    benchmarks = Benchmarks(data=NamedData())
    result = benchmarks.search()
    assert result["total"] == len(result["items"]) == 22
    assert result["selected"]["symbol"] == "000001.SH"
    assert not result["selected"]["fallback"]
    assert benchmarks.search("合成指数04")["items"] == [
        {"symbol": "000005.SH", "name": "合成指数04"}
    ]
    assert benchmarks.search("000005")["total"] == 1
    assert benchmarks.search("不存在")["items"] == []
    assert benchmarks.search("不存在")["default"] == result["default"]
    # 日期范围留在数据层；只传控件会显示的两个字段。
    assert all(set(item) == {"symbol", "name"} for item in result["items"])


def test_old_data_unknown_names_and_missing_default():
    benchmarks = Benchmarks(data=Data(5, default=False))
    result = benchmarks.search()
    assert result["default"]["fallback"]
    assert result["selected"]["symbol"] == result["items"][0]["symbol"]
    assert all(r["name"] == r["symbol"] for r in result["items"])


@pytest.mark.parametrize("future", [False, True])
def test_data_api_compatibility_and_http(tmp_path, monkeypatch, future):
    import ashare.data

    data = tmp_path / "data"
    source = write_panel(data)
    codes = write_indexes(data, source)
    monkeypatch.setenv("ASHARE_DATA_DIR", str(data))
    if future:

        def indexes(self, query=None):
            # 数据分支的公开方法负责取 catalog.BENCHMARKS 与本地日线的交集。
            # 故意只返回两个代码，确认平台不会绕开名单把日线全部暴露出来。
            return pl.DataFrame(
                {
                    "symbol": codes[:2],
                    "name": ["合成沪深300", "合成中证500"],
                    "first_date": ["20160101"] * 2,
                    "last_date": ["20251231"] * 2,
                }
            )

        monkeypatch.setattr(ashare.data.Data, "indexes", indexes, raising=False)
    directory = Benchmarks()
    result = directory.search()
    assert result["total"] == len(result["items"]) == (2 if future else 5)
    assert result["selected"]["fallback"]
    assert all(row["name"] != row["symbol"] for row in result["items"])
    returns = directory.daily(codes[0], "open")
    assert all(date < "20260101" for date in returns)
    assert np.isnan(returns[max(returns)])
    with TestClient(
        create_app(tmp_path / "results", data_root=str(data)),
        base_url="http://127.0.0.1",
    ) as client:
        response = client.get("/api/benchmarks")
        assert response.status_code == 200
        assert response.json() == result
        assert client.get("/api/benchmarks?search=沪深").json()["items"] == [
            {"symbol": "000300.SH", "name": "合成沪深300" if future else "沪深300"}
        ]
        assert (
            client.get("/api/settings").json()["defaults"]["benchmark"]
            == result["default"]["symbol"]
        )
        assert (
            client.get("/api/benchmarks", params={"search": "x" * 81}).status_code
            == 422
        )


def test_empty_local_directory():
    result = Benchmarks(data=Data(0)).search()
    assert result["items"] == []
    assert result["selected"]["symbol"] is None
    assert not result["default"]["fallback"]


def test_excess_return_and_information_ratio_use_first_long_group():
    class FakeStore:
        def series(self, run):
            arrays = {}
            for part in ("is", "validation"):
                arrays.update(
                    {
                        f"{part}.dates": np.array(["20240101", "20240102", "20240103"]),
                        f"{part}.groups": np.array(
                            [[0.03, 0.01, 0.04], [-0.7, -0.8, -0.9]]
                        ),
                    }
                )
            return arrays

    benchmarks = Benchmarks(data=Data(1))
    benchmarks.daily = lambda symbol, trade_at: {
        "20240101": 0.01,
        "20240102": 0.02,
        "20240103": 0.01,
    }
    result = benchmarks.comparison(
        FakeStore(),
        [{"id": "run", "status": "complete", "settings": {"trade_at": "open"}}],
    )
    values = result["relatives"]["run"]["is"]
    excess = np.array([0.02, -0.01, 0.03])
    assert values["n"] == 3
    assert values["excess"] == pytest.approx(excess.mean() * 240)
    assert values["information_ratio"] == pytest.approx(
        excess.mean() / excess.std(ddof=1) * np.sqrt(240)
    )
