"""真实数据目录接口、默认回退与相对指标均使用合成行情。"""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from ashare.data.catalog import BENCHMARKS, DEFAULT_BENCHMARK
from ashare.platform.app import create_app
from ashare.platform.benchmarks import Benchmarks
from bench.platform.synthetic import write_indexes, write_panel


class Data:
    version = "synthetic"

    def __init__(self, count=22):
        self.codes = list(BENCHMARKS)[:count]

    def indexes(self, query=None):
        assert query is None
        return pl.DataFrame(
            {
                "symbol": self.codes,
                "name": [BENCHMARKS[code] for code in self.codes],
                "first_date": ["20160101"] * len(self.codes),
                "last_date": ["20251231"] * len(self.codes),
            },
            schema={
                key: pl.String for key in ("symbol", "name", "first_date", "last_date")
            },
        )

    def table(self, name, **options):
        raise AssertionError("新名单接口可用时不能扫描日线猜测名单")


def test_curated_indexes_search_and_default():
    benchmarks = Benchmarks(data=Data())
    result = benchmarks.search()
    assert result["total"] == len(result["items"]) == 22
    assert result["selected"]["symbol"] == "000001.SH"
    assert not result["selected"]["fallback"]
    assert benchmarks.search("沪深300")["items"] == [
        {"symbol": "000300.SH", "name": "沪深300"}
    ]
    assert benchmarks.search("000300")["total"] == 1
    assert benchmarks.search("不存在")["items"] == []
    assert benchmarks.search("不存在")["default"] == result["default"]
    # 日期范围留在数据层；只传控件会显示的两个字段。
    assert all(set(item) == {"symbol", "name"} for item in result["items"])


@pytest.mark.parametrize("has_default", [False, True])
def test_data_indexes_and_http(tmp_path, monkeypatch, has_default):
    import ashare.data

    data = tmp_path / "data"
    source = write_panel(data)
    codes = ["000300.SH", "000905.SH", "000852.SH", "932000.CSI"]
    if has_default:
        codes.append(DEFAULT_BENCHMARK)
    write_indexes(data, source, codes=codes)
    monkeypatch.setenv("ASHARE_DATA_DIR", str(data))
    # 不替换 Data.indexes，守住真实的代码白名单和「代码_月份」清单分区约定。
    indexes = ashare.data.open(data).indexes()
    assert set(indexes["symbol"]) == set(codes)
    assert set(indexes["first_date"]) == {source.dates[0]}
    assert set(indexes["last_date"]) == {source.dates[-1]}
    directory = Benchmarks()
    result = directory.search()
    assert result["total"] == len(result["items"]) == len(codes)
    assert result["selected"]["fallback"] is (not has_default)
    assert result["selected"]["requested"] == DEFAULT_BENCHMARK
    assert result["selected"]["symbol"] == (
        DEFAULT_BENCHMARK if has_default else result["items"][0]["symbol"]
    )
    assert all(row["name"] == BENCHMARKS[row["symbol"]] for row in result["items"])
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
            {"symbol": "000300.SH", "name": "沪深300"}
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
