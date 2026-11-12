from pathlib import Path

import polars as pl
import pytest

import ashare
from ashare.data.catalog import DATASETS, INDEXES
from ashare.data.sources.base import FetchResult
from ashare.data.store import Store

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


@pytest.fixture(autouse=True)
def isolated_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ASHARE_DATA_DIR", str(tmp_path / "default-data"))
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)


@pytest.fixture
def tables():
    days = ["20260102", "20260105", "20260106", "20260107"]
    rows = []
    for symbol, available in (
        ("000001.SZ", [0, 1, 3]),
        ("000002.SZ", [1, 2]),
        ("000003.SZ", [0, 1, 2, 3]),
    ):
        for i in available:
            close = 10.0 if i < 3 else 9.0
            rows.append(
                dict(
                    ts_code=symbol,
                    trade_date=days[i],
                    open=close,
                    high=close + 1,
                    low=close - 1,
                    close=close,
                    pre_close=close,
                    vol=2.0,
                    amount=0.02,
                )
            )
    for row in rows:
        if row["ts_code"] == "000001.SZ" and row["trade_date"] == days[3]:
            row["pre_close"] = 10.0
    data = {"daily": pl.DataFrame(rows)}
    data["trade_cal"] = pl.DataFrame(
        dict(
            exchange=["SSE"] * 4,
            cal_date=days,
            is_open=[1] * 4,
            pretrade_date=["20251231"] + days[:-1],
        )
    )
    data["stock_basic"] = pl.DataFrame(
        dict(
            ts_code=["000001.SZ", "000002.SZ", "000003.SZ"],
            name=["甲", "乙", "丙"],
            market=["主板"] * 3,
            exchange=["SZSE"] * 3,
            list_date=[days[0], days[1], days[0]],
            delist_date=[None, days[3], None],
        )
    )
    data["adj_factor"] = (
        data["daily"]
        .select("ts_code", "trade_date")
        .with_columns(
            pl.when(
                (pl.col("ts_code") == "000003.SZ") & (pl.col("trade_date") == days[3])
            )
            .then(10.0 / 9.0)
            .otherwise(1.0)
            .alias("adj_factor")
        )
    )
    data["daily_basic"] = (
        data["daily"]
        .select("ts_code", "trade_date", "close")
        .with_columns(
            *[
                pl.lit(2.0).alias(c)
                for c in (
                    "turnover_rate",
                    "pe",
                    "pb",
                    "ps",
                    "dv_ratio",
                    "pe_ttm",
                    "ps_ttm",
                    "dv_ttm",
                    "total_share",
                    "float_share",
                    "free_share",
                )
            ],
            (pl.col("close") * 2).alias("total_mv"),
            (pl.col("close") * 2).alias("circ_mv"),
        )
    )
    data["stk_limit"] = (
        data["daily"]
        .select("ts_code", "trade_date", "pre_close")
        .with_columns(
            (pl.col("pre_close") * 1.1).alias("up_limit"),
            (pl.col("pre_close") * 0.9).alias("down_limit"),
        )
    )
    data["suspend_d"] = pl.DataFrame(
        dict(
            ts_code=["000001.SZ"],
            trade_date=[days[2]],
            suspend_type=["S"],
            suspend_timing=pl.Series([None], dtype=pl.String),
        )
    )
    data["stock_st"] = pl.DataFrame(
        dict(
            ts_code=["000003.SZ", "000001.SZ", "000003.SZ", "000003.SZ"],
            trade_date=days,
            type=["ST"] * 4,
        )
    )
    st_keys = (
        data["stock_st"]
        .select("ts_code", "trade_date")
        .with_columns(pl.lit(True).alias("_st"))
    )
    data["stk_limit"] = (
        data["stk_limit"]
        .join(st_keys, on=["ts_code", "trade_date"], how="left")
        .with_columns(
            (
                pl.col("pre_close")
                * (
                    1
                    + pl.when(pl.col("_st").fill_null(False)).then(0.05).otherwise(0.1)
                )
            ).alias("up_limit"),
            (
                pl.col("pre_close")
                * (
                    1
                    - pl.when(pl.col("_st").fill_null(False)).then(0.05).otherwise(0.1)
                )
            ).alias("down_limit"),
        )
        .drop("_st")
    )
    data["index_weight"] = pl.DataFrame(
        [
            dict(
                index_code=code,
                con_code="000001.SZ",
                trade_date="20251231",
                weight=60.0,
            )
            for code in INDEXES.values()
        ]
        + [
            dict(index_code=code, con_code="000001.SZ", trade_date=days[0], weight=60.0)
            for code in INDEXES.values()
        ]
        + [
            dict(
                index_code=code, con_code="000003.SZ", trade_date=days[2], weight=100.0
            )
            for code in INDEXES.values()
        ]
    )
    data["index_member_all"] = pl.DataFrame(
        dict(
            ts_code=["000001.SZ"] * 2,
            l1_code=["801010.SI", "801020.SI"],
            l3_code=["x", "y"],
            in_date=[days[0], days[2]],
            out_date=[days[1], None],
        )
    )
    data["dividend"] = pl.DataFrame(
        dict(
            ts_code=["000003.SZ"],
            end_date=["20251231"],
            ann_date=[days[0]],
            div_proc=["实施"],
            ex_date=[days[3]],
            imp_ann_date=[days[1]],
            stk_div=[0.1],
            cash_div_tax=[0.1],
        )
    )
    return data


@pytest.fixture
def make_store(tmp_path):
    def create(tables, name="root"):
        store = Store(tmp_path / name)
        for dataset, table in tables.items():
            spec = DATASETS[dataset]
            if spec.partition == "day":
                for day in ("20260102", "20260105", "20260106", "20260107"):
                    subset = table.filter(pl.col(spec.date) == day)
                    store.write(
                        dataset,
                        day,
                        {spec.date: day},
                        FetchResult(subset, "2026-01-08T00:00:00Z", 1),
                    )
            else:
                store.write(
                    dataset, "all", {}, FetchResult(table, "2026-01-08T00:00:00Z", 1)
                )
        return store

    return create
