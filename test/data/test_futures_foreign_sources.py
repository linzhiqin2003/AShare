"""境外期货的取数层：费用闸门、缺可选依赖、密钥只从环境变量读、假客户端与官方库的签名和返回形状一致、
Bloomberg 请求只用文档里的元素。

「默认不产生计费请求」的证明：假客户端记下每次调用，默认预算下 ``timeseries.get_range`` 的调用数为 0，
适配器的一切公开方法都在发请求前停下。没证明的：真实服务的 ``get_cost`` 估算与实际扣费是否一致
（官方说明按实际发送的字节计费，估算可能偏高）。

装了 ``databento`` 时多跑两项：假客户端的方法签名与官方库逐个参数相同；用 ``databento_dbn`` 把夹具编码成
真正的 DBN 字节、经官方 ``DBNStore.to_df()`` 读出，与假对象给的表在适配器用到的列上逐位相同。
"""

import datetime as dt
import inspect
import sys

import polars as pl
import pytest
from futures_foreign_fixtures import (
    CLOSED_DAYS,
    END,
    START,
    FakeBlpapi,
    FakeHistorical,
    FakeMetadata,
    FakeStore,
    FakeSymbology,
    FakeTimeseries,
    build,
    minute_bars,
)

from ashare.data.futures_foreign_bloomberg import BloombergFutures
from ashare.data.futures_foreign_databento import DatabentoFutures
from ashare.data.sources import bloomberg as bbg
from ashare.data.sources import databento as dbs
from ashare.data.sources.bloomberg import BloombergError, BloombergSource
from ashare.data.sources.databento import BudgetExceeded, DatabentoSource


# ---------------------------------------------------------------- 费用闸门
def test_default_budget_never_fetches():
    fake = FakeHistorical()
    source = DatabentoSource(client=fake)
    assert source.budget_usd == 0
    with pytest.raises(BudgetExceeded) as caught:
        source.fetch(
            "GLBX.MDP3",
            "ohlcv-1h",
            ["CL.FUT"],
            "2026-11-02",
            "2026-11-07",
            stype_in="parent",
        )
    assert caught.value.estimate > 0 and "估算费用" in str(caught.value)
    assert fake.count("get_cost") == 1 and fake.count("get_range") == 0
    # 估算为 0 的请求在预算 0 下也不取。
    free = FakeHistorical()
    free.metadata.get_cost = lambda **kw: 0.0
    with pytest.raises(BudgetExceeded):
        DatabentoSource(client=free).fetch(
            "GLBX.MDP3",
            "definition",
            ["CL.FUT"],
            "2026-11-02",
            "2026-11-03",
            stype_in="parent",
        )
    assert free.count("get_range") == 0


def test_adapter_default_budget_only_estimates():
    fake = FakeHistorical()
    adapter = DatabentoFutures(DatabentoSource(client=fake), ["CL", "B"], START, END)
    quote = adapter.estimate()
    assert len(quote["requests"]) == 6 and quote["total"] > 0
    for call in (
        adapter.contracts,
        adapter.daily,
        adapter.calendar,
        adapter.params,
        lambda: adapter.minute("20261104", "20261104"),
    ):
        with pytest.raises(BudgetExceeded):
            call()
    assert fake.count("get_range") == 0 and fake.count("resolve") == 0


def test_budget_is_cumulative():
    fake = FakeHistorical()
    source = DatabentoSource(client=fake)
    one = source.estimate(
        "GLBX.MDP3",
        "definition",
        ["CL.FUT"],
        "2026-11-02",
        "2026-11-12",
        stype_in="parent",
    )
    source.budget_usd = one * 1.5
    source.fetch(
        "GLBX.MDP3",
        "definition",
        ["CL.FUT"],
        "2026-11-02",
        "2026-11-12",
        stype_in="parent",
    )
    assert source.spent_usd == pytest.approx(one)
    # 同一个请求再取一次：会话内缓存，不估价、不取、不计费。
    again = source.fetch(
        "GLBX.MDP3",
        "definition",
        ["CL.FUT"],
        "2026-11-02",
        "2026-11-12",
        stype_in="parent",
    )
    assert again.height > 0 and source.spent_usd == pytest.approx(one)
    with pytest.raises(BudgetExceeded) as caught:
        source.fetch(
            "GLBX.MDP3",
            "definition",
            ["CL.FUT"],
            "2026-11-03",
            "2026-11-13",
            stype_in="parent",
        )
    assert caught.value.spent == pytest.approx(one)
    assert fake.count("get_range") == 1
    # 预算足够时，adapter 的全部请求逐个先估价再取，估价次数 = 取数次数。
    fake = FakeHistorical()
    DatabentoFutures(
        DatabentoSource(budget_usd=1e6, client=fake),
        ["CL"],
        START,
        END,
        closed_days=CLOSED_DAYS,
    ).daily()
    assert fake.count("get_cost") == fake.count("get_range") == 3


def test_get_cost_and_get_range_get_same_request():
    seen = []
    fake = FakeHistorical()
    real_cost, real_range = fake.metadata.get_cost, fake.timeseries.get_range
    fake.metadata.get_cost = lambda **kw: seen.append(("cost", kw)) or real_cost(**kw)
    fake.timeseries.get_range = lambda **kw: (
        seen.append(("range", kw)) or real_range(**kw)
    )
    DatabentoSource(budget_usd=100, client=fake).fetch(
        "GLBX.MDP3",
        "statistics",
        ["CL.FUT"],
        "2026-11-02",
        "2026-11-07",
        stype_in="parent",
    )
    assert seen[0][1] == seen[1][1]  # 估的就是要取的


# ---------------------------------------------------------------- 缺可选依赖、密钥
def test_missing_databento_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "databento", None)
    with pytest.raises(ImportError, match=r"ashare\[databento\]"):
        dbs.require()
    with pytest.raises(ImportError, match=r"ashare\[databento\]"):
        DatabentoSource().estimate(
            "GLBX.MDP3", "definition", ["CL.FUT"], "2026-11-02", "2026-11-03"
        )
    # 适配器模块本身不依赖官方库，导入不报错（给了假客户端就能用）。
    assert DatabentoFutures(
        DatabentoSource(client=FakeHistorical()), ["CL"], START, END
    )


def test_missing_blpapi_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "blpapi", None)
    # blpapi 不作可选依赖（锁文件不依赖 Bloomberg 的包索引），报错给出手动安装命令。
    with pytest.raises(ImportError, match="--index-url https://blpapi.bloomberg.com"):
        bbg.require()
    with pytest.raises(ImportError, match="uv pip install"):
        BloombergSource()


def test_key_only_from_environment(monkeypatch, capsys):
    secret = "db-SENTINEL-NOT-A-REAL-KEY"
    made = []

    class Module:
        @staticmethod
        def Historical(key=None, gateway=None):
            made.append(key)
            return FakeHistorical()

    monkeypatch.setitem(sys.modules, "databento", Module)
    monkeypatch.delenv(dbs.KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=dbs.KEY_ENV) as caught:
        DatabentoSource().estimate(
            "GLBX.MDP3",
            "definition",
            ["CL.FUT"],
            "2026-11-02",
            "2026-11-03",
            stype_in="parent",
        )
    assert secret not in str(caught.value)
    monkeypatch.setenv(dbs.KEY_ENV, secret)
    source = DatabentoSource()
    with pytest.raises(BudgetExceeded) as caught:
        source.fetch(
            "GLBX.MDP3",
            "definition",
            ["CL.FUT"],
            "2026-11-02",
            "2026-11-03",
            stype_in="parent",
        )
    assert made == [secret]
    text = str(caught.value) + repr(source.log) + capsys.readouterr().out
    assert secret not in text


# ---------------------------------------------------------------- 假对象与官方库一致（装了 databento 才跑）
def test_fake_signatures_match_official_library():
    databento = pytest.importorskip("databento")
    from databento.historical.api.metadata import MetadataHttpAPI
    from databento.historical.api.symbology import SymbologyHttpAPI
    from databento.historical.api.timeseries import TimeseriesHttpAPI

    for fake, real in (
        (FakeMetadata.get_cost, MetadataHttpAPI.get_cost),
        (FakeTimeseries.get_range, TimeseriesHttpAPI.get_range),
        (FakeSymbology.resolve, SymbologyHttpAPI.resolve),
        (FakeStore.to_df, databento.DBNStore.to_df),
    ):
        assert list(inspect.signature(fake).parameters) == list(
            inspect.signature(real).parameters
        ), fake.__qualname__


def _encode(schema, rows):
    """夹具记录 → 真正的 DBN 字节 → 官方 DBNStore。"""
    import databento as db
    import databento_dbn as d

    def ns(t):
        return int(t.timestamp()) * 10**9 if t is not None else d.UNDEF_TIMESTAMP

    def px(x):
        return d.UNDEF_PRICE if x != x else round(x * 1e9)

    records = []
    for r in rows:
        if schema.startswith("ohlcv"):
            rtype = 0x22 if schema == "ohlcv-1h" else 0x21
            records.append(
                d.OHLCVMsg(
                    rtype=rtype,
                    publisher_id=1,
                    instrument_id=r["instrument_id"],
                    ts_event=ns(r["ts_event"]),
                    open=px(r["open"]),
                    high=px(r["high"]),
                    low=px(r["low"]),
                    close=px(r["close"]),
                    volume=r["volume"],
                )
            )
        elif schema == "statistics":
            records.append(
                d.StatMsg(
                    publisher_id=1,
                    instrument_id=r["instrument_id"],
                    ts_recv=ns(r["ts_recv"]),
                    ts_event=ns(r["ts_event"]),
                    ts_ref=ns(r["ts_ref"]),
                    price=px(r["price"]),
                    quantity=r["quantity"],
                    sequence=0,
                    ts_in_delta=0,
                    stat_type=d.StatType(r["stat_type"]),
                    channel_id=0,
                    update_action=d.StatUpdateAction(r["update_action"]),
                    stat_flags=r["stat_flags"],
                )
            )
        else:
            records.append(
                d.InstrumentDefMsg(
                    publisher_id=1,
                    instrument_id=r["instrument_id"],
                    ts_event=ns(r["ts_recv"]),
                    ts_recv=ns(r["ts_recv"]),
                    min_price_increment=px(r["min_price_increment"]),
                    display_factor=px(r["display_factor"]),
                    raw_symbol=r["raw_symbol"],
                    asset=r["asset"],
                    security_type=r["security_type"],
                    instrument_class=d.InstrumentClass(r["instrument_class"]),
                    security_update_action=d.SecurityUpdateAction(
                        r["security_update_action"]
                    ),
                    expiration=ns(r["expiration"]),
                    activation=ns(r["activation"]),
                    maturity_year=r["maturity_year"],
                    maturity_month=r["maturity_month"],
                    unit_of_measure=r["unit_of_measure"],
                    unit_of_measure_qty=px(r["unit_of_measure_qty"]),
                    currency=r["currency"],
                    exchange=r["exchange"],
                )
            )
    meta = d.Metadata(
        dataset="GLBX.MDP3",
        schema=d.Schema(schema),
        start=0,
        stype_in=d.SType.PARENT,
        stype_out=d.SType.INSTRUMENT_ID,
        end=None,
        symbols=["CL.FUT"],
        partial=[],
        not_found=[],
        mappings=[],
        limit=None,
        ts_out=False,
        version=3,
    )
    return db.DBNStore.from_bytes(meta.encode() + b"".join(bytes(r) for r in records))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("schema", ["definition", "statistics", "ohlcv-1h", "ohlcv-1m"])
def test_fake_frames_match_official_dbnstore(schema):
    pytest.importorskip("databento")
    from futures_foreign_fixtures import _frame

    data = build()
    rows = {
        "definition": data.defs,
        "statistics": data.stats[:400],
        "ohlcv-1h": data.bars[:400],
        "ohlcv-1m": minute_bars()[:200],
    }[schema]
    fake = dbs.frame_of(_frame(rows, schema))
    real = dbs.frame_of(_encode(schema, rows))
    used = [c for c in fake.columns if c != "symbol"]
    assert set(used) <= set(real.columns)
    assert fake.select(used).equals(real.select(used)), schema
    # 用到的列就是这些（适配器多读一列，这里会因为假对象缺列而在别的测试里报错）。
    assert real.columns[0] == {"ohlcv-1h": "ts_event", "ohlcv-1m": "ts_event"}.get(
        schema, "ts_recv"
    )


def test_official_dbnstore_through_adapter():
    """官方 DBNStore 经适配器：用真实库读出的表替换假对象，统一日线与用假对象时逐位相同。"""
    pytest.importorskip("databento")
    import warnings

    data = build()

    class Real(FakeHistorical):
        pass

    fake = Real(data)
    original = fake.timeseries.get_range

    def get_range(
        dataset,
        start,
        end=None,
        symbols=None,
        schema="trades",
        stype_in="raw_symbol",
        stype_out="instrument_id",
        limit=None,
        path=None,
    ):
        rows = (
            original(
                dataset, start, end, symbols, schema, stype_in, stype_out, limit, path
            )
            ._frame.reset_index()
            .to_dict("records")
        )
        return _encode(schema, [_plain(r) for r in rows])

    fake.timeseries.get_range = get_range
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via_real = DatabentoFutures(
            DatabentoSource(budget_usd=100, client=fake),
            ["CL"],
            START,
            END,
            closed_days=CLOSED_DAYS,
        ).daily()
    via_fake = DatabentoFutures(
        DatabentoSource(budget_usd=100, client=FakeHistorical(data)),
        ["CL"],
        START,
        END,
        closed_days=CLOSED_DAYS,
    ).daily()
    assert via_real.equals(via_fake)


def _plain(r):
    out = {}
    for k, v in r.items():
        if hasattr(v, "to_pydatetime"):
            v = None if v != v else v.to_pydatetime()
        elif v is not None and not isinstance(v, str | float) and hasattr(v, "item"):
            v = v.item()
        out[k] = v
    return out


# ---------------------------------------------------------------- Bloomberg 取数层
def test_bloomberg_requests_use_documented_elements():
    api = FakeBlpapi()
    source = BloombergSource(api, host="10.0.0.1", port=8195)
    rows = source.historical(
        ["CLZ6 Comdty", "CL1 Comdty"], ["PX_SETTLE", "OPEN_INT"], "20261102", "20261106"
    )
    assert {r["security"] for r in rows} == {"CLZ6 Comdty", "CL1 Comdty"}
    assert all(isinstance(r["date"], dt.date) for r in rows)
    op, values = api.requests[-1]
    assert op == "HistoricalDataRequest" and values["periodicitySelection"] == "DAILY"
    assert values["startDate"] == "20261102" and values["securities"] == [
        "CLZ6 Comdty",
        "CL1 Comdty",
    ]
    assert (
        source._session.options.host == "10.0.0.1"
        and source._session.options.port == 8195
    )
    bars = source.intraday_bars(
        "CLZ6 Comdty",
        dt.datetime(2026, 11, 4, 0, tzinfo=dt.UTC),
        dt.datetime(2026, 11, 4, 1, tzinfo=dt.UTC),
    )
    assert len(bars) == 60 and api.requests[-1][1]["startDateTime"] == dt.datetime(
        2026, 11, 4, 0
    )
    with pytest.raises(ValueError, match="interval"):
        source.intraday_bars(
            "CLZ6 Comdty",
            dt.datetime(2026, 11, 4),
            dt.datetime(2026, 11, 5),
            interval=0,
        )


def test_fake_request_rejects_undocumented_elements():
    from futures_foreign_fixtures import FakeRequest

    req = FakeRequest("HistoricalDataRequest")
    with pytest.raises(ValueError, match="没有元素"):
        req.set("periodicity", "DAILY")
    with pytest.raises(ValueError, match="数组"):
        req.set("securities", "CLZ6 Comdty")


def test_bloomberg_errors_and_timeout():
    api = FakeBlpapi()
    original = api.respond
    api.respond = lambda request: [
        dict(responseError=dict(category="BAD_ARGS", message="bad"))
    ]
    with pytest.raises(BloombergError, match="请求出错"):
        BloombergSource(api).reference(["CLZ6 Comdty"], ["NAME"])
    api.respond = lambda request: [
        dict(
            securityData=[
                dict(
                    security="XXZ6 Comdty",
                    securityError=dict(message="Unknown/Invalid security"),
                )
            ]
        )
    ]
    with pytest.raises(BloombergError, match="XXZ6"):
        BloombergSource(api).reference(["XXZ6 Comdty"], ["NAME"])
    api.respond = original
    source = BloombergSource(api)
    source._service()
    source._session.queue = []
    source._session.sendRequest = lambda request, **kw: None  # 发出后什么都收不到
    with pytest.raises(BloombergError, match="超时"):
        source.reference(["CLZ6 Comdty"], ["NAME"])
    source.close()
    assert source._session is None


def test_bloomberg_adapter_requests():
    api = FakeBlpapi()
    adapter = BloombergFutures(BloombergSource(api), ["CL"], START, END)
    adapter.daily()
    ops = [op for op, _ in api.requests]
    assert ops == [
        "ReferenceDataRequest",
        "ReferenceDataRequest",
        "HistoricalDataRequest",
        "HistoricalDataRequest",
    ]
    history = [v for op, v in api.requests if op == "HistoricalDataRequest"][0]
    assert "CLZ16 Comdty" not in history["securities"]  # 区间外的到期合约不取行情
    assert pl.Series(history["securities"]).str.ends_with(" Comdty").all()


@pytest.mark.filterwarnings("ignore")
def test_dbn_v1_undefined_quantity_read_by_official_library_is_missing():
    """真实 DBN v1 记录经官方 ``to_df()``（不升级版本）读出 quantity = 2147483647（int32），适配器当空值。"""
    pytest.importorskip("databento")
    import databento as db
    import databento_dbn as d

    from ashare.data.futures_foreign import CME
    from ashare.data.futures_foreign_databento import quantity_rows

    ref = dt.datetime(2026, 11, 4, tzinfo=dt.UTC)
    when = dt.datetime(2026, 11, 5, 10, tzinfo=dt.UTC)
    records = [
        d.StatMsgV1(
            publisher_id=1,
            instrument_id=1002,
            ts_recv=int(when.timestamp()) * 10**9,
            ts_event=int(when.timestamp()) * 10**9,
            ts_ref=int(ref.timestamp()) * 10**9,
            price=d.UNDEF_PRICE,
            quantity=quantity,
            sequence=0,
            ts_in_delta=0,
            stat_type=d.StatType.OPEN_INTEREST,
            channel_id=0,
            update_action=d.StatUpdateAction.NEW,
            stat_flags=0,
        )
        for quantity in (2**31 - 1,)
    ]
    meta = d.Metadata(
        dataset="GLBX.MDP3",
        schema=d.Schema("statistics"),
        start=0,
        stype_in=d.SType.PARENT,
        stype_out=d.SType.INSTRUMENT_ID,
        end=None,
        symbols=["CL.FUT"],
        partial=[],
        not_found=[],
        mappings=[],
        limit=None,
        ts_out=False,
        version=1,
    )
    store = db.DBNStore.from_bytes(meta.encode() + b"".join(bytes(r) for r in records))
    frame = dbs.frame_of(store).drop("symbol", strict=False)
    assert frame["quantity"].to_list() == [2**31 - 1]
    rows = quantity_rows(frame, 9, CME, {"20261104", "20261105"}, "open_interest")
    assert rows.rows() == [(1002, "20261104", None)]
