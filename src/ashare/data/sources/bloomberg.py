"""Bloomberg 桌面 API 的取数层（官方客户端 ``blpapi``，``//blp/refdata`` 服务）。**未用真实终端验证。**

拿到终端后第一次该跑的核对清单见 ``docs/FOREIGN_FUTURES.md`` 的「首次核对清单」，最短的三条：

1. ``BloombergSource().reference(["CL1 Comdty"], ["FUT_CHAIN"], overrides={"INCLUDE_EXPIRED_CONTRACTS": "Y"})``，
   确认返回的合约代码写法（已到期的是一位还是两位年份）；
2. ``historical(["CLZ5 Comdty"], ["PX_SETTLE", "PX_LAST"], "20251103", "20251107")`` 与 CME 公布的结算价逐日核对，
   确认 ``PX_SETTLE`` 是结算价、``PX_LAST`` 是收盘（或最后成交）价、日期是 CME 交易日；
3. ``intraday_bars("CLZ5 Comdty", ...)`` 取一天的 1 分钟线，确认时间是 UTC、整日成交量之和与日线接近。

桌面 API 走本机终端（默认 127.0.0.1:8194），不需要密钥。这一层把请求和响应原样转成 Python 字典
（``Message.toPy()``），不做口径换算；换算在 ``ashare.data.futures_foreign`` 里做。不导入其他 ashare 模块。
``blpapi`` 只在 Bloomberg 自己的包索引上，不作为本仓库的可选依赖（锁文件不依赖那台主机）；没装时报错并给出
手动安装命令 ``uv pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi``。

字段错了不会悄悄变成空值：响应里的 ``fieldExceptions``（字段名无效、对该代码不适用）与 ``securityError``、
``responseError`` 一样直接报 ``BloombergError``，并列出出错的字段。

用到的官方接口（blpapi Python 3.24.11 文档）：``SessionOptions.setServerHost/setServerPort``、``Session.start/
openService/getService/sendRequest/nextEvent/stop``、``Service.createRequest``、``Request.set/append/getElement``、
``Element.appendElement/setElement``、``Event.eventType``（``RESPONSE``、``PARTIAL_RESPONSE``、``TIMEOUT``）、
``Message.messageType/toPy``。
"""

import datetime as dt

INSTALL = (
    "uv pip install --index-url "
    "https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi"
)
EXTRA_HINT = f"没有安装 blpapi（只在 Bloomberg 自己的包索引上，需手动安装）：{INSTALL}"
REFDATA = "//blp/refdata"


class BloombergError(RuntimeError):
    """请求级、证券级或字段级的错误（``responseError``、``securityError``、``fieldExceptions``）。"""


def require():
    try:
        import blpapi
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc
    return blpapi


class BloombergSource:
    """同步方式的 ``//blp/refdata`` 请求。``api`` 默认是导入的 ``blpapi`` 模块，测试时传入只实现了上面
    这些名字的假模块。

    例::

        src = BloombergSource()
        rows = src.historical(["CLZ5 Comdty"], ["PX_SETTLE", "OPEN_INT"], "20251103", "20251107")
    """

    def __init__(self, api=None, *, host="127.0.0.1", port=8194, timeout_ms=30000):
        self.api = api if api is not None else require()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._session = None

    # ------------------------------------------------------------------ 会话
    def _service(self):
        if self._session is None:
            options = self.api.SessionOptions()
            options.setServerHost(self.host)
            options.setServerPort(self.port)
            session = self.api.Session(options)
            if not session.start():
                raise BloombergError("无法连接 Bloomberg 桌面 API（终端是否已登录？）")
            if not session.openService(REFDATA):
                session.stop()
                raise BloombergError(f"无法打开服务 {REFDATA}")
            self._session = session
        return self._session.getService(REFDATA)

    def close(self):
        if self._session is not None:
            self._session.stop()
            self._session = None

    def _send(self, request):
        """发出请求，收齐 PARTIAL_RESPONSE 直到 RESPONSE；返回各条消息的 ``toPy()`` 字典。"""
        self._session.sendRequest(request)
        messages = []
        event_type = self.api.Event
        while True:
            event = self._session.nextEvent(self.timeout_ms)
            kind = event.eventType()
            if kind == event_type.TIMEOUT:
                raise BloombergError(f"等待响应超时（{self.timeout_ms} 毫秒）")
            if kind in (event_type.RESPONSE, event_type.PARTIAL_RESPONSE):
                for message in event:
                    body = message.toPy()
                    if "responseError" in body:
                        raise BloombergError(f"请求出错：{body['responseError']}")
                    messages.append(body)
            if kind == event_type.RESPONSE:
                return messages

    @staticmethod
    def _overrides(request, overrides):
        for name, value in (overrides or {}).items():
            item = request.getElement("overrides").appendElement()
            item.setElement("fieldId", name)
            item.setElement("value", value)

    # ------------------------------------------------------------------ 三种请求
    def historical(self, securities, fields, start, end, *, overrides=None):
        """``HistoricalDataRequest``（日频）。``start``/``end`` 写成 YYYYMMDD。
        返回长表行：``dict(security, date, <field>...)``，``date`` 为 ``datetime.date``。"""
        service = self._service()
        request = service.createRequest("HistoricalDataRequest")
        for s in securities:
            request.append("securities", s)
        for f in fields:
            request.append("fields", f)
        request.set("startDate", start)
        request.set("endDate", end)
        request.set("periodicitySelection", "DAILY")
        self._overrides(request, overrides)
        rows = []
        for body in self._send(request):
            data = body["securityData"]
            _check_security(data)
            for item in data.get("fieldData", []):
                rows.append(dict(security=data["security"], **item))
        return rows

    def reference(self, securities, fields, *, overrides=None):
        """``ReferenceDataRequest``。返回 ``dict(security, <field>...)``；批量字段（如 ``FUT_CHAIN``）
        是字典列表（如 ``[{"Security Description": "CLZ6 Comdty"}, ...]``）。"""
        service = self._service()
        request = service.createRequest("ReferenceDataRequest")
        for s in securities:
            request.append("securities", s)
        for f in fields:
            request.append("fields", f)
        self._overrides(request, overrides)
        rows = []
        for body in self._send(request):
            for data in body["securityData"]:
                _check_security(data)
                rows.append(
                    dict(security=data["security"], **data.get("fieldData", {}))
                )
        return rows

    def intraday_bars(self, security, start, end, *, interval=1, event_type="TRADE"):
        """``IntradayBarRequest``：``start``/``end`` 为 UTC 的 ``datetime``（官方说明请求与返回的时间都是
        GMT）。返回 ``dict(security, time, open, high, low, close, volume, numEvents, value)``，``time`` 为
        K 线开始时刻（UTC，未用真实终端验证）。"""
        if interval < 1 or interval > 1440:
            raise ValueError("interval 是分钟数，1 到 1440")
        service = self._service()
        request = service.createRequest("IntradayBarRequest")
        request.set("security", security)
        request.set("eventType", event_type)
        request.set("interval", interval)
        request.set("startDateTime", _naive_utc(start))
        request.set("endDateTime", _naive_utc(end))
        rows = []
        for body in self._send(request):
            for bar in body.get("barData", {}).get("barTickData", []):
                rows.append(dict(security=security, **bar))
        return rows


def _check_security(data):
    """证券级错误和字段级错误都报错。``fieldExceptions`` 形如
    ``[{"fieldId": "PX_SETTLE", "errorInfo": {"category": "BAD_FLD", "message": "Field not valid", ...}}]``。"""
    if data.get("securityError"):
        raise BloombergError(f"{data['security']}: {data['securityError']}")
    exceptions = data.get("fieldExceptions") or []
    if exceptions:
        detail = "；".join(
            f"{e.get('fieldId')}（{(e.get('errorInfo') or {}).get('message', '')}）"
            for e in exceptions
        )
        fields = sorted({str(e.get("fieldId")) for e in exceptions})
        raise BloombergError(
            f"{data['security']}: 字段出错 {fields}：{detail}；字段名或适用范围不对，不当成空值放过"
        )


def _naive_utc(value):
    if value.tzinfo is not None:
        value = value.astimezone(dt.UTC).replace(tzinfo=None)
    return value


__all__ = ["BloombergError", "BloombergSource", "INSTALL", "REFDATA", "require"]
