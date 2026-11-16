"""DATA_SPEC v1 的声明性词典；写入方和独立校验器唯一共享的规则。"""

SPEC_VERSION = "1"
KINDS = frozenset({"bars", "instruments", "calendar", "corporate_actions", "events"})
FREQUENCIES = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"})
INTRADAY = frozenset({"1m", "5m", "15m", "30m", "1h", "4h"})
TYPES = frozenset({"date", "datetime", "number", "integer", "boolean", "string"})
RESOURCE_NAME = r"[a-z0-9][a-z0-9._-]*"
NAME = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*"
PARTITION = r"(?:[0-9]{4}(?:-(?:0[1-9]|1[0-2]))?|all)"
KNOWN_AT = (
    r"(?:start|end|unknown|end\+[1-9][0-9]*(?:us|ms|s|m|h|d|w)|column:[a-z][a-z0-9_]*)"
)
SYMBOLOGY = {
    "cn": r"(?:[A-Z]?[0-9]{6}\.(?:SH|SZ|BJ|SI)|[A-Z0-9]{6}\.CSI)",
    "hk": r"[0-9]{5}\.HK",
    "us": r"[A-Z][A-Z0-9]*(?:[.\-][A-Z0-9]+)*",
    "futures": r"[A-Za-z]+[0-9]{2}(?:0[1-9]|1[0-2])\.[A-Z]+",
    "crypto": r"(?:spot|usdm):[A-Z0-9]+(?:_[0-9]{6})?",
    "custom": r"\S+",
}
CURRENCIES = frozenset(
    """AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BOV BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SLL SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD USN UYI UYU UYW UZS VED VES VND VUV WST XAF XAG XAU XBA XBB XBC XBD XCD XCG XDR XOF XPD XPF XPT XSU XTS XUA XXX YER ZAR ZMW ZWG ZWL USDT USDC""".split()
)
QUANTITIES = frozenset({"share", "contract", "coin", "bond"})
PRICE_UNITS = frozenset(f"{c}/{q}" for c in CURRENCIES for q in QUANTITIES)
UNITS = CURRENCIES | QUANTITIES | {"ratio", "1", "count"} | PRICE_UNITS
ASSET_UNITS = {
    "stock": "share",
    "index": "share",
    "etf": "share",
    "fund": "share",
    "bond": "bond",
    "convertible": "bond",
    "future": "contract",
    "option": "contract",
    "spot": "coin",
    "perpetual": "contract",
}
# (Table Schema 类型, 单位类别/固定单位, 默认可知时点)。
FIELDS = {
    "open": ("number", "price", "start"),
    "high": ("number", "price", "end"),
    "low": ("number", "price", "end"),
    "close": ("number", "price", "end"),
    "previous_close": ("number", "price", "start"),
    "volume": ("number", "quantity", "end"),
    "amount": ("number", "currency", "end"),
    "trade_count": ("integer", "count", "end"),
    "open_interest": ("number", "quantity", "end"),
    "settle": ("number", "price", "end"),
    "vwap": ("number", "price", "end"),
    "adj_factor": ("number", "1", "start"),
    "limit_up": ("number", "price", "start"),
    "limit_down": ("number", "price", "start"),
    "suspended": ("boolean", "1", "start"),
    "taker_buy_volume": ("number", "quantity", "end"),
}
PRICES = frozenset(k for k, v in FIELDS.items() if v[1] == "price")
NONNEGATIVE = frozenset(
    {"volume", "amount", "taker_buy_volume", "trade_count", "open_interest"}
)
QUALITY = {"SUSPECT": 1, "CORRECTED": 2, "SYNTHETIC": 4, "STALE": 8}
ACTION_KINDS = frozenset({"dividend", "split", "rights", "spinoff", "merger", "other"})
STRUCTURAL = {
    "symbol": "string",
    "date": "date",
    "ts": "datetime",
    "available_at": "datetime",
    "quality": "integer",
}
INSTRUMENT_FIELDS = {
    "symbol": "string",
    "asset_type": "string",
    "currency": "string",
    "name": "string",
    "list_date": "date",
    "delist_date": "date",
    "multiplier": "number",
    "tick_size": "number",
    "lot_size": "number",
    "successor": "string",
}
ACTION_FIELDS = {
    "symbol": "string",
    "ex_date": "date",
    "kind": "string",
    "split": "number",
    "cash_dividend": "number",
    "available_at": "datetime",
    "source": "string",
}
