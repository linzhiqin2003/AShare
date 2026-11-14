"""交易对、报价单位与原始下单规则。风险档位/费用考据留给 C3。"""

import json
import math
from decimal import Decimal
from pathlib import Path

from .timeutil import HOUR, millis

CONTRACT_TYPES = {"PERPETUAL", "CURRENT_QUARTER", "NEXT_QUARTER"}


def mapping_table():
    return json.loads(Path(__file__).with_name("mappings.json").read_text())


def eligible(symbol, market):
    if market == "spot":
        return symbol.get("quoteAsset") == "USDT" and symbol.get(
            "isSpotTradingAllowed", True
        )
    return (
        symbol.get("contractType") in CONTRACT_TYPES
        and symbol.get("underlyingType") == "COIN"
        and symbol.get("quoteAsset") == symbol.get("marginAsset") == "USDT"
    )


def snapshot_rule(symbol, known_at, source_url):
    filters = {r["filterType"]: r for r in symbol["filters"]}
    lot = filters.get("MARKET_LOT_SIZE", {})
    if Decimal(lot.get("stepSize", "0")) <= 0:
        lot = filters["LOT_SIZE"]
    notional = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {}))
    return dict(
        effective_at=millis(known_at),
        known_at=millis(known_at),
        qty_step=float(lot["stepSize"]),
        min_qty=float(lot["minQty"]),
        min_notional=float(notional.get("notional", notional.get("minNotional", 0))),
        source="snapshot",
        source_url=source_url,
        liquidation_fee_rate=float(symbol["liquidationFee"])
        if symbol.get("liquidationFee")
        else None,
    )


def fixed_unit(rules):
    exponents = [
        Decimal(str(r["qty_step"])).normalize().as_tuple().exponent for r in rules
    ]
    if not exponents:
        raise ValueError("不能从缺失的数量规则猜测计量单位")
    return float(Decimal(10) ** min(0, *exponents))


def validate_profile(profile):
    p = dict(profile)
    for field in (
        "instrument",
        "symbol",
        "market",
        "base_asset",
        "coin",
        "coin_units",
        "listed_at",
        "source_url",
        "rules",
    ):
        if field not in p:
            raise ValueError(f"交易对档案缺少 {field}")
    p["listed_at"] = millis(p["listed_at"])
    p.setdefault("mapping_source", p["source_url"])
    if (
        p["market"] not in ("spot", "usdm")
        or p["instrument"] != f"{p['market']}:{p['symbol']}"
    ):
        raise ValueError("交易对标识与市场不匹配")
    if (
        not p["mapping_source"]
        or not math.isfinite(p["coin_units"])
        or p["coin_units"] <= 0
    ):
        raise ValueError("币映射缺少出处或单位非法")
    if p["quote_asset"] != "USDT" or (
        p["market"] == "usdm"
        and (
            p["margin_asset"] != "USDT"
            or p["underlying_type"] != "COIN"
            or p["contract_type"] not in CONTRACT_TYPES
        )
    ):
        raise ValueError("只接受 COIN、永续/季度、报价与保证金均 USDT")
    p["rules"] = sorted(
        [
            dict(
                r,
                known_at=millis(r["known_at"]),
                effective_at=millis(r["effective_at"]),
            )
            for r in p["rules"]
        ],
        key=lambda r: (r["effective_at"], r["known_at"]),
    )
    if not p["rules"]:
        raise ValueError("档案缺规则；需有出处的显式假设")
    p.setdefault("multiplier", fixed_unit(p["rules"]) if p["market"] == "usdm" else 1.0)
    unit = Decimal(str(p["multiplier"]))
    if not unit.is_finite() or unit <= 0:
        raise ValueError("乘数必须为正")
    for r in p["rules"]:
        if (
            r["source"] not in ("announcement", "snapshot", "assumption")
            or not r["source_url"]
        ):
            raise ValueError("规则必须有逐条出处与来源等级")
        for f in ("qty_step", "min_qty", "min_notional"):
            value = Decimal(str(r[f]))
            if not value.is_finite() or value < 0 or (f == "qty_step" and value == 0):
                raise ValueError("数量规则值非法")
            if p["market"] == "usdm" and f != "min_notional" and value % unit:
                raise ValueError("新规则不能被冻结的乘数整除；需要显式建立新的数据系列")
    return p


def profile_from_snapshot(symbol, market, known_at, source_url, *, mapping=None):
    if not eligible(symbol, market):
        raise ValueError("交易对不属于研究范围")
    base = symbol["baseAsset"]
    # 数字前缀不能一概当作单位倍率（例如 1INCH）；必须逐条有来源。
    mapping = (
        mapping
        or mapping_table()["denominations"].get(f"{market}:{symbol['symbol']}")
        or dict(coin=base, coin_units=1, mapping_source=source_url)
    )
    if base.startswith("1000") and mapping["coin_units"] == 1:
        raise ValueError("1000 类合约需要有出处的币单位映射")
    return dict(
        instrument=f"{market}:{symbol['symbol']}",
        symbol=symbol["symbol"],
        market=market,
        base_asset=base,
        quote_asset=symbol["quoteAsset"],
        margin_asset=symbol.get("marginAsset"),
        contract_type=symbol.get("contractType", "SPOT"),
        underlying_type=symbol.get("underlyingType", "COIN"),
        listed_at=symbol.get("onboardDate"),
        status=symbol.get("status"),
        delivery_at=symbol.get("deliveryDate"),
        snapshot_known_at=millis(known_at),
        source_url=source_url,
        rules=[snapshot_rule(symbol, known_at, source_url)],
        **mapping,
    )


def rule_at(profile, at):
    """公告按已知时间和向上取整的生效时间；最早快照向前沿用要标假设。"""
    candidates = [
        r
        for r in profile["rules"]
        if r["known_at"] <= at and ((r["effective_at"] + HOUR - 1) // HOUR) * HOUR <= at
    ]
    if candidates:
        rule = max(candidates, key=lambda r: (r["effective_at"], r["known_at"]))
        return dict(rule, assumed=rule["source"] == "assumption")
    snapshots = [
        r for r in profile["rules"] if r["source"] in ("snapshot", "assumption")
    ]
    if not snapshots:
        raise ValueError(f"{profile['instrument']} 在 {at} 缺已知规则")
    return dict(
        min(snapshots, key=lambda r: r["known_at"]), assumed=True, source="assumption"
    )


def coin_price(price, profile):
    return price / profile["coin_units"]


def coin_quantity(quantity, profile):
    return quantity * profile["coin_units"]


def mapping_at(profile, events, at):
    """返回当时已生效且已知的代币转换关系；不替 C3 虚构成交或现金兑付。"""
    from .events import asof

    result = dict(
        coin=profile["coin"],
        coin_units=profile["coin_units"],
        source_url=profile.get("mapping_source", profile["source_url"]),
        conversions=[],
    )
    for e in asof(events, at, instrument=profile["instrument"]):
        if e["kind"] == "migration" and e["effective_at"] <= millis(at):
            result["coin"] = e["replacement"]
            result["coin_units"] *= e["ratio"]
            result["source_url"] = e["source_url"]
            result["conversions"].append(e["version_id"])
    return result
