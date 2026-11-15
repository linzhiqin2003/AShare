"""小批与增量同步；每个原始对象均可断点续跑，配置不包含任何凭据。"""

import json
from datetime import UTC, datetime

from ashare.paths import dataset_dir

from ..store import atomic_json, lock
from .announcements import crawl_announcements
from .archive import archive_prefix, download_series, funding_coverage, read_series
from .events import EventLedger
from .instruments import (
    eligible,
    profile_from_snapshot,
    snapshot_rule,
    validate_profile,
)
from .sources import FAPI, SPOT, PublicClient
from .storage import RawStore, encoded
from .timeutil import DAY, HOUR, millis


def prepare_profiles(client, store, config, now, snapshots):
    frozen_path = store.root / "profiles.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    out = []
    for name in config["instruments"]:
        market, symbol = name.split(":", 1)
        override = config.get("profiles", {}).get(name, {})
        if name in frozen:
            # 显式追加规则可以更新未来，单位仍沿用首版，历史修订生成内容版本。
            p = {**frozen[name], **override, "multiplier": frozen[name]["multiplier"]}
            current = next(
                (s for s in snapshots[market]["symbols"] if s["symbol"] == symbol), None
            )
            if current and eligible(current, market) and "rules" not in override:
                url = (
                    SPOT + "/api/v3/exchangeInfo"
                    if market == "spot"
                    else FAPI + "/fapi/v1/exchangeInfo"
                )
                rule = snapshot_rule(current, now, url)
                last = max(p["rules"], key=lambda r: r["known_at"])
                if any(
                    rule.get(f) != last.get(f)
                    for f in (
                        "qty_step",
                        "min_qty",
                        "min_notional",
                        "liquidation_fee_rate",
                    )
                ):
                    p["rules"] = [*p["rules"], rule]
            p = validate_profile(p)
            frozen[name] = p
            out.append(p)
            continue
        records = snapshots[market]["symbols"]
        original = next((r for r in records if r["symbol"] == symbol), None)
        if original is not None and not eligible(original, market):
            raise ValueError(
                f"{name} 被 contractType/underlyingType/quote/margin 字段剔除"
            )
        source = (
            (SPOT + "/api/v3/exchangeInfo")
            if market == "spot"
            else FAPI + "/fapi/v1/exchangeInfo"
        )
        if original is None:
            if "rules" not in override:
                raise ValueError(f"{name} 没有快照；请提供有出处的历史档案及规则假设")
            p = dict(override)
        else:
            p = profile_from_snapshot(
                original,
                market,
                now,
                source,
                mapping=config.get("mappings", {}).get(name),
            )
            p.update(override)
        if p.get("listed_at") is None:
            # 现货无 onboardDate：从归档目录最早文件的首条观测定覆盖起点。
            prefix = archive_prefix(market, "klines", symbol)
            candidates = sorted(k for k in client.listing(prefix) if k.endswith(".zip"))
            if not candidates:
                raise ValueError(f"{name} 无历史目录，无法定上市覆盖起点")
            first = candidates[0]
            client.archive(store, first)
            p["listed_at"] = read_series(store, [first], market, "klines")[0]["time"]
            p["listing_source"] = "https://data.binance.vision/" + first
            p["listing_basis"] = "first_archived_observation"
        p = validate_profile(p)
        frozen[name] = p
        out.append(p)
    atomic_json(frozen_path, frozen)
    return out


def sync_funding_info(client, store, now):
    """按响应字节冻结公开快照；updateTime 原样保存，绝不当历史证明。"""
    url = FAPI + "/fapi/v1/fundingInfo"
    payload = client.get(url)
    records = json.loads(payload)
    if not isinstance(records, list) or any(
        not isinstance(r, dict) or "symbol" not in r for r in records
    ):
        raise ValueError("fundingInfo 必须是交易对列表")
    if len({r["symbol"] for r in records}) != len(records):
        raise ValueError("fundingInfo 交易对重复")
    key = f"fundingInfo/{now}"
    entry = store.put(key, payload, source="rest", url=url, known_at=now)
    return dict(key=key, fetched_at=now, sha256=entry["sha256"])


def sync_funding_prices(
    client,
    store,
    symbol,
    start,
    end,
    *,
    previous=None,
    previous_sha=None,
    refresh=False,
    refetch=(),
    fetched_at=None,
):
    """增量尾部及新发布的归档区间分别取新快照，旧内容寻址对象保留。"""
    records = (
        json.loads(store.get(previous, sha=previous_sha))
        if previous and store.complete(previous) and not refresh
        else []
    )
    cursor = max((int(r["fundingTime"]) + 1 for r in records), default=start)
    for window in refetch:
        # 对账声明覆盖整月，即使调用方缩小了面板窗口，也要刷新该月旧推断价。
        lo = window["start"]
        hi = min(fetched_at if fetched_at is not None else end, window["end"] - 1)
        if lo <= hi:
            fresh = client.funding(symbol, lo, hi)
            # 替换整个已请求区间；新响应缺笔时不能用旧值填回去。
            records = [r for r in records if not lo <= int(r["fundingTime"]) <= hi]
            records += fresh
    if cursor <= end:
        extra = client.funding(symbol, cursor, end)
        present = {int(r["fundingTime"]) for r in records}
        records += [r for r in extra if int(r["fundingTime"]) not in present]
    records.sort(key=lambda r: int(r["fundingTime"]))
    key = f"funding_prices/{symbol}/{start}-{end}"
    store.put(
        key,
        encoded(records),
        source="rest",
        url=FAPI + "/fapi/v1/fundingRate",
        known_at=fetched_at,
        refreshed_archive_ranges=list(refetch),
    )
    return key


def sync(
    root, *, config=None, client=None, refresh=False, announcements=True, now=None
):
    root = dataset_dir("crypto", root)
    config = config or json.loads((root / "config.json").read_text())
    now = millis(now or datetime.now(UTC))
    start, end = millis(config["start"]), millis(config["end"])
    if start >= end or start % HOUR or end % HOUR or end > now // HOUR * HOUR:
        raise ValueError("同步区间必须是已完成的整小时")
    if not config.get("instruments"):
        raise ValueError("必须显式给 instruments；不隐式做全历史下载")
    own = client is None
    client = client or PublicClient(
        max_bytes=config.get("max_bytes", 40_000_000),
        cms_attempts=config.get("cms_attempts", 8),
        cms_backoff=config.get("cms_backoff", 30),
        cms_backoff_cap=config.get("cms_backoff_cap", 240),
    )
    store = RawStore(root)
    try:
        with lock(root / ".sync.lock"):
            previous_path = root / "sync_plan.json"
            previous_plan = (
                json.loads(previous_path.read_text()) if previous_path.exists() else {}
            )
            previous_manifest = (
                json.loads(
                    (
                        root / "raw_versions" / f"{previous_plan['raw_version']}.json"
                    ).read_text()
                )
                if previous_plan.get("raw_version")
                else {}
            )
            snapshots, universe = {}, {}
            for market in sorted({n.split(":")[0] for n in config["instruments"]}):
                if market not in ("spot", "usdm"):
                    raise ValueError("未知市场")
                url = (
                    SPOT + "/api/v3/exchangeInfo"
                    if market == "spot"
                    else FAPI + "/fapi/v1/exchangeInfo"
                )
                snapshots[market] = client.json(url)
                store.put(
                    f"exchangeInfo/{market}/{now}",
                    encoded(snapshots[market]),
                    source="rest",
                    url=url,
                    known_at=now,
                )
                prefix = archive_prefix(market, "klines")
                universe[market] = [
                    k[len(prefix) :].strip("/")
                    for k in client.listing(prefix, delimiter="/")
                ]
                store.put(
                    f"universe/{market}/{now}",
                    encoded(universe[market]),
                    source="s3",
                    known_at=now,
                )
            funding_info = (
                sync_funding_info(client, store, now) if "usdm" in snapshots else None
            )
            profiles = prepare_profiles(client, store, config, now, snapshots)
            ledger = EventLedger(root)
            for event in config.get("events", []):
                ledger.append(event)
            announcement_result = None
            if announcements:
                announcement_result = crawl_announcements(
                    client,
                    store,
                    since=config.get("announcement_since", config["start"]),
                    fetched_at=now,
                    page_limit=config.get("announcement_page_limit", 100),
                    until=end,
                    symbols=[p["coin"] for p in profiles],
                )
            plan = dict(
                start=start,
                end=end,
                profiles=profiles,
                exceptions=config.get("exceptions", []),
                series={},
                announcement_result=announcement_result,
                funding_info=funding_info,
            )
            for p in profiles:
                name, market, symbol = p["instrument"], p["market"], p["symbol"]
                if symbol not in universe[market]:
                    raise ValueError(f"{name} 未出现在历史归档目录中")
                targets = (
                    ["klines"]
                    if market == "spot"
                    else ["klines", "markPriceKlines", "indexPriceKlines"]
                )
                series = {}
                for dataset in targets:
                    pair = (
                        p.get("index_symbol", symbol.split("_")[0])
                        if dataset == "indexPriceKlines"
                        else symbol
                    )
                    series[dataset] = download_series(
                        client,
                        store,
                        market,
                        dataset,
                        pair,
                        start,
                        end + HOUR,
                        now=now,
                        refresh=refresh,
                    )
                if p["contract_type"] == "PERPETUAL":
                    series["fundingRate"] = download_series(
                        client,
                        store,
                        market,
                        "fundingRate",
                        symbol,
                        start - DAY,
                        end + DAY,
                        now=now,
                        refresh=refresh,
                    )
                    previous = (
                        previous_plan.get("series", {})
                        .get(name, {})
                        .get("funding_prices")
                    )
                    old_keys = (
                        previous_plan.get("series", {})
                        .get(name, {})
                        .get("fundingRate", [])
                    )
                    current_manifest = store.manifest.read()
                    new_archives = [
                        funding_coverage(key)
                        for key in series["fundingRate"]
                        if key not in old_keys
                        or previous_manifest.get(key, {}).get("sha256")
                        != current_manifest[key]["sha256"]
                    ]
                    series["funding_prices"] = sync_funding_prices(
                        client,
                        store,
                        symbol,
                        start - DAY,
                        min(end + DAY, now),
                        previous=previous,
                        previous_sha=previous_manifest.get(previous, {}).get("sha256"),
                        refresh=refresh,
                        refetch=new_archives if previous and not refresh else (),
                        fetched_at=now,
                    )
                elif market == "usdm":
                    pair = p.get("index_symbol", symbol.split("_")[0])
                    key = f"deliveries/{pair}/{now}"
                    delivery = client.json(
                        FAPI + "/futures/data/delivery-price", params=dict(pair=pair)
                    )
                    store.put(
                        key,
                        encoded(delivery),
                        source="rest",
                        url=FAPI + "/futures/data/delivery-price",
                    )
                    series["deliveries"] = key
                plan["series"][name] = series
            plan["events"] = ledger.read()
            plan["raw_version"] = store.freeze()
            atomic_json(root / "sync_plan.json", plan)
            return dict(
                raw_version=plan["raw_version"],
                instruments=len(profiles),
                downloaded_bytes=client.bytes,
                announcements=announcement_result,
            )
    finally:
        store.close()
        if own:
            client.client.close()


def load_plan(root):
    """从冻结清单加载，后续 refresh 不改变这次同步计划所指的数据。"""
    root = dataset_dir("crypto", root)
    plan = json.loads((root / "sync_plan.json").read_text())
    frozen = json.loads(
        (
            dataset_dir("crypto", root) / "raw_versions" / f"{plan['raw_version']}.json"
        ).read_text()
    )
    store = RawStore(root)

    class FrozenStore:
        def get(self, key):
            return store.get(key, sha=frozen[key]["sha256"])

    try:
        data = {}
        plan["funding_price_refs"] = {}
        plan["funding_archive_coverage"] = {}
        for p in plan["profiles"]:
            series = plan["series"][p["instrument"]]
            data[p["instrument"]] = {
                key: json.loads(FrozenStore().get(value))
                if isinstance(value, str)
                else read_series(FrozenStore(), value, p["market"], key)
                for key, value in series.items()
            }
            if p["contract_type"] == "PERPETUAL":
                key = series["funding_prices"]
                plan["funding_price_refs"][p["instrument"]] = dict(
                    key=key, sha256=frozen[key]["sha256"]
                )
                plan["funding_archive_coverage"][p["instrument"]] = [
                    dict(funding_coverage(key), sha256=frozen[key]["sha256"])
                    for key in series["fundingRate"]
                ]
        if plan.get("funding_info"):
            snapshot = plan["funding_info"]
            plan["funding_info"] = dict(
                snapshot, records=json.loads(FrozenStore().get(snapshot["key"]))
            )
        return plan, data
    finally:
        store.close()
