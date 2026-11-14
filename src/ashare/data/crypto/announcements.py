"""公开 CMS 列表与原文采集；多日期或复杂迁移进入待核对队列，不能猜历史。"""

import html
import json
import re

from .events import EventLedger, normalize_event
from .sources import CMS
from .storage import encoded
from .timeutil import DAY, millis


def body_text(value):
    if isinstance(value, dict):
        if value.get("node") == "text":
            return html.unescape(value.get("text", ""))
        return " ".join(
            body_text(v)
            for k, v in value.items()
            if k in ("child", "children", "text", "body")
        )
    if isinstance(value, list):
        return " ".join(body_text(v) for v in value)
    if isinstance(value, str):
        try:
            return body_text(json.loads(value))
        except (ValueError, TypeError):
            return html.unescape(re.sub("<[^>]+>", " ", value))
    return ""


def article_version(article, fetched_at):
    """有更新证据取最晚时间，否则以首发时间作为明确标注的假设。

    首发和更新都缺失才取抓取时刻。日期只有日时取该日结束，且不晚于已观察
    到该内容的时刻。仍标版本未知：没有更新说明不能证明不存在静默修改。
    """
    text = body_text(article.get("body", ""))
    publication = article.get("publishDate") or article.get("releaseDate")
    published = millis(publication) if publication else None
    proven = []
    if article.get("lastUpdateTime"):
        proven.append(int(article["lastUpdateTime"]))
    for found in re.finditer(
        r"\bupdated?\b\s*(?:on\s*)?[:：]?\s*\(?\s*"
        r"(\d{4}-\d{2}-\d{2})(?:[ T]+(\d{2}:\d{2}(?::\d{2})?))?",
        html.unescape(article.get("title", "")) + " " + text,
        re.I,
    ):
        proven.append(
            millis(found[1] + "T" + found[2]) if found[2] else millis(found[1]) + DAY
        )
    if proven:
        known, basis = max(proven), "latest_update_evidence"
    elif published is not None:
        known, basis = published, "publish_no_update_evidence"
    else:
        known, basis = millis(fetched_at), "fetched_no_time_evidence"
    known = min(known, millis(fetched_at))
    return dict(
        title=article.get("title", ""),
        original_text=text,
        published_at=published,
        known_at=known,
        known_at_basis=basis,
        version_unknown=True,
        source_url="https://www.binance.com/en/support/announcement/detail/"
        + article["code"],
    )


def extract_events(article, fetched_at):
    """只自动认定单句含明确合约、UTC 时刻的自动结算；其他内容留待人工标注。"""
    version = article_version(article, fetched_at)
    events = []
    pattern = r"(?:will close all positions and conduct an automatic settlement|will conduct an automatic settlement)([^.]+?)(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s*\(UTC\)"
    for match in re.finditer(pattern, version["original_text"], flags=re.I):
        symbols = re.findall(r"\b([A-Z0-9]+USDT)\b", match[1])
        for symbol in dict.fromkeys(symbols):
            events.append(
                normalize_event(
                    dict(
                        **version,
                        event_id=f"{article['code']}:usdm:{symbol}:settlement",
                        instrument=f"usdm:{symbol}",
                        kind="settlement",
                        effective_at=f"{match[2]}T{match[3]}:00Z",
                    )
                )
            )
    # 现货只认正文显式列出的 USDT 交易对，不能把其他钱包的终止时间拿来用。
    text = version["original_text"]
    pairs = re.search(
        r"exact trading pairs being removed are[:：]\s*([^.]+)", text, re.I
    )
    times = re.findall(
        r"delist and cease trading on all spot trading pairs[^.]*?at\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s*\(UTC\)",
        text,
        re.I,
    )
    times += re.findall(
        r"At\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s*\(UTC\),\s*Binance will delist (?:all existing|and cease trading on all) [^.]*?spot trading pairs",
        text,
        re.I,
    )
    if pairs and len(set(times)) == 1:
        date, hour = times[0]
        for coin in dict.fromkeys(re.findall(r"\b([A-Z0-9]+)/USDT\b", pairs[1])):
            symbol = coin + "USDT"
            events.append(
                normalize_event(
                    dict(
                        **version,
                        event_id=f"{article['code']}:spot:{symbol}:delist",
                        instrument=f"spot:{symbol}",
                        kind="delist",
                        effective_at=f"{date}T{hour}:00Z",
                    )
                )
            )
    return events


def crawl_announcements(
    client,
    store,
    *,
    since,
    fetched_at,
    catalogs=(161, 48, 49),
    page_limit=100,
    until=None,
    symbols=None,
):
    """下架、币种动态、合约动态；列表翻页，原文和无法自动解析的候选均保存。

    公开 CMS 非稳定开发者接口；失败显式上报，允许下次续跑或人工导入。
    """
    seen, pages, imported, pending = set(), 0, 0, 0
    ledger = EventLedger(store.root)
    for catalog in catalogs:
        finished = False
        for page in range(1, page_limit + 1):
            response = client.json(
                CMS + "/list/query",
                params=dict(type=1, pageNo=page, pageSize=50, catalogId=catalog),
            )
            if not response.get("success"):
                raise ValueError("公告列表返回失败")
            store.put(
                f"announcements/list/{catalog}/{page}",
                encoded(response),
                source="cms",
                fetched_at=millis(fetched_at),
            )
            pages += 1
            groups = response["data"]["catalogs"]
            articles = [a for g in groups for a in g["articles"]]
            if not articles:
                finished = True
                break
            for a in articles:
                if a["releaseDate"] < millis(since) or a["code"] in seen:
                    continue
                if until is not None and a["releaseDate"] >= millis(until):
                    continue
                if symbols and not any(
                    re.search(r"\b" + re.escape(s) + r"(?:USDT)?\b", a["title"], re.I)
                    for s in symbols
                ):
                    continue
                seen.add(a["code"])
                if not re.search(
                    r"delist|removal|swap|rebrand|migration|settlement|suspend|resum",
                    a["title"],
                    re.I,
                ):
                    continue
                response = client.json(
                    CMS + "/detail/query", params=dict(articleCode=a["code"])
                )
                if not response.get("success"):
                    raise ValueError("公告正文返回失败")
                detail = dict(response["data"], code=a["code"])
                key = f"announcements/article/{a['code']}"
                previous = store.manifest.read().get(key, {})
                payload = encoded(detail)
                entry = store.put(
                    key, payload, source="cms", fetched_at=millis(fetched_at)
                )
                if previous.get("sha256") == entry["sha256"]:
                    continue
                # 列表补充的首发时间只用于解析，不能改变原文 hash、伪造一次修订。
                parsed = extract_events(
                    dict(
                        detail,
                        releaseDate=detail.get("releaseDate") or a["releaseDate"],
                    ),
                    fetched_at,
                )
                old = ledger.read()
                for e in parsed:
                    prior = [
                        x
                        for x in old
                        if x["event_id"] == e["event_id"]
                        or (
                            x["source_url"] == e["source_url"]
                            and x["instrument"] == e["instrument"]
                            and x["kind"] == e["kind"]
                        )
                    ]
                    if prior:
                        last = max(prior, key=lambda x: x["known_at"])
                        if last["manual"] and last["effective_at"] == e["effective_at"]:
                            continue
                        e.pop("version_id")
                        e["event_id"] = last["event_id"]
                        e["supersedes"] = last["version_id"]
                        if e["known_at"] <= last["known_at"]:
                            e["known_at"] = millis(fetched_at)
                            e["known_at_basis"] = "observed_revision"
                    ledger.append(e)
                    imported += 1
                store.put(
                    f"announcements/pending/{a['code']}",
                    encoded(dict(article=detail, parsed=parsed, review_required=True)),
                    source="cms",
                )
                pending += 1
            if min(a["releaseDate"] for a in articles) < millis(since):
                finished = True
                break
        if not finished:
            raise ValueError(f"公告列表超过页数上限 {page_limit}，覆盖未完成")
    return dict(pages=pages, imported=imported, pending_review=pending)
