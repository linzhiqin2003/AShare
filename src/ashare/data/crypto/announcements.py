"""公开 CMS 列表与原文采集；多日期或复杂迁移进入待核对队列，不能猜历史。"""

import html
import json
import re

from ..store import atomic_json
from .events import EventLedger, normalize_event
from .sources import CMS
from .storage import digest, encoded
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


def _drain_article(store, ledger, listing):
    """按原文 SHA / 首次观察时间持久化导入队列，旧正文完成后才能消费新修订。"""
    code = listing["code"]
    key = f"announcements/article/{code}"
    entry = store.manifest.read().get(key)
    if entry is None:
        return 0, 0
    queue_key = f"announcements/ingestion/{code}"
    queue = json.loads(store.get(queue_key)) if store.complete(queue_key) else []
    known = {(r["sha256"], r["observed_at"]) for r in queue}
    previous_sha = None
    observations = []
    for saved in entry.get("revisions", [entry]):
        sha = saved["sha256"]
        if sha == previous_sha:
            continue
        previous_sha = sha
        observed = saved.get("first_observed_at", saved.get("fetched_at"))
        if observed is None:
            raise ValueError("已存公告缺首次观察时间，不能重新抓取后倒填")
        record = dict(
            sha256=sha,
            observed_at=observed,
            release_date=saved.get("release_date", listing["releaseDate"]),
            revision=bool(observations),
            complete=False,
        )
        observations.append(record)
        if (sha, observed) not in known:
            queue.append(record)
            known.add((sha, observed))
    store.put(queue_key, encoded(queue), source="cms_ingestion")
    imported = reviews = 0
    semantic = (
        "original_text",
        "effective_at",
        "settlement_price",
        "replacement",
        "ratio",
        "status",
        "kind",
        "instrument",
        "source_url",
    )
    for record in queue:
        if record["complete"]:
            continue
        detail = json.loads(store.get(key, sha=record["sha256"]))
        observed = record["observed_at"]
        parsed = extract_events(
            dict(
                detail, releaseDate=detail.get("releaseDate") or record["release_date"]
            ),
            observed,
        )
        for e in parsed:
            e.pop("version_id")
            if record["revision"]:
                e["known_at"] = observed
                e["known_at_basis"] = "observed_revision"
            prior = [
                x
                for x in ledger.read()
                if x["event_id"] == e["event_id"]
                or (
                    x["source_url"] == e["source_url"]
                    and x["instrument"] == e["instrument"]
                    and x["kind"] == e["kind"]
                )
            ]
            if any(
                x["known_at"] == e["known_at"]
                and all(x.get(k) == e.get(k) for k in semantic)
                for x in prior
            ):
                continue
            if prior:
                last = max(prior, key=lambda x: x["known_at"])
                if last["manual"] and last["effective_at"] == e["effective_at"]:
                    continue
                e["event_id"] = last["event_id"]
                e["supersedes"] = last["version_id"]
                if e["known_at"] <= last["known_at"]:
                    e["known_at"] = observed
                    e["known_at_basis"] = "observed_revision"
            e.update(body_sha256=record["sha256"], body_observed_at=observed)
            ledger.append(e)
            imported += 1
        store.put(
            f"announcements/pending/{code}",
            encoded(dict(article=detail, parsed=parsed, review_required=True)),
            source="cms",
        )
        record["complete"] = True
        store.put(queue_key, encoded(queue), source="cms_ingestion")
        store.put(
            f"announcements/processed/{code}",
            record["sha256"].encode(),
            source="cms_ingestion",
        )
        reviews += 1
    return imported, reviews


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
    query = dict(
        since=millis(since),
        until=millis(until) if until is not None else None,
        catalogs=list(catalogs),
        symbols=sorted(symbols or []),
    )
    progress_path = (
        store.root / "announcements" / ("progress-" + digest(encoded(query)) + ".json")
    )
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if progress.get("complete"):
        progress = {}
    progress.setdefault("next_pages", {})
    progress.setdefault("finished", [])
    progress.setdefault("cached_page", None)
    for catalog in catalogs:
        if catalog in progress["finished"]:
            continue
        finished = False
        for page in range(progress["next_pages"].get(str(catalog), 1), page_limit + 1):
            key = f"announcements/list/{catalog}/{page}"
            response = (
                json.loads(store.get(key))
                if progress["cached_page"] == key
                else client.json(
                    CMS + "/list/query",
                    params=dict(type=1, pageNo=page, pageSize=50, catalogId=catalog),
                )
            )
            if not response.get("success"):
                raise ValueError("公告列表返回失败")
            store.put(
                key,
                encoded(response),
                source="cms",
                fetched_at=millis(fetched_at),
            )
            progress["cached_page"] = key
            atomic_json(progress_path, progress)
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
                # 必须先消费落盘版本，再发起可能返回新正文的网络请求。
                count, reviews = _drain_article(store, ledger, a)
                imported += count
                pending += reviews
                response = client.json(
                    CMS + "/detail/query", params=dict(articleCode=a["code"])
                )
                if not response.get("success"):
                    raise ValueError("公告正文返回失败")
                detail = dict(response["data"], code=a["code"])
                key = f"announcements/article/{a['code']}"
                payload = encoded(detail)
                previous = store.manifest.read().get(key, {})
                observed_at = (
                    previous.get(
                        "first_observed_at",
                        previous.get("fetched_at", millis(fetched_at)),
                    )
                    if previous.get("sha256") == digest(payload)
                    else millis(fetched_at)
                )
                store.put(
                    key,
                    payload,
                    source="cms",
                    fetched_at=millis(fetched_at),
                    first_observed_at=observed_at,
                    release_date=a["releaseDate"],
                )
                count, reviews = _drain_article(store, ledger, a)
                imported += count
                pending += reviews
            progress["next_pages"][str(catalog)] = page + 1
            progress["cached_page"] = None
            atomic_json(progress_path, progress)
            if min(a["releaseDate"] for a in articles) < millis(since):
                finished = True
                break
        if not finished:
            raise ValueError(f"公告列表超过页数上限 {page_limit}，覆盖未完成")
        progress["finished"].append(catalog)
        progress["cached_page"] = None
        atomic_json(progress_path, progress)
    progress["complete"] = True
    atomic_json(progress_path, progress)
    return dict(pages=pages, imported=imported, pending_review=pending)
