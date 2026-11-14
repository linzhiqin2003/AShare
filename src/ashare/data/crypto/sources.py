"""币安只读 HTTP：代理、超时、重试、接口节流、S3 分页与校验和。"""

import hashlib
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx

VISION = "https://data.binance.vision/"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FAPI = "https://fapi.binance.com"
SPOT = "https://api.binance.com"
CMS = "https://www.binance.com/bapi/composite/v1/public/cms/article"


class PublicClient:
    """串行采集；资金费共享限额 ≤ 480/5min，其他 REST 合计 ≤ 5 请求/秒。

    单次 kline limit=1000 的权重不超过 5；全局 REST 间隔也保护 IP 权重。
    不读取密钥，不构造签名。测试可注入 MockTransport、时钟和 sleep。
    """

    def __init__(
        self,
        *,
        transport=None,
        sleep=time.sleep,
        clock=time.monotonic,
        max_bytes=40_000_000,
    ):
        self.client = httpx.Client(
            timeout=httpx.Timeout(30, connect=10),
            trust_env=True,
            transport=transport,
            follow_redirects=True,
        )
        self.sleep, self.clock = sleep, clock
        self.next_at = {}
        self.bytes = 0
        self.max_bytes = max_bytes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def get(self, url, *, params=None, missing_ok=False):
        path = urlparse(url).path
        group = (
            "funding"
            if path.endswith(("fundingRate", "fundingInfo"))
            else "cms"
            if "/cms/" in path
            else "rest"
            if "binance.com" in urlparse(url).netloc
            else "archive"
        )
        delay = {"funding": 0.65, "cms": 1.0, "rest": 0.25, "archive": 0.1}[group]
        for attempt in range(4):
            self.sleep(max(0, self.next_at.get(group, 0) - self.clock()))
            self.next_at[group] = self.clock() + delay
            try:
                with self.client.stream("GET", url, params=params) as response:
                    if missing_ok and response.status_code == 404:
                        return None
                    if (
                        response.status_code in (418, 429)
                        or response.status_code >= 500
                    ):
                        if attempt == 3:
                            response.raise_for_status()
                        pause = float(response.headers.get("Retry-After", 2**attempt))
                        if pause > 60:
                            raise RuntimeError(
                                "服务端要求等待超过 60 秒；停止本次同步，稍后续跑"
                            )
                        self.sleep(pause)
                        continue
                    response.raise_for_status()
                    chunks = []
                    for chunk in response.iter_bytes():
                        self.bytes += len(chunk)
                        if self.bytes > self.max_bytes:
                            raise RuntimeError(
                                "本次下载超过 max_bytes，停止以保留小规模采集限制"
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 3:
                    raise
                self.sleep(2**attempt)
        raise RuntimeError("HTTP 重试耗尽")

    def json(self, url, **kwargs):
        import json

        payload = self.get(url, **kwargs)
        if payload is None:
            return None
        value = json.loads(payload)
        if (
            isinstance(value, dict)
            and isinstance(value.get("code"), int)
            and value["code"] < 0
        ):
            raise ValueError(f"币安接口错误 {value['code']}")
        return value

    def listing(self, prefix, *, delimiter=None):
        """S3 ListObjects v1；无 NextMarker 时以最后 Key 续页，拒绝不前进。"""
        marker = None
        while True:
            params = dict(
                prefix=prefix, **({"delimiter": delimiter} if delimiter else {})
            )
            if marker:
                params["marker"] = marker
            root = ET.fromstring(self.get(S3, params=params))
            # 去掉 namespace，兼容实际 S3 与离线夹具。
            for node in root.iter():
                node.tag = node.tag.rsplit("}", 1)[-1]
            keys = [n.text for n in root.findall("Contents/Key")]
            prefixes = [n.text for n in root.findall("CommonPrefixes/Prefix")]
            yield from (prefixes if delimiter else keys)
            if root.findtext("IsTruncated", "false").lower() != "true":
                break
            next_marker = root.findtext("NextMarker") or (
                max([*keys, *prefixes]) if keys or prefixes else None
            )
            if not next_marker or (marker is not None and next_marker <= marker):
                raise ValueError("S3 分页没有前进")
            marker = next_marker

    def archive(self, store, key, *, refresh=False):
        if not refresh and store.complete(key):
            return store.get(key)
        try:
            checksum = self.get(VISION + key + ".CHECKSUM", missing_ok=True)
            if checksum is None:
                return None
            parts = checksum.decode().strip().split()
            if len(parts) != 2 or parts[1].lstrip("*") != key.rsplit("/", 1)[-1]:
                raise ValueError("CHECKSUM 文件名不匹配")
            payload = self.get(VISION + key)
            if hashlib.sha256(payload).hexdigest() != parts[0]:
                raise ValueError(f"CHECKSUM 不匹配: {key}")
            store.put(
                key,
                payload,
                source="archive",
                url=VISION + key,
                checksum=parts[0],
                checksum_text=checksum.decode(),
            )
            return payload
        except Exception as exc:
            previous = store.manifest.read().get(key, {})
            store.manifest.put(
                key,
                {
                    "error": type(exc).__name__,
                    **(
                        {}
                        if previous.get("status") == "complete"
                        else {"status": "failed"}
                    ),
                },
            )
            raise

    def funding(self, symbol, start, end):
        out, cursor = [], start
        while cursor <= end:
            page = self.json(
                FAPI + "/fapi/v1/fundingRate",
                params=dict(symbol=symbol, startTime=cursor, endTime=end, limit=1000),
            )
            if not page:
                break
            times = [int(r["fundingTime"]) for r in page]
            if min(times) < cursor or max(times) > end or times != sorted(set(times)):
                raise ValueError("资金费 REST 分页范围/顺序错误")
            out.extend(page)
            cursor = times[-1] + 1
            if len(page) < 1000:
                break
        return out

    def klines(self, market, dataset, symbol, start, end):
        host = SPOT if market == "spot" else FAPI
        path = "/api/v3/klines" if market == "spot" else "/fapi/v1/" + dataset
        key = "pair" if dataset == "indexPriceKlines" else "symbol"
        out, cursor = [], start
        while cursor < end:
            page = self.json(
                host + path,
                params={
                    key: symbol,
                    "interval": "1h",
                    "startTime": cursor,
                    "endTime": end - 1,
                    "limit": 1000,
                },
            )
            if not page:
                break
            times = [int(r[0]) for r in page]
            if min(times) < cursor or max(times) >= end or times != sorted(set(times)):
                raise ValueError("K 线 REST 分页范围/顺序错误")
            out.extend(page)
            cursor = times[-1] + 3_600_000
        return out
