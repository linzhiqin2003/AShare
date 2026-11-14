"""模拟盘页的只读接口。

账户目录由 ``python -m ashare.paper start --directory <账户根>/<名字>`` 建立，账户根默认是
``$ASHARE_DATA_DIR/paper``，可用 ``ASHARE_PAPER_DIR`` 指定。这里只读：日志以 ``read_only``
方式重放（不加锁、不截尾、不写文件），不启动或停止运行器，不连接任何券商。

金额在内核里是万分之一元的整数；接口统一换算成元，价格保留 4 位、金额保留 2 位。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response

LOG = logging.getLogger("ashare.platform.paper")
SCALE = 10_000
NAME = re.compile(r"\w[\w.-]{0,63}")
# 运行器每次处理行情都会在 status_interval（默认 30 秒）后刷新 status.json。
RUNNING_WITHIN = 120
BENCHMARK = "000300.SH"


def paper_root() -> Path:
    value = os.environ.get("ASHARE_PAPER_DIR")
    if value:
        return Path(value)
    return Path(os.environ.get("ASHARE_DATA_DIR", "data")) / "paper"


def _yuan(value, digits=2):
    return round(value / SCALE, digits)


def _day(value) -> str | None:
    text = str(value)
    return f"{text[:4]}-{text[4:6]}-{text[6:]}" if len(text) == 8 else None


def _clock(epoch_ms, fmt="%H:%M:%S"):
    """内核时间是北京时间的 epoch 毫秒；合成测试里的小整数时钟没有日期，返回 None。"""
    if epoch_ms <= 10**11:
        return None
    from ashare.paper.market import TZ

    return datetime.fromtimestamp(epoch_ms / 1000, TZ).strftime(fmt)


def _stat(path: Path):
    try:
        s = path.stat()
    except FileNotFoundError:
        return None
    return s.st_size, s.st_mtime_ns


def navs(points, initial_cash):
    """按资金进出调整的单位净值：r_t = (E_t − ΔF_t) / E_{t−1} − 1。

    points 为按日期升序的 (equity, flows)；flows 是内核累计的入金（含初始资金），
    初始时刻净值为 1、权益等于初始资金。权益不为正时后续净值记为 None。
    """
    result, nav = [], 1.0
    equity, flows = initial_cash, initial_cash
    for e, f in points:
        if nav is not None and equity > 0:
            nav *= (e - (f - flows)) / equity
        else:
            nav = None
        result.append(nav)
        equity, flows = e, f
    return result


def period_return(dates, values, now_day, now_nav, prefix_len):
    """以上一个周期最后一个净值为基；周期前没有记录时以开户（净值 1）为基。"""
    if now_nav is None:
        return None
    period = now_day[:prefix_len]
    base = 1.0
    for d, v in zip(dates, values, strict=True):
        if d[:prefix_len] < period and v is not None:
            base = v
    return now_nav / base - 1 if base else None


def activity(fills, orders, events, day_fees, equity):
    """一个交易日的成交统计；换手按（买入额 + 卖出额）÷ 2 ÷ 当日权益。"""
    bought = sum(f["qty"] * f["price"] for f in fills if f["side"] == "buy")
    sold = sum(f["qty"] * f["price"] for f in fills if f["side"] == "sell")
    rejected: dict[str, int] = {}
    for e in events:
        if e["type"] == "rejected":
            label = REASONS.get(e.get("reason"), "其他")
            rejected[label] = rejected.get(label, 0) + 1
    return dict(
        buys=sum(f["side"] == "buy" for f in fills),
        sells=sum(f["side"] == "sell" for f in fills),
        buy_amount=_yuan(bought),
        sell_amount=_yuan(sold),
        turnover=round((bought + sold) / 2 / equity, 6) if equity > 0 else None,
        fees=_yuan(sum(day_fees.values())),
        open_orders=sum(o["status"] == "open" for o in orders),
        unfilled=sum(o["remaining"] > 0 for o in orders),
        rejected=[[k, v] for k, v in sorted(rejected.items(), key=lambda kv: -kv[1])],
    )


class Account:
    """单个账户目录的读取与缓存；同一指纹只重放一次日志。"""

    def __init__(self, directory: Path):
        self.directory = directory
        self.lock = threading.Lock()
        self.state_key = None
        self.state = None
        self.days: dict[str, tuple] = {}

    def fingerprint(self) -> str:
        d = self.directory
        parts = [_stat(d / "events.jsonl"), _stat(d / "status.json")]
        parts += sorted(
            (p.name, _stat(p)) for p in d.glob("state-*.json") if p.is_file()
        )
        days = d / "days"
        parts.append(_stat(days) if days.is_dir() else None)
        return hashlib.sha256(repr(parts).encode()).hexdigest()[:24]

    def manifest(self) -> dict:
        return json.loads((self.directory / "manifest.json").read_bytes())

    def _replay(self) -> dict:
        from paperbroker import Broker

        from ashare.paper.profiles import from_native
        from ashare.paper.runner import report
        from ashare.paper.storage import ParquetStore

        manifest = self.manifest()
        cash, config = from_native(manifest["config"])
        with Broker(
            manifest["symbols"],
            cash,
            config,
            journal=self.directory,
            quote_store=ParquetStore(self.directory / "quotes"),
            read_only=True,
        ) as broker:
            state = json.loads(broker.dump())
            result = report(broker, broker.history, state)
            result["state"] = dict(
                day=state["day"],
                phase=state["phase"],
                settled_day=state["settled_day"],
                quote_time=max(
                    (q[1] for q in state["quotes"] if q is not None), default=0
                ),
                initial_cash=manifest["config"]["initial_cash"],
            )
            result["symbols"] = list(broker.symbols)
            return result

    def current(self) -> tuple[str, dict]:
        with self.lock:
            key = self.fingerprint()
            if key != self.state_key:
                started = time.perf_counter()
                for attempt in range(3):
                    try:
                        self.state = self._replay()
                        break
                    except FileNotFoundError:
                        # 运行器正在换快照或清理旧行情；稍后重读同一目录。
                        if attempt == 2:
                            raise
                        time.sleep(0.05)
                        key = self.fingerprint()
                self.state_key = key
                LOG.info(
                    "paper_replay account=%s ms=%.1f",
                    self.directory.name,
                    (time.perf_counter() - started) * 1000,
                )
            return self.state_key, self.state

    def archived(self) -> list[tuple]:
        """已结算日：(日期, 权益, 累计入金, 成交列表, 当日交易统计)。

        日报不可变；逐文件按大小与修改时间缓存，只保留页面用到的部分。
        """
        with self.lock:
            found = {}
            for path in sorted((self.directory / "days").glob("*.json")):
                if not re.fullmatch(r"[0-9]{8}", path.stem):
                    continue
                stamp = _stat(path)
                cached = self.days.get(path.stem)
                if cached is None or cached[0] != stamp:
                    summary = json.loads(path.read_bytes())
                    account = summary["account"]
                    fills = summary.get("fills", [])
                    cached = (
                        stamp,
                        (
                            path.stem,
                            account["equity"],
                            account["flows"],
                            fills,
                            activity(
                                fills,
                                summary.get("orders", []),
                                summary.get("events", []),
                                account["day_fees"],
                                account["equity"],
                            ),
                        ),
                    )
                found[path.stem] = cached
            self.days = found
            return [found[k][1] for k in sorted(found)]


class Reference:
    """证券名称、申万一级行业与基准指数；数据目录不可用时降级为空。"""

    def __init__(self, data_root=None):
        self.data_root = data_root
        self.lock = threading.Lock()
        self.loaded_at = float("-inf")
        self.names: dict[str, str] = {}
        self.industries: dict[str, str] = {}
        self.data = None

    def _load(self):
        with self.lock:
            if time.monotonic() - self.loaded_at < 3600:
                return
            self.loaded_at = time.monotonic()
            try:
                import polars as pl

                import ashare.data

                self.data = ashare.data.open(self.data_root)
                self.names = dict(
                    self.data.securities().select("symbol", "name").collect().rows()
                )
                members = (
                    self.data.table("index_member_all")
                    .filter(pl.col("effective_through").is_null())
                    .select("symbol", "industry_name")
                    .collect()
                )
                self.industries = dict(members.rows())
            except Exception:  # 数据目录缺失或损坏：页面只少名称与行业，不失败。
                LOG.warning("paper_reference_unavailable", exc_info=True)
                self.data, self.names, self.industries = None, {}, {}

    def lookup(self, symbol):
        self._load()
        return self.names.get(symbol, ""), self.industries.get(symbol)

    def benchmark(self, start, end):
        self._load()
        if self.data is None:
            return None
        try:
            from ashare.data.catalog import BENCHMARKS

            frame = (
                self.data.table(
                    "index_daily", start=start, end=end, symbols=[BENCHMARK]
                )
                .select("date", "close")
                .collect()
            )
        except Exception:
            LOG.warning("paper_benchmark_unavailable", exc_info=True)
            return None
        closes = {str(d): float(c) for d, c in frame.rows() if c is not None}
        return dict(symbol=BENCHMARK, name=BENCHMARKS.get(BENCHMARK, BENCHMARK)), closes


PHASES = {
    "preopen": "盘前",
    "open_auction": "开盘集合竞价",
    "continuous": "连续竞价",
    "break": "午间休市",
    "close_auction": "收盘集合竞价",
    "closed": "已收盘",
}
REASONS = {
    "session": "时段不接受",
    "unknown_symbol": "证券未登记",
    "no_quote": "无行情",
    "suspended": "停牌",
    "quantity": "数量不符",
    "tick": "价格档位不符",
    "price_band": "超出涨跌停",
    "cash": "资金不足",
    "sellable": "可卖不足",
    "unknown_order": "订单不存在",
    "time": "时间倒序",
    "already_settled": "已结算",
    "invalid_action": "公司行为无效",
}


def _fill(f, symbols, reference, day):
    symbol = symbols[f["symbol"]]
    return dict(
        day=_day(day),
        time=_clock(f["quote_time"]),
        symbol=symbol,
        name=reference.lookup(symbol)[0],
        side=f["side"],
        qty=f["qty"],
        price=_yuan(f["price"], 4),
        fee=_yuan(sum(f["fees"].values())),
    )


def summary(account: Account, reference: Reference, now: float | None = None):
    _, report = account.current()
    state, symbols = report["state"], report["symbols"]
    cash = report["account"]
    equity = cash["equity"]
    day = str(state["day"])
    archived = account.archived()
    settled = day in {d for d, *_ in archived}
    history = [(e, f) for d, e, f, *_ in archived if d != day]
    dates = [d for d, *_ in archived if d != day]
    values = navs(history + [(equity, cash["flows"])], state["initial_cash"])
    nav = values[-1]
    previous_equity, previous_flows = (
        history[-1] if history else (state["initial_cash"], state["initial_cash"])
    )
    day_pnl = equity - previous_equity - (cash["flows"] - previous_flows)

    positions, industries = [], {}
    for p in cash["positions"]:
        if not (p["total"] or p["frozen"]):
            continue
        name, industry = reference.lookup(p["symbol"])
        weight = round(p["market_value"] / equity, 6) if equity > 0 else None
        positions.append(
            dict(
                symbol=p["symbol"],
                name=name,
                industry=industry,
                total=p["total"],
                sellable=p["sellable"],
                cost=_yuan(p["cost_price"], 4),
                mark=_yuan(p["mark"], 4),
                value=_yuan(p["market_value"]),
                pnl=_yuan(p["market_value"] - p["cost"]),
                weight=weight,
            )
        )
        if weight is not None:
            key = industry or "未分类"
            industries[key] = industries.get(key, 0) + weight
    positions.sort(key=lambda p: -p["value"])
    ranked = sorted(industries.items(), key=lambda kv: -kv[1])

    orders = [
        dict(
            id=o["id"],
            symbol=symbols[o["symbol"]],
            name=reference.lookup(symbols[o["symbol"]])[0],
            side=o["side"],
            kind=o["kind"],
            qty=o["qty"],
            filled=o["qty"] - o["remaining"],
            price=_yuan(o["price"], 4),
            status=o["status"],
            time=_clock(o["time"]),
        )
        for o in report["orders"]
    ]
    # 成交面板与交易统计看同一天：今天有成交看今天，否则看最近一个有成交的交易日。
    fills, fills_day = report["fills"], day
    trading = activity(
        fills, report["orders"], report["events"], cash["day_fees"], equity
    )
    risk = dict(open_orders=trading["open_orders"], rejected=trading["rejected"])
    if not fills:
        for d, _, _, archived_fills, stats in reversed(archived):
            if archived_fills and d != day:
                fills, fills_day, trading = archived_fills, d, stats
                break
    status_stamp = _stat(account.directory / "status.json")
    now = time.time() if now is None else now
    running = status_stamp is not None and now - status_stamp[1] / 1e9 < RUNNING_WITHIN
    return dict(
        id=account.directory.name,
        day=_day(day),
        phase=PHASES.get(state["phase"], state["phase"]),
        settled=settled,
        running=running,
        quote_time=_clock(state["quote_time"], "%Y-%m-%d %H:%M:%S"),
        since=_day(archived[0][0]) if archived else _day(day),
        equity=_yuan(equity),
        cash=_yuan(cash["cash"]),
        frozen=_yuan(cash["frozen"]),
        market_value=_yuan(cash["market_value"]),
        invested=_yuan(cash["flows"]),
        day_pnl=_yuan(day_pnl),
        day_return=day_pnl / previous_equity if previous_equity > 0 else None,
        month_return=period_return(dates, values[:-1], day, nav, 6),
        year_return=period_return(dates, values[:-1], day, nav, 4),
        nav=nav,
        positions=positions,
        industries=[[k, round(v, 6)] for k, v in ranked],
        concentration=dict(
            max_weight=positions[0]["weight"] if positions else None,
            max_symbol=positions[0]["name"] or positions[0]["symbol"]
            if positions
            else None,
            max_industry=ranked[0][0] if ranked else None,
            max_industry_weight=ranked[0][1] if ranked else None,
            cash_weight=(cash["cash"] + cash["frozen"]) / equity
            if equity > 0
            else None,
        ),
        trading=trading,
        risk=risk,
        orders=orders,
        fills_day=_day(fills_day),
        fills=[_fill(f, symbols, reference, fills_day) for f in fills],
    )


def curve(account: Account, reference: Reference):
    """已结算日的单位净值与同日基准收盘价；当天盘中点由 summary 的 nav 补上。"""
    _, report = account.current()
    archived = account.archived()
    dates = [d for d, *_ in archived]
    values = navs([(e, f) for _, e, f, *_ in archived], report["state"]["initial_cash"])
    result = dict(
        dates=[_day(d) for d in dates], nav=values, benchmark=None, closes=None
    )
    if dates:
        found = reference.benchmark(
            dates[0], max(dates[-1], str(report["state"]["day"]))
        )
        if found is not None:
            result["benchmark"], closes = found
            result["closes"] = [closes.get(d) for d in dates]
            today = str(report["state"]["day"])
            result["today_close"] = closes.get(today)
    return result


def fill_history(account: Account, reference: Reference, offset: int, limit: int):
    _, report = account.current()
    symbols, day = report["symbols"], str(report["state"]["day"])
    days = [(d, fills) for d, _, _, fills, _ in account.archived() if d != day]
    days.append((day, report["fills"]))
    rows = [(d, f) for d, fills in reversed(days) for f in reversed(fills)]
    return dict(
        total=len(rows),
        items=[
            _fill(f, symbols, reference, d) for d, f in rows[offset : offset + limit]
        ],
    )


class _Failure(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


MISSING = "没有找到这个账户"
UNREADABLE = "账户暂时读取不到，请稍后再试"


def router(data_root=None, root: Path | None = None, reference=None) -> APIRouter:
    """错误只回一句中文（``{"message": …}``），不带路径或异常内容；详情写进日志。"""
    api = APIRouter(prefix="/api/paper")
    reference = reference or Reference(data_root)
    accounts: dict[str, Account] = {}
    guard = threading.Lock()

    def base() -> Path:
        return root if root is not None else paper_root()

    def find(name: str) -> Account:
        if not NAME.fullmatch(name):
            raise _Failure(404, MISSING)
        directory = base() / name
        if not (directory / "manifest.json").is_file():
            raise _Failure(404, MISSING)
        with guard:
            if name not in accounts or accounts[name].directory != directory:
                accounts[name] = Account(directory)
            return accounts[name]

    def respond(fn):
        try:
            return fn()
        except _Failure as failure:
            return JSONResponse({"message": failure.message}, failure.status)
        except Exception:
            LOG.exception("paper_read_failed")
            return JSONResponse({"message": UNREADABLE}, 503)

    def cached(request: Request, name: str, kind: str, build):
        account = find(name)
        tag = '"' + kind + ":" + account.fingerprint() + '"'
        headers = {"ETag": tag, "Cache-Control": "private, no-cache"}
        if request.headers.get("if-none-match") == tag:
            return Response(status_code=304, headers=headers)
        content = json.dumps(
            build(account), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        headers["X-Uncompressed-Bytes"] = str(len(content))
        return Response(content, media_type="application/json", headers=headers)

    @api.get("/accounts")
    def list_accounts():
        directory = base()
        items = []
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if not (
                    NAME.fullmatch(path.name) and (path / "manifest.json").is_file()
                ):
                    continue
                days = sorted(
                    p.stem
                    for p in (path / "days").glob("*.json")
                    if re.fullmatch(r"[0-9]{8}", p.stem)
                )
                stamp = _stat(path / "status.json")
                items.append(
                    dict(
                        id=path.name,
                        since=_day(days[0]) if days else None,
                        running=stamp is not None
                        and time.time() - stamp[1] / 1e9 < RUNNING_WITHIN,
                    )
                )
        return {"items": items}

    @api.get("/accounts/{name}")
    def account_summary(name: str, request: Request):
        return respond(
            lambda: cached(request, name, "s1", lambda a: summary(a, reference))
        )

    @api.get("/accounts/{name}/curve")
    def account_curve(name: str, request: Request):
        return respond(
            lambda: cached(request, name, "c1", lambda a: curve(a, reference))
        )

    @api.get("/accounts/{name}/fills")
    def account_fills(
        name: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
    ):
        return respond(lambda: fill_history(find(name), reference, offset, limit))

    return api
