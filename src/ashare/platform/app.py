"""研究平台 HTTP API：默认只面向本机，可经访问口令开放给指定网卡。

错误响应不含内部异常、SQL 或本地路径。
"""

import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.gzip import GZipMiddleware

from . import datastatus, market
from .access import Access, login_page
from .analysis import plot_payload
from .backtest.api import router as backtest_router
from .benchmarks import Benchmarks
from .compare import router as compare_router
from .config import CHOICES, expand_settings, platform_dir
from .holdings import Holdings
from .language import ExpressionError, completions, parse, suggested_name
from .paper_api import router as paper_router
from .preferences import Preferences
from .preferences import routes as preference_routes
from .runner import Manager, configure_logging
from .storage import Store, dumps

LOG = logging.getLogger("ashare.platform")
STATIC = Path(__file__).parent / "static"


def _same_origin(scope, host):
    origin = dict(scope["headers"]).get(b"origin", b"").decode("latin1")
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return not (
        parsed.scheme != "http"
        or parsed.netloc != host
        or parsed.path
        or parsed.query
        or parsed.fragment
        or sum(k == b"origin" for k, _ in scope["headers"]) != 1
    )


async def _read_body(receive):
    """读完正文并限长；客户端断开返回 None，超长返回 False。"""
    messages, size = [], 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        size += len(message.get("body", b""))
        if size > 32768:
            return False
        messages.append(message)
        if not message.get("more_body", False):
            return messages


def _redirect(location, cookie=None):
    headers = {"Location": location, "Cache-Control": "no-store"}
    if cookie:
        headers["Set-Cookie"] = cookie
    return Response(status_code=303, headers=headers)


class LocalOnly:
    """在读取正文前检查 Host / Origin 和登录状态，流式累计限制正文长度。"""

    def __init__(self, app, access=None):
        self.app = app
        self.access = access or Access()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        access = self.access
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode("latin1")
        if not access.host_allowed(host) or (
            sum(k == b"host" for k, _ in scope["headers"]) != 1
        ):
            return await JSONResponse({"message": "访问地址不受支持"}, 403)(
                scope, receive, send
            )
        client = (scope.get("client") or ("", 0))[0]
        if access.token is not None and scope["path"] == "/login":
            return await self.login(scope, receive, send, host, client)
        if access.needs_login(client, host) and not access.logged_in(
            headers.get(b"cookie", b"").decode("latin1")
        ):
            if scope["path"].startswith("/api/"):
                response = JSONResponse({"message": "请先登录"}, 401)
            else:
                response = _redirect("/login")
            return await response(scope, receive, send)
        if scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            if not _same_origin(scope, host):
                return await JSONResponse({"message": "请求来源不受支持"}, 403)(
                    scope, receive, send
                )
            messages = await _read_body(receive)
            if messages is None:
                return
            if messages is False:
                return await JSONResponse({"message": "请求过长"}, 413)(
                    scope, receive, send
                )
            iterator = iter(messages)

            async def replay():
                try:
                    return next(iterator)
                except StopIteration:
                    return await receive()

            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)

    async def login(self, scope, receive, send, host, client):
        access = self.access
        if scope["method"] in ("GET", "HEAD"):
            code = parse_qs(scope.get("query_string", b"").decode("latin1")).get(
                "e", [""]
            )[0]
            response = Response(
                login_page(code),
                media_type="text/html; charset=utf-8",
                headers={"Cache-Control": "no-store"},
            )
        elif scope["method"] != "POST" or not _same_origin(scope, host):
            response = JSONResponse({"message": "请求来源不受支持"}, 403)
        else:
            messages = await _read_body(receive)
            if messages is None:
                return
            if messages is False:
                response = JSONResponse({"message": "请求过长"}, 413)
            elif access.locked(client):
                response = _redirect("/login?e=2")
            else:
                body = b"".join(m.get("body", b"") for m in messages)
                given = parse_qs(body.decode("utf-8", "replace")).get("token", [""])[0]
                if access.check(client, given):
                    response = _redirect("/", access.cookie())
                else:
                    response = _redirect("/login?e=1")
        return await response(scope, receive, send)


class RenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(max_length=16000)
    settings: dict = Field(default_factory=dict)
    name: str = Field(
        default="alpha", min_length=1, max_length=80, pattern=r"^[^\x00-\x1f]+$"
    )


class TabRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


def create_app(root=None, *, data_root=None, manager_factory=Manager, access=None):
    store = Store(root or platform_dir())
    configure_logging(store.root)
    manager = manager_factory(store, data_root=data_root)
    benchmarks = Benchmarks(data_root)
    holdings = Holdings(store, data_root)
    preferences = Preferences(store.root)
    backtests = backtest_router(
        store.root, data_root, default_benchmark=preferences.benchmark
    )

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()
        store.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store, app.state.manager = store, manager
    app.state.benchmarks = benchmarks
    app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)
    app.add_middleware(LocalOnly, access=access)

    @app.middleware("http")
    async def timing(request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOG.exception("接口失败 method=%s", request.method)
            response = JSONResponse({"message": "操作失败，请稍后重试"}, 500)
        response.headers["Server-Timing"] = (
            f"app;dur={(time.perf_counter() - started) * 1000:.3f}"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/assets/") and response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(ExpressionError)
    async def expression_error(request, error):
        return JSONResponse(error.detail(), 422)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"message": "请求参数不正确"}, 422)

    @app.exception_handler(ValueError)
    async def value_error(request, error):
        # 只在受控 API 校验处主动抛 ValueError；未知内部异常统一走 500。
        return JSONResponse({"message": "请求参数不受支持"}, 422)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse({"message": error.detail}, error.status_code)

    app.include_router(datastatus.routes(data_root, store))
    app.include_router(market.routes(data_root))
    app.include_router(preference_routes(preferences, benchmarks))
    metadata = {"at": 0, "date": None}

    @app.get("/api/meta")
    def meta():
        if time.monotonic() - metadata["at"] > 60:
            try:
                import ashare.data

                dates = ashare.data.open(data_root).panel([]).dates
                day = str(dates[-1])
                metadata["date"] = f"{day[:4]}-{day[4:6]}-{day[6:]}"
            except (FileNotFoundError, ValueError, RuntimeError):
                metadata["date"] = None
            metadata["at"] = time.monotonic()
        return {
            "data_date": metadata["date"],
            "running": manager.running + backtests.jobs.running,
            "max_workers": manager.limit,
        }

    @app.get("/api/settings")
    def settings():
        selected = benchmarks.selection(preferences.benchmark())
        return {
            "defaults": preferences.defaults()
            | ({"benchmark": selected["symbol"]} if selected["symbol"] else {}),
            "choices": CHOICES,
            "benchmark": selected,
        }

    @app.get("/api/benchmarks")
    def benchmark_list(
        search: str = Query("", max_length=80),
        selected: str | None = Query(None, max_length=32),
    ):
        return benchmarks.search(search, selected)

    @app.get("/api/completions")
    def completion_list():
        return completions(store.best())

    @app.post("/api/validate")
    def validate(body: RunRequest):
        variants = parse(body.source)
        try:
            options = expand_settings(body.settings, len(variants))
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return {
            "count": len(variants) * len(options),
            "name": suggested_name(body.source),
        }

    @app.get("/api/tabs")
    def tabs():
        with manager.lock:
            return [
                {"id": key, "job": value["job"]} for key, value in manager.tabs.items()
            ]

    @app.post("/api/tabs")
    def new_tab(body: TabRequest | None = None):
        try:
            return {"id": manager.create_tab(body.id if body else None)}
        except ValueError as error:
            raise HTTPException(409, str(error)) from None

    @app.post("/api/tabs/{tab}/name")
    def rename_tab(tab: str, body: RenameRequest):
        with manager.lock:
            if tab not in manager.tabs:
                raise HTTPException(404, "研究 tab 已关闭")
            name = body.name.strip()
            if not name:
                raise HTTPException(422, "请输入因子名")
            store.rename_tab(tab, name)
        return {"ok": True}

    @app.delete("/api/tabs/{tab}")
    def close_tab(tab: str):
        manager.close_tab(tab)
        return {"ok": True}

    @app.post("/api/tabs/{tab}/runs", status_code=202)
    def run(tab: str, body: RunRequest):
        try:
            expand_settings(body.settings)
            selected = benchmarks.selection(body.settings.get("benchmark"))
            settings = body.settings | (
                {"benchmark": selected["symbol"]} if selected["symbol"] else {}
            )
            return {"id": manager.submit(tab, body.source, settings, body.name)}
        except ExpressionError:
            raise
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/jobs/{job}")
    def job_result(job: str):
        result = store.job(job)
        if result is None:
            raise HTTPException(404, "没有找到这次运行")
        return result

    @lru_cache(maxsize=64)
    def cached_benchmark(job, completed, symbol):
        return benchmarks.comparison(store, store.job(job)["runs"], symbol)

    @app.get("/api/jobs/{job}/benchmark")
    def job_benchmark(
        job: str, request: Request, symbol: str | None = Query(None, max_length=32)
    ):
        result = store.job(job)
        if result is None:
            raise HTTPException(404, "没有找到这次运行")
        completed = tuple(r["id"] for r in result["runs"] if r["status"] == "complete")
        content = dumps(cached_benchmark(job, completed, symbol)).encode()
        etag = '"' + hashlib.sha256(content).hexdigest() + '"'
        headers = {
            "ETag": etag,
            "Cache-Control": "private, no-cache",
            "X-Uncompressed-Bytes": str(len(content)),
        }
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(content, media_type="application/json", headers=headers)

    @app.post("/api/jobs/{job}/cancel")
    def cancel(job: str):
        return {"cancelled": manager.cancel(job)}

    @app.get("/api/jobs/{job}/events")
    async def events(job: str, request: Request, after: int = Query(0, ge=0)):
        if store.job(job) is None:
            raise HTTPException(404, "没有找到这次运行")
        try:
            after = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            raise HTTPException(422, "事件序号不正确") from None

        async def stream():
            cursor = after
            idle = 0
            while not await request.is_disconnected():
                batch = await asyncio.to_thread(store.events, job, cursor)
                for event in batch:
                    cursor = event["id"]
                    yield f"id: {cursor}\ndata: {dumps(event)}\n\n"
                    if event["kind"] in (
                        "complete",
                        "failed",
                        "cancelled",
                        "interrupted",
                    ):
                        return
                current = await asyncio.to_thread(store.job, job)
                if not batch and current["status"] != "running":
                    return
                idle += 1
                if idle % 40 == 0:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @lru_cache(maxsize=64)
    def cached_series(run, view):
        arrays = store.series(run)
        if arrays is None:
            raise HTTPException(404, "结果尚未就绪")
        return dumps(plot_payload(arrays, view)).encode()

    @app.get("/api/runs/{run}/series")
    def series(
        run: str,
        request: Request,
        view: Literal["returns", "long", "groups", "ic", "subs"] = "returns",
    ):
        etag = '"' + hashlib.sha256(f"v1:{run}:{view}".encode()).hexdigest() + '"'
        headers = {
            "ETag": etag,
            "Cache-Control": "private, max-age=31536000, immutable",
        }
        # 必须先确认运行存在，不能用伪造 ETag 探测任意路径。
        content = cached_series(run, view)
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        headers["X-Uncompressed-Bytes"] = str(len(content))
        return Response(content, media_type="application/json", headers=headers)

    @app.get("/api/runs/{run}/holdings")
    def positions(
        run: str,
        date: str | None = Query(None, pattern=r"^[0-9]{8}$"),
        compare: bool = False,
        search: str = Query("", max_length=100),
    ):
        result = holdings.read(run, date, compare=compare, search=search)
        if result is None:
            raise HTTPException(404, "结果尚未就绪")
        return result

    @app.get("/api/library")
    def library(
        search: str = Query("", max_length=200),
        status: str = "",
        universe: str = "",
        neutralize: str = "",
        category: str = "",
        rebalance: str = "",
        metric: str = "sharpe",
        minimum: float | None = None,
        maximum: float | None = None,
        sharpe_min: float | None = None,
        turnover_max: float | None = None,
        t_min: float | None = None,
        sort: str = "updated",
        descending: bool = True,
        group: str = "none",
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        return store.library(
            search=search,
            status=status,
            universe=universe,
            neutralize=neutralize,
            category=category,
            rebalance=rebalance,
            metric=metric,
            minimum=minimum,
            maximum=maximum,
            sharpe_min=sharpe_min,
            turnover_max=turnover_max,
            t_min=t_min,
            sort=sort,
            descending=descending,
            group=group,
            offset=offset,
            limit=limit,
        )

    @app.get("/api/library/correlation")
    def library_correlation(ids: str = Query("", max_length=800)):
        return store.correlation_matrix(ids.split(",") if ids else [])

    @app.get("/api/library/{entry}")
    def entry_detail(entry: str):
        result = store.entry(entry)
        if result is None:
            raise HTTPException(404, "没有找到这个因子")
        return result

    app.include_router(compare_router)
    app.include_router(paper_router(data_root))
    app.include_router(backtests)
    if STATIC.is_dir():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="web")
    return app
