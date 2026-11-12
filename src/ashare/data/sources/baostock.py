"""BaoStock 串行核对适配器；SDK 仅来自可选 data-verify 依赖组。"""

import asyncio
import contextlib
import io
import threading
from datetime import UTC, datetime

import polars as pl

from .base import FetchResult, SourceError

_LOCK = threading.Lock()


class BaoStock:
    def __init__(self, budget=12000):
        self.budget, self.calls = budget, 0
        self.logged_in = False

    def _fetch(self, api, params):
        import baostock as bs
        from baostock.common import context

        with _LOCK, contextlib.redirect_stdout(io.StringIO()):
            if self.calls >= self.budget:
                raise SourceError(api, "budget")
            if not self.logged_in:
                response = bs.login()
                if response.error_code != "0":
                    raise SourceError(api, "login", response.error_code)
                self.logged_in = True
                context.default_socket.settimeout(30)
            allowed = {
                "daily": bs.query_history_k_data_plus,
                "securities": bs.query_all_stock,
            }
            if api not in allowed:
                raise SourceError(api, "api_name")
            self.calls += 1
            try:
                result = allowed[api](**params)
                if result.error_code != "0":
                    raise SourceError(api, "api_error", result.error_code)
                rows = []
                while result.next():
                    rows.append(result.get_row_data())
                return FetchResult(
                    pl.DataFrame(rows, schema=result.fields, orient="row"),
                    datetime.now(UTC).isoformat(),
                    1,
                )
            except (OSError, TimeoutError):
                self.logged_in = False
                raise SourceError(api, "network") from None

    async def fetch(self, api, params, *, fields="", page_size=None):
        return await asyncio.to_thread(self._fetch, api, params)

    async def close(self):
        if self.logged_in:
            import baostock as bs

            with _LOCK, contextlib.redirect_stdout(io.StringIO()):
                bs.logout()
            self.logged_in = False
