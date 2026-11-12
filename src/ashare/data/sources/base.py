"""可独立复用的源协议；不依赖项目的存储、目录或读取接口。"""

from dataclasses import dataclass
from typing import Protocol

import polars as pl


@dataclass(frozen=True)
class FetchResult:
    table: pl.DataFrame
    fetched_at: str
    calls: int
    retries: int = 0


class Source(Protocol):
    async def fetch(
        self, api: str, params: dict, *, fields: str = "", page_size: int | None = None
    ) -> FetchResult: ...


class SourceError(RuntimeError):
    """安全的源错误，不附带可能包含鉴权信息的原始请求。"""

    def __init__(self, api: str, category: str, code=None, message=""):
        self.api, self.category, self.code = api, category, code
        super().__init__(f"{api}: {category} (code={code}) {message}")
