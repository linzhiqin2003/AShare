"""源适配层；不导入其他 ashare 模块。"""

from .base import FetchResult, Source, SourceError
from .tushare import Tushare

__all__ = ["FetchResult", "Source", "SourceError", "Tushare"]
