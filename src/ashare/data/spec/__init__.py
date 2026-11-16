"""外部数据包 v1：独立校验、原子构建与按需读取。"""

from .build import Adapter, ValidationError, build
from .check import check
from .reader import Package, open
from .seal import seal

__all__ = ["Adapter", "Package", "ValidationError", "build", "check", "open", "seal"]
