"""研究、回测和模拟盘的数据读取入口。价格量额均为统一单位。"""

from .reader import Data, Panel


def open(root=None, market="cn", **options) -> Data:
    """只读打开市场；已取得数组固定版本，旧代清理后再次读取须重新 open。

    ``market="hk"`` / ``"us"`` 打开港股、美股（``OverseasData``），可选 ``universe="sample"``
    取 2010 年起的样本面板，默认 ``"all"``（按交易日的全部证券）。
    """
    if market in ("hk", "us"):
        from .overseas_reader import OverseasData

        return OverseasData(root, market, **options)
    if options:
        raise TypeError(f"A 股不接受参数 {sorted(options)}")
    return Data(root, market)


def market_root(root=None, market="cn"):
    """数据目录下某个市场的目录（只解析路径，不读文件）。给数据状态页这类要看文件签名的只读用途。"""
    from .store import data_root

    return data_root(root) / market


def read_manifest(root=None, market="cn"):
    """原始库清单（只读，不加写锁）；还没有清单时返回空字典。"""
    from .store import Store, data_root

    base = data_root(root)
    folder = base / market
    if not (
        (folder / "manifest.sqlite3").exists() or (folder / "manifest.json").exists()
    ):
        return {}
    return Store(base, market, readonly=True).manifest()


def index_names():
    """指数代码 → 简称：基准名单的中文简称，股票池指数没有中文简称的用别名（如 csi300）。"""
    from .catalog import BENCHMARKS, INDEXES

    names = dict(BENCHMARKS)
    for alias, code in INDEXES.items():
        names.setdefault(code, alias)
    return names


def field_unit(name):
    """面板字段的单位（字段不在目录里时为空串）。"""
    from .catalog import FIELDS

    spec = FIELDS.get(name)
    return spec.unit if spec else ""


__all__ = [
    "Data",
    "Panel",
    "field_unit",
    "index_names",
    "read_manifest",
    "market_root",
    "open",
]
