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


__all__ = ["Data", "Panel", "open"]
