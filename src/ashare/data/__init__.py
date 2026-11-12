"""研究、回测和模拟盘的数据读取入口。价格量额均为统一单位。"""

from .reader import Data, Panel


def open(root=None, market="cn") -> Data:
    """只读打开市场；已取得数组固定版本，旧代清理后再次读取须重新 open。"""
    return Data(root, market)


__all__ = ["Data", "Panel", "open"]
