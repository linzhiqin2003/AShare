"""交割价接口的时间标签与实际交割时刻分开保存。"""

from .timeutil import DAY, HOUR


def delivery_record(records, effective_at):
    matches = [r for r in records if int(r["deliveryTime"]) == effective_at]
    if not matches and effective_at % DAY == 8 * HOUR:
        # 实测 delivery-price 的 deliveryTime 为 UTC 日期标签 00:00。
        # 只有显式事件为同日 08:00 时允许匹配，不能凭标签生成交割时刻。
        matches = [
            r for r in records if int(r["deliveryTime"]) == effective_at - 8 * HOUR
        ]
    if len(matches) != 1:
        raise ValueError("季度交割缺官方交割价或记录不唯一，不能用标记收盘替代")
    return matches[0]
