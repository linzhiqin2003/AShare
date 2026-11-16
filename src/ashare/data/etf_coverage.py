"""ETF 日覆盖的上市区间证据；未知区间不能用来证明空日合法。"""

from datetime import datetime

from .catalog import FUND_DATASETS


def _date(value):
    if not isinstance(value, str) or len(value) != 8:
        return None
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return None
    return value


class EtfCoverage:
    """当前 L/D/P 主表确定证券集合，基金表补退市日期；D 本身不能证明历史缺席。"""

    def __init__(self, basic, funds):
        self.inventory_known = bool(len(basic))
        self.intervals = {}
        fund_rows = {r["ts_code"]: r for r in funds.iter_rows(named=True)}
        rows = {r["ts_code"]: r for r in basic.iter_rows(named=True)}
        # 基金主表已识别的 ETF 也不能被遗漏；未知上市日期保持保守。
        rows.update(
            {
                c: r
                for c, r in fund_rows.items()
                if c not in rows and "ETF" in str(r.get("name", "")).upper()
            }
        )
        for code, row in rows.items():
            fund = fund_rows.get(code, {})
            start = _date(row.get("list_date")) or _date(fund.get("list_date"))
            end = _date(fund.get("delist_date"))
            if start and end and end < start:
                start, end = None, None
            self.intervals[code] = (start, end)

    def eligible(self, day):
        """None 表示集合未知；有集合时含区间未知的证券，避免错误豁免。"""
        if not self.inventory_known:
            return None
        return {
            code
            for code, (start, end) in self.intervals.items()
            if (start is None or start <= day) and (end is None or day <= end)
        }

    def required(self, day):
        if day < FUND_DATASETS["etf_share_size"].coverage_start:
            return False
        return self.eligible(day) != set()


def from_store(store):
    """只读已完成主表；不存在或不可读时无法证明空日合法。"""
    import polars as pl

    def read(name):
        try:
            return store.read(name)
        except (FileNotFoundError, OSError, pl.exceptions.PolarsError):
            return pl.DataFrame()

    return EtfCoverage(read("etf_basic"), read("fund_basic"))
