"""本包登记的核对「已知例外」（机制见 ``ashare.factor.formula.known_exceptions``）。"""

from ashare.factor import call
from ashare.factor.formula import KnownException, qlib


def _wvma_zero_window(window):
    # Q-Q-06：WVMA 的分子分母都是 |收益|×成交量 的 window 日统计；该量非负，窗口最大值为 0
    # 就是窗口内有效值全为 0（窗口里没有有效值时最大值为 NaN，条件不成立）。
    weighted = qlib.to_expr(f"Max(Abs($close/Ref($close, 1)-1)*$volume, {window})")
    return KnownException(
        ref="Q-Q-06",
        condition=call("eq", weighted, 0),
        candidate=0.0,
        description=(
            "窗口内 |收益|×成交量 全为 0：按公式字面 0/(0+1e-12) = 0；"
            "pandas 滚动标准差的残差除以 1e-12 会给出很大的数，不复现"
        ),
    )


ALPHA158 = {f"alpha158_wvma{n}": (_wvma_zero_window(n),) for n in (5, 10, 20, 30, 60)}
