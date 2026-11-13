"""港股、美股的公司行为：由源的前复权因子推出逐日事件，并用价格印证。

源（Tushare）两地都没有分红送转、拆并股的事件明细，只有前复权累计因子 F（最新日为 1）。相邻两个
有因子的日子之比 r=F[t]/F[t−1] 是这一天的总调整比例（每股变成 r 股的等价量），分不出现金与股数。

规则（``RULE`` 是给人看的同一段话）：

1. **噪声**：|r−1| ≤ 1e-5 不算事件（源因子只留 6 位小数）。
2. **事件窗口**：一个事件落在「上一个成交日 a（成交量 > 0）之后、下一个成交日 b 及之前」，同一窗口
   里的事件一起印证（停牌期内先合股后派息算一个窗口）。
3. **价格印证**：窗口合计比值 R=Π r，a 的收盘价 p0，b 的开盘价或收盘价 p1（两个都试，任一个通过就算；
   小盘股的开盘价常常沿用旧价）。持有者价值倍数 h=R·p1/p0（照引擎记账：股数 ×R、价格变成 p1）。max(R,1/R) ≥ 1.25 的窗口要求
   |ln h| ≤ min(½|ln R|, cap)：价格跳变要比「什么都没发生」更接近因子说的比例，cap 在 b−a ≤ 5 个交易日时
   为 ln 2、更长为 ln 3。更小的窗口价格分不出事件与日常波动，只要求 |ln h| ≤ ln 1.5（不明显矛盾）。
4. **不执行的事件**：印证不了的窗口（``action_source=6``）、最后一个成交日之后的因子变化（并购、退市
   日因子跳到 1，``action_source=7``）一律不生成拆并股或现金，``split=1``、``equivalent_cash=0``，
   ``action_suspect`` 为真。
5. **执行的事件**按 r 归类：max(r,1/r) ≥ 1.05 且是整数比（比值 < 3 时分母 ≤ 10、3–10 倍时分母 ≤ 2，相对差 ≤ 2e-4；
   比值 ≥ 10 时只认整数 n:1 / 1:n、绝对差 ≤ 2e-3；合股另放宽到因子比绝对差 6e-5，港股源只留 4 位小数）→ 拆并股（1，split=p/q）；
   其余 r>1 且 r < 1.10 → 分不出、小比值：港股按等效现金（2，equivalent_cash=除权前参考价×(1−1/r)），
   美股不执行（5，美股面板是价格收益口径，见文档）；其余 r ≥ 1.10 → 分不出、大比值，按股数调整
   （4，split=r，无现金）；r<1 且非整数比 → 3（split=r）。
6. **除权前参考价**：a 的收盘价，同一窗口里前面每个事件之后再除以它的 r（合股后派息用合股后的价位）。
7. **adj_factor**：只累乘执行了的事件（1、3、4 乘 split，2 乘 r），每列第一个有因子的日子为 1。
"""

import math

import numpy as np

SPLIT_MIN = 1.05
# 因子只留 6 位小数：相邻两日之比与 1 相差不到 1e-5 的视为舍入噪声，不算事件。
RATIO_NOISE = 1e-5
SPLIT_MAX_DEN = 10
# 比值越大越要收紧：分母放开到 10 时 [10,100] 里随机的 r 有一半以上能凑成 p/q。3 倍以上分母 ≤ 2，
# 10 倍以上只认整数。收紧只影响标签：印证过的非整数大比值照样按股数执行（action_source=4）。
SPLIT_INTEGER_FROM = 10
SPLIT_INTEGER_ABS_TOL = 2e-3
SPLIT_MAX = 1000
SPLIT_REL_TOL = 2e-4
SPLIT_ABS_TOL = 6e-5
SMALL = 1.10
CONFIRM_FROM = 1.25
LOOSE = math.log(1.5)
CAP_NEAR, CAP_FAR, NEAR_DAYS = math.log(2.0), math.log(3.0), 5

SOURCES = {
    0: "无事件",
    1: "拆并股",
    2: "分不出、小比值：等效现金（港股）",
    3: "因子下调且非整数比：按股数",
    4: "分不出、大比值：按股数（不扣税）",
    5: "美股现金类调整：不执行（价格收益口径）",
    6: "可疑：价格没有印证，不执行",
    7: "可疑：最后成交日之后的因子变化，不执行",
}
EXECUTED = (1, 2, 3, 4)
SUSPECT = (6, 7)

RULE = (
    "由前复权因子相邻两日之比 r 推出；|r−1|≤1e-5 不算事件。同一停牌窗口（上一个成交日之后到下一个成交日）"
    "里的事件合起来用价格印证：合计比值 R 达到 1.25 倍时要求持有者价值倍数 R·开盘价/上一收盘价 比「没有事件」"
    "更接近 1（且不超过 2 倍，窗口长于 5 个交易日时 3 倍）；印证不了的（action_source=6）和最后一个成交日之后的"
    "（7，并购、退市）不执行，action_suspect 为真。执行的：整数比拆并股 1（3 倍以上分母 ≤2，10 倍以上只认整数）；其余上调 r<1.10 "
    "港股为等效现金 2（除权前参考价×(1−1/r)）、美股不执行 5；r≥1.10 按股数 4（split=r）；非整数下调 3（split=r）。"
)


def classify_ratio(r, previous=None, current=None):
    """因子比 → (类别, 值)。类别 0 噪声、1 整数比拆并股（值为 p/q）、2 其余上调、3 其余下调。

    ``previous``、``current`` 是两日的因子值，用来放宽源 6 位小数舍入带来的误差。这里只看比值本身；
    是否执行由 ``corporate_actions`` 结合价格决定。
    """
    if not np.isfinite(r) or r <= 0:
        return 0, 1.0
    if abs(r - 1.0) <= RATIO_NOISE:
        return 0, 1.0
    tolerance = SPLIT_REL_TOL
    if previous is not None and current is not None:
        smallest = min(abs(previous), abs(current))
        if smallest > 0:
            tolerance = max(tolerance, 1e-6 / smallest)
    big = max(r, 1 / r)
    if r < 1:
        # 港股合股的因子比在源里只保留 4 位小数（1/12 写成 0.0833），按绝对误差放宽。
        tolerance = max(tolerance, SPLIT_ABS_TOL / r)
    if SPLIT_MIN <= big <= SPLIT_MAX:
        if big >= SPLIT_INTEGER_FROM:
            # 大比值只认整数，且按绝对差（相对差会随比值放大：100 倍时 2e-4 就是 0.02）。
            p, q = round(big), 1
            allowed = SPLIT_INTEGER_ABS_TOL
            if r < 1:
                allowed = max(allowed, big * tolerance)
            ok = abs(p - big) <= allowed
        else:
            p, q = _fraction(big, SPLIT_MAX_DEN if big < 3 else 2)
            ok = abs(p / q - big) / big <= tolerance
        if p > q and ok:
            return 1, (p / q if r > 1 else q / p)
    return (2, 1.0) if r > 1 else (3, float(r))


def _fraction(x, max_den):
    """分母不超过 ``max_den`` 的最近分数（逐个分母取最近的分子）。"""
    best = None
    for q in range(1, max_den + 1):
        p = round(x * q)
        if p <= 0:
            continue
        err = abs(p / q - x)
        if best is None or err < best[0] - 1e-15:
            best = (err, p, q)
    _, p, q = best
    g = math.gcd(p, q)
    return p // g, q // g


def _confirmed(R, moves, gap):
    """``moves`` 是除权后价格对除权前收盘的比（开盘、收盘各一个）：任一个能印证就算印证。

    开盘价在小盘股上常常失真（集合竞价没有成交、沿用旧价），收盘价则可能当天大涨大跌，两个都试。
    """
    lr = abs(math.log(R))
    if lr < math.log(CONFIRM_FROM):
        limit = LOOSE
    else:
        limit = min(0.5 * lr, CAP_NEAR if gap <= NEAR_DAYS else CAP_FAR)
    for move in moves:
        h = R * move
        if np.isfinite(h) and h > 0 and abs(math.log(h)) <= limit:
            return True
    return False


def corporate_actions(market, ratio, close, open_, tradable):
    """逐列推出公司行为字段。

    ``ratio``：T×N，有因子的格为当天的因子比（每列第一个因子日、换序列的拼接处为 1），无因子为 NaN。
    ``close``、``open_``：价格（NaN 为缺）；``tradable``：当天有成交（成交量 > 0 且有收盘价）。
    返回 dict：adj_factor、action_ratio、split、equivalent_cash、action_source、action_suspect，以及
    ``multiplier``（执行了的持有者倍数，供拆段与校验参考，不落盘）。
    """
    T, N = ratio.shape
    adj = np.full((T, N), np.nan)
    split = np.ones((T, N))
    cash = np.zeros((T, N))
    source = np.zeros((T, N), dtype=np.uint8)
    mult = np.ones((T, N))
    for j in range(N):
        column = ratio[:, j]
        valid = np.flatnonzero(np.isfinite(column))
        if not len(valid):
            continue
        events = [t for t in valid if classify_ratio(column[t])[0] != 0]
        trades = np.flatnonzero(tradable[:, j])
        windows = {}
        for t in events:
            k = int(np.searchsorted(trades, t))  # trades[k] 是 ≥ t 的第一个成交日
            a = int(trades[k - 1]) if k > 0 else None
            b = int(trades[k]) if k < len(trades) else None
            windows.setdefault((a, b), []).append(t)
        for (a, b), rows in windows.items():
            R = float(np.prod(column[rows]))
            if b is None:
                source[rows, j] = 7
                continue
            prices = [p for p in (open_[b, j], close[b, j]) if np.isfinite(p) and p > 0]
            if a is None:
                ok = max(R, 1 / R) < CONFIRM_FROM
                ref = prices[-1] * R
            else:
                ref = close[a, j]
                ok = _confirmed(R, [p / ref for p in prices], b - a)
            if not ok:
                source[rows, j] = 6
                continue
            for t in rows:
                r = float(column[t])
                kind, value = classify_ratio(r)
                if kind == 1:
                    code, s = 1, value
                elif kind == 3:
                    code, s = 3, r
                elif r >= SMALL:
                    code, s = 4, r
                elif market == "hk":
                    code, s = 2, 1.0
                    cash[t, j] = ref * (1 - 1 / r)
                    mult[t, j] = r
                else:
                    code, s = 5, 1.0
                source[t, j] = code
                split[t, j] = s
                if code in (1, 3, 4):
                    mult[t, j] = s
                ref /= r
        first = valid[0]
        adj[first:, j] = np.cumprod(mult[first:, j])
        adj[first:, j] /= adj[first, j]
    action_ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    return dict(
        adj_factor=adj,
        action_ratio=action_ratio,
        split=split,
        equivalent_cash=cash,
        action_source=source,
        action_suspect=np.isin(source, SUSPECT),
        multiplier=mult,
    )


def segments(close, tradable, multiplier, suspect, gap_days=60):
    """美股：价格不连续又没有被印证的事件解释时，把一列拆成两段（第二段起另起一列）。

    看相邻两次成交 a、b：缺口里执行了的事件合计倍数 M，持有者价值倍数 h = M·close[b]/close[a]
    （照引擎记账）。h 在 [1/3, 3] 之外，且「缺口长于 ``gap_days`` 个交易日」（代码复用、长停牌后复牌）
    或「缺口里有没执行的可疑事件」（源因子与价格对不上，如漏记的合股、分位以下价格的合股）时在 b 处
    拆开，不让引擎把这次跳变当成收益。返回 [(列, 起始行, 结束行)]，同一列按时间排。
    """
    T, N = close.shape
    out = []
    for j in range(N):
        trades = np.flatnonzero(tradable[:, j])
        start = 0
        for a, b in zip(trades[:-1], trades[1:], strict=True):
            long_gap = b - a - 1 > gap_days
            if not (long_gap or suspect[a + 1 : b + 1, j].any()):
                continue
            h = float(np.prod(multiplier[a + 1 : b + 1, j])) * close[b, j] / close[a, j]
            if not (1 / 3 <= h <= 3):
                out.append((j, start, int(b)))
                start = int(b)
        out.append((j, start, T))
    return out


__all__ = ["RULE", "SOURCES", "classify_ratio", "corporate_actions", "segments"]
