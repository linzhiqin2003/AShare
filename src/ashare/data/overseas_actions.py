"""港股、美股的公司行为：由源的前复权因子推出逐日事件，再用交易所前收或价格印证。

源（Tushare）两地都没有分红送转、拆并股的事件明细，只有前复权累计因子 F（最新日为 1）。相邻两个
有因子的日子之比 r=F[t]/F[t−1] 是这一天的总调整比例（每股变成 r 股的等价量），分不出现金与股数。

规则（``RULE`` 是给人看的同一段话）：

1. **噪声**：|r−1| ≤ 1e-5 不算事件（源因子只留 6 位小数）。
2. **事件窗口**：一个事件落在「上一个成交日 a（成交量 > 0）之后、下一个成交日 b 及之前」，同一窗口
   里的事件一起印证（停牌期内先合股后派息算一个窗口）。窗口合计比值 R=Π r。
3. **港股先看交易所前收**：b 的前收 pre_close 与 a 的收盘价之比和 1/R 相对差 ≤ 0.5%（大比值窗口 1%；再放宽半个价位的
   取整误差）就算印证，不再看 b 的开盘、收盘（合股、大比例供股时价格常被同时打压，价格印证不了）。前收
   缺失或对不上时退回第 4 条的价格印证：缺口窗口里交易所的前收是 b 前一天的按盘价（没有成交也会变），
   不是 a 的收盘；少数相邻窗口源的前收没有除权。实测只认前收会丢掉两百多个价格印证过的真实事件。
4. **价格印证**（美股；港股前收缺失时）：a 的收盘价 p0，b 的开盘价或收盘价 p1（两个都试，任一个通过
   就算；小盘股的开盘价常常沿用旧价）。持有者价值倍数 h=R·p1/p0（照引擎记账：股数 ×R、价格变成 p1）。
   max(R,1/R) ≥ 1.25 的窗口要求 |ln h| ≤ min(½|ln R|, cap)：价格跳变要比「什么都没发生」更接近因子说的
   比例，cap 在 b−a ≤ 5 个交易日时为 ln 2、更长为 ln 3。更小的窗口价格分不出事件与日常波动，只要求
   |ln h| ≤ ln 1.5（不明显矛盾）。只靠 b 的收盘价才印证的，``action_check`` 记 3（日内前视）。
5. **不执行的事件**：印证不了的窗口（``action_source=6``）、最后一个成交日之后的因子变化（并购、退市
   日因子跳到 1，``action_source=7``）一律不生成拆并股或现金，``split=1``、``equivalent_cash=0``，
   ``action_suspect`` 为真。
6. **执行的事件**按 r 归类：max(r,1/r) ≥ 1.05 且是整数比（比值 < 3 时分母 ≤ 10、3–10 倍时分母 ≤ 2，相对差 ≤ 2e-4；
   比值 ≥ 10 时只认整数 n:1 / 1:n、绝对差 ≤ 2e-3；合股另放宽到因子比绝对差 6e-5，港股源只留 4 位小数）→ 拆并股（1，split=p/q）；
   其余 r>1 且 r < 1.10 → 分不出、小比值：港股按等效现金（2，equivalent_cash=除权前参考价×(1−1/r)），
   美股不执行（5，美股面板是价格收益口径，见文档）；其余 r ≥ 1.10 → 分不出、大比值，按股数调整
   （4，split=r，无现金）；r<1 且非整数比 → 3（split=r）。
7. **美股小窗口里的非整数股数调整**：max(R,1/R) < 1.25 的窗口里本该记 3 或 4 的事件（多是特别股息、
   源把现金分红记成下调），只有 b 的开盘价印证 |ln h| ≤ ½|ln R| 时才执行，否则与 5 一样不执行。
8. **除权前参考价**：a 的收盘价，同一窗口里前面每个事件之后再除以它的 r（合股后派息用合股后的价位）。
9. **adj_factor**：只累乘执行了的事件（1、3、4 乘 split，2 乘 r），每列第一个有因子的日子为 1。
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
PRE_CLOSE_TOL = 5e-3
# 大比值窗口（max(R,1/R) ≥ 1.25）放到 1%：前收与价格隐含的比例对得上就是真实除权，开发库上 0.5%–2% 的
# 大窗口除 08121.HK 2024-08-20 外都已由价位放宽或价格印证执行，放宽只多执行这一件。
PRE_CLOSE_TOL_BIG = 1e-2
# 港交所价位表（2025-08-04 前较粗的一档）：前收按价位取整，只用来放宽前收印证的取整误差。
HK_TICKS = (
    (0.25, 0.001),
    (0.5, 0.005),
    (10, 0.01),
    (20, 0.02),
    (100, 0.05),
    (200, 0.1),
    (500, 0.2),
    (1000, 0.5),
    (2000, 1.0),
    (5000, 2.0),
    (math.inf, 5.0),
)

SOURCES = {
    0: "无事件",
    1: "拆并股",
    2: "分不出、小比值：等效现金（港股）",
    3: "因子下调且非整数比：按股数",
    4: "分不出、大比值：按股数（不扣税）",
    5: "美股不执行：现金类调整，或小窗口里价格没印证的非整数股数调整（价格收益口径）",
    6: "可疑：前收或价格没有印证，不执行",
    7: "可疑：最后成交日之后的因子变化，不执行",
}
EXECUTED = (1, 2, 3, 4)
SUSPECT = (6, 7)
# action_check：这一格所在窗口是怎样印证的（每个事件格一个值，无事件为 0）。
CHECKS = {
    0: "无事件",
    1: "交易所前收印证（港股）",
    2: "开盘价印证",
    3: "只靠当日收盘价印证（日内前视：开盘撮合时还不知道）",
    4: "没有上一个成交日的小比值窗口，不做印证",
    5: "没有印证（不执行，action_source=6）",
    6: "最后成交日之后（不执行，action_source=7）",
}

RULE = (
    "由前复权因子相邻两日之比 r 推出；|r−1|≤1e-5 不算事件。同一停牌窗口（上一个成交日之后到下一个成交日）"
    "里的事件合起来印证：港股先看交易所前收，前收/上一收盘与 1/R 相对差 ≤0.5%（大比值窗口 1%，加半个价位的取整误差）即印证，"
    "前收缺失或对不上时再看价格；美股看价格：合计比值 R 达到 1.25 倍时要求持有者价值倍数 R·开盘价（或收盘价）/上一收盘价 "
    "比「没有事件」更接近 1（且不超过 2 倍，窗口长于 5 个交易日时 3 倍），更小的窗口不超过 1.5 倍。印证不了的"
    "（action_source=6）和最后一个成交日之后的（7，并购、退市）不执行，action_suspect 为真；action_check 记印证方式。"
    "执行的：整数比拆并股 1（3 倍以上分母 ≤2，10 倍以上只认整数）；其余上调 r<1.10 港股为等效现金 2"
    "（除权前参考价×(1−1/r)）、美股不执行 5；r≥1.10 按股数 4（split=r）；非整数下调 3（split=r）。美股 R<1.25 的窗口里"
    "非整数的 3、4 还要开盘价走到隐含调整的一半以上（|ln h|≤½|ln R|），否则记 5 不执行。"
)


def hk_tick(price):
    for upper, tick in HK_TICKS:
        if price <= upper:
            return tick
    return HK_TICKS[-1][1]


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


def _limit(R, gap):
    lr = abs(math.log(R))
    if lr < math.log(CONFIRM_FROM):
        return LOOSE
    return min(0.5 * lr, CAP_NEAR if gap <= NEAR_DAYS else CAP_FAR)


def _price_check(R, opening, closing, limit):
    """开盘价、收盘价对除权前收盘的比 → 2（开盘印证）、3（只有收盘印证）或 None。

    开盘价在小盘股上常常失真（集合竞价没有成交、沿用旧价），收盘价则可能当天大涨大跌，两个都试；
    开盘先试，只靠收盘才通过的单独记出来（用到了 b 当天收盘才知道的信息）。
    """
    for code, move in ((2, opening), (3, closing)):
        h = R * move
        if np.isfinite(h) and h > 0 and abs(math.log(h)) <= limit:
            return code
    return None


def pre_close_confirms(R, previous_close, reference, coarse=False):
    """港股：交易所前收是否印证了合计比值 R（前收 ≈ 上一成交日收盘 / R）。

    相对差 ≤ 0.5%（大比值窗口 1%）；前收按价位取整（源日线四舍五入期内只到 0.01），再放宽前收半个价位折回除权前的误差。
    """
    if not (np.isfinite(previous_close) and previous_close > 0 and reference > 0):
        return False
    tick = hk_tick(previous_close)
    if coarse:
        tick = max(tick, 0.01)
    tol = PRE_CLOSE_TOL_BIG if max(R, 1 / R) >= CONFIRM_FROM else PRE_CLOSE_TOL
    allowed = max(tol * reference, 0.5 * tick * R) + 1e-9
    return abs(previous_close * R - reference) <= allowed


def corporate_actions(
    market, ratio, close, open_, tradable, previous_close=None, coarse=None
):
    """逐列推出公司行为字段。

    ``ratio``：T×N，有因子的格为当天的因子比（每列第一个因子日、换序列的拼接处为 1），无因子为 NaN。
    ``close``、``open_``：价格（NaN 为缺）；``tradable``：当天有成交（成交量 > 0 且有收盘价）。
    ``previous_close``（港股）：源的前收，港股先用它印证；``coarse``：长度 T 的布尔，前收只到 0.01 的日子。
    返回 dict：adj_factor、action_ratio、split、equivalent_cash、action_source、action_suspect、
    action_check，以及 ``multiplier``（执行了的持有者倍数，供拆段与校验参考，不落盘）。
    """
    T, N = ratio.shape
    adj = np.full((T, N), np.nan)
    split = np.ones((T, N))
    cash = np.zeros((T, N))
    source = np.zeros((T, N), dtype=np.uint8)
    check = np.zeros((T, N), dtype=np.uint8)
    mult = np.ones((T, N))
    use_pre = market == "hk" and previous_close is not None
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
                check[rows, j] = 6
                continue
            small = max(R, 1 / R) < CONFIRM_FROM
            opening = close_move = np.nan
            if a is None:
                p1 = close[b, j] if np.isfinite(close[b, j]) else open_[b, j]
                ref = p1 * R
                how = 4 if small else None
            else:
                ref = close[a, j]
                opening, close_move = open_[b, j] / ref, close[b, j] / ref
                pc = previous_close[b, j] if use_pre else np.nan
                rough = coarse is not None and bool(coarse[a] or coarse[b])
                if use_pre and pre_close_confirms(R, pc, ref, rough):
                    how = 1
                else:
                    how = _price_check(R, opening, close_move, _limit(R, b - a))
            if how is None:
                source[rows, j] = 6
                check[rows, j] = 5
                continue
            check[rows, j] = how
            # 美股小窗口里的非整数股数调整：开盘价要走到隐含调整的一半以上才执行（只认开盘：引擎在开盘
            # 执行，也免得用当天收盘才知道的信息）。
            tight = (
                _price_check(R, opening, np.nan, 0.5 * abs(math.log(R)))
                if market == "us" and small
                else None
            )
            for t in rows:
                r = float(column[t])
                kind, value = classify_ratio(r)
                shares = kind == 3 or (kind == 2 and r >= SMALL)
                if kind == 1:
                    code, s = 1, value
                elif shares and market == "us" and small and tight is None:
                    code, s = 5, 1.0
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
                if shares and code != 5 and market == "us" and small:
                    check[t, j] = tight
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
        action_check=check,
        multiplier=mult,
    )


def segments(
    close,
    tradable,
    multiplier,
    suspect,
    gap_days=60,
    *,
    volume=None,
    amount=None,
    quiet=None,
    found=None,
):
    """美股：价格不连续又没有被印证的事件解释时，把一列拆成两段（第二段起另起一列）。

    看相邻两次成交 a、b：缺口里执行了的事件合计倍数 M，持有者价值倍数 h = M·close[b]/close[a]
    （照引擎记账）。h 在 [1/3, 3] 之外，且「缺口长于 ``gap_days`` 个交易日」（代码复用、长停牌后复牌）
    或「缺口里有没执行的可疑事件」（源因子与价格对不上，如漏记的合股、分位以下价格的合股）时在 b 处
    拆开，不让引擎把这次跳变当成收益。

    ``quiet=(倍数, 成交量倍数, 成交额倍数, 基准天数)``：另一种情形，**疑似源因子漏记的合股**。a 到 b
    之间没有执行了的事件、价格上跳超过「倍数」、b 当天成交股数不超过前「基准天数」个成交日（含 a）中位数
    的「成交量倍数」、成交额不超过中位数的「成交额倍数」时也在 b 处拆开：真实的拉升都伴随成交额放大
    几十到几百倍，合股则是价格 ×N、股数 ÷N、成交额大致不变。只用到 b 当天及以前的数据。``found`` 给出
    时把这种拆开的 (列, b) 追加进去。

    返回 [(列, 起始行, 结束行)]，同一列按时间排。
    """
    T, N = close.shape
    out = []
    for j in range(N):
        trades = np.flatnonzero(tradable[:, j])
        start = 0
        for k in range(1, len(trades)):
            a, b = int(trades[k - 1]), int(trades[k])
            window = multiplier[a + 1 : b + 1, j]
            h = float(np.prod(window)) * close[b, j] / close[a, j]
            if b - a - 1 > gap_days or suspect[a + 1 : b + 1, j].any():
                cut = not (1 / 3 <= h <= 3)
            else:
                cut = quiet is not None and _quiet_jump(
                    h, window, volume, amount, trades[:k], b, j, quiet
                )
                if cut and found is not None:
                    found.append((j, b))
            if cut:
                out.append((j, start, b))
                start = b
        out.append((j, start, T))
    return out


def _quiet_jump(h, window, volume, amount, before, b, j, quiet):
    """没有执行了的事件、价格上跳超过门槛、成交量和成交额都没有放大（疑似漏记的合股）。"""
    jump, max_volume, max_amount, days = quiet
    if h <= jump or (window != 1).any():
        return False
    prev = before[-days:]
    base_volume = float(np.median(volume[prev, j]))
    base_amount = float(np.median(amount[prev, j]))
    return bool(
        volume[b, j] <= max_volume * base_volume
        and amount[b, j] <= max_amount * base_amount
    )


def cut_recovery(close, tradable, multiplier, j, b):
    """拆段处旧列的清算：在复牌日 b 退市，清算价取 min(停牌前收盘, 复牌收盘 × 缺口里执行了的倍数)。

    按对多头偏悲观处理：真实暴跌照实承受，暴涨不给。返回引擎口径的回收率：引擎在 b 日用记账价（旧列
    最后一个收盘价，再按其后执行的拆并股调整）× 回收率清算，结果恰好等于上面的清算价折成引擎股数。
    """
    trades = np.flatnonzero(tradable[:b, j])
    a = int(trades[-1])
    M = float(np.prod(multiplier[a + 1 : b + 1, j]))
    held = float(np.prod(multiplier[a + 1 : b, j]))  # 旧列在 b 之前已经执行了的股数倍数
    price = min(close[a, j], M * close[b, j]) / held
    quotes = np.flatnonzero(np.isfinite(close[:b, j]))
    last = int(quotes[-1])
    mark = close[last, j] / float(np.prod(multiplier[last + 1 : b, j]))
    return price / mark


__all__ = [
    "CHECKS",
    "RULE",
    "SOURCES",
    "classify_ratio",
    "corporate_actions",
    "cut_recovery",
    "pre_close_confirms",
    "segments",
]
