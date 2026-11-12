"""Tushare 期货原始库 → 统一字段（见 futures_schema）。Tushare 的代码、字段名和单位
只在这里出现；单位换算规则都由真实数据核对过，证据见 docs/FUTURES_DATA.md。
"""

import math
import re

import polars as pl

from .futures_schema import conform
from .futures_specs import MANUAL_MULTIPLIERS

# Tushare 合约代码后缀 → 交易所
SUFFIX = {
    "CFX": "CFFEX",
    "SHF": "SHFE",
    "DCE": "DCE",
    "ZCE": "CZCE",
    "INE": "INE",
    "GFE": "GFEX",
}
EXCHANGE_SUFFIX = {v: k for k, v in SUFFIX.items()}

# 成交量、持仓量、成交额的统计口径：在这些交易日之前源数据是双边计数（每笔成交买卖各记一次），
# 统一字段除以 2 换成单边。依据：双边计数时成交量和持仓量恒为偶数，切换后奇数约占一半
# （逐交易所逐年统计见 docs/FUTURES_DATA.md「单边与双边」）。中金所和广期所始终单边。
DOUBLE_SIDED_UNTIL = {
    "SHFE": "20200101",
    "INE": "20200101",
    "DCE": "20200101",
    "CZCE": "20200101",
}

# 大商所、广期所的 trading_fee 一列同时用于「元/手」和「万分之几」，源里没有标记。
# 下列品种按成交金额收费（交易所现行标准；历史上是否换过收费方式未核实），其余按手。
# 不在两个集合里的品种视为未知，费用字段留空。
RATE_BASED = {
    "DCE": {"I", "J", "JM", "JD", "BB", "FB", "LH"},
    "GFEX": {"SI", "LC", "PS"},
}
PER_CONTRACT = {
    "DCE": {
        "A",
        "B",
        "C",
        "CS",
        "EB",
        "EG",
        "L",
        "M",
        "P",
        "PG",
        "PP",
        "RR",
        "V",
        "Y",
        "L_F",
        "PP_F",
        "V_F",
    },
    "GFEX": set(),
}
# 中金所：股指期货按金额（万分之几），国债期货按手（元/手）；trading_fee_rate 与 trading_fee 两列填同一个数。
CFFEX_INDEX = {"IF", "IH", "IC", "IM"}

# 不进统一字段的品种：SCTAS 是原油的 TAS（按结算价交易）指令，源里当成独立合约
# （SCTAS2108.INE），没有开盘价、成交额折算的乘数是原油的两倍，不是可单独持有的合约。
# 原始分区照常保留。
EXCLUDED_PRODUCTS = {"SCTAS"}


def product_of(code):
    match = re.match(r"^([A-Z]+)", code)
    return match.group(1) if match else None


def unify_symbol(ts_code, product=None):
    """``RB2701.SHF`` → ``RB2701.SHFE``；``L2605F.DCE`` → ``L_F2605.DCE``；连续代码返回 None。"""
    base, suffix = ts_code.split(".")
    exchange = SUFFIX[suffix]
    match = re.match(r"^([A-Z]+)(\d{3,4})(F?)$", base)
    if not match:
        return None
    letters, digits, average = match.groups()
    if len(digits) == 3:
        raise ValueError(f"三位年月的合约代码须由合约表展开：{ts_code}")
    return f"{product or (letters + ('_F' if average else ''))}{digits}.{exchange}"


def _symbol_expr(column="ts_code"):
    code = pl.col(column)
    base = code.str.extract(r"^([A-Z]+\d{4}F?)\.")
    letters = base.str.extract(r"^([A-Z]+)")
    digits = base.str.extract(r"(\d{4})")
    average = base.str.ends_with("F")
    exchange = code.str.extract(r"\.(\w+)$").replace_strict(SUFFIX, default=None)
    return (
        pl.when(base.is_not_null())
        .then(
            pl.concat_str(
                [
                    letters,
                    pl.when(average).then(pl.lit("_F")).otherwise(pl.lit("")),
                    digits,
                    pl.lit("."),
                    exchange,
                ]
            )
        )
        .otherwise(None)
    )


def _product_expr(symbol="symbol"):
    return pl.col(symbol).str.extract(r"^([A-Z_]+)\d{4}\.")


def listing(store, raw=None):
    """合约表里每个合约的统一代码、品种、四位年月和上市窗口（不含 EXCLUDED_PRODUCTS）。"""
    raw = raw if raw is not None else store.read("futures_contracts")
    return (
        raw.with_columns(symbol=_symbol_expr())
        .filter(pl.col("symbol").is_not_null())
        .with_columns(
            product=_product_expr(),
            month=pl.col("symbol").str.extract(r"(\d{4})\.\w+$"),
            exchange=pl.col("symbol").str.extract(r"\.(\w+)$"),
        )
        .filter(~pl.col("product").is_in(list(EXCLUDED_PRODUCTS)))
        .select(
            "symbol",
            "exchange",
            "product",
            "month",
            "list_date",
            "delist_date",
        )
        .unique("symbol", keep="first", maintain_order=True)
    )


RESOLUTIONS = ("exact", "czce_decade", "exchange_suffix", "unmatched", "ambiguous")


def resolve(frame, table, *, code="ts_code", date="trade_date"):
    """把源代码对到合约表里「当天在上市期内」的唯一合约，加列 symbol 与 resolution。

    - exact：代码照字面换成统一代码后就是当天上市的合约。
    - czce_decade：郑商所原生代码只有三位年月，Tushare 补年代时会补错（2019 年的
      结算参数里 ``AP1003.ZCE`` 实为 AP2003，``WH1001`` 还与 2010 年的真合约同名）。
      按「品种 + 年份末位 + 月份」找当天上市的合约，唯一时采用。
    - exchange_suffix：交易所后缀写错（2024 年的结算参数里有 ``BC2407.SHF``，BC 在能源中心挂牌）。
      同品种同年月、当天上市且唯一时采用。
    - unmatched / ambiguous：找不到或不唯一，symbol 为空；调用方计数并在校验里逐条列出，不静默丢弃。
    """
    parsed = frame.with_row_index("_row").with_columns(_literal=_symbol_expr(code))
    parsed = parsed.with_columns(
        _product=_product_expr("_literal"),
        _month=pl.col("_literal").str.extract(r"(\d{4})\.\w+$"),
        _exchange=pl.col("_literal").str.extract(r"\.(\w+)$"),
    )
    window = (pl.col(date) >= pl.col("list_date")) & (
        pl.col(date) <= pl.col("delist_date")
    )
    exact = (
        parsed.select("_row", "_literal", date)
        .join(
            table.select("symbol", "list_date", "delist_date"),
            left_on="_literal",
            right_on="symbol",
        )
        .filter(window)
        .select("_row", symbol=pl.col("_literal"), resolution=pl.lit("exact"))
    )
    rest = parsed.join(exact.select("_row"), on="_row", how="anti").filter(
        pl.col("_product").is_not_null()
    )
    candidates = (
        rest.select("_row", "_product", "_month", "_exchange", date)
        .join(
            table.select(
                "symbol",
                "exchange",
                "list_date",
                "delist_date",
                _product=pl.col("product"),
                _year_digit=pl.col("month").str.slice(1, 3),
                _cand_month=pl.col("month"),
            ),
            left_on=["_product", pl.col("_month").str.slice(1, 3)],
            right_on=["_product", "_year_digit"],
        )
        .filter(window)
        .with_columns(
            same_exchange=pl.col("exchange") == pl.col("_exchange"),
            same_month=pl.col("_cand_month") == pl.col("_month"),
        )
        # 同交易所的候选优先；只换年代限郑商所，只换后缀要求四位年月相同。
        .filter(
            (pl.col("same_exchange") & (pl.col("_exchange") == "CZCE"))
            | (~pl.col("same_exchange") & pl.col("same_month"))
        )
    )
    counted = candidates.group_by("_row").agg(
        n=pl.len(),
        symbol=pl.col("symbol").first(),
        same=pl.col("same_exchange").first(),
    )
    fixed = counted.filter(pl.col("n") == 1).select(
        "_row",
        "symbol",
        resolution=pl.when(pl.col("same"))
        .then(pl.lit("czce_decade"))
        .otherwise(pl.lit("exchange_suffix")),
    )
    ambiguous = counted.filter(pl.col("n") > 1).select(
        "_row", symbol=pl.lit(None, pl.String), resolution=pl.lit("ambiguous")
    )
    found = pl.concat([exact, fixed, ambiguous])
    out = parsed.join(found, on="_row", how="left").with_columns(
        resolution=pl.when(pl.col("_literal").is_null())
        .then(pl.lit("continuous"))
        .otherwise(pl.col("resolution").fill_null("unmatched"))
    )
    return out.drop("_row", "_literal", "_product", "_month", "_exchange")


def contracts(store, history=None):
    """统一字段 contracts 表；``history`` 为已算好的 multipliers 表（不给就现算）。"""
    raw = store.read("futures_contracts")
    raw = raw.filter(
        ~pl.col("fut_code").str.to_uppercase().is_in(list(EXCLUDED_PRODUCTS))
    )
    tick = raw["quote_unit_desc"].str.extract(r"^([0-9.]+)").cast(pl.Float64)
    frame = raw.with_columns(
        symbol=_symbol_expr(),
        product=pl.col("fut_code").str.to_uppercase(),
        declared=pl.coalesce("multiplier", "per_unit").cast(pl.Float64)
        * pl.when(
            pl.col("quote_unit").str.contains("500千克")
            & (pl.col("trade_unit") == "吨")
        )
        .then(2.0)
        .otherwise(1.0),
        tick_size=tick,
        quote_unit=pl.col("quote_unit").str.replace("人民币", ""),
        settlement_type=pl.col("d_mode_desc").replace_strict(
            {"实物交割": "physical", "现金交割": "cash"}, default=None
        ),
        currency=pl.lit("CNY"),
        listing_date=pl.col("list_date"),
        last_trade_date=pl.col("delist_date"),
        delivery_date=pl.col("last_ddate"),
        delivery_month=pl.col("d_month"),
    ).filter(pl.col("symbol").is_not_null())
    history = history if history is not None else multipliers(store, raw)
    last = (
        history.sort("symbol", "start_date")
        .group_by("symbol")
        .agg(
            multiplier=pl.col("multiplier").last(),
            multiplier_source=pl.col("source").last(),
        )
    )
    frame = frame.join(last, on="symbol", how="left").with_columns(
        multiplier=pl.coalesce("multiplier_right", "declared"),
        multiplier_source=pl.col("multiplier_source").fill_null("contract_table"),
    )
    return (
        conform(
            "contracts",
            frame.select(
                "symbol",
                "exchange",
                "product",
                "name",
                "listing_date",
                "last_trade_date",
                "delivery_date",
                "delivery_month",
                "multiplier",
                "quote_unit",
                "tick_size",
                "settlement_type",
                "currency",
                "multiplier_source",
            ),
        )
        .unique("symbol", keep="first", maintain_order=True)
        .sort("symbol")
    )


# ---- 合约乘数（按日期） -------------------------------------------------------------
# 合约表的交易单位是现值；历史上改过交易单位的品种，旧合约甚至同一个合约存续期中途的乘数都
# 可能不同（PB1309–1311 在 2013-09-02 由 25 吨改 5 吨，FB2001 在 2019-12 由 500 张改 10 立方米）。
# 来源优先级：
#   1. 成交额反查：每个「合约 × 自然月」取「成交额 ÷ (成交量 × 结算价)」的中位数。与合约表相差
#      不超过 SPEC_NOISE 的月份视为与合约表一致；相差超过 SPEC_CHANGE 倍的月份视为交易单位不同，
#      取反查值（取整）；介于两者之间的是稀疏成交或成交额口径的噪声（见文档），不改乘数。
#      同一合约前后月份的取值不同，就在存续期内分段：分界日取「新值的第一个成交日」，此前
#      保持旧值。新旧值交错（不是一次切换）时不分段，整段用合约表值并标为 contract_table_unverified，
#      由校验列出。
#   2. 人工修正表（futures_specs.MANUAL_MULTIPLIERS）：没有任何可反查行的合约（上期所 2008 年前
#      成交额为空）且命中修正表时采用，每行写明依据。
#   3. 合约表现值：有反查行且一致时为 contract_table；没有可反查行时为 contract_table_unverified。
SPEC_NOISE = 0.02
SPEC_CHANGE = 1.5


def _declared(raw):
    return raw.with_columns(
        symbol=_symbol_expr(),
        declared=pl.coalesce("multiplier", "per_unit").cast(pl.Float64)
        * pl.when(
            pl.col("quote_unit").str.contains("500千克")
            & (pl.col("trade_unit") == "吨")
        )
        .then(2.0)
        .otherwise(1.0),
    ).filter(pl.col("symbol").is_not_null())


def implied_rows(store, raw_contracts=None, raw_daily=None):
    """逐行反查值：只取成交量、成交额、结算价都为正的行（统一代码、单边口径之前的原始量纲）。
    ``raw_daily`` 为已读入的全量原始日线（不给就从原始库扫描）。"""
    raw = (
        raw_contracts if raw_contracts is not None else store.read("futures_contracts")
    )
    listing_dates = raw.select("ts_code", "list_date", "delist_date")
    return (
        (raw_daily.lazy() if raw_daily is not None else store.scan("futures_daily"))
        .select("ts_code", "trade_date", "vol", "amount", "settle")
        .join(listing_dates.lazy(), on="ts_code", how="inner")
        .filter(
            (pl.col("trade_date") >= pl.col("list_date"))
            & (pl.col("trade_date") <= pl.col("delist_date"))
            & (pl.col("vol") > 0)
            & (pl.col("amount") > 0)
            & (pl.col("settle") > 0)
        )
        .with_columns(
            ratio=pl.col("amount") * 10000 / (pl.col("vol") * pl.col("settle")),
            month=pl.col("trade_date").str.slice(0, 6),
        )
        .select("ts_code", "trade_date", "month", "ratio")
        .collect()
    )


def multipliers(store, raw_contracts=None, raw_daily=None):
    """统一字段 multipliers 表：每个合约一段或多段 (start_date, multiplier, source, basis)。"""
    raw = (
        raw_contracts if raw_contracts is not None else store.read("futures_contracts")
    )
    raw = raw.filter(
        ~pl.col("fut_code").str.to_uppercase().is_in(list(EXCLUDED_PRODUCTS))
    )
    declared = _declared(raw).select(
        "ts_code",
        "symbol",
        "declared",
        "list_date",
        "delist_date",
        "exchange",
        "fut_code",
    )
    rows = implied_rows(store, raw, raw_daily).join(
        declared.select("ts_code", "declared"), on="ts_code"
    )
    months = (
        rows.group_by("ts_code", "month")
        .agg(median=pl.col("ratio").median(), first=pl.col("trade_date").min())
        .join(declared.select("ts_code", "declared"), on="ts_code")
        .with_columns(q=pl.col("median") / pl.col("declared"))
        .with_columns(
            value=pl.when((pl.col("q") - 1).abs() <= SPEC_NOISE)
            .then(pl.col("declared"))
            .when((pl.col("q") >= SPEC_CHANGE) | (pl.col("q") <= 1 / SPEC_CHANGE))
            .then(pl.col("median").map_elements(_nice, return_dtype=pl.Float64))
            .otherwise(None)
        )
        .filter(pl.col("value").is_not_null())
        .sort("ts_code", "month")
    )
    # 逐行把成交日归到该合约出现过的取值上，找分界日。
    values = months.group_by("ts_code").agg(
        vals=pl.col("value").unique().sort(), months=pl.len()
    )
    by_code = {r["ts_code"]: r["vals"] for r in values.iter_rows(named=True)}
    multi = {k: v for k, v in by_code.items() if len(v) > 1}
    change_rows = (
        rows.filter(pl.col("ts_code").is_in(list(multi)))
        .sort("ts_code", "trade_date")
        .to_dicts()
        if multi
        else []
    )
    days = {}
    for row in change_rows:
        days.setdefault(row["ts_code"], []).append(row)
    info = {r["ts_code"]: r for r in declared.iter_rows(named=True)}
    out, switches, labels_of = [], {}, {}
    for code, item in info.items():
        symbol, start = item["symbol"], item["list_date"]
        vals = by_code.get(code)
        if vals is None:
            manual = _manual(item)
            if manual is not None:
                out.append(
                    (symbol, start, manual["multiplier"], "manual", manual["basis"])
                )
            else:
                out.append(
                    (symbol, start, item["declared"], "contract_table_unverified", None)
                )
            continue
        if len(vals) == 1:
            value = vals[0]
            source = (
                "contract_table" if value == item["declared"] else "implied_from_amount"
            )
            out.append((symbol, start, value, source, None))
            continue
        # 每个成交日取离它最近（对数距离）的取值，要求取值序列只切换一次。
        labels = []
        for row in days[code]:
            best = min(vals, key=lambda v: abs(_log(row["ratio"]) - _log(v)))
            if abs(row["ratio"] / best - 1) <= SPEC_CHANGE - 1:
                labels.append((row["trade_date"], best))
        labels_of[code] = labels
        runs = [labels[0]] if labels else []
        for day, value in labels[1:]:
            if value != runs[-1][1]:
                runs.append((day, value))
        if len(runs) == len(vals):
            switches[code] = runs
            continue
        out.append(
            (
                symbol,
                start,
                item["declared"],
                "contract_table_unverified",
                f"未核实：反查值在 {vals} 之间交错，无法分段",
            )
        )
    # 同一品种同一种「旧值→新值」切换取最早的分界日：交易所改单位是全品种同日生效，
    # 某个合约在那天之后才有第一笔新值成交时，分界日落在它最后一笔旧值成交之后、第一笔新值成交
    # 之前的这个统一日期上。
    earliest = {}
    for code, runs in switches.items():
        product = info[code]["fut_code"]
        for (_, old), (day, new) in zip(runs, runs[1:], strict=False):
            key = (product, old, new)
            earliest[key] = min(earliest.get(key, day), day)
    for code, runs in switches.items():
        item = info[code]
        symbol, start, product = item["symbol"], item["list_date"], item["fut_code"]
        dated = []
        for k, (day, value) in enumerate(runs):
            if k:
                common = earliest[(product, runs[k - 1][1], value)]
                previous_old = max(
                    d for d, v in labels_of[code] if v == runs[k - 1][1] and d < day
                )
                if previous_old < common <= day:
                    day = common
            dated.append((day, value))
        for k, (day, value) in enumerate(dated):
            source = (
                "contract_table" if value == item["declared"] else "implied_from_amount"
            )
            out.append((symbol, start if k == 0 else day, value, source, None))
    return conform(
        "multipliers",
        pl.DataFrame(
            out,
            schema={
                "symbol": pl.String,
                "start_date": pl.String,
                "multiplier": pl.Float64,
                "source": pl.String,
                "basis": pl.String,
            },
            orient="row",
        ),
    ).sort("symbol", "start_date")


def _log(x):
    return math.log(x)


def _nice(x):
    """反查值取最短的有效数字（1–3 位）使相对误差不超过 1%：500.7 → 500，24.9 → 25。"""
    for digits in (1, 2, 3):
        scale = 10 ** (digits - 1 - math.floor(math.log10(abs(x))))
        value = round(x * scale) / scale
        if abs(value / x - 1) <= 0.01:
            return float(value)
    return float(round(x))


def _manual(item):
    exchange = SUFFIX.get(item["ts_code"].rsplit(".", 1)[-1])
    product = (item["fut_code"] or "").upper()
    for rule in MANUAL_MULTIPLIERS:
        if (
            rule["exchange"] == exchange
            and rule["product"] == product
            and (item["delist_date"] or "") <= rule["last_trade_before"]
        ):
            return rule
    return None


def _sided(frame):
    exchange = pl.col("exchange")
    double = pl.lit(False)
    for name, until in DOUBLE_SIDED_UNTIL.items():
        double = double | ((exchange == name) & (pl.col("date") < until))
    return frame.with_columns(_halve=pl.when(double).then(0.5).otherwise(1.0))


def daily(store, start=None, end=None, raw=None):
    """统一字段 daily 表；``raw`` 为已读入的原始日线（不给就按 start/end 读）。"""
    raw = raw if raw is not None else store.read("futures_daily", start, end)
    listing_dates = store.read("futures_contracts").select(
        "ts_code", "list_date", "delist_date"
    )
    # 只留上市期内的行：上期所 2018 年燃料油改合约时沿用了旧代码，新合约挂牌日之前的行属于
    # 同名旧合约（交易单位不同），丢弃；合约表里没有的代码（连续代码）在下面按代码形状丢弃。
    raw = raw.join(listing_dates, on="ts_code", how="left").filter(
        pl.col("list_date").is_null()
        | (
            (pl.col("trade_date") >= pl.col("list_date"))
            & (pl.col("trade_date") <= pl.col("delist_date").fill_null("99999999"))
        )
    )
    frame = (
        raw.with_columns(symbol=_symbol_expr(), date=pl.col("trade_date"))
        .filter(
            pl.col("symbol").is_not_null()
            & ~_product_expr().is_in(list(EXCLUDED_PRODUCTS))
        )
        .with_columns(exchange=pl.col("symbol").str.extract(r"\.(\w+)$"))
    )
    # 价格不可能为 0：2026-04-01 起大商所、郑商所、广期所（及中金所个别行）对零成交的行把开、高、
    # 低价写成 0（此前为空）。不置空的话 0 会被当成可成交的开盘价。
    frame = frame.with_columns(
        [
            pl.when(pl.col(c) > 0).then(pl.col(c)).alias(c)
            for c in ("open", "high", "low", "close", "settle", "pre_settle")
        ]
    )
    frame = _sided(frame).with_columns(
        volume=pl.col("vol") * pl.col("_halve"),
        amount=pl.col("amount") * 10000 * pl.col("_halve"),
        open_interest=pl.col("oi") * pl.col("_halve"),
        settlement=pl.col("settle"),
        previous_settlement=pl.col("pre_settle"),
    )
    return conform(
        "daily",
        frame.select(
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "settlement",
            "previous_settlement",
            "volume",
            "amount",
            "open_interest",
        ),
    ).sort("symbol", "date")


def tushare_main_daily(store, start=None, end=None):
    """Tushare 自己的主力连续日线（``RB.SHF`` 这类代码），只用于对账。"""
    raw = store.read("futures_daily", start, end)
    return raw.filter(pl.col("ts_code").str.contains(r"^[A-Z]+\.\w+$")).select(
        continuous=pl.col("ts_code"),
        date=pl.col("trade_date"),
        open="open",
        high="high",
        low="low",
        close="close",
        settlement="settle",
        volume="vol",
        open_interest="oi",
    )


ZERO_DAY_SHARE = 0.9
FEE_COLUMNS = ("trading_fee_rate", "trading_fee", "offset_today_fee", "delivery_fee")


def _clean_fees(frame):
    """手续费原始值的清洗，规则都由数据核对（见文档「手续费的空值与零」）：

    - 负数一律置空：郑商所 2014-12-23 至 2017-06-22 的 offset_today_fee 是 -1、-2
      （18 个品种），交易所没有负手续费，含义未查明，按缺失处理。
    - 某交易所某天有费用的行里至少 ZERO_DAY_SHARE 是「两列都为 0」时，把这些 0 置空：
      2026-09-28 上期所、能源中心、广期所全为 0，大商所 97% 为 0，次日恢复原值，是源错误
      而不是免收。全历史其余交易所-交易日的这一比例都不超过 0.2，阈值不敏感。
      单个品种为 0（如中金所国债平今免收）不在此列。
    """
    frame = frame.with_columns(
        [pl.when(pl.col(c) >= 0).then(pl.col(c)).alias(c) for c in FEE_COLUMNS]
    )
    present = (
        pl.col("trading_fee_rate").is_not_null() | pl.col("trading_fee").is_not_null()
    )
    zero = (
        present
        & (pl.col("trading_fee_rate").fill_null(0) == 0)
        & (pl.col("trading_fee").fill_null(0) == 0)
    )
    share = zero.sum().over("exchange", "date") / present.sum().over("exchange", "date")
    wipe = zero & (share >= ZERO_DAY_SHARE)
    return frame.with_columns(
        [pl.when(wipe).then(None).otherwise(pl.col(c)).alias(c) for c in FEE_COLUMNS]
    )


def resolved_settle(store, start=None, end=None):
    """结算参数原始行逐行对到合约（见 resolve），保留 resolution 列，供 params 与校验共用。"""
    raw = store.read("futures_settle", start, end)
    raw = raw.filter(
        ~pl.col("ts_code").str.extract(r"^([A-Z]+)").is_in(list(EXCLUDED_PRODUCTS))
    )
    return resolve(raw, listing(store))


def params(store, start=None, end=None):
    resolved = resolved_settle(store, start, end)
    # 同一天同一合约两行（错后缀与正确后缀并存）时保留字面正确的那行。
    priority = pl.col("resolution").replace_strict(
        {r: k for k, r in enumerate(RESOLUTIONS)}, default=len(RESOLUTIONS)
    )
    frame = (
        resolved.filter(pl.col("symbol").is_not_null())
        .with_columns(_priority=priority)
        .sort("trade_date", "symbol", "_priority")
        .unique(["trade_date", "symbol"], keep="first", maintain_order=True)
        .with_columns(
            date=pl.col("trade_date"),
            exchange=pl.col("symbol").str.extract(r"\.(\w+)$"),
            product=pl.col("symbol").str.extract(r"^([A-Z]+)"),
        )
    )
    frame = _clean_fees(frame)
    ex, product = pl.col("exchange"), pl.col("product")
    rate_col, unit_col = pl.col("trading_fee_rate"), pl.col("trading_fee")
    today = pl.col("offset_today_fee")
    margin_scale = pl.when(ex == "CFFEX").then(0.01).otherwise(1.0)
    is_index = (ex == "CFFEX") & product.is_in(list(CFFEX_INDEX))
    is_bond = (ex == "CFFEX") & ~product.is_in(list(CFFEX_INDEX))
    shfe = ex.is_in(["SHFE", "INE"])
    rate_based = pl.lit(False)
    per_contract = ex == "CZCE"
    for exchange, names in RATE_BASED.items():
        rate_based = rate_based | ((ex == exchange) & product.is_in(list(names)))
    for exchange, names in PER_CONTRACT.items():
        per_contract = per_contract | ((ex == exchange) & product.is_in(list(names)))
    # 开平仓（平昨）同一费率。
    open_rate = (
        pl.when(shfe)
        .then(rate_col * 1e-3)
        .when(is_index)
        .then(rate_col * 1e-4)
        .when(is_bond)
        .then(unit_col * 0.0)
        .when(rate_based)
        .then(unit_col * 1e-4)
        .when(per_contract)
        .then(unit_col * 0.0)
        .otherwise(None)
    )
    # 「这一部分为 0」只在源给了这一行费用时成立（乘 0 让空值保持为空）。
    open_unit = (
        pl.when(shfe)
        .then(unit_col)
        .when(is_index)
        .then(rate_col * 0.0)
        .when(is_bond)
        .then(unit_col)
        .when(rate_based)
        .then(unit_col * 0.0)
        .when(per_contract)
        .then(unit_col)
        .otherwise(None)
    )
    # 平今：中金所 offset_today_fee 是相对普通费的百分比（1000 = 10 倍，0 = 免收）；
    # 大商所/郑商所/广期所是与 trading_fee 同口径的数值；上期所/能源中心源里为空。
    cffex_multiple = today / 100
    close_today_rate = (
        pl.when(ex == "CFFEX")
        .then(open_rate * cffex_multiple)
        .when(rate_based)
        .then(today * 1e-4)
        .when(per_contract)
        .then(today * 0.0)
        .otherwise(None)
    )
    close_today_unit = (
        pl.when(ex == "CFFEX")
        .then(open_unit * cffex_multiple)
        .when(rate_based)
        .then(today * 0.0)
        .when(per_contract)
        .then(today)
        .otherwise(None)
    )
    close_today_known = today.is_not_null()
    delivery = pl.col("delivery_fee")
    frame = frame.with_columns(
        margin_long=pl.col("long_margin_rate") * margin_scale,
        margin_short=pl.col("short_margin_rate") * margin_scale,
        hedge_margin_long=pl.col("b_hedging_margin_rate") * margin_scale,
        hedge_margin_short=pl.col("s_hedging_margin_rate") * margin_scale,
        fee_open_rate=open_rate,
        fee_open_per_contract=open_unit,
        fee_close_rate=open_rate,
        fee_close_per_contract=open_unit,
        fee_close_today_rate=pl.when(close_today_known).then(close_today_rate),
        fee_close_today_per_contract=pl.when(close_today_known).then(close_today_unit),
        # 交割费：上期所/能源中心与 trading_fee_rate 同为千分比，按手收取的品种源里同样只给比例列，
        # 中金所、大商所、郑商所、广期所的交割费单位未核实，留空。
        delivery_fee_rate=pl.when(shfe).then(delivery * 1e-3),
        delivery_fee_per_contract=pl.lit(None, pl.Float64),
        price_limit=pl.lit(None, pl.Float64),
    )
    return conform(
        "params",
        frame.select(
            "date",
            "symbol",
            "margin_long",
            "margin_short",
            "hedge_margin_long",
            "hedge_margin_short",
            "fee_open_rate",
            "fee_open_per_contract",
            "fee_close_rate",
            "fee_close_per_contract",
            "fee_close_today_rate",
            "fee_close_today_per_contract",
            "delivery_fee_rate",
            "delivery_fee_per_contract",
            "price_limit",
        ),
    ).sort("symbol", "date")


def calendar(store):
    raw = store.read("futures_calendar")
    return conform(
        "calendar",
        raw.select(
            "exchange",
            date=pl.col("cal_date"),
            is_open=pl.col("is_open").cast(pl.Int64) == 1,
        ),
    ).sort("exchange", "date")


def mapping(store, start=None, end=None):
    """``RB.SHF`` 为主力、``RBL.SHF`` 为连续（近月）。

    不能只看代码末尾的 L（``AL.SHF``、``TL.CFX`` 本身就是品种）：被映射合约的品种等于连续代码
    去掉后缀的部分时为主力，等于再去掉末尾 L 时为连续。
    """
    if not store.entries("futures_mapping"):
        return conform(
            "mapping",
            pl.DataFrame(
                schema={c: pl.String for c in ("date", "continuous", "kind", "symbol")}
            ),
        )
    raw = store.read("futures_mapping", start, end)
    base = pl.col("ts_code").str.extract(r"^([A-Z_]+)\.")
    target = pl.col("mapping_ts_code").str.extract(r"^([A-Z]+)")
    frame = raw.with_columns(
        date=pl.col("trade_date"),
        continuous=pl.col("ts_code"),
        kind=pl.when(target == base)
        .then(pl.lit("main"))
        .when(
            base.str.ends_with("L")
            & (target == base.str.slice(0, base.str.len_chars() - 1))
        )
        .then(pl.lit("nearby"))
        .otherwise(pl.lit("other")),
    ).filter(~base.str.extract(r"^([A-Z]+?)L?$").is_in(list(EXCLUDED_PRODUCTS)))
    frame = resolve(frame, listing(store), code="mapping_ts_code").filter(
        pl.col("symbol").is_not_null()
    )
    return conform(
        "mapping", frame.select("date", "continuous", "kind", "symbol")
    ).sort("continuous", "date")


class TushareFutures:
    """把原始库包成统一字段的适配器。必备 contracts/daily/params/calendar，
    可选 mapping、multipliers、source_continuous、diagnostics。"""

    name = "Tushare Pro"

    def __init__(self, store):
        self.store = store
        self._cache = {}

    def _memo(self, name, compute):
        """同一份期货原始库（按 futures_version 判断）上只算一次：构建时全量原始日线要给合约表、
        乘数、日线三处用，原来各读一遍。原始库一变（版本不同）就重算，不会给出旧结果。"""
        version = self.store.futures_version
        hit = self._cache.get(name)
        if hit is None or hit[0] != version:
            hit = (version, compute())
            self._cache[name] = hit
        return hit[1]

    def _raw_daily(self):
        return self._memo("raw_daily", lambda: self.store.read("futures_daily"))

    def _used_raw(self, name):
        """全量原始日线只给乘数和全量日线两处用：两处都算过以后就放掉，不让它占着内存到构建结束。"""
        self._cache.setdefault("raw_users", set()).add(name)
        if {"multipliers", "daily"} <= self._cache["raw_users"]:
            self._cache.pop("raw_daily", None)
            self._cache.pop("raw_users", None)

    def contracts(self):
        return contracts(self.store, history=self.multipliers())

    def daily(self, start=None, end=None):
        if start is None and end is None:
            out = daily(self.store, raw=self._raw_daily())
            self._used_raw("daily")
            return out
        return daily(self.store, start, end)

    def params(self, start=None, end=None):
        return params(self.store, start, end)

    def calendar(self):
        return calendar(self.store)

    def mapping(self, start=None, end=None):
        return mapping(self.store, start, end)

    def multipliers(self):
        out = self._memo(
            "multipliers",
            lambda: multipliers(self.store, raw_daily=self._raw_daily()),
        )
        self._used_raw("multipliers")
        return out

    def diagnostics(self):
        """代码对齐的统计：每张表每种 resolution 的行数，以及没能对上的行（逐条，供校验）。"""
        table = listing(self.store)
        settle = self.store.read("futures_settle").filter(
            ~pl.col("ts_code").str.extract(r"^([A-Z]+)").is_in(list(EXCLUDED_PRODUCTS))
        )
        maps = (
            None
            if not self.store.entries("futures_mapping")
            else self.store.read("futures_mapping").filter(
                ~pl.col("mapping_ts_code")
                .str.extract(r"^([A-Z]+)")
                .is_in(list(EXCLUDED_PRODUCTS))
            )
        )
        out = {}
        for name, frame, code in (
            ("params", settle, "ts_code"),
            ("mapping", maps, "mapping_ts_code"),
        ):
            if frame is None:
                continue
            done = resolve(frame, table, code=code)
            counts = done.group_by("resolution").len()
            failed = done.filter(pl.col("resolution").is_in(["unmatched", "ambiguous"]))
            out[name] = dict(
                rows={r["resolution"]: r["len"] for r in counts.iter_rows(named=True)},
                unresolved=failed.select(
                    code=pl.col(code),
                    date=pl.col("trade_date"),
                    resolution="resolution",
                )
                .sort("date", "code")
                .to_dicts(),
            )
        return out

    def source_continuous(self):
        """来源自带的主力连续日线，只用于对账，不进统一字段。"""
        return tushare_main_daily(self.store)


__all__ = ["TushareFutures", "multipliers", "resolve", "unify_symbol"]
