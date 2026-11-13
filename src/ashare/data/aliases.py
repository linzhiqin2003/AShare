"""证券换码只在规范化读取时拼接；原始库和原始内容哈希不改。"""

import polars as pl

from .catalog import DATASETS, SYMBOL_ALIASES


def alias_context(store):
    result = []
    for alias in SYMBOL_ALIASES:
        old, new, before, after = (
            alias[k] for k in ("old", "new", "old_last", "switch")
        )
        covered = [
            store.manifest().get(f"daily/{day}", {}).get("status") == "complete"
            for day in (before, after)
        ]
        if not all(covered):
            # 原始库不覆盖接缝时，库里至多只有一边的代码，不需要因子基准。
            continue
        quotes = store.read("daily", before, after)
        a = quotes.filter((pl.col("ts_code") == old) & (pl.col("trade_date") == before))
        b = quotes.filter((pl.col("ts_code") == new) & (pl.col("trade_date") == after))
        if not len(a) or not len(b):
            raise ValueError(
                f"换码 {old}→{new} 接缝 {before}/{after} 缺行情，不能确定因子基准；请核对 catalog.SYMBOL_ALIASES 的日期"
            )
        if abs(a["close"][0] - b["pre_close"][0]) > 0.011:
            raise ValueError(f"换码 {old}→{new} 接缝参考价不连续，不能自动拼接")
        events = store.read("dividend", after, after)
        if len(
            events.filter(
                pl.col("ts_code").is_in([old, new])
                & (pl.col("ex_date") == after)
                & (pl.col("div_proc") == "实施")
                & (
                    (pl.col("stk_div").fill_null(0) != 0)
                    | (pl.col("cash_div_tax").fill_null(0) != 0)
                )
            )
        ):
            raise ValueError(f"换码 {old}→{new} 接缝有公司行为，请人工确认因子基准")
        factors = store.read("adj_factor", before, after)
        x = factors.filter(
            (pl.col("ts_code") == old) & (pl.col("trade_date") == before)
        )["adj_factor"]
        y = factors.filter(
            (pl.col("ts_code") == new) & (pl.col("trade_date") == after)
        )["adj_factor"]
        scale = float(y[0] / x[0]) if len(x) and len(y) else 1.0
        if scale <= 0:
            raise ValueError("换码因子必须为正数")
        result.append(
            dict(
                **alias,
                factor_scale=scale,
                old_close=float(a["close"][0]),
                new_previous_close=float(b["pre_close"][0]),
            )
        )
    return result


def canonical(table, name, context):
    columns = (
        table.columns
        if isinstance(table, pl.DataFrame)
        else table.collect_schema().names()
    )
    column = "con_code" if name in ("index_weight", "index_member") else "ts_code"
    if column not in columns or name in (
        "stock_basic",
        "index_daily",
        "fut_basic",
        "fut_daily",
    ):
        return table
    aliases = SYMBOL_ALIASES
    affected_codes = [a[k] for a in aliases for k in ("old", "new")]
    affected = table.filter(pl.col(column).is_in(affected_codes)).with_columns(
        pl.col(column).alias("_original_code")
    )
    unaffected = table.filter(~pl.col(column).is_in(affected_codes))
    mapped, priority = pl.col(column), pl.lit(0)
    scales = {a["old"]: a["factor_scale"] for a in context}
    for alias in aliases:
        old, new = alias["old"], alias["new"]
        if name == "adj_factor":
            affected = affected.with_columns(
                pl.when(pl.col(column) == old)
                .then(pl.col("adj_factor") * scales.get(old, 1.0))
                .otherwise(pl.col("adj_factor"))
                .alias("adj_factor")
            )
        preferred = pl.col(column) == old
        if "trade_date" in columns:
            preferred = preferred == (pl.col("trade_date") < alias["switch"])
        priority = (
            pl.when(pl.col(column).is_in([old, new]))
            .then(preferred.cast(pl.Int8))
            .otherwise(priority)
        )
        mapped = pl.when(pl.col(column) == old).then(pl.lit(new)).otherwise(mapped)
    affected = affected.with_columns(priority.alias("_priority"), mapped.alias(column))
    keys = [c for c in DATASETS[name].keys if c in columns]
    if (
        DATASETS[name].partition != "day" and name != "index_weight"
    ) or name == "dividend":
        keys = [c for c in columns if c != "_fetched_at"]
    affected = (
        affected.sort("_priority", "_original_code")
        .unique(subset=keys, keep="last", maintain_order=True)
        .drop("_priority", "_original_code")
    )
    return pl.concat([unaffected, affected], how="diagonal_relaxed")


class CanonicalStore:
    def __init__(self, store, context=None):
        self.raw = store
        self.aliases = alias_context(store) if context is None else context

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def read(self, name, start=None, end=None):
        return canonical(self.raw.read(name, start, end), name, self.aliases)

    def scan(self, name, start=None, end=None, codes=None):
        return canonical(self.raw.scan(name, start, end, codes), name, self.aliases)

    def snapshot(self):
        return CanonicalStore(self.raw.snapshot(), self.aliases)
