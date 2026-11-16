"""从冻结 A 股面板按月导出；不写源目录，不补没有报价的行。"""

import numpy as np
import polars as pl

from ashare.data import open as open_cn

from .constants import FIELDS
from .export_common import clean, field, manifest, resource

CN_MAPPING = {
    name: name
    for name in FIELDS
    if name
    not in {
        "limit_up",
        "limit_down",
        "suspended",
        "trade_count",
        "open_interest",
        "settle",
        "taker_buy_volume",
    }
}
CN_MAPPING.update(
    limit_up="upper_limit", limit_down="lower_limit", suspended="is_suspended"
)
CN_EXTENSIONS = {
    "total_market_cap": "CNY",
    "float_market_cap": "CNY",
    "total_shares": "share",
    "float_shares": "share",
    "free_shares": "share",
    "turnover": "ratio",
    "pe": "1",
    "pb": "1",
    "ps": "1",
    "pe_ttm": "1",
    "ps_ttm": "1",
    "dividend_yield": "ratio",
    "dividend_yield_ttm": "ratio",
}
ACTION_SOURCE = {
    0: "无事件",
    1: "分红表原样",
    2: "经济去重/候选裁决/日期对齐",
    3: "复权因子合成（等效再投资）",
    4: "分红表候选被否决",
    5: "差异化分红，持有人实得公告值",
}


class CNAdapter:
    def __init__(self, root=None, *, start=None, end=None, symbols=None):
        self.data = open_cn(root)
        axes = self.data.panel(["has_quote"], start, end, symbols)
        self.dates, self.symbols = axes.dates.copy(), axes.symbols.copy()
        self.mapping = {**CN_MAPPING, **{k: k for k in CN_EXTENSIONS}}
        self.months = sorted({d[:4] + "-" + d[4:6] for d in self.dates})
        self._cached = None

    def describe(self):
        from ashare.data.catalog import FIELDS as SOURCE_FIELDS

        fields = [field("symbol", "string"), field("date", "date")]
        for name, original in self.mapping.items():
            if name in FIELDS:
                kind, category, known = FIELDS[name]
                unit = {
                    "price": "CNY/share",
                    "quantity": "share",
                    "currency": "CNY",
                }.get(category, category)
                description = field(name, kind, unit)
                # 源复权因子实际收盘后裁决，不能利用词典默认值把它冒充盘前数据。
                if name == "adj_factor":
                    description["knownAt"] = "end"
            else:
                description = field(
                    name,
                    unit=CN_EXTENSIONS[name],
                    description=SOURCE_FIELDS[original].known_at,
                )
            fields.append(description)
        instruments = [
            field(k, t)
            for k, t in [
                ("symbol", "string"),
                ("asset_type", "string"),
                ("currency", "string"),
                ("name", "string"),
                ("list_date", "date"),
                ("delist_date", "date"),
            ]
        ]
        actions = [
            dict(
                name=k,
                type=t,
                knownAt="end" if k == "available_at" else "column:available_at",
            )
            for k, t in [
                ("symbol", "string"),
                ("ex_date", "date"),
                ("kind", "string"),
                ("split", "number"),
                ("cash_dividend", "number"),
                ("available_at", "datetime"),
                ("source", "string"),
            ]
        ]
        return manifest(
            "ashare-cn",
            f"AShare A股冻结面板 {self.data.version}",
            "cn",
            [
                resource("bars", "bars", fields, ["symbol", "date"], "1d"),
                resource("instruments", "instruments", instruments, ["symbol"]),
                resource("calendar", "calendar", [field("date", "date")], ["date"]),
                resource(
                    "corporate_actions",
                    "corporate_actions",
                    actions,
                    ["symbol", "ex_date"],
                ),
            ],
        )

    def partitions(self):
        yield "instruments", "all"
        yield "calendar", "all"
        for month in self.months:
            yield "bars", month
            yield "corporate_actions", month

    def _panel(self, partition):
        if self._cached is None or self._cached[0] != partition:
            dates = self.dates[
                np.char.startswith(self.dates, partition.replace("-", ""))
            ]
            fields = [
                *self.mapping.values(),
                "has_quote",
                "split",
                "cash_dividend",
                "action_source",
            ]
            panel = self.data.panel(fields, dates[0], dates[-1], self.symbols)
            self._cached = (partition, panel)
        return self._cached[1]

    def read(self, table, partition):
        if table == "calendar":
            return pl.DataFrame({"date": self.dates}).with_columns(
                pl.col("date").str.to_date("%Y%m%d")
            )
        if table == "instruments":
            return (
                self.data.securities()
                .filter(pl.col("symbol").is_in(self.symbols.tolist()))
                .select(
                    "symbol",
                    pl.lit("stock").alias("asset_type"),
                    pl.lit("CNY").alias("currency"),
                    "name",
                    pl.col("listing_date").str.to_date("%Y%m%d").alias("list_date"),
                    pl.col("delisting_date").str.to_date("%Y%m%d").alias("delist_date"),
                )
                .sort("symbol")
                .collect()
            )
        panel = self._panel(partition)
        if table == "bars":
            i, j = np.nonzero(panel["has_quote"])
            columns = {"symbol": panel.symbols[j], "date": panel.dates[i]}
            columns.update(
                {
                    name: panel[original][i, j].astype(
                        bool if name == "suspended" else np.float64
                    )
                    for name, original in self.mapping.items()
                }
            )
            return clean(
                pl.DataFrame(columns)
                .with_columns(
                    pl.col("date").str.to_date("%Y%m%d"),
                    *[
                        pl.when(pl.col(k) == 0).then(None).otherwise(pl.col(k)).alias(k)
                        for k in ("limit_up", "limit_down")
                    ],
                )
                .sort(["symbol", "date"])
            )
        i, j = np.nonzero((panel["split"] != 1) | (panel["cash_dividend"] != 0))
        result = pl.DataFrame(
            {
                "symbol": pl.Series(panel.symbols[j], dtype=pl.String),
                "ex_date": pl.Series(panel.dates[i], dtype=pl.String),
                "kind": pl.Series(
                    [
                        "split" if panel["split"][a, b] != 1 else "dividend"
                        for a, b in zip(i, j, strict=True)
                    ],
                    dtype=pl.String,
                ),
                "split": pl.Series(panel["split"][i, j], dtype=pl.Float64),
                "cash_dividend": pl.Series(
                    panel["cash_dividend"][i, j], dtype=pl.Float64
                ),
                "source": pl.Series(
                    [
                        "AShare日终裁决："
                        + ACTION_SOURCE[int(panel["action_source"][a, b])]
                        for a, b in zip(i, j, strict=True)
                    ],
                    dtype=pl.String,
                ),
            }
        )
        return clean(
            result.with_columns(pl.col("ex_date").str.to_date("%Y%m%d"))
            .with_columns(
                (pl.col("ex_date").cast(pl.Datetime("us")) + pl.duration(hours=15))
                .dt.replace_time_zone("Asia/Shanghai")
                .dt.convert_time_zone("UTC")
                .alias("available_at")
            )
            .select(
                "symbol",
                "ex_date",
                "kind",
                "split",
                "cash_dividend",
                "available_at",
                "source",
            )
            .sort(["symbol", "ex_date"])
        )
