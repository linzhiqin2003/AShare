"""端到端测试用的研究脚本：模拟研究者在命令行里跑一次完整研究。

    python test/e2e/research_cli.py --root DATA --out OUT [--source "-ts_std(returns, 10)"]

数据 → 表达式求值 → 分组回测（样本内）→ 第 1 组持仓交给 panelbt 做账户级回测 → 报告。
产物：groupbt.html、groupbt.json、tearsheet.json、report.html、summary.json；
最后把 summary.json 的内容打印到标准输出。只用公开接口，不读真实数据、不联网。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REBALANCE = 5
BENCHMARK = "000300.SH"


def clean(value):
    return (
        value
        if isinstance(value, (int, str)) or value is None or math.isfinite(value)
        else None
    )


def research(root, out, source):
    import panelbt

    import ashare.data
    import ashare.report
    from ashare.backtest import execution_targets, market_from_panel
    from ashare.factor import Evaluator, groupbt
    from ashare.platform.language import parse

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = ashare.data.open(root)
    expression = parse(source)[0].expression
    fields = sorted(set(groupbt.FIELDS) | set(expression.fields))
    panel = data.panel(fields, end=str(np.asarray(data.panel([]).dates)[-1]))
    alpha = Evaluator().evaluate(expression, panel)

    result = groupbt.run(alpha, panel, "TOP1800", rebalance=REBALANCE)
    (out / "groupbt.html").write_text(result.html(), encoding="utf-8")
    record = result.to_dict()
    record["long_short_pnl"] = [clean(float(x)) for x in result.pnl["long_short"]]
    record["dates"] = [str(d) for d in result.dates]
    (out / "groupbt.json").write_text(json.dumps(record, ensure_ascii=False), "utf-8")

    # 账户级回测：样本内每 REBALANCE 天等权持有因子值最高的 1/5（主池、可买的股票）。
    market = market_from_panel(
        data,
        end=str(max(d for d in panel.dates if d < groupbt.IS[1])),
        fields=groupbt.FIELDS,
    )
    dates = np.asarray(market.dates)
    signal = np.full(market.shape, np.nan)
    pool = groupbt.universe_mask(market.panel, "TOP1800")
    rows = np.arange(len(dates))
    for t in rows[(rows % REBALANCE == 0) & (dates < groupbt.IS[1])]:
        values = np.where(pool[t], alpha[t], np.nan)
        finite = np.isfinite(values)
        if finite.sum() < 5:
            continue
        cut = np.nanquantile(values, 0.8)
        chosen = finite & (values >= cut)
        signal[t] = np.where(chosen, 0.98 / chosen.sum(), 0.0)
    targets = execution_targets(signal)
    costs = market.costs(commission_rate=3e-4, commission_min=5.0, slippage=5e-4)
    config = market.config(initial_cash=1e7, record_trades=True)
    backtest = panelbt.run(market.market, targets, costs, config)
    sheet = ashare.report.tearsheet(
        backtest,
        market.market,
        targets,
        [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates],
        list(market.symbols),
        data=data,
        benchmark=BENCHMARK,
        benchmark_kind="price",
        costs=costs,
        config=config,
        title="合成数据低波动",
    )
    sheet.save_json(out / "tearsheet.json")
    sheet.save(out / "report.html")
    summary = dict(
        data_version=data.version,
        groupbt_sharpe=clean(result.table().to_dicts()[-1]["sharpe"]),
        groupbt_days=int(np.isfinite(result.pnl["long_short"]).sum()),
        final_equity=float(np.asarray(backtest.equity)[-1]),
        tearsheet_final_nav=sheet.summary["final_nav"],
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), "utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="合成数据上的一次完整研究")
    parser.add_argument("--root", required=True, help="数据根目录")
    parser.add_argument("--out", required=True, help="产物目录")
    parser.add_argument("--source", default="-ts_std(returns, 10)", help="因子表达式")
    args = parser.parse_args(argv)
    summary = research(args.root, args.out, args.source)
    sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
