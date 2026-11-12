"""事件驱动研究：事件表、内置事件提取、事件转目标持仓、事件研究。

十行用法（真实数据）::

    import ashare.data as ad
    import panelbt
    from ashare.events import extract, event_study, to_targets
    from ashare.events.market import load

    ds = ad.open(); m = load(ds, start="2015-01-05")
    ev = extract.forecasts(ds).filter(forecast_type="预增", is_revision=False)
    study = event_study(ev, m.returns, m.market_return, m.dates, m.symbols, window=(-20, 60))
    print(study.calendar_time(1, 20), study.test(1, 20))   # 日历时间 NW t；按月聚类 t
    plan = to_targets(ev, m.dates, m.symbols, m["can_buy_open"], can_sell=m["can_sell_open"],
                      holding=20, max_positions=50, priority="change_low")
    bt = m.backtest
    costs = bt.costs(commission_rate=2.5e-4, commission_min=5.0, slippage=5e-4)
    result = panelbt.run(bt.market, plan.targets, costs, bt.config())

细节见 ``docs/DESIGN.md`` 6.6 与 ``docs/EVENTS_NOTES.md``。
"""

from . import extract
from .study import (
    EventStudy,
    abnormal_returns,
    daily_returns,
    event_study,
    group_benchmark,
    size_industry_groups,
)
from .table import AlignedEvents, align, dedupe, validate, with_timing
from .targets import EventTargets, tie_key, to_targets

__all__ = [
    "AlignedEvents",
    "EventStudy",
    "EventTargets",
    "abnormal_returns",
    "align",
    "daily_returns",
    "dedupe",
    "event_study",
    "extract",
    "group_benchmark",
    "size_industry_groups",
    "tie_key",
    "to_targets",
    "validate",
    "with_timing",
]
