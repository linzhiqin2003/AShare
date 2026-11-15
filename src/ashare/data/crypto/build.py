"""从同步计划构建；也可从离线规范化输入构建共享测试面板。"""

from .dataset import write_panel
from .instruments import validate_profile
from .panel import build_frame
from .recent import prepare_recent
from .reconcile import reconcile
from .sync import load_plan


def build(root, *, allow_missing=False, allow_inferred_funding=False):
    plan, data = load_plan(root)
    profiles = [validate_profile(p) for p in plan["profiles"]]
    data, end, recent = prepare_recent(
        profiles,
        data,
        plan["end"],
        start=plan["start"],
        events=plan["events"],
        allow_inferred_funding=allow_inferred_funding,
        funding_info=plan.get("funding_info"),
    )
    recent.update(
        source_contract=1,
        funding_price_refs=plan.get("funding_price_refs", {}),
        funding_archive_coverage=plan.get("funding_archive_coverage", {}),
    )
    frame, notes = build_frame(
        profiles,
        data,
        plan["events"],
        plan["start"],
        end,
        allow_missing=allow_missing,
    )
    ds = write_panel(
        root,
        frame,
        profiles=profiles,
        events=plan["events"],
        data=data,
        notes=notes,
        raw_version=plan["raw_version"],
        exceptions=plan.get("exceptions", []),
        recent=recent,
    )
    reconcile(ds)
    return ds
