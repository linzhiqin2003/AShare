"""从同步计划构建；也可从离线规范化输入构建共享测试面板。"""

from .dataset import write_panel
from .instruments import validate_profile
from .panel import build_frame
from .sync import load_plan


def build(root, *, allow_missing=False):
    plan, data = load_plan(root)
    profiles = [validate_profile(p) for p in plan["profiles"]]
    frame, notes = build_frame(
        profiles,
        data,
        plan["events"],
        plan["start"],
        plan["end"],
        allow_missing=allow_missing,
    )
    return write_panel(
        root,
        frame,
        profiles=profiles,
        events=plan["events"],
        data=data,
        notes=notes,
        raw_version=plan["raw_version"],
        exceptions=plan.get("exceptions", []),
    )
