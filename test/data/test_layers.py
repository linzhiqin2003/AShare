import ast
import importlib.util
import sys
from pathlib import Path

import ashare

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)
ROOT = Path(__file__).resolve().parents[2] / "src/ashare"
INTERNAL = {p.stem for p in (ROOT / "data").glob("*.py")} - {"__init__", "__main__"} | {
    "sources"
}


def violations(content, module, *, source=False):
    tree = ast.parse(content)
    package = module.rsplit(".", 1)[0]
    failures, aliases = [], {}
    for node in ast.walk(tree):
        imports = []
        if isinstance(node, ast.Import):
            imports = [a.name for a in node.names]
            aliases.update(
                {
                    a.asname or a.name.split(".")[0]: a.name
                    if a.asname
                    else a.name.split(".")[0]
                    for a in node.names
                }
            )
        elif isinstance(node, ast.ImportFrom):
            base = (
                importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
                if node.level
                else node.module or ""
            )
            imports = [base + "." + a.name for a in node.names]
            aliases.update(
                {a.asname or a.name: base + "." + a.name for a in node.names}
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("ashare.")
        ):
            imports = [node.value]
        for imported in imports:
            if source:
                allowed = set(sys.stdlib_module_names) | {
                    "httpx",
                    "polars",
                    "pyarrow",
                    "baostock",
                }
                if imported.startswith("ashare.") and not imported.startswith(
                    "ashare.data.sources."
                ):
                    failures.append(imported)
                elif (
                    not imported.startswith("ashare.")
                    and imported.split(".")[0] not in allowed
                ):
                    failures.append(imported)
            elif (
                imported.startswith("ashare.data.")
                and imported.split(".")[2] in INTERNAL
            ):
                failures.append(imported)
    if not source:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                parts, head = [], node
                while isinstance(head, ast.Attribute):
                    parts.insert(0, head.attr)
                    head = head.value
                if isinstance(head, ast.Name) and head.id in aliases:
                    full = aliases[head.id] + "." + ".".join(parts)
                    if (
                        full.startswith("ashare.data.")
                        and full.split(".")[2] in INTERNAL
                    ):
                        failures.append(full)
        if "api.tushare.pro" in content:
            failures.append("source_url")
    return failures


def test_source_independence():
    for path in (ROOT / "data/sources").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT.parent).with_suffix("").parts)
        assert not violations(path.read_text(), module, source=True), path


def test_public_boundary():
    # paper 在另一 worktree 开发；合并后由维护者接入共享源客户端，再移除豁免。
    for path in ROOT.rglob("*.py"):
        if path.is_relative_to(ROOT / "data") or path.is_relative_to(ROOT / "paper"):
            continue
        assert not violations(
            path.read_text(),
            ".".join(path.relative_to(ROOT.parent).with_suffix("").parts),
        ), path


def test_layer_checker_cannot_be_bypassed():
    for content, module, source in [
        ("from .data.store import Store", "ashare.test", False),
        ("from ..data.catalog import FIELDS", "ashare.factors.test", False),
        ("import ashare.data as d; x = d.store.Store", "ashare.test", False),
        ("from ...catalog import DATASETS", "ashare.data.sources.sub.test", True),
        ("import pandas", "ashare.data.sources.test", True),
        (
            'import importlib; importlib.import_module("ashare.data.store")',
            "ashare.data.sources.test",
            True,
        ),
    ]:
        assert violations(content, module, source=source), content
    assert not violations("from .data import open", "ashare.test")
    assert not violations(
        "from .base import Source", "ashare.data.sources.test", source=True
    )
