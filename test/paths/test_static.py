"""D1 静态门禁：数据路径字面量与根目录拼接只允许统一入口定义。"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 币安远端对象键不是本地目录；三个平台脚本在临时目录生成合成夹具。
ALLOW = {
    "src/ashare/data/crypto/archive.py": {"data/spot", "data/futures/um"},
    # 币安归档的 URL 路径片段，不是本地数据目录。
    "src/ashare/data/crypto/verify_minute.py": {"data/spot", "data/futures/um"},
    "bench/platform/holdings_size.py": {'root / "data"'},
    "bench/platform/http_perf.py": {'root / "data"'},
    "bench/platform/models.py": {'root / "data"'},
}


def violations(source, allowed=()):
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if (
            isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and node.body
        ):
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
    bad = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            if re.match(r"^data/[^/]+", node.value) and node.value not in allowed:
                bad.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and node.args
        ):
            if isinstance(node.args[0], ast.Constant) and node.args[0].value == "data":
                bad.append(node.lineno)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if (
                isinstance(node.right, ast.Constant)
                and node.right.value == "data"
                and ast.unparse(node).replace("'", '"') not in allowed
            ):
                bad.append(node.lineno)
        if isinstance(node, ast.Call):
            function = ast.unparse(node.func)
            if function in {"os.environ.get", "os.getenv"}:
                defaults = node.args[1:] + [
                    kw.value for kw in node.keywords if kw.arg == "default"
                ]
                if any(
                    isinstance(v, ast.Constant) and v.value == "data" for v in defaults
                ):
                    bad.append(node.lineno)
            if function == "os.path.join" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value == "data":
                    bad.append(node.lineno)
    return sorted(set(bad))


def test_no_literal_data_locations():
    found = []
    for folder in ("src", "bench"):
        for path in (ROOT / folder).rglob("*.py"):
            name = path.relative_to(ROOT).as_posix()
            if name == "src/ashare/paths.py":
                continue
            for line in violations(path.read_text(), ALLOW.get(name, ())):
                found.append(f"{name}:{line}")
    assert found == []


def test_detector_catches_split_and_literal_paths():
    assert violations('x = Path("data") / "dev"') == [1]
    assert violations('x = ROOT / "data" / "dev"') == [1]
    assert violations('x = "data/crypto"') == [1]
    assert (
        violations(
            '"""data/dev is documentation"""\nx = "https://data.binance.vision/data/spot"'
        )
        == []
    )


def test_allowlist_only_exempts_exact_expression():
    assert violations('x = root / "data"', {'root / "data"'}) == []
    assert violations('x = "data/dev"', {'root / "data"'}) == [1]
