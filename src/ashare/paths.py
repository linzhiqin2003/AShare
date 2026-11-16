"""数据位置的唯一入口；默认只解析，搬迁必须显式调用 move。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
POINTER = "LOCATION.json"


@dataclass(frozen=True)
class Dataset:
    hot: bool
    env: str | None = None


# 未在设计中列冷的数据按热数据保护；新增数据集必须在此登记。
DATASETS = MappingProxyType(
    {
        "cn": Dataset(True),
        "live": Dataset(True),
        "paper": Dataset(True, "ASHARE_PAPER_DIR"),
        "platform": Dataset(True, "ASHARE_PLATFORM_DIR"),
        "mining": Dataset(True, "ASHARE_MINING_DIR"),
        "research": Dataset(True),
        "dev": Dataset(False),
        "crypto": Dataset(False, "CRYPTO_DATA_ROOT"),
        "crypto_minute": Dataset(False),
        "crypto_metrics": Dataset(False),
        "ml": Dataset(True, "ASHARE_ML_DIR"),
        "hk": Dataset(True),
        "us": Dataset(True),
        "foreign": Dataset(True),
        "fixtures": Dataset(True),
    }
)


class DataLocationError(RuntimeError):
    """位置配置错误或外置数据不可用；调用方不得捕获后退回默认目录。"""

    def __init__(self, message, *, path=None, redirected=False):
        super().__init__(message)
        self.path = path
        self.redirected = redirected


def _spec(name):
    if name not in DATASETS:
        raise DataLocationError(f"未登记的数据集：{name!r}")
    return DATASETS[name]


def _test_guard(path):
    # 沿用股票库的保护，并覆盖重定向后的目标，防止测试意外写真实库。
    if "pytest" in sys.modules:
        repository = Path(__file__).resolve().parents[2]
        forbidden = {p / "data" for p in (repository, *repository.parents)}
        if any(path == p or path.is_relative_to(p) for p in forbidden):
            raise RuntimeError(
                "测试禁止访问真实数据根目录；请传 tmp_path 或设置临时 ASHARE_DATA_DIR"
            )
    return path


def _env(name):
    """读位置类环境变量：未设置返回 None；设了但为空直接报错，不静默换位置。

    来源：D1 审查第二轮 P1。旧代码对空值的处理各入口不一（有的退回 cwd、有的视同未设），
    任何一种都会让数据悄悄落到另一个目录，所以统一拒绝。
    """
    value = os.environ.get(name)
    if value is not None and not value.strip():
        raise DataLocationError(
            f"环境变量 {name} 设了但为空：要么不设，要么写成一个目录"
        )
    return value


def _checked_path(value, *, cwd_default=False):
    path = Path(value).expanduser()
    absolute = _test_guard(path.resolve())
    return path if cwd_default else absolute


def data_root(root=None, *, cwd_default=False) -> Path:
    """整根目录（兼容股票 root 参数）；不代替 dataset_dir 解析数据集。"""
    value = (
        root
        if root is not None
        else _env("ASHARE_DATA_DIR")
        or (Path("data") if cwd_default else DEFAULT_DATA_ROOT)
    )
    return _checked_path(value, cwd_default=cwd_default)


def _location(name, path=None, *, root=None, cwd_default=False):
    spec = _spec(name)
    if path is not None:
        return _checked_path(path, cwd_default=cwd_default), False
    legacy = _env(spec.env) if spec.env else None
    if root is None and legacy:
        return _checked_path(legacy, cwd_default=cwd_default), False
    folder = data_root(root, cwd_default=cwd_default) / name
    pointer = folder / POINTER
    # lexists 也识别断开的软链接，不能把它当成没有配置。
    if not os.path.lexists(pointer):
        return folder, False
    try:
        value = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            k in value for k in ("path", "moved_at", "files", "bytes")
        ):
            raise ValueError("缺少 path、moved_at、files 或 bytes 字段")
        if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
            raise ValueError("path 必须是绝对路径")
        if not isinstance(value["moved_at"], str):
            raise ValueError("moved_at 必须是时间字符串")
        datetime.fromisoformat(value["moved_at"])
        if any(type(value[k]) is not int or value[k] < 0 for k in ("files", "bytes")):
            raise ValueError("files、bytes 必须是非负整数")
        target = Path(value["path"])
    except (OSError, ValueError, TypeError) as exc:
        raise DataLocationError(
            f"数据集 {name} 的指针无效：{pointer}（{exc}）",
            path=folder,
            redirected=True,
        ) from exc
    return _test_guard(target.resolve()), True


def dataset_dir(
    name: str, path=None, *, root=None, create=False, cwd_default=False
) -> Path:
    """显式数据集路径 > 兼容变量 > 所选根下指针 > 所选根下目录。

    root 为旧 API 的整根参数，优先于环境变量；替换整根后仍解析该根下的指针。
    显式 path 和兼容变量直指数据集，跳过指针。只有非指针位置允许 create。
    cwd_default 仅供旧的 cwd 相对入口使用，保留相对路径表示；不改变覆盖优先级。
    """
    folder, redirected = _location(name, path, root=root, cwd_default=cwd_default)
    if redirected and not folder.is_dir():
        raise DataLocationError(
            f"数据集 {name} 的目标目录不可用，期望路径：{folder}；请挂载外置盘，禁止回退或自动创建"
        )
    if create and not redirected:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def bench_data_root(path=None, *, checkout=None) -> Path:
    """兼容基准环境变量；旧端到端脚本可指定 checkout 保留主检出回退。"""
    selected = path if path is not None else _env("ASHARE_BENCH_DATA")
    if selected is not None or _env("ASHARE_DATA_DIR") or checkout is None:
        return data_root(selected)
    root = data_root(Path(checkout) / "data")
    # 显式指针优先；失效的外置位置不能被主检出回退掩盖。
    has_pointer = os.path.lexists(root / "cn" / POINTER)
    market = dataset_dir("cn", root=root)
    if has_pointer or (market / "panel/current.json").exists():
        return root
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout.strip()
    return data_root(Path(common).parent / "data")


def bench_ml_dir(*, root=None) -> Path:
    """ML 基准沿用 dev/ml 子目录和 ASHARE_ML_DIR 的覆盖语义。"""
    override = _env("ASHARE_ML_DIR")
    if override:
        return dataset_dir("ml", override)
    if root is None:
        root = _env("ASHARE_BENCH_DATA")
    return dataset_dir("dev", root=root) / "ml"


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory(folder):
    """递归核对清单包含空目录；拒绝软链接与设备文件，不跟随外部树。"""
    entries = {}
    for base, dirs, files in os.walk(
        folder, followlinks=False, onerror=lambda e: _raise(e)
    ):
        for name in sorted([*dirs, *files]):
            child = Path(base) / name
            mode = child.lstat().st_mode
            relative = child.relative_to(folder).as_posix()
            if stat.S_ISDIR(mode):
                entries[relative] = None
            elif stat.S_ISREG(mode):
                entries[relative] = (child.stat().st_size, _sha256(child))
            else:
                raise DataLocationError(f"搬迁拒绝符号链接或特殊文件：{child}")
    return entries


def _raise(error):
    raise error


def _verify_copy(source, target, before):
    if _inventory(source) != before or _inventory(target) != before:
        raise DataLocationError(
            "搬迁核对失败：文件大小、SHA-256 或目录清单不一致（源数据也不得变化）"
        )


def _write_pointer(source, value):
    fd, temporary = tempfile.mkstemp(prefix=".LOCATION-", suffix=".tmp", dir=source)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, source / POINTER)
    finally:
        Path(temporary).unlink(missing_ok=True)


def move(name, target, *, root=None, delete_source=False, force=False):
    """复制、逐文件核对后才发布指针；调用前必须停止对该数据集的写入。"""
    spec = _spec(name)
    if spec.hot and not force:
        raise DataLocationError(
            f"数据集 {name} 是热数据，拒绝搬迁；确需搬迁请显式使用 --force"
        )
    if root is None and spec.env and _env(spec.env):
        raise DataLocationError(f"数据集 {name} 由 {spec.env} 覆盖，请先取消覆盖再搬迁")
    base = data_root(root)
    source = base / name
    if os.path.lexists(source / POINTER):
        raise DataLocationError(
            f"数据集 {name} 已有指针，拒绝重复搬迁：{source / POINTER}"
        )
    if source.is_symlink() or not source.is_dir():
        raise DataLocationError(f"数据集 {name} 的源目录无效：{source}")
    destination = Path(target).expanduser()
    if destination.is_symlink():
        raise DataLocationError(f"目标不能是符号链接：{destination}")
    destination = _test_guard(destination.resolve())
    if destination.is_relative_to(base) or base.is_relative_to(destination):
        raise DataLocationError("目标必须在数据根目录之外，且不能与源目录互相包含")
    # 父目录必须已存在，拔盘时不在挂载点下自动造目录。
    if not destination.parent.is_dir():
        raise DataLocationError(
            f"目标父目录不存在，请先挂载外置盘：{destination.parent}"
        )
    existed = destination.exists()
    if existed and (not destination.is_dir() or any(destination.iterdir())):
        raise DataLocationError(f"目标目录已存在且非空或不是目录：{destination}")
    lock = base / f".move-{name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise DataLocationError(f"数据集 {name} 正在搬迁（锁：{lock}）") from exc
    os.close(fd)
    owned = False
    try:
        if os.path.lexists(source / POINTER):
            raise DataLocationError(
                f"数据集 {name} 已有指针，拒绝重复搬迁：{source / POINTER}"
            )
        if source.is_symlink() or not source.is_dir():
            raise DataLocationError(f"数据集 {name} 的源目录已变化：{source}")
        before = _inventory(source)
        if existed:
            destination.rmdir()  # 再次保证是空目录；失败时不取得其所有权。
        destination.mkdir()
        owned = True
        shutil.copytree(
            source, destination, dirs_exist_ok=True, copy_function=shutil.copy2
        )
        _verify_copy(source, destination, before)
        files = [v for v in before.values() if v is not None]
        value = dict(
            path=str(destination),
            moved_at=datetime.now(UTC).isoformat(),
            files=len(files),
            bytes=sum(v[0] for v in files),
        )
        _write_pointer(source, value)
        if delete_source:
            for child in source.iterdir():
                if child.name != POINTER:
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
        return value
    finally:
        # 检查实际发布状态，不能在 replace 与内存标志赋值间留下中断窗口。
        # 即使出现非预期的指针也保守保留目标；读者可能已开始使用它。
        if owned and not os.path.lexists(source / POINTER):
            shutil.rmtree(destination)
            if existed:
                destination.mkdir()
        lock.unlink(missing_ok=True)


def status(*, root=None):
    """逐数据集报告；不可读项目也保留一行，不做写操作。"""
    rows = []
    for name in DATASETS:
        row = dict(name=name, path="?", bytes=None, readable=False, redirected=False)
        try:
            folder, redirected = _location(name, root=root)
            row.update(path=str(folder), redirected=redirected)
            folder = dataset_dir(name, root=root)
            if not folder.is_dir():
                raise DataLocationError(f"目录不存在：{folder}")
            total = 0
            for base, _, files in os.walk(folder, onerror=_raise):
                for filename in files:
                    file = Path(base) / filename
                    with file.open("rb"):
                        total += file.stat().st_size
            row.update(bytes=total, readable=True)
        except (OSError, DataLocationError) as exc:
            row["error"] = str(exc)
            if isinstance(exc, DataLocationError) and exc.path is not None:
                row.update(path=str(exc.path), redirected=exc.redirected)
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="替换整根数据目录（通常使用 ASHARE_DATA_DIR）"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    mover = sub.add_parser("move")
    mover.add_argument("name", choices=DATASETS)
    mover.add_argument("target", type=Path)
    mover.add_argument("--delete-source", action="store_true")
    mover.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "move":
            print(
                json.dumps(
                    move(
                        args.name,
                        args.target,
                        root=args.root,
                        delete_source=args.delete_source,
                        force=args.force,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        rows = status(root=args.root)
        print("数据集\t位置\t大小（字节）\t可读\t重定向")
        for row in rows:
            print(
                f"{row['name']}\t{row['path']}\t{row['bytes'] if row['bytes'] is not None else '?'}\t{'是' if row['readable'] else '否'}\t{'是' if row['redirected'] else '否'}"
            )
            if "error" in row:
                print(f"  {row['error']}")
        return int(any(r["redirected"] and not r["readable"] for r in rows))
    except (DataLocationError, OSError) as exc:
        print(f"数据位置错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
