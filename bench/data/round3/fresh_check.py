"""在全新临时目录安装并运行项目检查；测试默认数据根及密钥环境完全隔离。"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(
        prefix=".fresh-", dir=Path(__file__).parent
    ) as temporary:
        clone = Path(temporary)
        ignored = {
            ".git",
            ".venv",
            "target",
            ".cache",
            ".pytest_cache",
            ".ruff_cache",
            ".hypothesis",
            "__pycache__",
            ".DS_Store",
        }

        def ignore(directory, names):
            return [
                n
                for n in names
                if n in ignored
                or n.startswith(".fresh-")
                or n.endswith((".pyc", ".so", ".dylib"))
                or (Path(directory) == root and n == "data")
            ]

        for path in root.iterdir():
            if path.name in ignore(root, [path.name]):
                continue
            if path.is_dir():
                shutil.copytree(path, clone / path.name, ignore=ignore)
            else:
                shutil.copy2(path, clone / path.name)
        env = dict(os.environ, ASHARE_DATA_DIR=str(clone / "isolated-test-data"))
        env.pop("TUSHARE_TOKEN", None)
        env.pop("PYTHONPATH", None)
        subprocess.run(["make", "dev"], cwd=clone, env=env, check=True)
        subprocess.run(["make", "check"], cwd=clone, env=env, check=True)
        assert not (clone / "isolated-test-data").exists()
        print(
            "Fresh copy: make dev && make check passed; default test data root untouched."
        )


if __name__ == "__main__":
    main()
