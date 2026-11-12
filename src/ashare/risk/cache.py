"""一代、无损、按需重建暴露的日模型缓存。"""

import argparse
import json
import os
import re
from pathlib import Path
from uuid import uuid4

import numpy as np

from .model import RiskModel


def _snapshot(path):
    return re.fullmatch(r"(?:[0-9]{6,}-)?[a-f0-9]{64}\.npz", path.name) is not None


STATE_FIELDS = ("f", "residual", "base_f", "base_d", "bias", "estimation", "cap")


class ModelCache:
    """历史保存 F/D 和递推状态，B 只保留最新一天；历史 B 由调用方按需重建。"""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        for path in self.directory.glob("*.npz"):
            parts = path.stem.split("-", 1)
            if _snapshot(path) and len(parts) == 2 and parts[0].isdigit():
                self.entries[int(parts[0])] = parts[1]
        marker = self.directory / ".generation.json"
        self.generation = (
            json.loads(marker.read_text())["generation"] if marker.exists() else None
        )

    def _json(self, name, value):
        path = self.directory / name
        temporary = self.directory / f".{uuid4().hex}.tmp"
        temporary.write_text(json.dumps(value, ensure_ascii=False))
        os.replace(temporary, path)

    def prepare(self, generation):
        if self.generation != generation:
            self.clear()
            self.generation = generation
            self._json(".generation.json", {"generation": generation, "schema": 3})

    def clear(self):
        """删除本目录内缓存文件；不递归删除调用方的其他目录。"""
        removed = 0
        for path in self.directory.iterdir():
            if path.is_file() and (
                _snapshot(path)
                or path.name
                in (
                    ".latest.npy",
                    ".latest.json",
                    ".checkpoint.cache",
                    ".generation.json",
                )
            ):
                removed += path.stat().st_size
                path.unlink()
        self.entries.clear()
        self.generation = None
        return removed

    def stats(self):
        files = [
            p
            for p in self.directory.iterdir()
            if p.is_file()
            and (
                _snapshot(p)
                or p.name
                in (
                    ".latest.npy",
                    ".latest.json",
                    ".checkpoint.cache",
                    ".generation.json",
                )
            )
        ]
        return {
            "snapshots": len(self.entries),
            "bytes": sum(p.stat().st_size for p in files),
            "latest_exposure_bytes": (self.directory / ".checkpoint.cache")
            .stat()
            .st_size
            if (self.directory / ".checkpoint.cache").exists()
            else 0,
            "generations": int(self.generation is not None),
        }

    def load(self, key, *, day=None, with_model=True, exposure_loader=None):
        if day is None:
            day = next((i for i, value in self.entries.items() if value == key), None)
        if day is None:
            return None
        if day in self.entries and self.entries[day] != key:
            # 历史修订使其后所有前缀失效；立即清理旧尾部，不留下第二代。
            for i in [i for i in self.entries if i >= day]:
                (self.directory / f"{i:06d}-{self.entries.pop(i)}.npz").unlink(
                    missing_ok=True
                )
            (self.directory / ".latest.npy").unlink(missing_ok=True)
            (self.directory / ".latest.json").unlink(missing_ok=True)
            (self.directory / ".checkpoint.cache").unlink(missing_ok=True)
        path = self.directory / f"{day:06d}-{key}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            state = {name: data[name].copy() for name in STATE_FIELDS}
            model = None
            if with_model:
                latest = self._latest(key)
                if latest is not None:
                    b = latest["B"]
                elif exposure_loader is not None:
                    b = exposure_loader()
                else:
                    raise ValueError(
                        "历史 B 按需重建：请提供 exposure_loader 或使用 model_series"
                    )
                meta = json.loads(str(data["meta"]))
                model = RiskModel(
                    tuple(data["symbols"].tolist()),
                    b,
                    tuple(data["factors"].tolist()),
                    data["F"],
                    data["D"],
                    meta["date"],
                    meta["metadata"],
                )
        return model, state

    def save(self, key, model, state, *, day, latest=True, checkpoint=None):
        path = self.directory / f"{day:06d}-{key}.npz"
        temporary = self.directory / f".{uuid4().hex}.npz"
        try:
            np.savez_compressed(
                temporary,
                symbols=np.asarray(model.symbols),
                factors=np.asarray(model.factors),
                F=model.F,
                D=model.D,
                meta=json.dumps({"date": model.date, "metadata": model.metadata}),
                **state,
            )
            os.replace(temporary, path)
            if latest:
                exposure = self.directory / f".{uuid4().hex}.tmp"
                values = (
                    {}
                    if checkpoint is None
                    else {
                        "rolling_" + name: value for name, value in checkpoint.items()
                    }
                )
                with exposure.open("wb") as stream:
                    np.savez_compressed(stream, key=key, B=model.B, **values)
                os.replace(exposure, self.directory / ".checkpoint.cache")
            self.entries[day] = key
        finally:
            temporary.unlink(missing_ok=True)

    def _latest(self, key):
        path = self.directory / ".checkpoint.cache"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != key:
                return None
            return {name: data[name] for name in data.files}

    def checkpoint(self, key):
        latest = self._latest(key)
        if latest is None:
            return None
        values = {
            name.removeprefix("rolling_"): value
            for name, value in latest.items()
            if name.startswith("rolling_")
        }
        return values or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stats", "clear"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    cache = ModelCache(args.directory)
    print(
        json.dumps(
            cache.stats()
            if args.action == "stats"
            else {"removed_bytes": cache.clear()}
        )
    )


if __name__ == "__main__":
    main()
