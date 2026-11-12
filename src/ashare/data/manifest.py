"""SQLite 事务清单；旧 JSON 原样保留，迁移与每次提交均可崩溃恢复。"""

import json
import sqlite3
import threading


class Manifest:
    def __init__(self, root, *, readonly=False):
        self.path = root / "manifest.sqlite3"
        self.guard = threading.RLock()
        self.readonly = readonly
        self.cache, self.epoch = None, None
        if readonly:
            if not self.path.exists():
                legacy = root / "manifest.json"
                self.db = None
                self.cache = json.loads(legacy.read_text()) if legacy.exists() else {}
                return
            # 已关闭的库不创建 WAL/SHM；有活跃 WAL 时使用 SQLite 的只读快照。
            wal = root / "manifest.sqlite3-wal"
            suffix = "" if wal.exists() and wal.stat().st_size else "&immutable=1"
            self.db = sqlite3.connect(
                self.path.as_uri() + "?mode=ro" + suffix,
                uri=True,
                check_same_thread=False,
            )
            return
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA fullfsync=ON")
        self.db.execute("PRAGMA checkpoint_fullfsync=ON")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS partitions (key TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        with self.db:
            # BEGIN IMMEDIATE 防止两个进程同时导入；完成标志与数据同事务提交。
            self.db.execute("BEGIN IMMEDIATE")
            if not self.db.execute(
                "SELECT 1 FROM metadata WHERE key='imported'"
            ).fetchone():
                legacy = root / "manifest.json"
                if legacy.exists():
                    records = json.loads(legacy.read_text())
                    self.db.executemany(
                        "INSERT OR IGNORE INTO partitions VALUES (?, ?)",
                        (
                            (k, json.dumps(v, ensure_ascii=False))
                            for k, v in records.items()
                        ),
                    )
                self.db.execute("INSERT INTO metadata VALUES ('imported', '1')")
        self.cache = None
        self.epoch = None

    def read(self):
        with self.guard:
            if self.db is None:
                return self.cache
            epoch = self.db.execute("PRAGMA data_version").fetchone()[0]
            if self.cache is None or self.epoch != epoch:
                self.cache = {
                    k: json.loads(v)
                    for k, v in self.db.execute("SELECT key, body FROM partitions")
                }
                self.epoch = epoch
            return self.cache

    def put(self, key, changes):
        if self.readonly:
            raise PermissionError("只读数据句柄不能修改清单")
        with self.guard:
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                row = self.db.execute(
                    "SELECT body FROM partitions WHERE key=?", (key,)
                ).fetchone()
                entry = {**(json.loads(row[0]) if row else {}), **changes}
                self.db.execute(
                    "INSERT OR REPLACE INTO partitions VALUES (?, ?)",
                    (key, json.dumps(entry, ensure_ascii=False)),
                )
            self.read()[key] = entry
            return entry

    def predecessor(self, name, partition):
        with self.guard:
            row = self.db.execute(
                "SELECT body FROM partitions WHERE key>=? AND key<? ORDER BY key DESC LIMIT 1",
                (name + "/", name + "/" + partition),
            ).fetchone()
            return json.loads(row[0]) if row else {}
