"""SQLite WAL 元数据、可续传事件和逐试验 NPZ；SQL 值始终使用绑定参数。"""

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .analysis import checks


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "series").mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS entries(
                    id TEXT PRIMARY KEY, expression_id TEXT NOT NULL,
                    name TEXT NOT NULL, source TEXT NOT NULL, settings TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '自定义', status TEXT NOT NULL DEFAULT 'pending',
                    latest TEXT, sharpe REAL, validation REAL, turnover REAL, fitness REAL,
                    annual_return REAL, drawdown REAL, coverage REAL, updated REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS entry_expression ON entries(expression_id);
                CREATE INDEX IF NOT EXISTS entry_sort ON entries(sharpe, id);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, tab TEXT NOT NULL,
                    status TEXT NOT NULL, total INTEGER NOT NULL, created REAL NOT NULL, elapsed REAL);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, job TEXT NOT NULL,
                    entry TEXT NOT NULL, ordinal INTEGER NOT NULL, label TEXT NOT NULL,
                    status TEXT NOT NULL, summary TEXT, version TEXT NOT NULL, data_version TEXT,
                    elapsed REAL, FOREIGN KEY(entry) REFERENCES entries(id));
                CREATE INDEX IF NOT EXISTS run_job ON runs(job, ordinal);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    job TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS event_job ON events(job, seq);
            """)
            for table, additions in {
                "entries": {
                    "family": "TEXT NOT NULL DEFAULT ''",
                    "t": "REAL",
                    "margin": "REAL",
                },
                "runs": {"benchmark": "TEXT"},
            }.items():
                columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                for name, kind in additions.items():
                    if name not in columns:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
            from .language import family_id, parse

            for row in db.execute(
                "SELECT id,source,expression_id FROM entries WHERE family=''"
            ).fetchall():
                try:
                    family = family_id(parse(row["source"])[0].expression)
                except ValueError:
                    family = row["expression_id"]
                db.execute(
                    "UPDATE entries SET family=? WHERE id=?", (family, row["id"])
                )
            db.execute("""UPDATE entries SET t=(SELECT json_extract(summary,'$.is.t') FROM runs WHERE id=entries.latest),
                       margin=(SELECT json_extract(summary,'$.is.margin') FROM runs WHERE id=entries.latest)
                       WHERE latest IS NOT NULL AND (t IS NULL OR margin IS NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS entry_family ON entries(family)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "platform.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover(self):
        with self.connect() as db:
            jobs = db.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
        for job in jobs:
            self.finish(job["id"], "interrupted")

    def begin(self, tab, name, variants, version):
        from .language import family_id, registry

        registered = {factor.expression.digest for factor in registry().values()}
        job = uuid.uuid4().hex
        runs = []
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,NULL)",
                (job, tab, "running", len(variants), time.time()),
            )
            for i, (variant, settings) in enumerate(variants):
                config = dumps(
                    dict(
                        sorted((k, v) for k, v in settings.items() if k != "benchmark")
                    )
                )
                expression_id = variant.expression.digest
                entry = hashlib.sha256((expression_id + config).encode()).hexdigest()
                run = uuid.uuid4().hex
                label = " · ".join(
                    (*variant.parameters, settings["group_neutralize"] or "无")
                )
                db.execute(
                    """INSERT INTO entries(id,expression_id,name,source,settings,updated)
                              VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING""",
                    (entry, expression_id, name, variant.source, config, time.time()),
                )
                db.execute(
                    "UPDATE entries SET family=? WHERE id=?",
                    (family_id(variant.expression), entry),
                )
                if expression_id in registered:
                    db.execute(
                        "UPDATE entries SET category='登记因子' WHERE id=?", (entry,)
                    )
                # 名字属于表达式，不把配置拼进名字；修改同步到同表达式的所有配置。
                db.execute(
                    "UPDATE entries SET name=? WHERE expression_id=?",
                    (name, expression_id),
                )
                db.execute(
                    "INSERT INTO runs(id,job,entry,ordinal,label,status,summary,version,data_version,elapsed,benchmark) VALUES(?,?,?,?,?,'running',NULL,?,NULL,NULL,?)",
                    (
                        run,
                        job,
                        entry,
                        i,
                        label,
                        dumps(version),
                        settings.get("benchmark"),
                    ),
                )
                runs.append(
                    {
                        "id": run,
                        "entry": entry,
                        "label": label,
                        "source": variant.source,
                        "settings": settings,
                    }
                )
            self._event(db, job, {"kind": "started", "total": len(runs)})
        return job, runs

    def rename_tab(self, tab, name):
        """重命名最近一次运行的表达式，同一表达式的配置条目同步。"""
        with self.connect() as db:
            db.execute(
                """UPDATE entries SET name=? WHERE expression_id IN (
                    SELECT e.expression_id FROM entries e JOIN runs r ON r.entry=e.id
                    WHERE r.job=(SELECT id FROM jobs WHERE tab=? ORDER BY created DESC LIMIT 1))""",
                (name, tab),
            )

    @staticmethod
    def _event(db, job, payload):
        db.execute("INSERT INTO events(job,payload) VALUES(?,?)", (job, dumps(payload)))

    def event(self, job, payload):
        with self.connect() as db:
            self._event(db, job, payload)

    def events(self, job, after):
        with self.connect() as db:
            return [
                {"id": r["seq"], **json.loads(r["payload"])}
                for r in db.execute(
                    "SELECT seq,payload FROM events WHERE job=? AND seq>? ORDER BY seq LIMIT 128",
                    (job, after),
                )
            ]

    def trials(self):
        with self.connect() as db:
            return db.execute("SELECT count(*) FROM entries").fetchone()[0]

    def correlations(self, entry, dates, pnl, job=None):
        with self.connect() as db:
            rows = db.execute(
                """SELECT e.id,e.name,e.latest,e.sharpe FROM entries e
                   WHERE e.latest IS NOT NULL AND e.id!=?
                   AND e.family != (SELECT family FROM entries WHERE id=?)
                   AND NOT EXISTS (SELECT 1 FROM runs r WHERE r.entry=e.id AND r.job=?)""",
                (entry, entry, job),
            ).fetchall()
        values = []
        for row in rows:
            with np.load(
                self.root / "series" / f"{row['latest']}.npz", allow_pickle=False
            ) as old:
                _, ix, iy = np.intersect1d(dates, old["is.dates"], return_indices=True)
                x, y = pnl[ix], old["is.pnl.long_short"][iy]
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
                    continue
                value = float(np.corrcoef(x[valid], y[valid])[0, 1])
                values.append(
                    {
                        "entry": row["id"],
                        "name": row["name"],
                        "value": value,
                        "sharpe": row["sharpe"],
                    }
                )
        return sorted(values, key=lambda r: -abs(r["value"]))

    def save(self, run, summary, arrays, elapsed, data_version):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
            if not row or row["status"] != "running":
                return False
        correlation = self.correlations(
            row["entry"], arrays["is.dates"], arrays["is.pnl.long_short"], row["job"]
        )
        summary["correlations"] = correlation[:20]
        summary["checks"] = checks(summary, self.trials(), correlation)
        summary["trial_count"] = self.trials()
        path = self.root / "series" / f"{run}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        a = summary["is"]
        status = (
            "fail" if any(r["status"] == "fail" for r in summary["checks"]) else "pass"
        )
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET status='complete',summary=?,elapsed=?,data_version=? WHERE id=?",
                (dumps(summary), elapsed, data_version, run),
            )
            db.execute(
                """UPDATE entries SET latest=?,status=?,sharpe=?,validation=?,turnover=?,fitness=?,
                       annual_return=?,drawdown=?,coverage=?,updated=? WHERE id=?""",
                (
                    run,
                    status,
                    a["sharpe"],
                    summary["validation"]["sharpe"],
                    a["turnover"],
                    a["fitness"],
                    a["return"],
                    a["maxdrawdown"],
                    a["coverage"],
                    time.time(),
                    row["entry"],
                ),
            )
            db.execute(
                "UPDATE entries SET t=?,margin=? WHERE id=?",
                (a["t"], a["margin"], row["entry"]),
            )
            self._event(
                db,
                row["job"],
                {"kind": "variant", "run": run, "ordinal": row["ordinal"]},
            )
        return True

    def finish(self, job, status, elapsed=None):
        if status == "complete":
            # 结束时刷新独立因子的相关性；同批和数值参数同族始终排除。
            with self.connect() as db:
                completed = db.execute(
                    "SELECT id,entry,summary FROM runs WHERE job=? AND status='complete'",
                    (job,),
                ).fetchall()
            trials = self.trials()
            for row in completed:
                arrays = self.series(row["id"])
                summary = json.loads(row["summary"])
                correlation = self.correlations(
                    row["entry"], arrays["is.dates"], arrays["is.pnl.long_short"], job
                )
                summary.update(
                    correlations=correlation[:20],
                    checks=checks(summary, trials, correlation),
                    trial_count=trials,
                )
                entry_status = (
                    "fail"
                    if any(c["status"] == "fail" for c in summary["checks"])
                    else "pass"
                )
                with self.connect() as db:
                    db.execute(
                        "UPDATE runs SET summary=? WHERE id=?",
                        (dumps(summary), row["id"]),
                    )
                    db.execute(
                        "UPDATE entries SET status=? WHERE id=? AND latest=?",
                        (entry_status, row["entry"], row["id"]),
                    )
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,elapsed=? WHERE id=? AND status='running'",
                (status, elapsed, job),
            )
            db.execute(
                "UPDATE runs SET status=? WHERE job=? AND status='running'",
                (status, job),
            )
            self._event(db, job, {"kind": status, "elapsed": elapsed})

    def job(self, job):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None:
                return None
            runs = db.execute(
                """SELECT r.id,r.ordinal,r.label,r.status,r.summary,r.elapsed,r.benchmark,e.settings
                                 FROM runs r JOIN entries e ON e.id=r.entry WHERE job=? ORDER BY ordinal""",
                (job,),
            ).fetchall()
        return dict(row) | {
            "runs": [
                dict(r)
                | {
                    "summary": json.loads(r["summary"]) if r["summary"] else None,
                    "settings": json.loads(r["settings"])
                    | ({"benchmark": r["benchmark"]} if r["benchmark"] else {}),
                }
                for r in runs
            ]
        }

    def series(self, run):
        # ID 来自数据库；不允许用户提供任意文件路径。
        with self.connect() as db:
            row = db.execute("SELECT status FROM runs WHERE id=?", (run,)).fetchone()
        if row is None or row["status"] != "complete":
            return None
        with np.load(self.root / "series" / f"{run}.npz", allow_pickle=False) as file:
            return {k: file[k] for k in file.files}

    def best(self):
        from .language import registry

        with self.connect() as db:
            names = {
                r[0]: r[1]
                for r in db.execute(
                    "SELECT name,max(sharpe) FROM entries GROUP BY name HAVING max(sharpe) IS NOT NULL"
                )
            }
            expressions = dict(
                db.execute(
                    "SELECT expression_id,max(sharpe) FROM entries GROUP BY expression_id HAVING max(sharpe) IS NOT NULL"
                )
            )
        for factor in registry().values():
            if factor.expression.digest in expressions:
                names[factor.name] = max(
                    names.get(factor.name, -float("inf")),
                    expressions[factor.expression.digest],
                )
        return names

    def entry(self, entry):
        with self.connect() as db:
            row = db.execute(
                "SELECT e.*,r.summary,r.benchmark FROM entries e LEFT JOIN runs r ON r.id=e.latest WHERE e.id=?",
                (entry,),
            ).fetchone()
        if row is None:
            return None
        return dict(row) | {
            "settings": json.loads(row["settings"])
            | ({"benchmark": row["benchmark"]} if row["benchmark"] else {}),
            "summary": json.loads(row["summary"]) if row["summary"] else None,
        }

    def correlation_matrix(self, ids):
        # 只计算当前页能画下的条目，避免全库平方级响应和文件读取。
        if len(ids) > 12 or len(set(ids)) != len(ids):
            raise ValueError("相关矩阵最多选择 12 个不同因子")
        series = {}
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,latest FROM entries WHERE latest IS NOT NULL AND id IN ("
                + ",".join("?" for _ in ids)
                + ")",
                ids,
            ).fetchall()
        for row in rows:
            with np.load(
                self.root / "series" / f"{row['latest']}.npz", allow_pickle=False
            ) as file:
                series[row["id"]] = (file["is.dates"], file["is.pnl.long_short"])
        matrix = [[None for _ in ids] for _ in ids]
        for i, left in enumerate(ids):
            for j in range(i, len(ids)):
                right = ids[j]
                if left not in series or right not in series:
                    continue
                dates, x = series[left]
                other_dates, y = series[right]
                _, ix, iy = np.intersect1d(dates, other_dates, return_indices=True)
                x, y = x[ix], y[iy]
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
                    continue
                value = round(float(np.corrcoef(x[valid], y[valid])[0, 1]), 4)
                matrix[i][j] = matrix[j][i] = value
        return {"ids": ids, "matrix": matrix}

    def library(
        self,
        *,
        search="",
        status="",
        universe="",
        neutralize="",
        category="",
        metric="sharpe",
        minimum=None,
        maximum=None,
        sharpe_min=None,
        turnover_max=None,
        t_min=None,
        sort="updated",
        descending=True,
        group="none",
        offset=0,
        limit=50,
    ):
        columns = {
            "name",
            "source",
            "status",
            "sharpe",
            "validation",
            "turnover",
            "fitness",
            "annual_return",
            "drawdown",
            "coverage",
            "updated",
            "t",
            "margin",
            "universe",
            "neutralize",
            "decay",
        }
        if (
            sort not in columns
            or metric
            not in columns
            - {"name", "source", "status", "updated", "universe", "neutralize", "decay"}
            or group not in ("none", "expression", "config")
        ):
            raise ValueError("不支持的列表选项")
        clauses, args = [], []
        if search:
            clauses.append("(name LIKE ? ESCAPE '\\' OR source LIKE ? ESCAPE '\\')")
            escaped = (
                search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            args += [f"%{escaped}%"] * 2
        for value, sql in [
            (status, "status"),
            (universe, "json_extract(settings,'$.universe')"),
            (
                neutralize,
                "coalesce(json_extract(settings,'$.group_neutralize'),'none')",
            ),
            (category, "category"),
        ]:
            if value:
                values = value.split(",")
                if len(values) > 16:
                    raise ValueError("筛选项过多")
                clauses.append(sql + " IN (" + ",".join("?" for _ in values) + ")")
                args.extend(values)
        for value, op in ((minimum, ">="), (maximum, "<=")):
            if value is not None:
                clauses.append(f"{metric}{op}?")
                args.append(value)
        for column, op, value in (
            ("sharpe", ">=", sharpe_min),
            ("turnover", "<=", turnover_max),
            ("t", ">=", t_min),
        ):
            if value is not None:
                clauses.append(f"{column}{op}?")
                args.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        group_key = {"expression": "expression_id", "config": "settings", "none": None}[
            group
        ]
        sort_sql = {
            "universe": "json_extract(settings,'$.universe')",
            "neutralize": "json_extract(settings,'$.group_neutralize')",
            "decay": "json_extract(settings,'$.decay')",
        }.get(sort, sort)
        order = (
            f"{group_key}," if group_key else ""
        ) + f"{sort_sql} {'DESC' if descending else 'ASC'},id"
        with self.connect() as db:
            # 一个读事务，计数、分组和分页看到同一版筛选结果。
            db.execute("BEGIN")
            facets = {
                key: {} for key in ("status", "universe", "neutralize", "category")
            }
            population = db.execute(
                "SELECT status,json_extract(settings,'$.universe'),coalesce(json_extract(settings,'$.group_neutralize'),'none'),category,count(*) FROM entries"
                + where
                + " GROUP BY 1,2,3,4",
                args,
            )
            for values in population:
                for index, key in enumerate(facets):
                    value = values[index]
                    facets[key][value] = facets[key].get(value, 0) + values[4]
            counts = facets["status"]
            rows = db.execute(
                "SELECT id,name,source,settings,expression_id,status,sharpe,validation,turnover,fitness,annual_return,drawdown,coverage,t,margin,latest FROM entries"
                + where
                + f" ORDER BY {order} LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
            groups = []
            if group_key and rows:
                visible = list(dict.fromkeys(row[group_key] for row in rows))
                group_where = where + (" AND " if where else " WHERE ")
                group_where += f"{group_key} IN ({','.join('?' for _ in visible)})"
                # 只传当前页会画的分组头；每个分组的计数仍基于全部已筛选条目。
                groups = [
                    dict(r)
                    for r in db.execute(
                        f"SELECT {group_key} AS key,count(*) AS count FROM entries"
                        + group_where
                        + f" GROUP BY {group_key}",
                        [*args, *visible],
                    )
                ]
        return {
            "items": [dict(r) | {"settings": json.loads(r["settings"])} for r in rows],
            "total": sum(counts.values()),
            "counts": counts,
            "groups": groups,
            "facets": facets,
            "offset": offset,
            "limit": limit,
        }
