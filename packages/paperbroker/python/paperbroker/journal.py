"""Crash-safe WAL. Quote blobs are durable before their references enter the log."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ._core import BrokerCore

if TYPE_CHECKING:
    from .api import Broker, QuoteArray


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sync_file(fd: int) -> None:
    """Flush through the device cache on macOS; propagate unsupported-drive errors."""
    os.fsync(fd)
    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, getattr(fcntl, "F_FULLFSYNC", 51))


def durable_write(path: Path, data: bytes) -> None:
    """Atomically replace a file and fsync both the file and its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        sync_file(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class QuoteStore(Protocol):
    directory: Path

    def put(self, sequence: int, quotes: QuoteArray) -> dict[str, Any]: ...
    def get(self, reference: dict[str, Any]) -> QuoteArray: ...


class NumpyQuoteStore:
    """Dependency-free immutable recordings for the standalone package."""

    def __init__(self, directory: Path):
        self.directory = directory

    def put(self, sequence: int, quotes: QuoteArray) -> dict[str, Any]:
        stream = io.BytesIO()
        np.savez_compressed(stream, quotes=quotes)
        data = stream.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        name = f"{sequence:012d}-{digest}.npz"
        durable_write(self.directory / name, data)
        return {"file": name, "sha256": digest}

    def get(self, reference: dict[str, Any]) -> QuoteArray:
        name = reference["file"]
        if Path(name).name != name:
            raise ValueError("unsafe quote reference")
        data = (self.directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != reference["sha256"]:
            raise ValueError("quote checksum mismatch")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            return archive["quotes"]


class Journal:
    """Single-writer journal; incomplete tails are truncated, corruption fails closed."""

    def __init__(
        self,
        directory: Path,
        broker: Broker,
        store: QuoteStore | None = None,
        *,
        read_only: bool = False,
    ):
        import fcntl

        self.directory, self.broker = directory, broker
        self.read_only = read_only
        if not read_only:
            directory.mkdir(parents=True, exist_ok=True)
        self.store = store or NumpyQuoteStore(directory / "quotes")
        self.handle = (directory / "events.jsonl").open("rb" if read_only else "a+b")
        self.sequence, self.digest, self.broken = 0, "", False
        try:
            if not read_only:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            broker.journal = self
            self._recover()
        except BaseException:
            self.handle.close()
            raise

    def _entries(self):
        self.handle.seek(0)
        for line in self.handle:
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)

    def _recover(self) -> None:
        # Stream the immutable chain: startup memory is independent of WAL age.
        self.handle.seek(0)
        end = 0
        for line in self.handle:
            if not line.endswith(b"\n"):
                break
            end += len(line)
        self.handle.seek(0, os.SEEK_END)
        if end != self.handle.tell() and not self.read_only:
            self.handle.truncate(end)
            self.handle.flush()
            sync_file(self.handle.fileno())
        snapshots = []
        for path in sorted(self.directory.glob("state-*.json"), reverse=True):
            try:
                snap = json.loads(path.read_bytes())
                payload = snap["payload"]
                if (
                    payload.get("version") == 3
                    and hashlib.sha256(canonical(payload)).hexdigest() == snap["sha256"]
                ):
                    snapshots.append(payload)
            except (ValueError, KeyError, TypeError, FileNotFoundError):
                continue
        heads = {0: ""}
        wanted = {p["seq"] for p in snapshots}
        digest, count = "", 0
        for index, entry in enumerate(self._entries(), 1):
            body = entry["body"]
            actual = hashlib.sha256(digest.encode() + canonical(body)).hexdigest()
            if (
                body.get("version") != 3
                or body["seq"] != index
                or entry["sha256"] != actual
            ):
                raise ValueError("journal checksum/sequence mismatch")
            digest, count = actual, index
            if index in wanted:
                heads[index] = actual
        manifest = self.directory / "manifest.json"
        identity = {
            "version": 3,
            "symbols": self.broker.symbols,
            "config": json.loads(self.broker.dump())["config"],
        }
        if manifest.exists():
            if json.loads(manifest.read_bytes()) != identity:
                raise ValueError("journal symbols/configuration mismatch")
        else:
            if count or self.read_only:
                raise ValueError("missing journal manifest")
            durable_write(manifest, canonical(identity))
        floor_path = self.directory / "gc.json"
        floor = (
            json.loads(floor_path.read_bytes())["recovery_floor"]
            if floor_path.exists()
            else 0
        )
        skip = 0
        for payload in snapshots:
            seq = payload["seq"]
            if not floor <= seq <= count or payload["head"] != heads.get(seq):
                continue
            try:
                engine = BrokerCore.restore(payload["state"])
                self.broker._core = engine
                self.broker.daily_states = payload["daily_states"]
                self.broker.history = payload["outputs"]
                self.broker._pending = list(payload["outputs"])
                self.broker.request_id = payload["request_id"]
                skip = seq
                break
            except (ValueError, KeyError, TypeError, FileNotFoundError):
                continue
        if skip < floor:
            raise ValueError(
                "retained quotes require a valid post-GC snapshot; restore an archive"
            )
        for entry in self._entries():
            body = entry["body"]
            if body["seq"] > count:
                break
            if body["seq"] <= skip:
                continue
            if body["kind"] == "quote":
                self.broker._core.feed(
                    self.store.get(body["reference"]), body["received_at"]
                )
            else:
                self.broker._core.apply(body["event"])
                self.broker.request_id += 1
            self.broker._capture()
        # A crash may have occurred after snapshot publication but before archive publication.
        if not self.read_only:
            for summary in self.broker.daily_states.values():
                self.archive(summary)
        self.sequence, self.digest = count, digest
        self.handle.seek(0, os.SEEK_END)

    def archive(self, summary: dict) -> None:
        path = self.directory / "days" / f"{summary['day']}.json"
        if not path.exists():
            durable_write(path, canonical(summary))

    def _append(self, body: dict[str, Any]) -> None:
        if self.read_only:
            raise RuntimeError("read-only journal")
        if self.broken:
            raise RuntimeError("journal failed; close and recover before continuing")
        body = body | {"version": 3, "seq": self.sequence + 1}
        digest = hashlib.sha256(self.digest.encode() + canonical(body)).hexdigest()
        self.handle.seek(0, os.SEEK_END)
        size = self.handle.tell()
        if size:
            self.handle.seek(-1, os.SEEK_END)
            if self.handle.read(1) != b"\n":
                self.broken = True
                raise ValueError("journal does not end with newline; recover first")
        try:
            self.handle.write(canonical({"body": body, "sha256": digest}) + b"\n")
            self.handle.flush()
            sync_file(self.handle.fileno())
        except BaseException:
            self.broken = True
            raise
        self.sequence += 1
        self.digest = digest

    def apply(self, encoded: str) -> None:
        # Validate on a disposable engine before committing malformed input.
        trial = BrokerCore.restore(self.broker._core.dump())
        trial.apply(encoded)
        self._append({"kind": "event", "event": encoded})
        try:
            self.broker._core.apply(encoded)
        except BaseException:
            self.broken = True
            raise

    def feed(self, quotes: QuoteArray, received_at: int | None = None) -> None:
        BrokerCore.validate_quotes(quotes, len(self.broker.symbols))
        if self.read_only:
            raise RuntimeError("read-only journal")
        if (
            received_at is not None
            and received_at < json.loads(self.broker.dump())["time"]
        ):
            raise ValueError("receipt time precedes engine time")
        # Execute on a disposable engine before either quotes or WAL are written.
        trial = BrokerCore.restore(self.broker._core.dump())
        trial.feed(quotes, received_at)
        reference = self.store.put(self.sequence + 1, quotes)
        self._append(
            {
                "kind": "quote",
                "reference": reference,
                "received_at": received_at,
                "day": self.broker.account()["day"],
            }
        )
        try:
            self.broker._core.feed(quotes, received_at)
        except BaseException:
            self.broken = True
            raise

    def checkpoint(self) -> None:
        if self.read_only:
            raise RuntimeError("read-only journal")
        payload = {
            "version": 3,
            "outputs": self.broker.history,
            "daily_states": self.broker.daily_states,
            "request_id": self.broker.request_id,
            "seq": self.sequence,
            "head": self.digest,
            "state": self.broker._core.dump(),
        }
        path = self.directory / f"state-{self.sequence:012d}.json"
        durable_write(
            path,
            canonical(
                {
                    "payload": payload,
                    "sha256": hashlib.sha256(canonical(payload)).hexdigest(),
                }
            ),
        )
        # Validate the published snapshot before retiring any previous recovery point.
        loaded = json.loads(path.read_bytes())
        if hashlib.sha256(canonical(loaded["payload"])).hexdigest() != loaded["sha256"]:
            raise ValueError("published snapshot checksum mismatch")
        BrokerCore.restore(loaded["payload"]["state"])
        paths = sorted(self.directory.glob("state-*.json"), reverse=True)
        for old in paths[self.broker.snapshot_retention :]:
            old.unlink()

    def gc(self) -> dict[str, int]:
        """Retire old recordings only behind all retained, validated snapshots.

        The WAL is never deleted. After quote GC, replay requires a retained
        snapshot; full historical replay needs a separately retained recording.
        """
        self.checkpoint()
        keep = self.broker.quote_retention_days
        if not keep:
            return {"quotes_deleted": 0, "bytes_deleted": 0}
        snapshots = []
        for path in self.directory.glob("state-*.json"):
            snap = json.loads(path.read_bytes())
            if hashlib.sha256(canonical(snap["payload"])).hexdigest() != snap["sha256"]:
                raise ValueError("invalid snapshot during GC")
            BrokerCore.restore(snap["payload"]["state"])
            snapshots.append(snap["payload"]["seq"])
        safe = min(snapshots)
        days = set()
        for entry in self._entries():
            body = entry["body"]
            if body["kind"] == "quote":
                days.add(body["day"])
                if len(days) > keep:
                    days.remove(min(days))
        # Publish the recovery floor before removing the first dependency.
        previous = self.directory / "gc.json"
        old_floor = (
            json.loads(previous.read_bytes())["recovery_floor"]
            if previous.exists()
            else 0
        )
        deleted, size = 0, 0
        for entry in self._entries():
            body = entry["body"]
            if body["kind"] != "quote" or body["day"] in days or body["seq"] > safe:
                continue
            name = body["reference"]["file"]
            if Path(name).name != name:
                raise ValueError("unsafe quote reference")
            path = self.store.directory / name
            if not path.exists():
                continue
            if deleted == 0:
                durable_write(
                    previous,
                    canonical({"version": 3, "recovery_floor": max(old_floor, safe)}),
                )
            size += path.stat().st_size
            path.unlink()
            deleted += 1
        self.handle.seek(0, os.SEEK_END)
        return {"quotes_deleted": deleted, "bytes_deleted": size}

    def close(self) -> None:
        self.handle.close()
