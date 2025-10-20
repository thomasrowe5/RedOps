"""Audit log helpers for orchestrator runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# The orchestrator stores run state underneath ``data/runs`` relative to the
# repository root.  We intentionally resolve this path at runtime so that the
# audit helper can be used independently of the FastAPI application module.
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = DATA_DIR / "runs"

AUDIT_FILENAME = "audit.ndjson"
IO_TIMEOUT_SECONDS = 5.0

_RUN_LOCKS: Dict[str, asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()


async def _get_run_lock(run_id: str) -> asyncio.Lock:
    async with _LOCKS_GUARD:
        lock = _RUN_LOCKS.get(run_id)
        if lock is None:
            lock = asyncio.Lock()
            _RUN_LOCKS[run_id] = lock
    return lock


async def _ensure_run_dir(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    async with asyncio.timeout(IO_TIMEOUT_SECONDS):
        await asyncio.to_thread(run_dir.mkdir, parents=True, exist_ok=True)
    return run_dir


async def _audit_path(run_id: str, *, create: bool) -> Path:
    if create:
        run_dir = await _ensure_run_dir(run_id)
    else:
        run_dir = RUNS_DIR / run_id
    return run_dir / AUDIT_FILENAME


async def _path_exists(path: Path) -> bool:
    async with asyncio.timeout(IO_TIMEOUT_SECONDS):
        return await asyncio.to_thread(path.exists)


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_dict(payload: Dict[str, Any]) -> str:
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def _read_last_record(path: Path) -> Optional[Dict[str, Any]]:
    if not await _path_exists(path):
        return None

    async with asyncio.timeout(IO_TIMEOUT_SECONDS):
        def _reader() -> Optional[Dict[str, Any]]:
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle.readlines() if line.strip()]
            if not lines:
                return None
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                return None

        return await asyncio.to_thread(_reader)


async def _append_record(path: Path, record: Dict[str, Any]) -> None:
    line = _canonical_json(record) + "\n"

    def _writer() -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    async with asyncio.timeout(IO_TIMEOUT_SECONDS):
        await asyncio.to_thread(_writer)


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


async def write_audit(run_id: str, actor: str, action: str, payload: Dict[str, Any]) -> None:
    """Append an audit record for ``run_id`` with hash chaining."""

    lock = await _get_run_lock(run_id)
    async with lock:
        path = await _audit_path(run_id, create=True)
        last_record = await _read_last_record(path)
        prev_hash = last_record.get("hash") if last_record else None

        payload_sha = _hash_dict(payload)
        record_body = {
            "ts": _utc_timestamp(),
            "actor": actor,
            "action": action,
            "payload_sha256": payload_sha,
            "prev_hash": prev_hash,
        }
        record_hash = _hash_dict(record_body)
        record = {**record_body, "hash": record_hash}
        await _append_record(path, record)


async def verify_audit(run_id: str) -> bool:
    """Verify the append-only hash chain for ``run_id``."""

    path = await _audit_path(run_id, create=False)
    if not await _path_exists(path):
        return True

    lock = await _get_run_lock(run_id)
    async with lock:
        async with asyncio.timeout(IO_TIMEOUT_SECONDS):
            try:
                contents = await asyncio.to_thread(path.read_text, encoding="utf-8")
            except FileNotFoundError:
                return True

        previous_hash: Optional[str] = None
        for raw_line in contents.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                record: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                return False

            body = {
                "ts": record.get("ts"),
                "actor": record.get("actor"),
                "action": record.get("action"),
                "payload_sha256": record.get("payload_sha256"),
                "prev_hash": record.get("prev_hash"),
            }
            expected_hash = _hash_dict(body)
            if record.get("hash") != expected_hash:
                return False

            if body["prev_hash"] != previous_hash:
                return False

            previous_hash = record.get("hash")

        return True


__all__ = ["verify_audit", "write_audit"]

