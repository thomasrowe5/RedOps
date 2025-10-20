"""Async job infrastructure for run event ingestion."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, Optional, cast


class QueueFullError(Exception):
    """Raised when a job queue cannot accept additional events."""


_DEFAULT_TIMEOUT = object()


@dataclass
class JobBus:
    """Lightweight async bus that manages per-run queues."""

    maxsize: int = int(os.getenv("REDOPS_QUEUE_MAX", "1000"))
    default_timeout: float = float(os.getenv("REDOPS_QUEUE_TIMEOUT", "0.1"))

    def __post_init__(self) -> None:
        try:
            self.maxsize = int(self.maxsize)
        except (TypeError, ValueError):
            self.maxsize = 1000
        if self.maxsize <= 0:
            self.maxsize = 1000
        try:
            self.default_timeout = float(self.default_timeout)
        except (TypeError, ValueError):
            self.default_timeout = 0.1
        if self.default_timeout < 0:
            self.default_timeout = 0.1
        self._queues: Dict[str, asyncio.Queue] = {}

    def _ensure_queue(self, run_id: str) -> asyncio.Queue:
        queue = self._queues.get(run_id)
        if queue is None:
            queue = asyncio.Queue(maxsize=self.maxsize)
            self._queues[run_id] = queue
        return queue

    async def put_event(
        self,
        run_id: str,
        event: Dict[str, object],
        *,
        timeout: Optional[float] | object = _DEFAULT_TIMEOUT,
    ) -> bool:
        """Attempt to enqueue an event for a run."""

        queue = self._ensure_queue(run_id)
        if timeout is _DEFAULT_TIMEOUT:
            effective_timeout: Optional[float] = self.default_timeout
        else:
            effective_timeout = cast(Optional[float], timeout)
        try:
            await asyncio.wait_for(queue.put(event), timeout=effective_timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def get_event(self, run_id: str) -> AsyncIterator[Dict[str, object]]:
        """Yield events for the given run indefinitely."""

        queue = self._ensure_queue(run_id)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            # Allow cancellation via aclose without re-raising.
            pass

    def stats(self) -> Dict[str, int]:
        """Return the queued depth for each run."""

        return {run_id: queue.qsize() for run_id, queue in self._queues.items()}


class BatchWriter:
    """Persist queued events to NDJSON files with batching and rotation."""

    BATCH_FSYNC_INTERVAL = 50
    BATCH_FSYNC_SECS = 2
    MAX_MB = 50

    def __init__(self, bus: JobBus, base_dir: Path) -> None:
        self._bus = bus
        self._base_dir = base_dir
        self._tasks: Dict[str, asyncio.Task] = {}
        self._stopping = False

    def start(self) -> None:
        self._stopping = False

    def ensure_writer(self, run_id: str) -> None:
        if self._stopping:
            return
        if run_id in self._tasks:
            return
        self._tasks[run_id] = asyncio.create_task(self._writer_loop(run_id))

    async def shutdown(self) -> None:
        self._stopping = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    async def _writer_loop(self, run_id: str) -> None:
        generator = self._bus.get_event(run_id)
        pending: list[Dict[str, object]] = []
        last_flush = time.monotonic()
        max_bytes = self.MAX_MB * 1024 * 1024
        try:
            while True:
                timeout = max(self.BATCH_FSYNC_SECS - (time.monotonic() - last_flush), 0.01)
                try:
                    event = await asyncio.wait_for(generator.__anext__(), timeout=timeout)
                except asyncio.TimeoutError:
                    if pending:
                        await self._flush(run_id, pending, max_bytes)
                        pending.clear()
                        last_flush = time.monotonic()
                    continue
                except StopAsyncIteration:
                    break
                pending.append(event)
                now = time.monotonic()
                if (
                    len(pending) >= self.BATCH_FSYNC_INTERVAL
                    or (now - last_flush) >= self.BATCH_FSYNC_SECS
                ):
                    await self._flush(run_id, pending, max_bytes)
                    pending.clear()
                    last_flush = time.monotonic()
        except asyncio.CancelledError:
            pass
        finally:
            if pending:
                await self._flush(run_id, pending, max_bytes)
            with contextlib.suppress(Exception):
                await generator.aclose()
            self._tasks.pop(run_id, None)

    async def _flush(
        self,
        run_id: str,
        events: Iterable[Dict[str, object]],
        max_bytes: int,
    ) -> None:
        if not events:
            return
        await asyncio.to_thread(self._write_batch, run_id, list(events), max_bytes)

    def _write_batch(
        self,
        run_id: str,
        events: list[Dict[str, object]],
        max_bytes: int,
    ) -> None:
        run_dir = self._base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "events.ndjson"
        lines = [json.dumps(event, sort_keys=True) for event in events]
        payload = "\n".join(lines) + "\n"
        self._append_with_rotation(path, payload, max_bytes)
        legacy_path = path.with_suffix(".json")
        self._append_with_rotation(legacy_path, payload, max_bytes)

    def _append_with_rotation(self, path: Path, payload: str, max_bytes: int) -> None:
        self._rotate_if_needed(path, max_bytes)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def _rotate_if_needed(self, path: Path, max_bytes: int) -> None:
        if not path.exists():
            return
        try:
            if path.stat().st_size <= max_bytes:
                return
        except FileNotFoundError:
            return
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        rotated = path.with_name(f"{path.stem}-{timestamp}{path.suffix}")
        tmp_rotated = rotated.with_suffix(rotated.suffix + ".tmp")
        path.replace(tmp_rotated)
        tmp_rotated.replace(rotated)

