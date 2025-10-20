"""Redis-backed implementation of the JobBus API."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterable, Optional

import redis
from redis.exceptions import ResponseError

from .jobs import _DEFAULT_TIMEOUT


class _MissingGroupError(Exception):
    """Raised internally when a stream consumer group is absent."""


@dataclass
class RedisBus:
    """High-throughput job bus built on Redis streams."""

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    stream_prefix: str = "redops:queue"
    consumer_prefix: str = "redops-consumer"
    default_timeout: float = float(os.getenv("REDOPS_QUEUE_TIMEOUT", "0.1"))
    maxsize: int = int(os.getenv("REDOPS_QUEUE_MAX", "1000"))
    poll_interval: float = 0.05

    def __post_init__(self) -> None:
        if self.default_timeout < 0:
            self.default_timeout = 0.1
        if self.maxsize <= 0:
            # Treat non-positive values as unbounded.
            self.maxsize = 0
        self._block_ms = max(int(self.default_timeout * 1000), 100)
        self._streams_registry = f"{self.stream_prefix}:streams"
        self._pool = redis.ConnectionPool.from_url(
            self.redis_url,
            decode_responses=True,
        )
        self._redis = redis.Redis(connection_pool=self._pool)

    # ------------------------------------------------------------------
    # Helpers

    def _stream_key(self, run_id: str) -> str:
        return f"{self.stream_prefix}:{run_id}"

    def _group_name(self, run_id: str) -> str:
        return f"{self.stream_prefix}:{run_id}:group"

    def _ensure_group(self, key: str, group: str) -> None:
        try:
            self._redis.xgroup_create(key, group, id="0", mkstream=True)
        except ResponseError as exc:  # pragma: no cover - networking guard
            if "BUSYGROUP" in str(exc):
                return
            raise

    def _read_group(self, key: str, group: str, consumer: str):
        try:
            response = self._redis.xreadgroup(
                group,
                consumer,
                {key: ">"},
                count=1,
                block=self._block_ms,
            )
        except ResponseError as exc:
            if "NOGROUP" in str(exc):
                raise _MissingGroupError from exc
            raise
        if not response:
            return []
        _, entries = response[0]
        return entries

    def _ack(self, key: str, group: str, message_id: str) -> None:
        self._redis.xack(key, group, message_id)
        self._redis.xdel(key, message_id)

    # ------------------------------------------------------------------
    # JobBus API

    async def put_event(
        self,
        run_id: str,
        event: Dict[str, object],
        *,
        timeout: Optional[float] | object = _DEFAULT_TIMEOUT,
    ) -> bool:
        """Enqueue an event for a run via Redis streams."""

        key = self._stream_key(run_id)
        await asyncio.to_thread(self._redis.sadd, self._streams_registry, run_id)
        if timeout is _DEFAULT_TIMEOUT:
            effective_timeout: Optional[float] = self.default_timeout
        else:
            effective_timeout = timeout if timeout is None else float(timeout)

        deadline = None if effective_timeout is None else time.monotonic() + effective_timeout
        payload = json.dumps(event, separators=(",", ":"))

        while True:
            if self.maxsize:
                current_length = await asyncio.to_thread(self._redis.xlen, key)
                if current_length >= self.maxsize:
                    if effective_timeout is None:
                        await asyncio.sleep(self.poll_interval)
                        continue
                    remaining = (deadline - time.monotonic()) if deadline is not None else 0
                    if remaining <= 0:
                        return False
                    await asyncio.sleep(min(self.poll_interval, remaining))
                    continue
            await asyncio.to_thread(self._redis.xadd, key, {"event": payload})
            return True

    async def get_event(self, run_id: str) -> AsyncIterator[Dict[str, object]]:
        """Yield events for the given run by consuming a Redis stream."""

        key = self._stream_key(run_id)
        group = self._group_name(run_id)
        consumer = f"{self.consumer_prefix}-{uuid.uuid4().hex}"

        await asyncio.to_thread(self._ensure_group, key, group)

        while True:
            try:
                entries = await asyncio.to_thread(self._read_group, key, group, consumer)
            except _MissingGroupError:
                await asyncio.to_thread(self._ensure_group, key, group)
                continue
            if not entries:
                await asyncio.sleep(self.poll_interval)
                continue
            for message_id, data in entries:
                raw = data.get("event")
                if raw is None:
                    await asyncio.to_thread(self._ack, key, group, message_id)
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    event = {"raw": raw}
                await asyncio.to_thread(self._ack, key, group, message_id)
                yield event

    def stats(self) -> Dict[str, int]:
        """Return stream depths for all known run queues."""

        run_ids: Iterable[str] = self._redis.smembers(self._streams_registry)
        run_id_list = list(run_ids)
        if not run_id_list:
            return {}
        pipeline = self._redis.pipeline()
        for run_id in run_id_list:
            pipeline.xlen(self._stream_key(run_id))
        lengths = pipeline.execute()
        stats: Dict[str, int] = {}
        for run_id, length in zip(run_id_list, lengths):
            try:
                stats[run_id] = int(length)
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                stats[run_id] = 0
        return stats

