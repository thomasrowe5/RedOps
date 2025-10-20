"""Utilities for profiling orchestrator code paths.

This module exposes a small context manager that wraps ``cProfile``
instrumentation and a CLI helper that can execute the orchestrator's event
ingestion handler against sample data.  The CLI is useful for quickly
understanding the runtime characteristics of the pipeline without having to
spin up the full FastAPI application.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pstats


class Profiler:
    """Context manager that profiles the enclosed block using ``cProfile``."""

    def __init__(
        self,
        *,
        sort: str = "tottime",
        limit: Optional[int] = None,
        stream = None,
        autoprint: bool = True,
    ) -> None:
        self.sort = sort
        self.limit = limit
        self.stream = stream or sys.stdout
        self.autoprint = autoprint
        self._profiler: Optional[cProfile.Profile] = None
        self._stats: Optional[pstats.Stats] = None

    def __enter__(self) -> "Profiler":
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._profiler is None:
            return
        self._profiler.disable()
        self._stats = pstats.Stats(self._profiler, stream=self.stream)
        self._stats.sort_stats(self.sort)
        if self.autoprint:
            self.print_stats(self.limit)

    @property
    def stats(self) -> Optional[pstats.Stats]:
        """Return the underlying :class:`pstats.Stats` object, if available."""

        return self._stats

    def print_stats(self, limit: Optional[int] = None) -> None:
        """Pretty-print the collected statistics."""

        if self._stats is None:
            return
        if limit is None:
            self._stats.print_stats()
        else:
            self._stats.print_stats(limit)


def _load_events(path: Path) -> List[dict[str, Any]]:
    """Load a list of JSON events from ``path``.

    The loader accepts JSON arrays as well as newline-delimited JSON payloads.
    Each event can either be a full request body containing ``schema`` and
    ``payload`` keys or a bare payload dictionary.
    """

    raw = path.read_text(encoding="utf-8")
    events: List[dict[str, Any]] = []

    if path.suffix.lower() == ".json":
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            events = [dict(item) for item in parsed]
        elif isinstance(parsed, dict):
            events = [dict(parsed)]
        else:
            raise ValueError("JSON input must be an object or an array of objects")
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    normalised: List[dict[str, Any]] = []
    for event in events:
        if "schema" in event and "payload" in event:
            normalised.append(event)
            continue
        payload = dict(event)
        schema = payload.pop("schema", "redops/v1")
        normalised.append({"schema": schema, "payload": payload})
    return normalised


async def _ingest_events(events: Iterable[dict[str, Any]], run_id: str) -> None:
    """Replay ``events`` through the orchestrator ingestion handler."""

    from orchestrator.app import main

    await main.ensure_directories()
    main.BATCH_WRITER.start()

    async with main.runs_lock:
        record = main.runs.get(run_id)
        if record is None:
            events_path = await main.run_events_path(run_id)
            record = main.RunRecord(
                run_id=run_id,
                scenario_id="profiling",  # sentinel scenario identifier
                status="profiling",
                events_path=events_path,
            )
            main.runs[run_id] = record

    try:
        for raw_event in events:
            model = main.VersionedEventIn.parse_obj(raw_event)
            await main.ingest_run_event(run_id, model)
    finally:
        await asyncio.sleep(0)  # allow background tasks to make progress
        await main.BATCH_WRITER.shutdown()
        async with main.runs_lock:
            main.runs.pop(run_id, None)


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Profile orchestrator handlers")
    parser.add_argument("events", type=Path, help="Path to sample event data (JSON or NDJSON)")
    parser.add_argument("--run-id", default="profiling-run", help="Run identifier used for ingestion")
    parser.add_argument("--sort", default="tottime", help="Stat sorting field passed to pstats")
    parser.add_argument("--limit", type=int, default=30, help="Number of rows to print from the profile")
    args = parser.parse_args(argv)

    events = _load_events(args.events)
    if not events:
        raise SystemExit("No events loaded from input file")

    with Profiler(sort=args.sort, limit=args.limit, autoprint=True) as profiler:
        asyncio.run(_ingest_events(events, args.run_id))
    if profiler.stats is None:
        raise SystemExit("Profiling failed to produce statistics")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_cli())
