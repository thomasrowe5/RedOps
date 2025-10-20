"""Prometheus metrics helpers for the orchestrator service."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

__all__ = [
    "EVENTS_TOTAL",
    "QUEUE_DEPTH",
    "WRITE_BATCHES_TOTAL",
    "WRITE_ERRORS_TOTAL",
    "REQUEST_LATENCY_SECONDS",
    "track_request_latency",
    "record_event",
    "set_queue_depth",
    "record_write_success",
    "record_write_error",
    "record_detections",
    "latest",
    "CONTENT_TYPE_LATEST",
]

EVENTS_TOTAL = Counter(
    "redops_events_total",
    "Total number of events processed per run.",
    ["run_id"],
)

QUEUE_DEPTH = Gauge(
    "redops_queue_depth",
    "Current depth of the event queue per run.",
    ["run_id"],
)

WRITE_BATCHES_TOTAL = Counter(
    "redops_write_batches_total",
    "Total number of event batches written to disk.",
)

WRITE_ERRORS_TOTAL = Counter(
    "redops_write_errors_total",
    "Total number of errors encountered while writing event batches.",
)

REQUEST_LATENCY_SECONDS = Histogram(
    "redops_requests_latency_seconds",
    "Latency of orchestrator requests in seconds.",
    ["method", "endpoint"],
)


@contextmanager
def track_request_latency(method: str, endpoint: str):
    """Context manager to observe elapsed time for a logical request."""

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        REQUEST_LATENCY_SECONDS.labels(method=method, endpoint=endpoint).observe(duration)


def record_event(run_id: str, *, count: int = 1, queue_depth: Optional[int] = None) -> None:
    """Increment the event counter and optionally update the queue depth gauge."""

    if count <= 0:
        return
    EVENTS_TOTAL.labels(run_id=run_id).inc(count)
    if queue_depth is not None:
        QUEUE_DEPTH.labels(run_id=run_id).set(queue_depth)


def set_queue_depth(run_id: str, depth: int) -> None:
    """Update the queue depth gauge for a run."""

    QUEUE_DEPTH.labels(run_id=run_id).set(depth)


def record_write_success() -> None:
    """Increment the counter tracking successful batch writes."""

    WRITE_BATCHES_TOTAL.inc()


def record_write_error() -> None:
    """Increment the counter tracking failed batch writes."""

    WRITE_ERRORS_TOTAL.inc()


def record_detections(run_id: str, count: int) -> None:
    """Record detections for a run using the event counter."""

    if count <= 0:
        return
    EVENTS_TOTAL.labels(run_id=run_id).inc(count)


def latest() -> bytes:
    """Return the Prometheus text exposition for all metrics."""

    return generate_latest()
