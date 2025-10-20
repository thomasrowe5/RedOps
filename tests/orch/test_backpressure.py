"""Tests for orchestrator backpressure behavior."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
import types
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def orchestrator_app(monkeypatch, tmp_path):
    """Load a fresh orchestrator app instance with isolated configuration."""

    monkeypatch.setenv("REDOPS_JWT_SECRET", "test-secret")
    monkeypatch.setenv("REDOPS_QUEUE_MAX", "2")
    monkeypatch.setenv("REDOPS_QUEUE_TIMEOUT", "1")

    # Ensure modules pick up the new environment configuration.
    for module_name in [
        "orchestrator.app.main",
        "orchestrator.app.jobs",
        "orchestrator.app.auth",
    ]:
        sys.modules.pop(module_name, None)

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    prometheus_stub = types.ModuleType("prometheus_client")

    class _Metric:
        def labels(self, **_kwargs):  # pragma: no cover - lightweight stub
            return self

        def inc(self, *_args, **_kwargs):  # pragma: no cover
            return None

        def set(self, *_args, **_kwargs):  # pragma: no cover
            return None

        def observe(self, *_args, **_kwargs):  # pragma: no cover
            return None

    prometheus_stub.Counter = lambda *args, **kwargs: _Metric()
    prometheus_stub.Gauge = lambda *args, **kwargs: _Metric()
    prometheus_stub.Histogram = lambda *args, **kwargs: _Metric()
    prometheus_stub.generate_latest = lambda: b""
    prometheus_stub.CONTENT_TYPE_LATEST = "text/plain"

    sys.modules.setdefault("prometheus_client", prometheus_stub)

    structlog_stub = types.ModuleType("structlog")

    class _TimeStamper:
        def __init__(self, **_kwargs):  # pragma: no cover - lightweight stub
            pass

        def __call__(self, _logger, _method_name, event_dict):  # pragma: no cover
            return event_dict

    class _JSONRenderer:
        def __call__(self, _logger=None, _method_name=None, event_dict=None):  # pragma: no cover
            return event_dict or {}

    structlog_stub.processors = types.SimpleNamespace(
        TimeStamper=lambda **kwargs: _TimeStamper(),
        add_log_level=lambda *_args, **_kwargs: {},
        JSONRenderer=lambda *args, **kwargs: _JSONRenderer(),
    )
    structlog_stub.configure = lambda **_kwargs: None

    stdlib_module = types.ModuleType("structlog.stdlib")

    class _BoundLogger:  # pragma: no cover - stub only
        pass

    class _LoggerFactory:  # pragma: no cover - stub only
        def __call__(self, *_args, **_kwargs):
            return _BoundLogger()

    class _ProcessorFormatter:  # pragma: no cover - stub only
        remove_processors_meta = staticmethod(lambda _logger, _method, event_dict: event_dict)

        def __init__(self, *_args, **_kwargs):
            pass

        def format(self, record):  # pragma: no cover - basic logging stub
            return getattr(record, "getMessage", lambda: "")()

    stdlib_module.BoundLogger = _BoundLogger
    stdlib_module.LoggerFactory = _LoggerFactory
    stdlib_module.ProcessorFormatter = _ProcessorFormatter
    stdlib_module.filter_by_level = lambda _logger, _method, event_dict: event_dict

    contextvars_module = types.ModuleType("structlog.contextvars")
    contextvars_module.get_contextvars = lambda: {}
    contextvars_module.merge_contextvars = lambda _logger, _method, event_dict: event_dict

    structlog_stub.stdlib = stdlib_module
    structlog_stub.contextvars = contextvars_module

    sys.modules.setdefault("structlog", structlog_stub)
    sys.modules.setdefault("structlog.stdlib", stdlib_module)
    sys.modules.setdefault("structlog.contextvars", contextvars_module)

    try:
        import pydantic  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - tests rely on compatibility stub
        pydantic = None  # type: ignore

    if pydantic is not None:
        try:
            from pydantic import utils as pydantic_utils  # type: ignore
        except Exception:  # pragma: no cover - fallback if utils missing
            pydantic_utils = None  # type: ignore

        if pydantic_utils is not None:
            original_validate = pydantic_utils.validate_field_name

            def _validate_field_name(bases, field_name):  # pragma: no cover - simple override
                if field_name == "schema":
                    return None
                return original_validate(bases, field_name)

            monkeypatch.setattr(
                pydantic_utils,
                "validate_field_name",
                _validate_field_name,
                raising=False,
            )

            try:
                import pydantic.main as pydantic_main  # type: ignore
            except Exception:  # pragma: no cover - fallback if module missing
                pydantic_main = None  # type: ignore

            if pydantic_main is not None:
                monkeypatch.setattr(
                    pydantic_main,
                    "validate_field_name",
                    _validate_field_name,
                    raising=False,
                )

    main = importlib.import_module("orchestrator.app.main")

    data_dir = tmp_path / "data"
    runs_dir = data_dir / "runs"
    scenarios_dir = data_dir / "scenarios"
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(main, "SCENARIOS_DIR", scenarios_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    main.BATCH_WRITER._base_dir = runs_dir

    async def _noop_worker() -> None:  # pragma: no cover - background worker disabled for tests
        return None

    monkeypatch.setattr(main, "run_worker", _noop_worker)

    return main


def _event_payload(schema: str, when: datetime) -> dict[str, object]:
    return {
        "schema": schema,
        "payload": {
            "timestamp": when.isoformat().replace("+00:00", "Z"),
            "agent_id": "agent-007",
            "kind": "simulated",
            "tactic": "discovery",
        },
    }


def test_queue_backpressure_and_flush(orchestrator_app, monkeypatch):
    main = orchestrator_app

    # Prevent the BatchWriter from starting its background consumer so that the
    # in-memory queue can reach its configured capacity.
    original_ensure_writer = main.BATCH_WRITER.ensure_writer

    def _noop_ensure_writer(self, run_id):
        return None

    monkeypatch.setattr(
        main.BATCH_WRITER,
        "ensure_writer",
        types.MethodType(_noop_ensure_writer, main.BATCH_WRITER),
    )

    flushed_batches: list[list[dict[str, object]]] = []

    def _capture_write_batch(self, run_id, events, max_bytes):
        flushed_batches.append(list(events))

    monkeypatch.setattr(
        main.BATCH_WRITER,
        "_write_batch",
        types.MethodType(_capture_write_batch, main.BATCH_WRITER),
    )

    monkeypatch.setattr(main, "_install_signal_handlers", lambda _loop: None)

    original_put_event = main.JOB_BUS._bus.put_event

    async def _put_event_with_zero_timeout(
        run_id: str,
        event: dict[str, object],
        *,
        timeout: object = None,
    ) -> bool:
        if timeout == 0:
            queue = main.JOB_BUS._bus._ensure_queue(run_id)
            try:
                queue.put_nowait(event)
                return True
            except asyncio.QueueFull:
                return False
        return await original_put_event(run_id, event, timeout=timeout)

    monkeypatch.setattr(main.JOB_BUS._bus, "put_event", _put_event_with_zero_timeout)

    with TestClient(main.app) as client:
        scenario_resp = client.post(
            "/scenarios",
            data="name: example",
            headers={"content-type": "text/plain"},
        )
        scenario_resp.raise_for_status()
        scenario_id = scenario_resp.json()["scenario_id"]

        run_resp = client.post(f"/runs/{scenario_id}/start")
        run_resp.raise_for_status()
        run_id = run_resp.json()["run_id"]

        token = main.create_token("agent", "agent_red")
        headers = {"Authorization": f"Bearer {token}"}
        base_time = datetime.now(timezone.utc)

        inner_queue = main.JOB_BUS._bus._queues[run_id]
        assert main.JOB_BUS._bus.maxsize == 2
        priming_events = []
        while not inner_queue.empty():
            priming_events.append(inner_queue.get_nowait())
        if priming_events:
            asyncio.run(
                main.BATCH_WRITER._flush(
                    run_id,
                    priming_events,
                    main.BATCH_WRITER.MAX_MB * 1024 * 1024,
                )
            )
        assert inner_queue.qsize() == 0

        # Fill the queue to its maximum size.
        for offset in range(2):
            payload = _event_payload(main.SCHEMA_V1, base_time + timedelta(seconds=offset))
            response = client.post(f"/runs/{run_id}/events", json=payload, headers=headers)
            assert response.status_code == 202

        # Next request should exceed capacity and return HTTP 429.
        overflow_payload = _event_payload(main.SCHEMA_V1, base_time + timedelta(seconds=2))
        overflow_response = client.post(
            f"/runs/{run_id}/events",
            json=overflow_payload,
            headers=headers,
        )
        assert overflow_response.status_code == 429

        inner_queue = main.JOB_BUS._bus._queues[run_id]
        assert inner_queue.qsize() == 2

        # Manually drain the queued events and flush them through the BatchWriter.
        drained_events = []
        while not inner_queue.empty():
            drained_events.append(inner_queue.get_nowait())
        assert len(drained_events) == 2

        asyncio.run(
            main.BATCH_WRITER._flush(
                run_id,
                drained_events,
                main.BATCH_WRITER.MAX_MB * 1024 * 1024,
            )
        )

        assert inner_queue.qsize() == 0
        assert flushed_batches, "Expected BatchWriter to capture at least one flush"

        # Restore the real ensure_writer so future enqueues behave normally.
        monkeypatch.setattr(main.BATCH_WRITER, "ensure_writer", original_ensure_writer)

        # A subsequent event should succeed now that the queue has available capacity.
        recovery_payload = _event_payload(
            main.SCHEMA_V1,
            base_time + timedelta(seconds=3),
        )
        recovery_response = client.post(
            f"/runs/{run_id}/events",
            json=recovery_payload,
            headers=headers,
        )
        assert recovery_response.status_code == 202

        # Verify the stats endpoint reflects the drained queue depth.
        operator_token = main.create_token("operator", "operator")
        stats_headers = {"Authorization": f"Bearer {operator_token}"}
        stats_response = client.get(f"/runs/{run_id}/queue_stats", headers=stats_headers)
        stats_response.raise_for_status()
        assert stats_response.json()["depth"] == 0
