"""FastAPI application for the RedOps orchestrator."""

from __future__ import annotations

LAB_MODE = True

import asyncio
import contextlib
import ipaddress
import json
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional, Protocol
from typing import Tuple

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import detector, metrics
from .audit import verify_audit, write_audit
from .auth import ALLOWED_ROLES, Actor, create_token, require_roles
from .jobs import BatchWriter, JobBus, QueueFullError, _DEFAULT_TIMEOUT
from .logging_config import configure_uvicorn
from .schemas import (
    Detection,
    SchemaVersion,
    VersionedEventIn,
    VersionedResponseIn,
)

try:  # Optional dependency used to parse scenario YAML.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML may be absent in some environments.
    yaml = None  # type: ignore

LOGGER = logging.getLogger("redops.orchestrator")

_ALLOWED_SUBNETS = (
    ipaddress.ip_network("127.0.0.1/32"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("172.16.0.0/12"),
)

app = FastAPI(
    title="RedOps Orchestrator",
    version="0.2.0",
    description=(
        "API surface for scheduling and tracking simulated RedOps scenarios. "
        "All actions are file-local and are intended for demonstration only."
    ),
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
SCENARIOS_DIR = DATA_DIR / "scenarios"
RUNS_DIR = DATA_DIR / "runs"

IO_TIMEOUT_SECONDS = 5.0
BACKOFF_MAX_ATTEMPTS = 5
BACKOFF_BASE_DELAY = 0.05


@app.middleware("http")
async def _enforce_lab_mode(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    if LAB_MODE:
        client_host = request.client.host if request.client else None
        if client_host and not _is_host_allowed(client_host):
            LOGGER.warning("Denied request from disallowed host %s", client_host)
            raise HTTPException(status_code=403, detail="LAB MODE: external access prohibited")
    return await call_next(request)


@dataclass
class RunRecord:
    """In-memory representation of a scheduled run."""

    run_id: str
    scenario_id: str
    status: str
    events_path: Path
    actions: List[Dict[str, Any]] = field(default_factory=list)
    action_pointer: int = 0
    error: Optional[str] = None


job_queue: asyncio.Queue[str] = asyncio.Queue()
worker_task: Optional[asyncio.Task[None]] = None
runs: Dict[str, RunRecord] = {}
runs_lock = asyncio.Lock()

class JobBusProtocol(Protocol):
    async def put_event(
        self,
        run_id: str,
        event: Dict[str, object],
        *,
        timeout: Optional[float] | object = _DEFAULT_TIMEOUT,
    ) -> bool:
        ...

    async def get_event(self, run_id: str) -> AsyncIterator[Dict[str, object]]:
        ...

    def stats(self) -> Dict[str, int]:
        ...


class InstrumentedJobBus:
    def __init__(self, bus: JobBusProtocol) -> None:
        self._bus = bus

    async def put_event(
        self,
        run_id: str,
        event: Dict[str, object],
        *,
        timeout: Optional[float] | object = _DEFAULT_TIMEOUT,
    ) -> bool:
        success = await self._bus.put_event(run_id, event, timeout=timeout)
        if success:
            metrics.set_queue_depth(run_id, self.stats().get(run_id, 0))
        return success

    async def get_event(self, run_id: str) -> AsyncIterator[Dict[str, object]]:
        async for event in self._bus.get_event(run_id):
            metrics.set_queue_depth(run_id, self.stats().get(run_id, 0))
            yield event

    def stats(self) -> Dict[str, int]:
        return self._bus.stats()

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough helper
        return getattr(self._bus, item)


class InstrumentedBatchWriter(BatchWriter):
    async def _flush(
        self,
        run_id: str,
        events: Iterable[Dict[str, object]],
        max_bytes: int,
    ) -> None:
        event_list = list(events)
        if not event_list:
            return
        try:
            await super()._flush(run_id, event_list, max_bytes)
            metrics.record_write_success()
        except Exception:
            metrics.record_write_error()
            raise

# Job bus selection -------------------------------------------------------


def _create_job_bus() -> InstrumentedJobBus:
    driver = os.getenv("REDOPS_QUEUE_DRIVER", "memory").strip().lower()
    base_bus: JobBusProtocol
    if driver == "redis":
        from .redis_bus import RedisBus

        base_bus = RedisBus()
    else:
        base_bus = JobBus()
    return InstrumentedJobBus(base_bus)


JOB_BUS = _create_job_bus()
BATCH_WRITER = InstrumentedBatchWriter(JOB_BUS, RUNS_DIR)
SHUTDOWN_TRIGGERED = asyncio.Event()


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _is_host_allowed(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in subnet for subnet in _ALLOWED_SUBNETS)


MAX_POST_BYTES = 64 * 1024
RATE_LIMIT_BURST = max(0, _get_env_int("REDOPS_RATE_LIMIT_BURST", 100))
RATE_LIMIT_RATE = max(0.0, _get_env_float("REDOPS_RATE_LIMIT_RATE", 50.0))
_TOKENS_PER_SECOND = RATE_LIMIT_RATE / 60.0 if RATE_LIMIT_RATE > 0 else 0.0


@dataclass
class _TokenBucketState:
    tokens: float
    updated_at: float


_RATE_LIMIT_BUCKETS: Dict[Tuple[str, str], _TokenBucketState] = {}
_RATE_LIMIT_LOCK = asyncio.Lock()


def _should_rate_limit(path: str) -> bool:
    return path.endswith("/events") or path.endswith("/responses")


def _extract_run_id(path: str, path_params: Dict[str, Any]) -> Optional[str]:
    run_id = path_params.get("run_id") if isinstance(path_params, dict) else None
    if isinstance(run_id, str) and run_id:
        return run_id
    segments = [segment for segment in path.strip("/").split("/") if segment]
    if len(segments) >= 2 and segments[0] == "runs":
        return segments[1]
    return None


def _extract_agent_id(path: str, body: bytes) -> Optional[str]:
    if not _should_rate_limit(path) or not body:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    envelope = payload.get("payload")
    if isinstance(envelope, dict):
        agent_id = envelope.get("agent_id")
        if isinstance(agent_id, str) and agent_id.strip():
            return agent_id.strip()
    return None


async def _consume_rate_limit_token(run_id: str, agent_id: str) -> bool:
    if RATE_LIMIT_BURST <= 0 or _TOKENS_PER_SECOND <= 0:
        return True
    key = (run_id, agent_id)
    now = time.monotonic()
    async with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_BUCKETS.get(key)
        if bucket is None:
            bucket = _TokenBucketState(tokens=float(RATE_LIMIT_BURST), updated_at=now)
            _RATE_LIMIT_BUCKETS[key] = bucket
        else:
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(float(RATE_LIMIT_BURST), bucket.tokens + elapsed * _TOKENS_PER_SECOND)
            bucket.updated_at = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False


@app.middleware("http")
async def enforce_request_limits(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
):
    body_bytes: Optional[bytes] = None
    if request.method.upper() == "POST":
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                return JSONResponse(status_code=413, content={"error": "payload too large"})
            if declared_length > MAX_POST_BYTES:
                return JSONResponse(status_code=413, content={"error": "payload too large"})
        else:
            body_bytes = await request.body()
            if len(body_bytes) > MAX_POST_BYTES:
                return JSONResponse(status_code=413, content={"error": "payload too large"})

        path = request.url.path
        if _should_rate_limit(path):
            if body_bytes is None:
                body_bytes = await request.body()
            if len(body_bytes) > MAX_POST_BYTES:
                return JSONResponse(status_code=413, content={"error": "payload too large"})
            run_id = _extract_run_id(path, request.scope.get("path_params", {}))
            agent_id = _extract_agent_id(path, body_bytes)
            if run_id and agent_id:
                allowed = await _consume_rate_limit_token(run_id, agent_id)
                if not allowed:
                    return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    return await call_next(request)


@app.middleware("http")
async def record_request_latency(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
):
    """Observe request latency for all inbound HTTP traffic."""

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start
        metrics.REQUEST_LATENCY_SECONDS.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)
        raise
    duration = time.perf_counter() - start
    metrics.REQUEST_LATENCY_SECONDS.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)
    return response

def _require_schema_v1(schema_value: str) -> None:
    if schema_value != SCHEMA_V1:
        raise HTTPException(status_code=400, detail="unsupported schema version")


def _wrap_payload(payload: Any) -> Dict[str, Any]:
    return {"schema": SCHEMA_V1, "payload": payload}

SCHEMA_V1 = SchemaVersion.REDOPS_V1.value


async def ensure_directories() -> None:
    """Create the directories required for storing scenarios and run artifacts."""

    await _mkdir_with_timeout(SCENARIOS_DIR)
    await _mkdir_with_timeout(RUNS_DIR)


def scenario_path_from_id(scenario_id: str) -> Path:
    """Return the filesystem path used to persist a scenario body."""

    return SCENARIOS_DIR / f"{scenario_id}.yaml"


async def run_events_path(run_id: str) -> Path:
    """Return the filesystem path for storing JSON events for a run."""

    run_dir = await _ensure_run_dir(run_id)
    return run_dir / "events.ndjson"


async def run_responses_path(run_id: str) -> Path:
    """Return the filesystem path for storing response actions for a run."""

    run_dir = await _ensure_run_dir(run_id)
    return run_dir / "responses.jsonl"


async def run_policy_path(run_id: str) -> Path:
    """Return the filesystem path for storing run policy state."""

    run_dir = await _ensure_run_dir(run_id)
    return run_dir / "policy.json"


async def _mkdir_with_timeout(path: Path) -> None:
    async with asyncio.timeout(IO_TIMEOUT_SECONDS):
        await asyncio.to_thread(path.mkdir, parents=True, exist_ok=True)


async def _ensure_run_dir(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    await _mkdir_with_timeout(run_dir)
    return run_dir


async def _path_exists(path: Path) -> bool:
    async with asyncio.timeout(IO_TIMEOUT_SECONDS):
        return await asyncio.to_thread(path.exists)


async def _read_text(path: Path) -> str:
    async with asyncio.timeout(IO_TIMEOUT_SECONDS):
        return await asyncio.to_thread(path.read_text, encoding="utf-8")


async def _write_with_backoff(operation: Callable[[], None]) -> None:
    delay = BACKOFF_BASE_DELAY
    last_error: Optional[Exception] = None
    for attempt in range(1, BACKOFF_MAX_ATTEMPTS + 1):
        try:
            async with asyncio.timeout(IO_TIMEOUT_SECONDS):
                await asyncio.to_thread(operation)
            return
        except Exception as exc:  # pragma: no cover - safety net
            last_error = exc
            if attempt == BACKOFF_MAX_ATTEMPTS:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    if last_error:
        raise last_error


async def _write_text(path: Path, content: str) -> None:
    await _write_with_backoff(lambda: path.write_text(content, encoding="utf-8"))


async def _append_text(path: Path, content: str) -> None:
    def _writer() -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)

    await _write_with_backoff(_writer)


async def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    if not await _path_exists(path):
        return []
    raw = await _read_text(path)
    events: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"raw": line})
    return events


def _encode_sse_event(payload: Dict[str, Any]) -> bytes:
    data = json.dumps(payload, sort_keys=True)
    return f"data: {data}\n\n".encode("utf-8")


async def _tail_events_stream(path: Path, start_index: int) -> AsyncIterator[bytes]:
    last_index = max(0, start_index)
    keepalive_counter = 0
    try:
        while not SHUTDOWN_TRIGGERED.is_set():
            events = await _read_ndjson(path)
            if last_index < len(events):
                for event in events[last_index:]:
                    yield _encode_sse_event(event)
                last_index = len(events)
                keepalive_counter = 0
            else:
                keepalive_counter += 1
                if keepalive_counter >= 15:
                    yield b": keep-alive\n\n"
                    keepalive_counter = 0
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        raise


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    for signame in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signame, None)
        if signum is None:
            continue
        try:
            loop.add_signal_handler(
                signum,
                lambda s=signame: loop.create_task(_initiate_shutdown(s)),
            )
        except NotImplementedError:  # pragma: no cover - platform dependent
            LOGGER.debug("Signal handler %s not supported on this platform", signame)


async def _initiate_shutdown(reason: Optional[str] = None) -> None:
    if SHUTDOWN_TRIGGERED.is_set():
        return
    SHUTDOWN_TRIGGERED.set()
    LOGGER.info("Initiating graceful shutdown%s", f" ({reason})" if reason else "")
    global worker_task
    if worker_task is not None:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        worker_task = None
    await BATCH_WRITER.shutdown()


async def enqueue_run_event(
    run_id: str,
    event: Dict[str, Any],
    *,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Enqueue an enriched event for asynchronous persistence."""

    enriched = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **event,
    }
    BATCH_WRITER.ensure_writer(run_id)
    success = await JOB_BUS.put_event(run_id, enriched, timeout=timeout)
    if not success:
        raise QueueFullError(f"queue full for run {run_id}")
    return enriched


async def append_event(record: RunRecord, event: Dict[str, Any]) -> Dict[str, Any]:
    """Append a JSON event with timestamp metadata to the run's event log."""

    return await enqueue_run_event(record.run_id, event)


def parse_actions_from_scenario(body: str) -> List[Dict[str, Any]]:
    """Extract a sequence of actions from a scenario definition."""

    parsed: Any = None
    if yaml is not None:
        try:
            parsed = yaml.safe_load(body)
        except Exception:
            parsed = None

    actions: List[Any]
    if isinstance(parsed, dict):
        raw_actions = parsed.get("steps") or parsed.get("actions") or []
        actions = raw_actions if isinstance(raw_actions, list) else [raw_actions]
    elif isinstance(parsed, list):
        actions = parsed
    else:
        # Fallback: treat each non-empty line as an action description.
        actions = [line for line in body.splitlines() if line.strip()]

    normalised: List[Dict[str, Any]] = []
    for index, action in enumerate(actions):
        if isinstance(action, dict):
            payload = dict(action)
        elif isinstance(action, str):
            payload = {"description": action}
        else:
            payload = {"description": repr(action)}
        payload.setdefault("id", index)
        normalised.append(payload)
    return normalised


async def run_worker() -> None:
    """Background worker processing queued run executions."""

    while True:
        try:
            run_id = await job_queue.get()
        except asyncio.CancelledError:
            break
        try:
            async with runs_lock:
                record = runs.get(run_id)
                if record is None:
                    continue
                record.status = "running"
            await append_event(record, {"type": "run_started", "run_id": run_id})

            scenario_file = scenario_path_from_id(record.scenario_id)
            try:
                scenario_body = await _read_text(scenario_file)
            except FileNotFoundError:
                async with runs_lock:
                    record.status = "error"
                    record.error = "Scenario definition not found"
                await append_event(
                    record,
                    {
                        "type": "run_error",
                        "run_id": run_id,
                        "message": record.error,
                    },
                )
                continue

            actions = parse_actions_from_scenario(scenario_body)
            async with runs_lock:
                record.actions = actions
                record.action_pointer = 0
            await append_event(
                record,
                {
                    "type": "actions_loaded",
                    "run_id": run_id,
                    "count": len(actions),
                },
            )

            if not actions:
                async with runs_lock:
                    record.status = "completed"
                await append_event(
                    record,
                    {"type": "run_completed", "run_id": run_id, "reason": "no_actions"},
                )
        except Exception as exc:  # pragma: no cover - safety net
            async with runs_lock:
                record = runs.get(run_id)
                if record is not None:
                    record.status = "error"
                    record.error = str(exc)
            if record is not None:
                await append_event(
                    record,
                    {"type": "run_error", "run_id": run_id, "message": record.error},
                )
        finally:
            job_queue.task_done()

    while True:
        try:
            job_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        else:
            job_queue.task_done()


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise directories and launch the background worker."""

    if not LAB_MODE:
        raise RuntimeError("LAB_MODE must remain enabled for orchestrator startup")
    env_lab_mode = os.getenv("LAB_MODE", "1").strip().lower()
    if env_lab_mode in {"0", "false", "no"}:
        raise RuntimeError("LAB_MODE environment variable must be enabled for lab operation")
    configure_uvicorn()
    SHUTDOWN_TRIGGERED.clear()
    await ensure_directories()
    LOGGER.info("LAB MODE enforced: external egress disabled")
    BATCH_WRITER.start()
    global worker_task
    worker_task = asyncio.create_task(run_worker())
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cancel the background worker on application shutdown."""

    await _initiate_shutdown("lifespan")


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    dependencies=[Depends(require_roles("operator"))],
)
def metrics_endpoint() -> Response:
    """Expose Prometheus-formatted metrics for scraping."""

    payload = metrics.latest()
    return Response(content=payload, media_type=metrics.CONTENT_TYPE_LATEST)


if LAB_MODE:

    class TokenRequest(BaseModel):
        actor: str
        role: str
        expires_in: Optional[int] = 3600

    @app.post("/token", summary="Issue JWT for local development")
    def issue_token(payload: TokenRequest) -> Dict[str, str]:
        actor = payload.actor.strip()
        if not actor:
            raise HTTPException(status_code=400, detail="actor is required")
        if payload.role not in ALLOWED_ROLES:
            raise HTTPException(status_code=400, detail="invalid role")
        expires_in = payload.expires_in if payload.expires_in is not None else 3600
        if expires_in <= 0:
            raise HTTPException(status_code=400, detail="expires_in must be positive")
        try:
            token = create_token(actor, payload.role, expires_in=expires_in)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"token": token}


@app.post("/scenarios", summary="Create a scenario")
async def create_scenario(body: str = Body(..., media_type="text/plain")) -> Dict[str, str]:
    """Persist an uploaded scenario definition and return its identifier."""

    await ensure_directories()
    scenario_id = uuid.uuid4().hex
    path = scenario_path_from_id(scenario_id)
    await _write_text(path, body)
    return {"scenario_id": scenario_id}


@app.post("/runs/{scenario_id}/start", summary="Schedule a run")
async def schedule_run(scenario_id: str) -> Dict[str, Any]:
    """Schedule execution of a stored scenario and return the run identifier."""

    scenario_file = scenario_path_from_id(scenario_id)
    if not await _path_exists(scenario_file):
        raise HTTPException(status_code=404, detail="Scenario not found")

    run_id = uuid.uuid4().hex
    events_path = await run_events_path(run_id)
    await _write_text(events_path, "")
    legacy_events_path = events_path.with_suffix(".json")
    await _write_text(legacy_events_path, "")
    record = RunRecord(
        run_id=run_id,
        scenario_id=scenario_id,
        status="queued",
        events_path=events_path,
    )
    async with runs_lock:
        runs[run_id] = record
    metrics.set_queue_depth(run_id, 0)
    await append_event(record, {"type": "run_created", "run_id": run_id, "scenario_id": scenario_id})
    await job_queue.put(run_id)
    return {"run_id": run_id, "status": record.status}


@app.get("/runs/{run_id}/status", summary="Run status")
async def run_status(run_id: str) -> Dict[str, Any]:
    """Return the current status of a run."""

    async with runs_lock:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        status_payload = {
            "run_id": run_id,
            "scenario_id": record.scenario_id,
            "status": record.status,
            "pending_actions": max(len(record.actions) - record.action_pointer, 0),
        }
        if record.error:
            status_payload["error"] = record.error
    return status_payload


@app.get("/runs/{run_id}/events", summary="Run events")
async def run_events(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: Optional[int] = Query(100, ge=0),
    tail: bool = Query(False),
):
    """Return stored events for a run or stream updates via server-sent events."""

    async with runs_lock:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        events_file = record.events_path

    if tail:
        stream = _tail_events_stream(events_file, offset)
        return StreamingResponse(stream, media_type="text/event-stream")

    events = await _read_ndjson(events_file)
    total = len(events)
    end = offset + limit if limit is not None else None
    sliced = events[offset:end]
    return {
        "run_id": run_id,
        "events": sliced,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.post(
    "/runs/{run_id}/events",
    summary="Ingest run event",
)
async def ingest_run_event(
    run_id: str,
    event: VersionedEventIn,
    actor: Actor = Depends(require_roles("agent_red", "agent_blue")),
) -> JSONResponse:
    """Record an externally produced event for a run."""

    async with runs_lock:
        record = runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    _require_schema_v1(event.schema)
    event_payload_model = event.payload
    if event_payload_model.kind != "simulated":
        return JSONResponse(status_code=400, content={"error": "non-simulated event rejected"})

    event_payload = json.loads(event_payload_model.json(exclude_none=True))
    payload = {"type": "agent_event", "run_id": run_id, **event_payload}
    try:
        enriched_event = await enqueue_run_event(run_id, payload, timeout=0)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail="queue full") from exc
    metrics.record_event(run_id)
    await write_audit(run_id, actor.name, "event_ingested", payload)
    with metrics.track_request_latency("internal", "detections"):
        detections = detector.process_event_for_detections(enriched_event)
    metrics.record_detections(run_id, len(detections))
    return JSONResponse(status_code=202, content=_wrap_payload({"status": "enqueued"}))


@app.get(
    "/runs/{run_id}/queue_stats",
    summary="Run queue stats",
    dependencies=[Depends(require_roles("operator"))],
)
async def run_queue_stats(run_id: str) -> Dict[str, int]:
    """Expose the current depth of the run's event queue."""

    async with runs_lock:
        record_exists = run_id in runs
    if not record_exists and not await _path_exists(RUNS_DIR / run_id):
        raise HTTPException(status_code=404, detail="Run not found")

    depth = JOB_BUS.stats().get(run_id, 0)
    return {"depth": depth}


@app.get("/runs/{run_id}/detections", summary="Run detections")
async def run_detections(run_id: str, since_ts: Optional[str] = None) -> Dict[str, Any]:
    """Return detections generated for a run."""

    async with runs_lock:
        record_exists = run_id in runs
    run_dir = RUNS_DIR / run_id
    if not record_exists and not await _path_exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    raw_detections = detector.get_detections_since(run_id, since_ts)
    validated = [
        json.loads(Detection.parse_obj(detection).json(exclude_none=True))
        for detection in raw_detections
    ]
    return _wrap_payload(validated)


@app.post(
    "/runs/{run_id}/responses",
    summary="Record response action",
)
async def record_run_response(
    run_id: str,
    response: VersionedResponseIn,
    actor: Actor = Depends(require_roles("agent_red", "agent_blue")),
) -> Dict[str, Any]:
    """Persist a blue-team style response associated with a run."""

    async with runs_lock:
        record_exists = run_id in runs
    run_dir = RUNS_DIR / run_id
    if not record_exists and not await _path_exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    _require_schema_v1(response.schema)
    response_payload_model = response.payload
    response_payload = json.loads(response_payload_model.json(exclude_none=True))

    responses_path = await run_responses_path(run_id)
    await _append_text(responses_path, json.dumps(response_payload) + "\n")

    policy_changes = response_payload_model.apply_policy_changes
    if isinstance(policy_changes, dict):
        policy_path = await run_policy_path(run_id)
        if await _path_exists(policy_path):
            try:
                existing_policy = json.loads(await _read_text(policy_path))
            except json.JSONDecodeError:
                existing_policy = {}
        else:
            existing_policy = {}
        existing_policy.update(policy_changes)
        await _write_text(
            policy_path,
            json.dumps(existing_policy, indent=2, sort_keys=True),
        )

    await write_audit(run_id, actor.name, "response_recorded", response_payload)

    return _wrap_payload({"status": "recorded"})


@app.get(
    "/runs/{run_id}/audit/verify",
    summary="Verify run audit log",
    dependencies=[Depends(require_roles("operator"))],
)
async def audit_verify(run_id: str) -> Dict[str, bool]:
    async with runs_lock:
        record_exists = run_id in runs
    run_dir = RUNS_DIR / run_id
    if not record_exists and not await _path_exists(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    ok = await verify_audit(run_id)
    return {"ok": ok}


@app.get("/runs/{run_id}/next", summary="Next action")
async def run_next_action(run_id: str) -> Dict[str, Any]:
    """Return the next queued action for the run or its completion status."""

    event_to_record: Optional[Dict[str, Any]] = None
    completion_event: Optional[Dict[str, Any]] = None

    async with runs_lock:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")

        if record.status in {"queued"}:
            return {"status": record.status}
        if record.status == "error":
            return {"status": "error", "error": record.error}
        if record.action_pointer < len(record.actions):
            record.action_pointer += 1
            action = record.actions[record.action_pointer - 1]
            response = {
                "status": "action",
                "sequence": record.action_pointer,
                "action": action,
            }
            event_to_record = {
                "type": "action_dispatched",
                "run_id": run_id,
                "sequence": record.action_pointer,
                "action": action,
            }
            return_value = response
        else:
            if record.status != "completed":
                record.status = "completed"
                completion_event = {
                    "type": "run_completed",
                    "run_id": run_id,
                    "reason": "actions_exhausted",
                }
            return_value = {"status": "completed"}

    if event_to_record is not None:
        await append_event(record, event_to_record)
    if completion_event is not None:
        await append_event(record, completion_event)
    return return_value


@app.get("/health", summary="Service health probe")
async def health() -> Dict[str, str]:
    """Simple readiness probe returning the orchestrator status."""

    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
