"""FastAPI application for the RedOps orchestrator."""

from __future__ import annotations

LAB_MODE = True

import asyncio
import contextlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import detector

try:  # Optional dependency used to parse scenario YAML.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML may be absent in some environments.
    yaml = None  # type: ignore

LOGGER = logging.getLogger("redops.orchestrator")

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


@dataclass
class RunRecord:
    """In-memory representation of a scheduled run."""

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


def ensure_directories() -> None:
    """Create the directories required for storing scenarios and run artifacts."""

    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def scenario_path_from_id(scenario_id: str) -> Path:
    """Return the filesystem path used to persist a scenario body."""

    return SCENARIOS_DIR / f"{scenario_id}.yaml"


def run_events_path(run_id: str) -> Path:
    """Return the filesystem path for storing JSON events for a run."""

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "events.json"


def run_responses_path(run_id: str) -> Path:
    """Return the filesystem path for storing response actions for a run."""

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "responses.jsonl"


def run_policy_path(run_id: str) -> Path:
    """Return the filesystem path for storing run policy state."""

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "policy.json"


def append_event(record: RunRecord, event: Dict[str, Any]) -> None:
    """Append a JSON event with timestamp metadata to the run's event log."""

    enriched = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **event,
    }
    record.events_path.parent.mkdir(parents=True, exist_ok=True)
    with record.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(enriched) + "\n")


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
        run_id = await job_queue.get()
        try:
            async with runs_lock:
                record = runs.get(run_id)
                if record is None:
                    continue
                record.status = "running"
            append_event(record, {"type": "run_started", "run_id": run_id})

            scenario_file = scenario_path_from_id(record.scenario_id)
            try:
                scenario_body = scenario_file.read_text(encoding="utf-8")
            except FileNotFoundError:
                async with runs_lock:
                    record.status = "error"
                    record.error = "Scenario definition not found"
                append_event(
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
            append_event(
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
                append_event(
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
                append_event(
                    record,
                    {"type": "run_error", "run_id": run_id, "message": record.error},
                )
        finally:
            job_queue.task_done()


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise directories and launch the background worker."""

    ensure_directories()
    LOGGER.info("RedOps orchestrator running in LAB MODE (local only)")
    global worker_task
    worker_task = asyncio.create_task(run_worker())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cancel the background worker on application shutdown."""

    if worker_task is not None:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


@app.post("/scenarios", summary="Create a scenario")
async def create_scenario(body: str = Body(..., media_type="text/plain")) -> Dict[str, str]:
    """Persist an uploaded scenario definition and return its identifier."""

    ensure_directories()
    scenario_id = uuid.uuid4().hex
    path = scenario_path_from_id(scenario_id)
    path.write_text(body, encoding="utf-8")
    return {"scenario_id": scenario_id}


@app.post("/runs/{scenario_id}/start", summary="Schedule a run")
async def schedule_run(scenario_id: str) -> Dict[str, Any]:
    """Schedule execution of a stored scenario and return the run identifier."""

    scenario_file = scenario_path_from_id(scenario_id)
    if not scenario_file.exists():
        raise HTTPException(status_code=404, detail="Scenario not found")

    run_id = uuid.uuid4().hex
    events_path = run_events_path(run_id)
    events_path.write_text("", encoding="utf-8")
    record = RunRecord(
        scenario_id=scenario_id,
        status="queued",
        events_path=events_path,
    )
    async with runs_lock:
        runs[run_id] = record
    append_event(record, {"type": "run_created", "run_id": run_id, "scenario_id": scenario_id})
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
async def run_events(run_id: str) -> Dict[str, Any]:
    """Return the stored events for a run as a list of JSON objects."""

    async with runs_lock:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        events_file = record.events_path
    events: List[Dict[str, Any]] = []
    if events_file.exists():
        with events_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        events.append({"raw": line})
    return {"run_id": run_id, "events": events}


@app.post("/runs/{run_id}/events", summary="Ingest run event")
async def ingest_run_event(run_id: str, event: Dict[str, Any] = Body(...)) -> Dict[str, str]:
    """Record an externally produced event for a run."""

    async with runs_lock:
        record = runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if event.get("kind") != "simulated":
        return JSONResponse(status_code=400, content={"error": "non-simulated event rejected"})

    enriched_event = {"type": "agent_event", "run_id": run_id, **event}
    append_event(record, enriched_event)
    detector.process_event_for_detections(enriched_event)
    return {"status": "accepted"}


@app.get("/runs/{run_id}/detections", summary="Run detections")
async def run_detections(run_id: str, since_ts: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return detections generated for a run."""

    async with runs_lock:
        record_exists = run_id in runs
    run_dir = RUNS_DIR / run_id
    if not record_exists and not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    return detector.get_detections_since(run_id, since_ts)


@app.post("/runs/{run_id}/responses", summary="Record response action")
async def record_run_response(run_id: str, response: Dict[str, Any] = Body(...)) -> Dict[str, str]:
    """Persist a blue-team style response associated with a run."""

    async with runs_lock:
        record_exists = run_id in runs
    run_dir = RUNS_DIR / run_id
    if not record_exists and not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    responses_path = run_responses_path(run_id)
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with responses_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(response) + "\n")

    policy_changes = response.get("apply_policy_changes")
    if isinstance(policy_changes, dict):
        policy_path = run_policy_path(run_id)
        if policy_path.exists():
            try:
                existing_policy = json.loads(policy_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_policy = {}
        else:
            existing_policy = {}
        existing_policy.update(policy_changes)
        policy_path.write_text(json.dumps(existing_policy, indent=2, sort_keys=True), encoding="utf-8")

    return {"status": "recorded"}


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
        append_event(record, event_to_record)
    if completion_event is not None:
        append_event(record, completion_event)
    return return_value


@app.get("/health", summary="Service health probe")
async def health() -> Dict[str, str]:
    """Simple readiness probe returning the orchestrator status."""

    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
