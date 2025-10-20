"""Local RedOps scenario runner for lab-mode experimentation.

This utility automates end-to-end execution of the orchestrator and a single
agent for a configurable number of runs.  The implementation is intentionally
limited to "lab mode" where every component is executed on the local machine
and only loopback network addresses are contacted.

For each run the script performs the following steps:

* ensure the orchestrator API is running locally (starting it if required),
* create a temporary scenario through the orchestrator API,
* launch the Python agent with the requested policy,
* allow the scenario to execute for a fixed duration before stopping the agent,
* collect the generated ``events.json`` artefact, and
* compute detection metrics using :mod:`analysis.score`.

The aggregated results are written to ``results/summary_<timestamp>.json`` so
subsequent analysis can be performed offline.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import httpx

from analysis.score import compute_metrics
from agents.agent_python.agent_llm import ACTION_LIBRARY


LOGGER = logging.getLogger("redops.analysis.runner")


DEFAULT_ORCHESTRATOR_HOST = "127.0.0.1"
DEFAULT_ORCHESTRATOR_PORT = 8000
RUN_DURATION_SECONDS = 30.0


@dataclass
class RunSummary:
    """Container for the final metrics associated with a single run."""

    run_id: str
    policy: str
    num_events: int
    detection_rate: float
    mttd: Optional[float]
    avg_impact: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the runner."""

    parser = argparse.ArgumentParser(description="Execute RedOps lab scenarios locally")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of independent runs to execute")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for stochastic policies")
    parser.add_argument(
        "--policy",
        choices=("random", "rule", "llm"),
        default="llm",
        help="Policy used by the Python agent when generating actions",
    )
    return parser.parse_args(argv)


def ensure_lab_mode(url: str) -> None:
    """Abort execution if the orchestrator URL points to a non-local host."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in {"localhost", "127.0.0.1"} and not host.startswith("127."):
        raise SystemExit("Lab mode restriction: orchestrator host must resolve to localhost")


class OrchestratorProcess:
    """Manage the lifecycle of the local orchestrator subprocess."""

    def __init__(self, project_root: Path, host: str, port: int) -> None:
        self.project_root = project_root
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self._process: Optional[subprocess.Popen[bytes]] = None

    def ensure_running(self) -> None:
        if self._is_healthy():
            LOGGER.info("Using orchestrator already running at %s", self.url)
            return

        LOGGER.info("Starting orchestrator locally at %s", self.url)
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "orchestrator.app.main:app",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(self.project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_health()

    def _wait_for_health(self, timeout: float = 15.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_healthy():
                return
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("Timed out waiting for orchestrator health check")

    def _is_healthy(self) -> bool:
        try:
            with httpx.Client(base_url=self.url, timeout=1.0) as client:
                response = client.get("/health")
            return response.status_code == httpx.codes.OK
        except httpx.HTTPError:
            return False

    def stop(self) -> None:
        if self._process is None:
            return
        LOGGER.info("Stopping orchestrator process")
        self._terminate_process(self._process)
        self._process = None

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes], timeout: float = 5.0) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            LOGGER.warning("Process did not terminate gracefully; forcing kill")
            process.kill()
            process.wait()


def create_scenario(client: httpx.Client, base_steps: Iterable[Dict[str, Any]]) -> str:
    """Create a transient scenario through the orchestrator API."""

    # Build a compact JSON document that is also valid YAML.
    steps_list = []
    for index, step in enumerate(base_steps, start=1):
        payload = {"id": index}
        payload.update(step)
        steps_list.append(payload)
    body = json.dumps({"steps": steps_list}, indent=2)

    response = client.post("/scenarios", content=body, headers={"Content-Type": "text/plain"})
    response.raise_for_status()
    payload = response.json()
    scenario_id = payload.get("scenario_id")
    if not scenario_id:
        raise RuntimeError("Scenario creation did not return an identifier")
    LOGGER.debug("Created scenario %s", scenario_id)
    return str(scenario_id)


def schedule_run(client: httpx.Client, scenario_id: str) -> str:
    """Request execution of ``scenario_id`` and return the run identifier."""

    response = client.post(f"/runs/{scenario_id}/start")
    response.raise_for_status()
    payload = response.json()
    run_id = payload.get("run_id")
    if not run_id:
        raise RuntimeError("Run scheduling failed to return a run_id")
    LOGGER.info("Scheduled run %s for scenario %s", run_id, scenario_id)
    return str(run_id)


def launch_agent(project_root: Path, run_id: str, policy: str, seed: Optional[int], run_index: int) -> subprocess.Popen[bytes]:
    """Start the standalone Python agent for ``run_id``."""

    agent_script = project_root / "agents" / "agent_python" / "agent_llm.py"
    agent_id = f"agent_{run_index:03d}"
    command = [
        sys.executable,
        str(agent_script),
        "--agent-id",
        agent_id,
        "--run-id",
        run_id,
        "--policy",
        policy,
        "--orchestrator",
        f"http://{DEFAULT_ORCHESTRATOR_HOST}:{DEFAULT_ORCHESTRATOR_PORT}",
        "--poll-interval",
        "1.5",
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])

    LOGGER.info("Launching agent process for run %s with policy %s", run_id, policy)
    return subprocess.Popen(command, cwd=str(project_root))


def terminate_agent(process: subprocess.Popen[bytes]) -> None:
    """Stop the agent process gracefully."""

    if process.poll() is not None:
        return
    LOGGER.debug("Terminating agent process (pid=%s)", process.pid)
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        LOGGER.warning("Agent did not exit on SIGINT; sending terminate")
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            LOGGER.warning("Agent still running; killing process")
            process.kill()
            process.wait()


def read_event_log(path: Path) -> List[Dict[str, Any]]:
    """Return the list of JSON objects stored in ``path`` as JSON lines."""

    events: List[Dict[str, Any]] = []
    if not path.exists():
        LOGGER.warning("Event log %s does not exist", path)
        return events

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                LOGGER.debug("Skipping malformed event line: %s", text)
                continue
            if isinstance(payload, dict):
                events.append(payload)
    LOGGER.debug("Loaded %s raw events from %s", len(events), path)
    return events


def load_detection_file(path: Path) -> List[Dict[str, Any]]:
    """Load detection results stored alongside a run, if present."""

    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        LOGGER.warning("Failed to read detections file %s: %s", path, exc)
        return []

    if not content:
        return []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            detections = data.get("detections")
            if isinstance(detections, list):
                return [item for item in detections if isinstance(item, dict)]
            return [data]
    except json.JSONDecodeError:
        detections: List[Dict[str, Any]] = []
        for line in content.splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                detections.append(payload)
        return detections
    return []


def _parse_action_payload(action: Any) -> Optional[Dict[str, Any]]:
    """Coerce the stored action payload into a dictionary."""

    if isinstance(action, dict):
        if set(action.keys()) == {"description"} and isinstance(action["description"], str):
            try:
                parsed = json.loads(action["description"])
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return action
    if isinstance(action, str):
        try:
            parsed = json.loads(action)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def normalise_events(raw_events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract action level events suitable for scoring metrics."""

    events: List[Dict[str, Any]] = []
    for entry in raw_events:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "action_dispatched":
            continue
        action = _parse_action_payload(entry.get("action"))
        if not isinstance(action, dict):
            continue
        record = {
            "timestamp": entry.get("timestamp"),
            "tactic": action.get("tactic"),
            "technique": action.get("technique"),
            "impact": action.get("impact"),
            "detectability": action.get("detectability"),
            "note": action.get("note"),
        }
        events.append(record)
    LOGGER.debug("Normalised %s action events", len(events))
    return events


def build_scenario_steps() -> List[Dict[str, Any]]:
    """Create a deterministic list of scenario steps from the action library."""

    steps: List[Dict[str, Any]] = []
    for action in ACTION_LIBRARY[:5]:
        steps.append(
            {
                "tactic": action.tactic,
                "technique": action.technique,
                "impact": action.impact,
                "detectability": action.detectability,
                "note": f"seed_step_{action.tactic}",
            }
        )
    return steps


def execute_run(
    project_root: Path,
    client: httpx.Client,
    run_index: int,
    policy: str,
    seed: Optional[int],
) -> RunSummary:
    """Execute a single run and return the aggregated metrics."""

    scenario_steps = build_scenario_steps()
    scenario_id = create_scenario(client, scenario_steps)
    run_id = schedule_run(client, scenario_id)

    agent_process = launch_agent(project_root, run_id, policy, seed, run_index)
    try:
        LOGGER.info("Allowing run %s to execute for %s seconds", run_id, RUN_DURATION_SECONDS)
        time.sleep(RUN_DURATION_SECONDS)
    finally:
        terminate_agent(agent_process)

    # Allow the orchestrator a brief window to flush any pending writes.
    time.sleep(1.0)

    run_dir = project_root / "data" / "runs" / run_id
    events_path = run_dir / "events.json"
    raw_events = read_event_log(events_path)
    action_events = normalise_events(raw_events)

    detections_path = run_dir / "detections.json"
    detection_hits = load_detection_file(detections_path)

    metrics = compute_metrics(action_events, detection_hits)

    avg_impact = 0.0
    impacts = [event.get("impact") for event in action_events if isinstance(event.get("impact"), (int, float))]
    if impacts:
        avg_impact = float(sum(float(value) for value in impacts) / len(impacts))

    summary = RunSummary(
        run_id=run_id,
        policy=policy,
        num_events=len(action_events),
        detection_rate=float(metrics.overall_detection_rate),
        mttd=float(metrics.mttd_seconds) if metrics.mttd_seconds is not None else None,
        avg_impact=avg_impact,
    )
    LOGGER.info(
        "Run %s summary: events=%s detection_rate=%.2f avg_impact=%.2f",
        run_id,
        summary.num_events,
        summary.detection_rate,
        summary.avg_impact,
    )
    return summary


def summarise_results(results: Sequence[RunSummary], project_root: Path) -> Path:
    """Persist the aggregated run summaries to disk."""

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = results_dir / f"summary_{timestamp}.json"

    payload = [
        {
            "run_id": item.run_id,
            "policy": item.policy,
            "num_events": item.num_events,
            "detection_rate": item.detection_rate,
            "mttd": item.mttd,
            "avg_impact": item.avg_impact,
        }
        for item in results
    ]

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote results summary to %s", output_path)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    orchestrator_url = f"http://{DEFAULT_ORCHESTRATOR_HOST}:{DEFAULT_ORCHESTRATOR_PORT}"
    ensure_lab_mode(orchestrator_url)

    orchestrator = OrchestratorProcess(project_root, DEFAULT_ORCHESTRATOR_HOST, DEFAULT_ORCHESTRATOR_PORT)
    orchestrator.ensure_running()

    client = httpx.Client(base_url=orchestrator.url, timeout=10.0)
    results: List[RunSummary] = []

    try:
        for index in range(args.num_runs):
            run_seed = args.seed + index if args.seed is not None else None
            results.append(execute_run(project_root, client, index, args.policy, run_seed))
    finally:
        client.close()
        orchestrator.stop()

    summarise_results(results, project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

