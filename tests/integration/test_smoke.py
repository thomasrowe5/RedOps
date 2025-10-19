"""Integration smoke test covering orchestrator and agent interaction."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import httpx


def _reserve_free_port() -> int:
    """Ask the OS for an available TCP port bound to localhost."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for(predicate: Callable[[], bool], *, timeout: float = 30.0, interval: float = 0.2) -> None:
    """Poll ``predicate`` until it returns ``True`` or ``timeout`` expires."""

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as exc:  # pragma: no cover - defensive logging path
            last_error = exc
        time.sleep(interval)
    message = "Timed out waiting for condition"
    if last_error is not None:
        raise TimeoutError(f"{message}: {last_error}") from last_error
    raise TimeoutError(message)


def _terminate_processes(processes: Iterable[subprocess.Popen]) -> None:
    """Attempt to terminate all subprocesses gracefully."""

    for proc in processes:
        if proc is None:
            continue
        if proc.poll() is not None:
            continue
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_smoke_orchestrator_and_agent() -> None:
    """Exercise orchestrator API together with a real agent subprocess."""

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    port = _reserve_free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH"), str(repo_root)]))

    orchestrator_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "orchestrator.app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    orchestrator_proc = subprocess.Popen(
        orchestrator_cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    agent_proc: subprocess.Popen | None = None
    client = httpx.Client(base_url=base_url, timeout=5.0)

    try:
        def orchestrator_ready() -> bool:
            if orchestrator_proc.poll() is not None:
                raise RuntimeError(f"Orchestrator exited early with code {orchestrator_proc.returncode}")
            response = client.get("/health")
            response.raise_for_status()
            return response.json().get("status") == "ok"

        _wait_for(orchestrator_ready, timeout=30.0)

        scenario_body = "\n".join(
            [
                "steps:",
                "  - description: Test agent action",
                "    tactic: execution",
                "    technique: T1059",
            ]
        )
        response = client.post("/scenarios", content=scenario_body, headers={"Content-Type": "text/plain"})
        response.raise_for_status()
        scenario_id = response.json()["scenario_id"]

        response = client.post(f"/runs/{scenario_id}/start")
        response.raise_for_status()
        run_id = response.json()["run_id"]

        def run_is_active() -> bool:
            status_response = client.get(f"/runs/{run_id}/status")
            status_response.raise_for_status()
            payload = status_response.json()
            if payload.get("status") == "error":
                raise RuntimeError(f"Run entered error state: {payload}")
            return payload.get("status") in {"running", "completed"}

        _wait_for(run_is_active, timeout=30.0)

        agent_cmd = [
            sys.executable,
            "-m",
            "agents.agent_python.agent",
            "--orchestrator",
            base_url,
            "--agent-id",
            "test-agent",
            "--run-id",
            run_id,
            "--poll-interval",
            "0.2",
            "--action-delay",
            "0.1",
        ]

        agent_proc = subprocess.Popen(
            agent_cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        events_path = data_dir / "runs" / run_id / "events.json"
        expected_keys = {"timestamp", "type", "run_id", "sequence", "action"}

        def agent_event_present() -> bool:
            if agent_proc and agent_proc.poll() is not None:
                raise RuntimeError(f"Agent exited early with code {agent_proc.returncode}")
            if not events_path.exists():
                return False
            lines = [line.strip() for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                return False
            for line in lines:
                event = json.loads(line)
                if event.get("type") == "action_dispatched" and expected_keys.issubset(event.keys()):
                    action = event.get("action", {})
                    return action.get("description") == "Test agent action"
            return False

        _wait_for(agent_event_present, timeout=60.0, interval=0.5)

        assert events_path.exists(), "Expected events.json to be created"
        with events_path.open("r", encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]

        assert any(event.get("type") == "action_dispatched" for event in events), "Agent event not recorded"
        for event in events:
            assert "timestamp" in event
            assert "type" in event
            assert "run_id" in event

    finally:
        client.close()
        _terminate_processes([agent_proc, orchestrator_proc])

