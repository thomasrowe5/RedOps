"""Concurrent co-simulation runner for Red and Blue agents in lab mode.

This utility coordinates a Red agent (either the heuristic LLM policy or a
pre-trained RL policy) together with the Blue response agent against an
existing orchestrator run.  Both agents are executed concurrently for a fixed
simulation window before being stopped.  Once the window elapses the script
collates telemetry artefacts produced by the orchestrator and computes summary
metrics describing the detection and response timeline.

Usage example
-------------

.. code-block:: bash

    python analysis/cosim_runner.py --run-id cosim-1 --red-policy llm \
        --seed 42 --duration 45 --orchestrator http://localhost:8000 \
        --blue-config configs/blue_policy.yaml

The resulting JSON summary is written to
``results/cosim_summary_<run-id>.json``.
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
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

LOGGER = logging.getLogger("redops.analysis.cosim")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DATA_DIR = PROJECT_ROOT / "orchestrator" / "data" / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DURATION = 45.0
DEFAULT_RED_AGENT_ID = "red-lab"
DEFAULT_BLUE_AGENT_ID = "blue-lab"
DEFAULT_POLL_INTERVAL = 2.0


@dataclass
class Arguments:
    run_id: str
    orchestrator: str
    duration: float
    red_policy: str
    red_agent_id: str
    blue_agent_id: str
    seed: Optional[int]
    blue_config: Path
    red_rl_policy_path: Optional[Path]
    red_poll_interval: float
    blue_poll_interval: float


def parse_args(argv: Optional[Sequence[str]] = None) -> Arguments:
    parser = argparse.ArgumentParser(description="Run Red and Blue agents concurrently in lab mode")
    parser.add_argument("--run-id", required=True, help="Run identifier to attach agents to")
    parser.add_argument(
        "--orchestrator",
        default="http://localhost:8000",
        help="Base URL for the orchestrator service (must be localhost/127.x)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Number of seconds to allow the co-simulation to run",
    )
    parser.add_argument(
        "--red-policy",
        choices=("llm", "rule", "random"),
        default="llm",
        help="Policy used when launching the heuristic Red agent",
    )
    parser.add_argument("--red-agent-id", default=DEFAULT_RED_AGENT_ID, help="Identifier for the Red agent instance")
    parser.add_argument("--blue-agent-id", default=DEFAULT_BLUE_AGENT_ID, help="Identifier for the Blue agent instance")
    parser.add_argument("--seed", type=int, default=None, help="Random seed passed to stochastic Red policies")
    parser.add_argument("--blue-config", required=True, type=Path, help="Path to the Blue policy YAML configuration")
    parser.add_argument(
        "--red-rl-policy-path",
        type=Path,
        default=None,
        help="Path to a trained RL policy (enables the RL agent instead of the heuristic one)",
    )
    parser.add_argument(
        "--red-poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Polling interval for the heuristic Red agent in seconds",
    )
    parser.add_argument(
        "--blue-poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Polling interval for the Blue agent in seconds",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the runner (DEBUG, INFO, WARNING, ERROR)",
    )

    raw_args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, raw_args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    ensure_lab_mode(raw_args.orchestrator)

    return Arguments(
        run_id=raw_args.run_id,
        orchestrator=raw_args.orchestrator.rstrip("/"),
        duration=max(float(raw_args.duration), 0.0),
        red_policy=raw_args.red_policy,
        red_agent_id=raw_args.red_agent_id,
        blue_agent_id=raw_args.blue_agent_id,
        seed=raw_args.seed,
        blue_config=raw_args.blue_config,
        red_rl_policy_path=raw_args.red_rl_policy_path,
        red_poll_interval=max(raw_args.red_poll_interval, 0.1),
        blue_poll_interval=max(raw_args.blue_poll_interval, 0.1),
    )


def ensure_lab_mode(orchestrator_url: str) -> None:
    parsed = urlparse(orchestrator_url)
    host = (parsed.hostname or "").lower()
    if host not in {"localhost", "127.0.0.1"} and not host.startswith("127."):
        raise SystemExit("Lab-only safety: orchestrator host must be localhost/127.x")


def build_red_command(args: Arguments) -> List[str]:
    if args.red_rl_policy_path is not None:
        policy_path = args.red_rl_policy_path
        if not policy_path.exists():
            raise FileNotFoundError(f"RL policy not found at {policy_path}")
        return [
            sys.executable,
            str(PROJECT_ROOT / "agents" / "agent_rl.py"),
            "--policy-path",
            str(policy_path),
            "--env-url",
            args.orchestrator,
            "--agent-id",
            args.red_agent_id,
            "--run-id",
            args.run_id,
            "--num-episodes",
            "1",
        ]

    command = [
        sys.executable,
        str(PROJECT_ROOT / "agents" / "agent_python" / "agent_llm.py"),
        "--orchestrator",
        args.orchestrator,
        "--agent-id",
        args.red_agent_id,
        "--run-id",
        args.run_id,
        "--policy",
        args.red_policy,
        "--poll-interval",
        f"{args.red_poll_interval}",
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    return command


def build_blue_command(args: Arguments) -> List[str]:
    if not args.blue_config.exists():
        raise FileNotFoundError(f"Blue policy configuration not found at {args.blue_config}")
    return [
        sys.executable,
        str(PROJECT_ROOT / "agents" / "agent_blue.py"),
        "--orchestrator",
        args.orchestrator,
        "--run-id",
        args.run_id,
        "--agent-id",
        args.blue_agent_id,
        "--policy-config",
        str(args.blue_config),
        "--poll-interval",
        f"{args.blue_poll_interval}",
    ]


def launch_process(command: Sequence[str]) -> subprocess.Popen[str]:
    LOGGER.info("Launching process: %s", " ".join(command))
    return subprocess.Popen(command, text=True)


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    LOGGER.debug("Terminating process pid=%s", process.pid)
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            LOGGER.warning("Process %s did not terminate gracefully; killing", process.pid)
            process.kill()
            process.wait()


def read_json_lines(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def parse_timestamp(value: object) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def earliest_timestamp(
    entries: Iterable[Dict[str, object]],
    key: str,
    predicate: Optional[Callable[[Dict[str, object]], bool]] = None,
) -> Optional[datetime]:
    earliest: Optional[datetime] = None
    for entry in entries:
        if predicate and not predicate(entry):
            continue
        ts = parse_timestamp(entry.get(key))
        if ts is None:
            continue
        if earliest is None or ts < earliest:
            earliest = ts
    return earliest


def count_exfiltration_events(events: Iterable[Dict[str, object]]) -> Tuple[int, List[Tuple[datetime, Dict[str, object]]]]:
    results: List[Tuple[datetime, Dict[str, object]]] = []
    count = 0
    for event in events:
        tactic = event.get("tactic")
        if isinstance(tactic, str) and "exfil" in tactic.lower():
            ts = parse_timestamp(event.get("timestamp"))
            if ts is not None:
                results.append((ts, event))
            count += 1
    return count, results


def determine_containment(responses: List[Dict[str, object]], run_dir: Path) -> Tuple[bool, Optional[datetime]]:
    containment_time: Optional[datetime] = None
    containment_active = False
    for response in responses:
        changes = response.get("apply_policy_changes")
        if not isinstance(changes, dict):
            continue
        if bool(changes.get("net_egress_block")):
            ts = parse_timestamp(response.get("timestamp"))
            if ts is not None and (containment_time is None or ts < containment_time):
                containment_time = ts
            containment_active = True
    if not containment_active:
        policy_path = run_dir / "policy.json"
        if policy_path.exists():
            try:
                policy_data = json.loads(policy_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                policy_data = {}
            if isinstance(policy_data, dict) and bool(policy_data.get("net_egress_block")):
                containment_active = True
    return containment_active, containment_time


def compute_metrics(run_id: str) -> Dict[str, object]:
    run_dir = RUN_DATA_DIR / run_id
    events = read_json_lines(run_dir / "events.json")
    detections = read_json_lines(run_dir / "detections.json")
    responses = read_json_lines(run_dir / "responses.jsonl")

    simulated_events = [event for event in events if event.get("kind") == "simulated"]

    first_event_ts = earliest_timestamp(simulated_events, "timestamp")
    first_detection_ts = earliest_timestamp(detections, "timestamp")
    first_response_ts = earliest_timestamp(responses, "timestamp", predicate=lambda item: "response" in item)

    ttd_seconds = (
        (first_detection_ts - first_event_ts).total_seconds()
        if first_event_ts and first_detection_ts
        else None
    )
    mttr_seconds = (
        (first_response_ts - first_detection_ts).total_seconds()
        if first_detection_ts and first_response_ts
        else None
    )

    exfil_attempts, exfil_events = count_exfiltration_events(simulated_events)
    containment_active, containment_time = determine_containment(responses, run_dir)
    if containment_active and containment_time is not None:
        exfil_after_containment = sum(1 for ts, _ in exfil_events if ts >= containment_time)
    else:
        exfil_after_containment = 0

    return {
        "first_event_timestamp": first_event_ts.isoformat() if first_event_ts else None,
        "first_detection_timestamp": first_detection_ts.isoformat() if first_detection_ts else None,
        "first_response_timestamp": first_response_ts.isoformat() if first_response_ts else None,
        "time_to_first_detection_seconds": ttd_seconds,
        "time_to_first_response_seconds": mttr_seconds,
        "exfil_attempts": exfil_attempts,
        "exfil_after_containment": exfil_after_containment,
        "containment_active": containment_active,
        "containment_timestamp": containment_time.isoformat() if containment_time else None,
        "events_observed": len(simulated_events),
        "detections_observed": len(detections),
        "responses_observed": len(responses),
    }


def run_simulation(args: Arguments) -> Dict[str, object]:
    red_cmd = build_red_command(args)
    blue_cmd = build_blue_command(args)

    red_proc = launch_process(red_cmd)
    time.sleep(0.5)  # allow red agent to establish connection before blue starts
    blue_proc = launch_process(blue_cmd)

    start_time = time.time()
    try:
        deadline = start_time + args.duration
        LOGGER.info("Co-simulation running for %.1f seconds", args.duration)
        while time.time() < deadline:
            if red_proc.poll() is not None and blue_proc.poll() is not None:
                LOGGER.info("Both agents exited before duration elapsed")
                break
            time.sleep(1.0)
    finally:
        LOGGER.info("Stopping agents")
        stop_process(red_proc)
        stop_process(blue_proc)

    metrics = compute_metrics(args.run_id)
    return {
        "run_id": args.run_id,
        "orchestrator": args.orchestrator,
        "duration_seconds": args.duration,
        "red_agent": {
            "agent_id": args.red_agent_id,
            "policy": args.red_policy,
            "rl_policy_path": str(args.red_rl_policy_path) if args.red_rl_policy_path else None,
            "seed": args.seed,
        },
        "blue_agent": {
            "agent_id": args.blue_agent_id,
            "policy_config": str(args.blue_config),
        },
        "metrics": metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_summary(summary: Dict[str, object]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"cosim_summary_{summary['run_id']}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    LOGGER.info("Summary written to %s", output_path)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        summary = run_simulation(args)
        write_summary(summary)
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Execution interrupted by user")
        return 130
    except Exception as exc:
        LOGGER.exception("Co-simulation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
