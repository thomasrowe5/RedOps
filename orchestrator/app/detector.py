"""Simple detection utilities for the orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = BASE_DIR / "data" / "runs"

DETECTION_WINDOW_SECONDS = 10
EXFIL_BURST_THRESHOLD = 3
LATERAL_MOVEMENT_THRESHOLD = 4

_recent_events: Dict[str, List[Dict[str, Any]]] = {}


def _now() -> datetime:
    return datetime.utcnow()


def _parse_iso8601(value: str) -> Optional[datetime]:
    try:
        if value.endswith("Z"):
            value = value[:-1]
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _run_directory(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _append_detection(run_id: str, detection: Dict[str, Any]) -> None:
    detections_path = _run_directory(run_id) / "detections.json"
    with detections_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(detection) + "\n")


def process_event_for_detections(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply lightweight rule detections to an incoming event."""

    run_id = event.get("run_id")
    if not run_id:
        return []

    tactic = event.get("tactic")
    timestamp = _now()

    window_start = timestamp - timedelta(seconds=DETECTION_WINDOW_SECONDS)
    recent_for_run = _recent_events.setdefault(run_id, [])
    recent_for_run.append({"timestamp": timestamp, "tactic": tactic})
    _recent_events[run_id] = [
        entry for entry in recent_for_run if entry["timestamp"] >= window_start
    ]

    detections: List[Dict[str, Any]] = []

    if tactic == "exfiltration":
        count = sum(1 for entry in _recent_events[run_id] if entry["tactic"] == tactic)
        if count > EXFIL_BURST_THRESHOLD:
            detections.append(
                {
                    "timestamp": timestamp.isoformat() + "Z",
                    "run_id": run_id,
                    "type": "exfil_burst",
                    "tactic": tactic,
                    "source_event": event,
                }
            )

    if tactic == "lateral-movement":
        count = sum(1 for entry in _recent_events[run_id] if entry["tactic"] == tactic)
        if count >= LATERAL_MOVEMENT_THRESHOLD:
            detections.append(
                {
                    "timestamp": timestamp.isoformat() + "Z",
                    "run_id": run_id,
                    "type": "lateral_suspect",
                    "tactic": tactic,
                    "source_event": event,
                }
            )

    if tactic == "privilege-escalation":
        detections.append(
            {
                "timestamp": timestamp.isoformat() + "Z",
                "run_id": run_id,
                "type": "priv_esc",
                "tactic": tactic,
                "source_event": event,
            }
        )

    for detection in detections:
        _append_detection(run_id, detection)

    return detections


def get_detections_since(run_id: str, since_ts_iso: Optional[str]) -> List[Dict[str, Any]]:
    """Return detections for a run optionally filtered by timestamp."""

    detections_path = _run_directory(run_id) / "detections.json"
    if not detections_path.exists():
        return []

    since_dt = _parse_iso8601(since_ts_iso) if since_ts_iso else None
    results: List[Dict[str, Any]] = []
    with detections_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_dt is not None:
                ts_value = record.get("timestamp")
                if not isinstance(ts_value, str):
                    continue
                parsed_ts = _parse_iso8601(ts_value)
                if parsed_ts is None or parsed_ts <= since_dt:
                    continue
            results.append(record)
    return results
