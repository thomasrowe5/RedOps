import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest

from analysis import cosim_runner


@pytest.fixture()
def fake_run_dir(tmp_path):
    run_dir = tmp_path / "fake-run"
    run_dir.mkdir()
    yield run_dir


def write_json_lines(path: Path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_compute_metrics_from_run_directory(fake_run_dir):
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    detection_time = base_time + timedelta(seconds=5)
    response_time = base_time + timedelta(seconds=12)

    events = [
        {"timestamp": base_time.isoformat(), "kind": "simulated", "tactic": "Data Exfiltration"},
        {
            "timestamp": (base_time + timedelta(seconds=30)).isoformat(),
            "kind": "simulated",
            "tactic": "Discovery",
        },
        {
            "timestamp": (base_time + timedelta(seconds=40)).isoformat(),
            "kind": "simulated",
            "tactic": "Follow-up Exfil",
        },
    ]
    detections = [
        {"timestamp": detection_time.isoformat(), "id": "det-1", "description": "exfil detected"}
    ]
    responses = [
        {
            "timestamp": response_time.isoformat(),
            "response": "block_egress",
            "agent_id": "blue",
        },
        {
            "timestamp": response_time.isoformat(),
            "apply_policy_changes": {"net_egress_block": True},
        },
    ]

    write_json_lines(fake_run_dir / "events.json", events)
    write_json_lines(fake_run_dir / "detections.json", detections)
    write_json_lines(fake_run_dir / "responses.jsonl", responses)

    metrics = cosim_runner._compute_metrics(fake_run_dir)

    assert metrics["time_to_first_detection_seconds"] == pytest.approx(5.0)
    assert metrics["time_to_first_response_seconds"] == pytest.approx(7.0)
    assert metrics["exfil_attempts"] == 2
    assert metrics["exfil_after_containment"] == 1
    assert metrics["containment_active"] is True
    assert metrics["containment_timestamp"] == response_time.isoformat()
    assert metrics["events_observed"] == 3
    assert metrics["detections_observed"] == 1
    assert metrics["responses_observed"] == 2
