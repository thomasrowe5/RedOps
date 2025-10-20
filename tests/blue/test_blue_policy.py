import sys
import types
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

requests_stub = types.ModuleType("requests")
requests_stub.Session = lambda: types.SimpleNamespace(get=lambda *a, **k: None, post=lambda *a, **k: None)
requests_stub.RequestException = Exception
sys.modules.setdefault("requests", requests_stub)

import pytest

from agents.agent_blue import BlueAgent, load_policy_config


@pytest.fixture()
def policy_config():
    return load_policy_config("configs/blue_policy.yaml")


class TimeSequencer:
    def __init__(self, values):
        self._iterator = iter(values)

    def __call__(self):
        return next(self._iterator)


def test_policy_actions_and_cooldown(monkeypatch, policy_config):
    agent = BlueAgent(
        orchestrator_base="http://localhost:8000",
        run_id="run-1",
        agent_id="blue-test",
        policy_config=policy_config,
        poll_interval=0.1,
    )

    recorded_responses = []

    def fake_post(url, payload):
        if "response" in payload:
            recorded_responses.append(payload)
        return True

    monkeypatch.setattr(agent, "_post_json", fake_post)
    monkeypatch.setattr(agent, "_run_url", lambda path: path)

    timestamps = [
        datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, 3, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, 12, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, 20, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, 21, tzinfo=timezone.utc),
    ]

    time_values = TimeSequencer([t.timestamp() for t in timestamps])
    monkeypatch.setattr("agents.agent_blue.time.time", time_values)

    detections = [
        {"id": "det-1", "tactic": "Exfiltration", "timestamp": timestamps[0].isoformat()},
        {"id": "det-2", "tactic": "Command and Control", "timestamp": timestamps[1].isoformat()},
        {"id": "det-3", "tactic": "Exfiltration", "timestamp": timestamps[2].isoformat()},
        {"id": "det-4", "tactic": "Exfiltration", "timestamp": timestamps[3].isoformat()},
        {"id": "det-5", "tactic": "Collection", "timestamp": timestamps[4].isoformat()},
        {"id": "det-6", "tactic": "Lateral Movement", "timestamp": timestamps[5].isoformat(), "asset": "db-1"},
        {
            "id": "det-7",
            "tactic": "Credential Access",
            "timestamp": timestamps[6].isoformat(),
            "asset": "db-1",
        },
    ]

    agent.process_detections(detections)

    assert [r["response"] for r in recorded_responses[:2]] == ["block_egress", "rotate_creds"]
    assert all(r["reason"] == "det-3" for r in recorded_responses[:2])

    assert not any(r["reason"] == "det-4" for r in recorded_responses)

    assert [r["response"] for r in recorded_responses[2:4]] == ["block_egress", "rotate_creds"]
    assert all(r["reason"] == "det-5" for r in recorded_responses[2:4])

    assert recorded_responses[4]["response"] == "isolate_service"
    assert recorded_responses[4]["reason"] == "det-7"
