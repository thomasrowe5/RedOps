import random
import sys
from pathlib import Path

import pytest

# Ensure the repository root is importable when tests are run in isolation.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.reward_model import reward_for_event


@pytest.fixture
def sample_events():
    return [
        {
            "tactic": "initial-access",
            "technique": "T1566",
            "kind": "phishing",
            "agent_id": "agent-001",
            "timestamp": "2024-04-22T10:00:00Z",
            "detect_flag": True,
        },
        {
            "tactic": "lateral-movement",
            "technique": "T1105",
            "kind": "remote-service",
            "agent_id": "agent-002",
            "timestamp": "2024-04-22T10:05:00Z",
            "impact_est": 3.2,
            "detect_flag": False,
        },
        {
            "tactic": "collection",
            "technique": "T1041",
            "kind": "exfil",
            "agent_id": "agent-003",
            "timestamp": "2024-04-22T10:10:00Z",
            "impact_est": 2.8,
            "detect_flag": True,
        },
        {
            "tactic": "unknown",
            "technique": "T9999",
            "detect_flag": False,
        },
    ]


def test_reward_model_deterministic_outputs(sample_events):
    rng = random.Random(42)
    chosen_events = rng.sample(sample_events, 3)

    exposure = 0.0
    rewards = []
    for event in chosen_events:
        rewards.append(reward_for_event(event, exposure))
        exposure += 0.1

    assert rewards == pytest.approx([1.2, 0.95, 3.1])
