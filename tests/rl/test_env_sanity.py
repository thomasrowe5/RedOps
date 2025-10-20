from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure the repository root is importable when tests are run in isolation.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

np = pytest.importorskip("numpy")
import httpx

from gym_redops.redops_env import RedOpsEnv


class DummyResponse:
    def __init__(self, json_data: Dict[str, Any] | None = None, status_code: int = 200):
        self._json_data = json_data or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:  # pragma: no cover - defensive, nothing to raise
        return None

    def json(self) -> Dict[str, Any]:
        return self._json_data


class DummyClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.posts: List[Dict[str, Any]] = []

    def post(self, url: str, json: Dict[str, Any] | None = None) -> DummyResponse:
        self.posts.append({"url": url, "json": json})
        return DummyResponse()

    def get(self, url: str) -> DummyResponse:
        return DummyResponse({"score": 0.25})

    def close(self) -> None:
        return None


@pytest.fixture
def mocked_http_client(monkeypatch: pytest.MonkeyPatch) -> DummyClient:
    dummy_client = DummyClient()

    def factory(*args: Any, **kwargs: Any) -> DummyClient:
        return dummy_client

    monkeypatch.setattr(httpx, "Client", factory)
    return dummy_client


def test_env_reset_and_step(mocked_http_client: DummyClient) -> None:
    env = RedOpsEnv(
        orchestrator="http://localhost:8000",
        run_id="test-run",
        agent_id="agent-007",
        max_steps=3,
    )

    observation, info = env.reset(seed=123)

    assert env.observation_space.contains(observation)
    assert observation["last_tactic_onehot"].shape == env.observation_space["last_tactic_onehot"].shape
    assert info["exposure"] == pytest.approx(0.0)

    next_obs, reward, done, step_info = env.step(0)

    assert env.observation_space.contains(next_obs)
    assert isinstance(next_obs["last_detection_score"], np.floating)
    assert isinstance(reward, float)
    assert done in {False, True}
    assert "event" in step_info
    assert mocked_http_client.posts[0]["url"].endswith("/runs/test-run/reset")
    assert mocked_http_client.posts[1]["url"].endswith("/runs/test-run/events")

    env.close()
