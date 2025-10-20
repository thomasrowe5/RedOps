"""Gym environment wrapper around the RedOps orchestrator."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import gym
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
import httpx
import numpy as np

from agents.agent_python.agent_llm import ACTION_LIBRARY

LOGGER = logging.getLogger("redops.gym.env")


@dataclass
class RewardOutcome:
    """Container describing the outcome of a reward calculation."""

    reward: float
    exposure: float
    detection_score: float


class SimpleRewardModel:
    """Heuristic reward model for the RedOps simulation environment."""

    def __init__(self) -> None:
        self._impact: Dict[tuple[str, str], float] = {}
        self._detectability: Dict[tuple[str, str], float] = {}
        for action in ACTION_LIBRARY:
            key = (action.tactic, action.technique)
            self._impact[key] = float(action.impact)
            self._detectability[key] = float(action.detectability)

    def reward_for_event(
        self,
        event: Dict[str, Any],
        detection: Optional[Dict[str, Any]] = None,
        previous_exposure: float = 0.0,
    ) -> RewardOutcome:
        """Compute a reward for ``event`` given optional detection feedback."""

        tactic = str(event.get("tactic", "unknown"))
        technique = str(event.get("technique", "unknown"))
        key = (tactic, technique)
        impact = self._impact.get(key, 1.0)
        detectability = self._detectability.get(key, 0.5)

        detection_score = 0.0
        if isinstance(detection, dict):
            score = detection.get("score")
            if score is not None:
                try:
                    detection_score = max(0.0, float(score))
                except (TypeError, ValueError):
                    detection_score = 0.0

        detection_penalty = detection_score * (0.5 + detectability)
        reward = impact - detection_penalty

        exposure_delta = impact * 0.05 + detectability * 0.02 - detection_score * 0.03
        exposure = float(np.clip(previous_exposure + exposure_delta, 0.0, 1.0))

        return RewardOutcome(reward=reward, exposure=exposure, detection_score=detection_score)


reward_model = SimpleRewardModel()


CANDIDATE_ACTIONS: List[Dict[str, Any]] = [
    {
        "tactic": action.tactic,
        "technique": action.technique,
        "delay_ms": 0,
        "note": f"library:{action.technique.replace(' ', '_').lower()}",
        "impact": float(action.impact),
        "detectability": float(action.detectability),
    }
    for action in ACTION_LIBRARY
]


class SafetyError(RuntimeError):
    """Raised when a safety constraint is violated."""


class RedOpsEnv(gym.Env):
    """Reinforcement-learning compatible interface for the RedOps orchestrator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        orchestrator: str = "http://localhost:8000",
        run_id: str = "",
        agent_id: str = "",
        max_steps: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._base_url = self._sanitize_orchestrator_url(orchestrator)
        self.run_id = run_id
        self.agent_id = agent_id
        self.max_steps = int(max(1, max_steps))
        self._step_count = 0
        self._exposure = 0.0
        self._last_detection_score = 0.0
        self._last_tactic: Optional[str] = None
        self._last_observation: Optional[Dict[str, Any]] = None

        self._client = httpx.Client(base_url=self._base_url, timeout=10.0)

        tactics = list(dict.fromkeys(action["tactic"] for action in CANDIDATE_ACTIONS))
        if not tactics:
            raise ValueError("No candidate actions available for environment")
        self._tactic_index = {tactic: idx for idx, tactic in enumerate(tactics)}

        self.action_space = spaces.Discrete(len(CANDIDATE_ACTIONS))
        self.observation_space = spaces.Dict(
            {
                "exposure_estimate": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "last_detection_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "last_tactic_onehot": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(self._tactic_index),),
                    dtype=np.float32,
                ),
                "step_index": spaces.Box(low=0, high=self.max_steps, shape=(), dtype=np.int32),
            }
        )

        self._np_random: Optional[np.random.Generator] = None
        self._python_rng = random.Random()
        self.seed(seed)

    @staticmethod
    def _sanitize_orchestrator_url(raw_url: str) -> str:
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise SafetyError("Orchestrator URL must include http or https scheme")
        host = parsed.hostname or ""
        if host not in {"localhost"} and not host.startswith("127."):
            raise SafetyError("Unsafe orchestrator host; only localhost or 127.* are allowed")
        return raw_url.rstrip("/")

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the environment's random number generators."""

        if seed is None:
            seed = random.randrange(0, 2**32 - 1)
        self._np_random, actual_seed = seeding.np_random(seed)
        self._python_rng.seed(actual_seed)
        return [actual_seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment state and notify the orchestrator."""

        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._step_count = 0
        self._exposure = 0.0
        self._last_detection_score = 0.0
        self._last_tactic = None

        self._notify_orchestrator_reset(options)

        self._last_observation = self._build_observation()
        info = {"exposure": self._exposure}
        return self._last_observation, info

    def step(self, action: int):
        """Execute a simulated action and update the environment state."""

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action index: {action}")

        candidate = CANDIDATE_ACTIONS[action]
        self._step_count += 1
        self._last_tactic = candidate["tactic"]

        event_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "tactic": candidate["tactic"],
            "technique": candidate["technique"],
            "delay_ms": candidate["delay_ms"],
            "note": candidate["note"],
            "kind": "simulated",
        }
        self._post_event(event_payload)

        detection_feedback = self._fetch_detection_feedback()
        reward_outcome = reward_model.reward_for_event(
            candidate,
            detection=detection_feedback,
            previous_exposure=self._exposure,
        )

        reward = float(reward_outcome.reward)
        self._exposure = float(reward_outcome.exposure)
        self._last_detection_score = float(reward_outcome.detection_score)

        observation = self._build_observation()
        self._last_observation = observation
        done = self._step_count >= self.max_steps
        info = {
            "event": event_payload,
            "detection": detection_feedback,
            "exposure": self._exposure,
        }
        return observation, reward, done, info

    def close(self) -> None:
        """Release network resources associated with the environment."""

        try:
            self._client.close()
        except Exception:  # pragma: no cover - defensive cleanup
            LOGGER.debug("Exception while closing HTTP client", exc_info=True)
        finally:
            super().close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_observation(self) -> Dict[str, Any]:
        onehot = np.zeros(len(self._tactic_index), dtype=np.float32)
        if self._last_tactic in self._tactic_index:
            onehot[self._tactic_index[self._last_tactic]] = 1.0
        observation = {
            "exposure_estimate": np.float32(self._exposure),
            "last_detection_score": np.float32(self._last_detection_score),
            "last_tactic_onehot": onehot,
            "step_index": np.int32(self._step_count),
        }
        return observation

    def _notify_orchestrator_reset(self, options: Optional[Dict[str, Any]]) -> None:
        if not self.run_id:
            return
        payload = {
            "agent_id": self.agent_id,
            "kind": "simulated",
        }
        if isinstance(options, dict):
            payload.update({k: v for k, v in options.items() if isinstance(k, str)})
        try:
            response = self._client.post(f"/runs/{self.run_id}/reset", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.debug("Reset notification failed: %s", exc)

    def _post_event(self, event: Dict[str, Any]) -> None:
        if not self.run_id:
            return
        try:
            response = self._client.post(f"/runs/{self.run_id}/events", json=event)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.debug("Failed to post event: %s", exc)

    def _fetch_detection_feedback(self) -> Optional[Dict[str, Any]]:
        if not self.run_id:
            return None
        try:
            response = self._client.get(f"/runs/{self.run_id}/detections/latest")
        except httpx.HTTPError as exc:
            LOGGER.debug("Detection feedback request failed: %s", exc)
            return None
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            return None
        data = response.json()
        return data if isinstance(data, dict) else None


try:
    register("redops-v0", entry_point="gym_redops.redops_env:RedOpsEnv")
except gym.error.Error:
    LOGGER.debug("redops-v0 environment already registered")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = gym.make(
        "redops-v0",
        orchestrator="http://localhost:8000",
        run_id="demo-run",
        agent_id="demo-agent",
        max_steps=5,
    )
    try:
        observation, info = env.reset()
        LOGGER.info("Initial observation: %s", observation)
        LOGGER.info("Reset info: %s", info)
        for step in range(3):
            action = env.action_space.sample()
            observation, reward, done, step_info = env.step(action)
            LOGGER.info(
                "Step %s action=%s reward=%.3f done=%s info=%s",
                step + 1,
                action,
                reward,
                done,
                step_info,
            )
            if done:
                break
    finally:
        env.close()
