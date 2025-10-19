"""Standalone Python agent with local planning policies for simulated actions."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import httpx


LOGGER = logging.getLogger("redops.agent.llm")


@dataclasses.dataclass(frozen=True)
class ActionDefinition:
    """Metadata describing a simulated action."""

    tactic: str
    technique: str
    impact: float
    detectability: float

    @property
    def score(self) -> float:
        """Heuristic score for the action."""

        return self.impact - self.detectability


ACTION_LIBRARY: Sequence[ActionDefinition] = (
    ActionDefinition("recon", "Network Scanning", impact=1.0, detectability=0.4),
    ActionDefinition("recon", "OSINT Profiling", impact=0.6, detectability=0.2),
    ActionDefinition("access", "Phishing Email", impact=1.5, detectability=0.6),
    ActionDefinition("access", "Exploit Web Vulnerability", impact=2.0, detectability=0.8),
    ActionDefinition("persistence", "Create Scheduled Task", impact=1.2, detectability=0.7),
    ActionDefinition("persistence", "Modify Startup Scripts", impact=1.1, detectability=0.5),
    ActionDefinition("lateral", "Pass-the-Hash", impact=1.8, detectability=0.9),
    ActionDefinition("lateral", "Remote Service Abuse", impact=1.6, detectability=0.7),
    ActionDefinition("exfil", "Archive and Compress Data", impact=1.4, detectability=0.6),
    ActionDefinition("exfil", "Exfiltrate Over HTTPS", impact=2.1, detectability=0.9),
)


class Policy:
    """Abstract base class for simulated action policies."""

    name: str

    def next_action(self) -> ActionDefinition:
        raise NotImplementedError


class RandomPolicy(Policy):
    """Policy that selects random actions from the library."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.name = "random"

    def next_action(self) -> ActionDefinition:
        action = self._rng.choice(list(ACTION_LIBRARY))
        LOGGER.debug("Random policy selected action: %s", action)
        return action


class RulePolicy(Policy):
    """Policy that walks through a deterministic sequence of tactics."""

    _ORDER = ("recon", "access", "persistence", "lateral", "exfil")

    def __init__(self) -> None:
        self._index = 0
        self.name = "rule"

    def next_action(self) -> ActionDefinition:
        tactic = self._ORDER[self._index]
        self._index = (self._index + 1) % len(self._ORDER)
        for action in ACTION_LIBRARY:
            if action.tactic == tactic:
                LOGGER.debug("Rule policy selected action: %s", action)
                return action
        # Fallback in case the library is altered unexpectedly.
        LOGGER.debug("Rule policy fell back to first action in library")
        return ACTION_LIBRARY[0]


class LlmHeuristicPolicy(Policy):
    """Local heuristic planner using a simple beam search loop."""

    def __init__(self, beam_width: int = 3, max_depth: int = 4) -> None:
        self.name = "llm"
        self._beam_width = beam_width
        self._max_depth = max_depth
        self._plan: Deque[ActionDefinition] = deque()

    def _generate_plan(self) -> None:
        LOGGER.debug("LLM policy generating new plan with beam_width=%s max_depth=%s", self._beam_width, self._max_depth)
        # Start with each action as a candidate path.
        beams: List[List[ActionDefinition]] = [[action] for action in ACTION_LIBRARY]
        for depth in range(1, self._max_depth):
            expanded: List[List[ActionDefinition]] = []
            for path in beams:
                used_tactics = {step.tactic for step in path}
                for action in ACTION_LIBRARY:
                    if action.tactic in used_tactics:
                        continue
                    expanded.append(path + [action])
            if not expanded:
                break
            beams = sorted(expanded, key=self._path_score, reverse=True)[: self._beam_width]
            LOGGER.debug("LLM policy beam depth %s retained %s candidates", depth + 1, len(beams))

        best_path = max(beams, key=self._path_score)
        LOGGER.debug("LLM policy selected plan: %s", best_path)
        self._plan = deque(best_path)

    @staticmethod
    def _path_score(path: Iterable[ActionDefinition]) -> float:
        return sum(action.score for action in path)

    def next_action(self) -> ActionDefinition:
        if not self._plan:
            self._generate_plan()
        action = self._plan.popleft()
        LOGGER.debug("LLM policy emitted action: %s", action)
        return action


POLICY_TYPES = {
    "random": RandomPolicy,
    "rule": RulePolicy,
    "llm": LlmHeuristicPolicy,
}


@dataclasses.dataclass
class AgentConfig:
    orchestrator: str
    agent_id: str
    run_id: str
    policy: str
    seed: Optional[int]
    poll_interval: float


class Agent:
    """Synchronous agent that polls the orchestrator or generates actions locally."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client = httpx.Client(base_url=self._sanitize_url(config.orchestrator), timeout=10.0)
        self._policy = self._build_policy(config)

    @staticmethod
    def _sanitize_url(raw_url: str) -> str:
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Orchestrator URL must include http or https scheme")
        host = parsed.hostname or ""
        if host not in {"localhost"} and not host.startswith("127."):
            raise ValueError("Unsafe orchestrator host; only localhost or 127.* are allowed")
        return raw_url.rstrip("/")

    def _build_policy(self, config: AgentConfig) -> Policy:
        policy_name = config.policy.lower()
        policy_cls = POLICY_TYPES.get(policy_name)
        if policy_cls is None:
            raise ValueError(f"Unknown policy '{config.policy}'")
        if policy_cls is RandomPolicy:
            rng = random.Random(config.seed)
            return policy_cls(rng)
        if policy_cls is LlmHeuristicPolicy:
            return policy_cls()
        return policy_cls()

    def _fetch_next_action(self) -> Optional[Dict[str, str]]:
        endpoint = f"/runs/{self.config.run_id}/next"
        try:
            response = self._client.get(endpoint)
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            LOGGER.error("Failed to contact orchestrator: %s", exc)
            return None

        if response.status_code == httpx.codes.NO_CONTENT:
            LOGGER.debug("No action available from orchestrator")
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            LOGGER.error("Unexpected response when polling orchestrator: %s", exc)
            return None

        data = response.json()
        if not isinstance(data, dict):
            LOGGER.error("Unexpected action payload: %s", data)
            return None
        LOGGER.debug("Received orchestrator action: %s", data)
        return {
            "tactic": str(data.get("tactic", "unknown")),
            "technique": str(data.get("technique", "unknown")),
            "note": str(data.get("note", "orchestrator")),
        }

    def _post_event(self, payload: Dict[str, str]) -> None:
        endpoint = f"/runs/{self.config.run_id}/events"
        try:
            response = self._client.post(endpoint, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            LOGGER.error("Failed to post event: %s", exc)

    def _build_event(self, action: Dict[str, str], note: str) -> Dict[str, str]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.config.agent_id,
            "tactic": action.get("tactic", "unknown"),
            "technique": action.get("technique", "unknown"),
            "note": note,
            "kind": "simulated",
        }
        LOGGER.debug("Constructed event payload: %s", event)
        return event

    def run(self) -> None:
        LOGGER.info(
            "Agent %s connecting to orchestrator at %s for run %s with policy %s",
            self.config.agent_id,
            self.config.orchestrator,
            self.config.run_id,
            self._policy.name,
        )
        while True:
            action = self._fetch_next_action()
            if action is None:
                generated = self._policy.next_action()
                action = {"tactic": generated.tactic, "technique": generated.technique}
                note = self._policy.name
                LOGGER.info("Generated simulated action via %s policy: %s", self._policy.name, action)
            else:
                note = action.get("note", "orchestrator")
                LOGGER.info("Received action from orchestrator: %s", action)

            event_payload = self._build_event(action, note)
            self._post_event(event_payload)
            time.sleep(self.config.poll_interval)


def parse_args(argv: Optional[Sequence[str]] = None) -> AgentConfig:
    parser = argparse.ArgumentParser(description="RedOps heuristic simulated agent")
    parser.add_argument("--orchestrator", default="http://localhost:8000", help="Base URL for the orchestrator")
    parser.add_argument("--agent-id", required=True, help="Identifier for this agent")
    parser.add_argument("--run-id", required=True, help="Run identifier to join")
    parser.add_argument("--policy", choices=sorted(POLICY_TYPES.keys()), default="llm", help="Policy used when generating actions")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for stochastic policies")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between action cycles")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., INFO, DEBUG)")

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    return AgentConfig(
        orchestrator=args.orchestrator,
        agent_id=args.agent_id,
        run_id=args.run_id,
        policy=args.policy,
        seed=args.seed,
        poll_interval=args.poll_interval,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    agent = Agent(config)
    try:
        agent.run()
    except KeyboardInterrupt:  # pragma: no cover - manual interruption path
        LOGGER.info("Agent interrupted by user")


if __name__ == "__main__":
    main(sys.argv[1:])
