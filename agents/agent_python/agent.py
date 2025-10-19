"""Asynchronous Python agent that communicates with the RedOps orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

LOGGER = logging.getLogger("redops.agent")


@dataclass
class AgentConfig:
    """Runtime configuration for the agent."""

    orchestrator_url: str
    agent_id: str
    run_id: str
    poll_interval: float = 2.0
    action_delay: float = 1.0


class Agent:
    """Agent that polls the orchestrator for simulated actions and posts events."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(base_url=self.config.orchestrator_url.rstrip("/"), timeout=10.0)
        self._running = False

    async def run(self) -> None:
        """Main loop that continually polls for new actions."""

        self._running = True
        LOGGER.info(
            "Agent %s connected to orchestrator at %s for run %s",
            self.config.agent_id,
            self.config.orchestrator_url,
            self.config.run_id,
        )
        try:
            while self._running:
                action = await self._fetch_next_action()
                if action is None:
                    await asyncio.sleep(self.config.poll_interval)
                    continue

                LOGGER.info("Received action: %s", action)
                await asyncio.sleep(self.config.action_delay)
                event_payload = self._build_event(action)
                await self._post_event(event_payload)
        finally:
            await self._client.aclose()

    async def _fetch_next_action(self) -> Optional[Dict[str, Any]]:
        """Retrieve the next action from the orchestrator."""

        endpoint = f"/runs/{self.config.run_id}/next"
        try:
            response = await self._client.get(endpoint)
            if response.status_code == httpx.codes.NO_CONTENT:
                LOGGER.debug("No action available yet")
                return None
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.error("Failed to fetch next action: %s", exc)
            return None

        data = response.json()
        if not data:
            LOGGER.debug("Empty action payload received")
            return None
        return data

    async def _post_event(self, payload: Dict[str, Any]) -> None:
        """Send an event back to the orchestrator."""

        endpoint = f"/runs/{self.config.run_id}/events"
        try:
            response = await self._client.post(endpoint, json=payload)
            response.raise_for_status()
            LOGGER.info("Event sent successfully")
        except httpx.HTTPError as exc:
            LOGGER.error("Failed to send event: %s", exc)

    def _build_event(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Construct the event payload from an action description."""

        tactic = str(action.get("tactic", "unknown"))
        technique = str(action.get("technique", "unknown"))
        note = self._extract_note(action)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.config.agent_id,
            "tactic": tactic,
            "technique": technique,
            "note": note,
        }
        LOGGER.debug("Built event payload: %s", event)
        return event

    @staticmethod
    def _extract_note(action: Dict[str, Any]) -> str:
        """Extract a human-readable note from the action payload."""

        for key in ("note", "description", "details", "message"):
            value = action.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(action)


async def _async_main(config: AgentConfig) -> None:
    agent = Agent(config)
    try:
        await agent.run()
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        LOGGER.info("Agent cancelled")


def parse_args() -> AgentConfig:
    parser = argparse.ArgumentParser(description="RedOps simulated agent")
    parser.add_argument("--orchestrator", required=True, help="Base URL of the orchestrator, e.g. http://localhost:8000")
    parser.add_argument("--agent-id", required=True, help="Identifier for this agent instance")
    parser.add_argument("--run-id", required=True, help="Identifier of the run to participate in")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between polling for new actions",
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=1.0,
        help="Seconds to wait before emitting an event after receiving an action",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    return AgentConfig(
        orchestrator_url=args.orchestrator,
        agent_id=args.agent_id,
        run_id=args.run_id,
        poll_interval=args.poll_interval,
        action_delay=args.action_delay,
    )


def main() -> None:
    config = parse_args()
    try:
        asyncio.run(_async_main(config))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        LOGGER.info("Agent interrupted by user")


if __name__ == "__main__":
    main()
