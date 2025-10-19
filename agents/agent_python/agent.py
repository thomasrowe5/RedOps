"""Template Python agent that communicates with the RedOps orchestrator."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("redops.agent")


class Agent:
    """Simple agent that emits simulated heartbeat events to the orchestrator."""

    def __init__(self, agent_id: str, orchestrator_url: str) -> None:
        self.agent_id = agent_id
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.orchestrator_url, timeout=10.0)
        self._running = False

    async def start(self) -> None:
        """Start sending heartbeat events until stopped."""

        LOGGER.info("Starting agent %s", self.agent_id)
        self._running = True
        while self._running:
            await self.send_heartbeat(
                tactic="initial-access",
                technique="T0001",
                description="simulated step",
            )
            await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the agent's heartbeat loop."""

        LOGGER.info("Stopping agent %s", self.agent_id)
        self._running = False
        await self._client.aclose()

    async def send_heartbeat(self, tactic: str, technique: str, description: str) -> None:
        """Send a simulated heartbeat event to the orchestrator."""

        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "status": "active",
            "detail": "heartbeat",
            "event": {
                "tactic": tactic,
                "technique": technique,
                "description": description,
            },
        }
        endpoint = "/events/heartbeat"
        LOGGER.debug("Sending heartbeat: %s", payload)
        try:
            response = await self._client.post(endpoint, json=payload)
            response.raise_for_status()
            LOGGER.info("Heartbeat acknowledged: %s", response.json())
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            LOGGER.error("Failed to send heartbeat: %s", exc)


async def main() -> None:
    agent = Agent(agent_id="agent-python", orchestrator_url="http://localhost:8000")
    try:
        await agent.start()
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
