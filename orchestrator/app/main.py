"""FastAPI application for the RedOps orchestrator."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RedOps Orchestrator", version="0.1.0")


class HeartbeatEvent(BaseModel):
    """Schema representing a simulated heartbeat from an agent."""

    agent_id: str
    status: str
    detail: str


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness probe returning the orchestrator status."""

    return {"status": "ok"}


@app.post("/events/heartbeat")
def receive_heartbeat(event: HeartbeatEvent) -> dict[str, str]:
    """Accept simulated heartbeat events from registered agents."""

    # In a real system this would persist to telemetry or message bus.
    # For the skeleton we simply acknowledge receipt.
    return {"status": "received", "agent_id": event.agent_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
