import os, base64, json, time
import pytest
from fastapi.testclient import TestClient

from orchestrator.app.main import app


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("REDOPS_JWT_SECRET", "devsecret")
    yield


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def mk_token(role="agent_red"):
    # Minimal HS256-like mock if app provides dev token endpoint else stub token "dev.agent_red"
    return "dev.agent_red"


def test_missing_jwt_401():
    c = TestClient(app)
    r = c.post(
        "/runs/xx/events",
        json={
            "schema": "redops/v1",
            "payload": {
                "kind": "simulated",
                "agent_id": "A",
                "tactic": "reconnaissance",
                "technique": "T1595",
                "timestamp": "2025-01-01T00:00:00Z",
                "note": "ok",
            },
        },
    )
    assert r.status_code in (401, 403)


def test_rate_limit_and_size(monkeypatch):
    c = TestClient(app)
    tok = mk_token()
    # Oversized payload
    big = "x" * (70_000)
    r = c.post(
        "/runs/rl/events",
        headers=bearer(tok),
        json={
            "schema": "redops/v1",
            "payload": {
                "kind": "simulated",
                "agent_id": "A",
                "tactic": "reconnaissance",
                "technique": "T1595",
                "timestamp": "2025-01-01T00:00:00Z",
                "note": big,
            },
        },
    )
    assert r.status_code in (400, 413)

    # Burst requests
    ok = {
        "schema": "redops/v1",
        "payload": {
            "kind": "simulated",
            "agent_id": "A",
            "tactic": "reconnaissance",
            "technique": "T1595",
            "timestamp": "2025-01-01T00:00:00Z",
            "note": "ok",
        },
    }
    hits = []
    for _ in range(120):
        hits.append(c.post("/runs/rl/events", headers=bearer(tok), json=ok).status_code)
    assert any(code == 429 for code in hits)
