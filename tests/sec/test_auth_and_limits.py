"""Tests for orchestrator authentication and request limits."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

_BACKPRESSURE_MODULE = None


def _load_backpressure_module():
    global _BACKPRESSURE_MODULE
    if _BACKPRESSURE_MODULE is None:
        module_path = Path(__file__).resolve().parents[1] / "orch" / "test_backpressure.py"
        spec = importlib.util.spec_from_file_location("tests.orch.test_backpressure", module_path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Unable to load orchestrator test utilities")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, module)
        spec.loader.exec_module(module)
        _BACKPRESSURE_MODULE = module
    return _BACKPRESSURE_MODULE


@pytest.fixture
def security_app(monkeypatch, tmp_path):
    """Provide a configured orchestrator application instance for security tests."""

    monkeypatch.setenv("REDOPS_JWT_SECRET", "test-secret")
    monkeypatch.setenv("REDOPS_RATE_LIMIT_BURST", "2")
    monkeypatch.setenv("REDOPS_RATE_LIMIT_RATE", "1")

    backpressure = _load_backpressure_module()
    orchestrator_factory = getattr(backpressure, "orchestrator_app")
    factory_callable = getattr(orchestrator_factory, "__wrapped__", orchestrator_factory)
    main_module = factory_callable(monkeypatch, tmp_path)
    monkeypatch.setattr(main_module, "_install_signal_handlers", lambda *_args, **_kwargs: None)

    async def _enqueue_stub(run_id: str, event: Dict[str, object], *, timeout=None) -> Dict[str, object]:
        return {"timestamp": datetime.utcnow().isoformat() + "Z", **event}

    monkeypatch.setattr(main_module, "enqueue_run_event", _enqueue_stub)
    return main_module


def _auth_header(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _load_auth_module():
    return importlib.import_module("orchestrator.app.auth")


def _base_event_payload(timestamp: datetime) -> Dict[str, object]:
    return {
        "schema": "redops/v1",
        "payload": {
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "agent_id": "agent-007",
            "kind": "simulated",
            "tactic": "discovery",
        },
    }


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer not-a-valid-token"},
    ],
)
def test_missing_or_invalid_jwt_returns_401(security_app, headers):
    client = TestClient(security_app.app)
    with client:
        response = client.get("/metrics", headers=headers)
    assert response.status_code == 401


def test_valid_jwt_with_wrong_role_returns_403(security_app):
    auth = _load_auth_module()
    token = auth.create_token("blue-agent", "agent_red")
    client = TestClient(security_app.app)
    with client:
        response = client.get("/metrics", headers=_auth_header(token))
    assert response.status_code == 403


def test_rate_limit_returns_429_when_bucket_exhausted(security_app):
    auth = _load_auth_module()
    token = auth.create_token("rate-limited-agent", "agent_red")
    client = TestClient(security_app.app)

    with client:
        scenario = client.post(
            "/scenarios",
            content="name: scenario\n",
            headers={"content-type": "text/plain"},
        )
        scenario_id = scenario.json()["scenario_id"]
        run = client.post(f"/runs/{scenario_id}/start")
        run_id = run.json()["run_id"]

        assert security_app.RATE_LIMIT_BURST == 2
        base_time = datetime.now(timezone.utc)
        for attempt in range(2):
            payload = _base_event_payload(base_time + timedelta(seconds=attempt))
            response = client.post(
                f"/runs/{run_id}/events",
                headers=_auth_header(token),
                json=payload,
            )
            assert response.status_code == 202, response.json()

        payload = _base_event_payload(base_time + timedelta(seconds=2))
        response = client.post(
            f"/runs/{run_id}/events",
            headers=_auth_header(token),
            json=payload,
        )
    assert response.status_code == 429


def test_oversized_payload_returns_413(security_app):
    auth = _load_auth_module()
    token = auth.create_token("oversize-agent", "agent_red")
    client = TestClient(security_app.app)

    with client:
        scenario = client.post(
            "/scenarios",
            content="name: scenario\n",
            headers={"content-type": "text/plain"},
        )
        scenario_id = scenario.json()["scenario_id"]
        run = client.post(f"/runs/{scenario_id}/start")
        run_id = run.json()["run_id"]

        payload = _base_event_payload(datetime.now(timezone.utc))
        payload["payload"]["details"] = "x" * (70 * 1024)

        response = client.post(
            f"/runs/{run_id}/events",
            headers=_auth_header(token),
            json=payload,
        )
    assert response.status_code == 413
