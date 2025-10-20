"""Audit log verification tests for tamper detection."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

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
    """Provide a configured orchestrator application instance for audit tests."""

    monkeypatch.setenv("REDOPS_JWT_SECRET", "test-secret")
    monkeypatch.setenv("REDOPS_RATE_LIMIT_BURST", "3")
    monkeypatch.setenv("REDOPS_RATE_LIMIT_RATE", "3")

    backpressure = _load_backpressure_module()
    orchestrator_factory = getattr(backpressure, "orchestrator_app")
    factory_callable = getattr(orchestrator_factory, "__wrapped__", orchestrator_factory)
    main_module = factory_callable(monkeypatch, tmp_path)
    monkeypatch.setattr(main_module, "_install_signal_handlers", lambda *_args, **_kwargs: None)

    audit_module = importlib.import_module("orchestrator.app.audit")
    monkeypatch.setattr(audit_module, "DATA_DIR", main_module.DATA_DIR)
    monkeypatch.setattr(audit_module, "RUNS_DIR", main_module.RUNS_DIR)

    async def _enqueue_stub(run_id: str, event: Dict[str, object], *, timeout=None) -> Dict[str, object]:
        return {"timestamp": datetime.utcnow().isoformat() + "Z", **event}

    monkeypatch.setattr(main_module, "enqueue_run_event", _enqueue_stub)
    return main_module


def _auth_header(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _load_auth_module():
    return importlib.import_module("orchestrator.app.auth")


def _event_payload(when: datetime) -> Dict[str, object]:
    return {
        "schema": "redops/v1",
        "payload": {
            "timestamp": when.isoformat().replace("+00:00", "Z"),
            "agent_id": "agent-007",
            "kind": "simulated",
            "tactic": "discovery",
        },
    }


def test_audit_verify_detects_tampering(security_app):
    auth = _load_auth_module()
    agent_token = auth.create_token("audit-agent", "agent_red")
    operator_token = auth.create_token("audit-operator", "operator")

    client = TestClient(security_app.app)

    with client:
        scenario = client.post(
            "/scenarios",
            content="name: scenario\n",
            headers={"content-type": "text/plain"},
        )
        scenario_id = scenario.json()["scenario_id"]

        run = client.post(f"/runs/{scenario_id}/start")
        run.raise_for_status()
        run_id = run.json()["run_id"]

        base_time = datetime.now(timezone.utc)
        for offset in range(3):
            payload = _event_payload(base_time + timedelta(seconds=offset))
            response = client.post(
                f"/runs/{run_id}/events",
                headers=_auth_header(agent_token),
                json=payload,
            )
            assert response.status_code == 202, response.json()

        audit_path = security_app.RUNS_DIR / run_id / "audit.ndjson"
        assert audit_path.exists(), "expected audit log to be created"

        lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        records: List[Dict[str, object]] = [json.loads(line) for line in lines]
        assert len(records) >= 3, "expected at least three audit records"

        records[1]["prev_hash"] = "tampered"
        tampered = "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
        audit_path.write_text(tampered, encoding="utf-8")

        verify = client.get(
            f"/runs/{run_id}/audit/verify",
            headers=_auth_header(operator_token),
        )

    assert verify.status_code == 200
    assert verify.json() == {"ok": False}
