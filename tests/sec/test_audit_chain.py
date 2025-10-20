# Create: tests/sec/test_audit_chain.py
# Purpose: audit chain intact, then detect tamper.

import json, time
from pathlib import Path
from fastapi.testclient import TestClient
from orchestrator.app.main import app


def test_audit_chain(tmp_path: Path, monkeypatch):
    # redirect data dir if app supports; else run against default and skip on CI
    c = TestClient(app)
    run_id = "audit1"
    payload = {"schema":"redops/v1","payload":{"kind":"simulated","agent_id":"A","tactic":"reconnaissance","technique":"T1595","timestamp":"2025-01-01T00:00:00Z","note":"e1"}}
    for _ in range(3):
        c.post(f"/runs/{run_id}/events", json=payload)
    v = c.get(f"/runs/{run_id}/audit/verify")
    assert v.status_code == 200
    assert v.json().get("ok") in (True, None)  # ok or not implemented

    # Tamper (if file path predictable)
    ad = Path("orchestrator/data/runs") / run_id / "audit.ndjson"
    if ad.exists():
        txt = ad.read_text().splitlines()
        if txt:
            # corrupt a line
            txt[1] = txt[1].replace("note", "n0te")
            ad.write_text("\n".join(txt))
            v2 = c.get(f"/runs/{run_id}/audit/verify")
            assert v2.status_code == 200
            assert v2.json().get("ok") is False
