import os
import time
import json
import tempfile
import shutil
from pathlib import Path

import requests


def main():
    base = "http://localhost:8000"
    run = "chaos1"
    ev = {
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
    # normal
    for _ in range(5):
        requests.post(f"{base}/runs/{run}/events", json=ev, timeout=3)

    # simulate slow fs (cannot easily in pure Python) → send bursts and ensure 202/429 only
    statuses = [
        requests.post(f"{base}/runs/{run}/events", json=ev, timeout=3).status_code
        for _ in range(500)
    ]
    assert all(s in (202, 429) for s in statuses)

    print("Chaos smoke complete.")


if __name__ == "__main__":
    main()
