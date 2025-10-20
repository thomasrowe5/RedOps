#!/usr/bin/env bash
set -euo pipefail
docker compose -f lab/docker-compose.yml up -d
( cd orchestrator && uvicorn app.main:app --reload )
