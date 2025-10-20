# RedOps

RedOps is a research-focused platform for exploring defensive automation patterns in a controlled environment.

## Safety Notice
This project is for **lab use only**. Do not deploy in production or expose to untrusted networks.

## Development Setup

We recommend installing [pre-commit](https://pre-commit.com/) to automatically run formatters and linters before each commit:

```bash
pip install pre-commit
pre-commit install
```

The configured hooks will enforce code style and run static analysis for the `agents`, `analysis`, and `orchestrator` modules.

## Orchestrator Authentication

The orchestrator API now requires JWT bearer tokens for write and operational endpoints.

1. Set a shared secret for signing tokens before starting the service:

   ```bash
   export REDOPS_JWT_SECRET="change-me"
   ```

2. While running in lab mode (the default), request a development token:

   ```bash
   curl -X POST http://localhost:8000/token \
     -H 'Content-Type: application/json' \
     -d '{"actor": "demo-agent", "role": "agent_red"}'
   ```

3. Call protected endpoints with the `Authorization: Bearer <token>` header.

   - Agent roles (`agent_red`, `agent_blue`) are required for `POST /runs/{run_id}/events` and `POST /runs/{run_id}/responses`.
   - The `operator` role is required for `GET /metrics` and `GET /runs/{run_id}/queue_stats`.

Tokens expire one hour after issuance by default. Adjust the lifetime by including an `expires_in` value (in seconds) when requesting a token.
