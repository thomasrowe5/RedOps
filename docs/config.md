# Configuration Reference

RedOps relies on a handful of environment variables and runtime constants to
tune queueing, persistence, authentication, and rate limiting. This document
summarises the supported knobs, their defaults, and how to adjust them for
local development versus a hardened deployment.

## Queueing and Persistence

| Setting | Default | How to change | Notes |
| --- | --- | --- | --- |
| `REDOPS_QUEUE_MAX` | `1000` | Environment variable | Maximum number of pending events per run queue in both the in-memory and Redis-backed buses. Values ≤ 0 fall back to 1000 for in-memory queues and remove the cap for Redis streams.【F:orchestrator/app/jobs.py†L24-L49】【F:orchestrator/app/redis_bus.py†L23-L41】 |
| `BATCH_FSYNC_INTERVAL` | `50` events | Assign to `orchestrator.app.main.BATCH_WRITER.BATCH_FSYNC_INTERVAL` before the writer starts | Controls how many events are buffered before forcing a flush to disk. Exposed as a class attribute on the batch writer; there is currently no environment variable hook.【F:orchestrator/app/jobs.py†L90-L149】 |
| `BATCH_FSYNC_SECS` | `2` seconds | Assign to `orchestrator.app.main.BATCH_WRITER.BATCH_FSYNC_SECS` before the writer starts | Maximum wall-clock interval between flushes regardless of batch size.【F:orchestrator/app/jobs.py†L90-L149】 |
| `MAX_MB` | `50` MiB | Assign to `orchestrator.app.main.BATCH_WRITER.MAX_MB` before the writer starts | Upper bound for a single NDJSON file produced by the batch writer; rotation happens when the limit is exceeded.【F:orchestrator/app/jobs.py†L90-L149】 |
| `REDOPS_QUEUE_DRIVER` | `memory` | Environment variable | Selects the queue backend. Use `redis` to enable the Redis stream implementation.【F:orchestrator/app/main.py†L166-L179】 |
| `REDIS_URL` | `redis://localhost:6379/0` | Environment variable | Connection string used when `REDOPS_QUEUE_DRIVER=redis`. Supports authentication and alternate databases.【F:orchestrator/app/redis_bus.py†L23-L47】 |

## Authentication and Operational Modes

| Setting | Default | How to change | Notes |
| --- | --- | --- | --- |
| `REDOPS_JWT_SECRET` | _required_ | Environment variable | Shared HS256 secret used to sign and validate all API JWTs. The orchestrator refuses to start without it.【F:orchestrator/app/auth.py†L32-L116】 |
| `LAB_MODE` | `1` (`true`) | Environment variable (must remain truthy) | Guards lab-only behaviours such as localhost-only access and the `/token` helper endpoint. The server aborts startup if the variable is unset or falsy.【F:orchestrator/app/main.py†L1-L120】【F:orchestrator/app/main.py†L680-L740】 |

## Rate Limiting and Payload Size

| Setting | Default | How to change | Notes |
| --- | --- | --- | --- |
| `RATE_LIMIT_BURST` (`REDOPS_RATE_LIMIT_BURST`) | `100` tokens | Environment variable | Token bucket size for write endpoints. Values ≤ 0 disable rate limiting.【F:orchestrator/app/main.py†L205-L218】【F:orchestrator/app/main.py†L308-L364】 |
| `RATE_LIMIT_PER_MIN` (`REDOPS_RATE_LIMIT_RATE`) | `50` requests per minute | Environment variable | Refill rate for the token bucket shared by each `(run_id, agent_id)` pair. Setting ≤ 0 disables rate limiting.【F:orchestrator/app/main.py†L205-L218】【F:orchestrator/app/main.py†L308-L364】 |
| `CONTENT_MAX_BYTES` (`MAX_POST_BYTES`) | `65536` bytes | Modify `orchestrator.app.main.MAX_POST_BYTES` before app startup | Hard cap enforced on all POST request bodies; payloads above the limit are rejected with HTTP 413.【F:orchestrator/app/main.py†L205-L364】 |

> **Note:** The batch-writer tuning knobs (`BATCH_FSYNC_INTERVAL`, `BATCH_FSYNC_SECS`,
> `MAX_MB`) and the payload size guard (`CONTENT_MAX_BYTES`) are currently exposed as
> module-level attributes rather than environment variables. Override them by
> assigning new values prior to starting the FastAPI application (for example, in a
> custom entrypoint that imports the app module and adjusts the attributes before
> handing control to Uvicorn).

## Example Configurations

### Local Development

A minimal `.env` for iterating locally while keeping the defaults that favour
higher throughput:

```bash
# Queue in memory and keep generous buffers
export REDOPS_QUEUE_DRIVER="memory"
export REDOPS_QUEUE_MAX="1000"

# Authentication and lab-only helpers
export LAB_MODE="1"
export REDOPS_JWT_SECRET="change-me-dev"

# Leave rate limits relaxed
export REDOPS_RATE_LIMIT_BURST="100"
export REDOPS_RATE_LIMIT_RATE="50"
```

Optional Python bootstrap (for example in `main_local.py`) if you want faster
flushes during debugging:

```python
from orchestrator.app import main

main.BATCH_WRITER.BATCH_FSYNC_INTERVAL = 10
main.BATCH_WRITER.BATCH_FSYNC_SECS = 0.5
```

### Hardened Deployment

A hardened environment typically enables Redis-backed buffering, reduces queue
sizes, and tightens limits while keeping lab safeguards enabled:

```bash
# Harden queueing and persistence
export REDOPS_QUEUE_DRIVER="redis"
export REDIS_URL="redis://redops-redis:6379/0"
export REDOPS_QUEUE_MAX="250"

# Mandatory lab safety and strong authentication
export LAB_MODE="1"
export REDOPS_JWT_SECRET="$(openssl rand -hex 32)"

# Aggressive rate limiting for production-like load
export REDOPS_RATE_LIMIT_BURST="20"
export REDOPS_RATE_LIMIT_RATE="120"
```

Pair the environment variables above with a bootstrap tweak to limit disk usage:

```python
from orchestrator.app import main

main.BATCH_WRITER.MAX_MB = 10  # rotate files sooner
```

These examples are intended as starting points—adjust the values to match your
infrastructure capacity and operational risk tolerance.
