"""Async load generator for the orchestrator event ingestion endpoint."""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import httpx


TACTICS = [
    "reconnaissance",
    "initial-access",
    "execution",
    "privilege-escalation",
    "lateral-movement",
    "credential-access",
    "command-and-control",
    "collection",
    "exfiltration",
    "persistence",
    "discovery",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _percentile(samples: List[float], percentile: float) -> Optional[float]:
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    index = (len(samples) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(samples) - 1)
    if lower == upper:
        return samples[lower]
    fraction = index - lower
    return samples[lower] + (samples[upper] - samples[lower]) * fraction


@dataclass
class AgentMetrics:
    latencies: List[float]
    errors: int = 0
    sent: int = 0


def _build_payload(run_id: str, agent_id: int, sequence: int) -> dict:
    tactic = random.choice(TACTICS)
    payload = {
        "schema": "redops/v1",
        "payload": {
            "timestamp": _now_iso(),
            "agent_id": f"agent-{agent_id}",
            "kind": "simulated",
            "tactic": tactic,
            "note": f"run={run_id} seq={sequence} tactic={tactic}",
            "sequence": sequence,
            "run_id": run_id,
        },
    }
    return payload


async def _run_agent(
    client: httpx.AsyncClient,
    *,
    run_id: str,
    agent_index: int,
    events_per_second: float,
    duration: float,
    endpoint: str,
) -> AgentMetrics:
    metrics = AgentMetrics(latencies=[], errors=0, sent=0)
    if events_per_second <= 0 or duration <= 0:
        return metrics

    interval = 1.0 / events_per_second
    start = time.perf_counter()
    deadline = start + duration
    sequence = 0

    while True:
        now = time.perf_counter()
        if now >= deadline:
            break

        target = start + sequence * interval
        if target > now:
            await asyncio.sleep(target - now)

        payload = _build_payload(run_id, agent_index, sequence)
        request_started = time.perf_counter()
        try:
            response = await client.post(endpoint, json=payload, timeout=10.0)
            latency = time.perf_counter() - request_started
            if response.is_success:
                metrics.latencies.append(latency)
            else:
                metrics.errors += 1
        except Exception:
            metrics.errors += 1
        metrics.sent += 1
        sequence += 1

    return metrics


async def _run_load_test(
    *,
    base_url: str,
    endpoint_path: str,
    agents: int,
    events_per_second: float,
    duration: float,
    run_id: str,
) -> None:
    endpoint = base_url.rstrip("/") + "/" + endpoint_path.lstrip("/")
    async with httpx.AsyncClient() as client:
        tasks = [
            _run_agent(
                client,
                run_id=run_id,
                agent_index=index,
                events_per_second=events_per_second,
                duration=duration,
                endpoint=endpoint,
            )
            for index in range(agents)
        ]
        results = await asyncio.gather(*tasks)

    total_sent = sum(r.sent for r in results)
    total_errors = sum(r.errors for r in results)
    latencies: List[float] = sorted(lat for r in results for lat in r.latencies)

    actual_duration = duration if duration > 0 else 0.0
    throughput = total_sent / actual_duration if actual_duration else 0.0
    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)

    print("Load test results")
    print("=================")
    print(f"Agents: {agents}")
    print(f"Requested throughput: {agents * events_per_second:.2f} events/sec")
    print(f"Actual throughput: {throughput:.2f} events/sec")
    print(f"Total events sent: {total_sent}")
    print(f"Errors: {total_errors}")
    if latencies:
        print(f"p50 latency: {p50 * 1000:.2f} ms")
        print(f"p95 latency: {p95 * 1000:.2f} ms")
        print(f"Average latency: {statistics.mean(latencies) * 1000:.2f} ms")
    else:
        print("No successful responses recorded; latency metrics unavailable.")


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Async load generator for /events")
    parser.add_argument("--agents", type=int, default=1, help="Number of concurrent agents")
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="Events per second produced by each agent",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of the test in seconds")
    parser.add_argument("--run-id", default="bench", help="Logical run identifier embedded in events")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Server base URL",
    )
    parser.add_argument(
        "--endpoint",
        default="/events",
        help="Endpoint path relative to the base URL",
    )
    args = parser.parse_args()

    asyncio.run(
        _run_load_test(
            base_url=args.base_url,
            endpoint_path=args.endpoint,
            agents=args.agents,
            events_per_second=args.eps,
            duration=args.duration,
            run_id=args.run_id,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_cli())
