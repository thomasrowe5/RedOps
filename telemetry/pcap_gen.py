#!/usr/bin/env python3
"""Synthetic network flow generator for RedOps telemetry experiments.

The generator produces deterministic, labeled flow records suitable for
training and experimentation. It focuses on two classes:

* ``benign``  – routine workstation and service traffic
* ``exfil``   – high-volume transfers representing data exfiltration

Running the script writes a CSV file to ``telemetry/out`` containing the
requested number of flows for each class. The random seed is configurable to
ensure reproducible datasets.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
from typing import Iterable, List

# Directory relative to this script where generated CSV files will be written.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "out"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "synthetic_flows.csv"

FIELDNAMES = [
    "timestamp",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "bytes_sent",
    "bytes_received",
    "packets",
    "duration_seconds",
    "label",
]


@dataclass(frozen=True)
class FlowConfig:
    """Configuration describing how to synthesize a flow for a label."""

    label: str
    src_networks: Iterable[str]
    dst_networks: Iterable[str]
    dst_ports: Iterable[int]
    bytes_range: tuple[int, int]
    response_ratio: tuple[float, float]
    duration_range: tuple[float, float]
    protocols: Iterable[str]


BENIGN_CONFIG = FlowConfig(
    label="benign",
    src_networks=("10", "192.168"),
    dst_networks=("172.16", "198.51.100"),
    dst_ports=(80, 443, 53, 22, 123),
    bytes_range=(2_000, 80_000),
    response_ratio=(0.4, 0.9),
    duration_range=(0.3, 90.0),
    protocols=("tcp", "udp"),
)

EXFIL_CONFIG = FlowConfig(
    label="exfil",
    src_networks=("10", "192.168"),
    dst_networks=("203.0.113", "198.18"),
    dst_ports=(443, 8443, 8080),
    bytes_range=(500_000, 5_000_000),
    response_ratio=(0.02, 0.15),
    duration_range=(60.0, 900.0),
    protocols=("tcp",),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic network flows")
    parser.add_argument(
        "--benign",
        type=int,
        default=200,
        metavar="N",
        help="Number of benign flows to generate (default: 200)",
    )
    parser.add_argument(
        "--exfil",
        type=int,
        default=50,
        metavar="N",
        help="Number of exfiltration flows to generate (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed controlling deterministic output (default: 1337)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=(
            "Path to write the CSV file (default: telemetry/out/synthetic_flows.csv). "
            "Directories will be created as needed."
        ),
    )
    return parser.parse_args()


def _random_host(rng: random.Random, network_prefix: str) -> str:
    """Return a random host IP within a dotted prefix such as "10" or "198.51.100"."""

    octets = network_prefix.split(".")
    while len(octets) < 4:
        octets.append(str(rng.randint(0, 255)))
    return ".".join(octets[:4])


def _choose(values: Iterable[str | int], rng: random.Random):
    """Helper that returns a random element from an iterable."""

    sequence = list(values)
    if not sequence:
        raise ValueError("Values iterable must not be empty")
    return rng.choice(sequence)


def _synthesize_flow(rng: random.Random, config: FlowConfig, base_time: datetime) -> dict[str, object]:
    """Create a single flow record based on the provided configuration."""

    timestamp = base_time + timedelta(seconds=rng.uniform(0, 3600))
    duration = rng.uniform(*config.duration_range)
    bytes_sent = int(rng.uniform(*config.bytes_range))
    response_factor = rng.uniform(*config.response_ratio)
    bytes_received = max(0, int(bytes_sent * response_factor))

    # Estimate packets based on a typical payload size distribution.
    avg_payload = rng.uniform(250, 900)
    packets = max(1, int(round(bytes_sent / avg_payload)))

    flow = {
        "timestamp": timestamp,
        "src_ip": _random_host(rng, _choose(config.src_networks, rng)),
        "src_port": rng.randint(1024, 65535),
        "dst_ip": _random_host(rng, _choose(config.dst_networks, rng)),
        "dst_port": int(_choose(config.dst_ports, rng)),
        "protocol": _choose(config.protocols, rng),
        "bytes_sent": bytes_sent,
        "bytes_received": bytes_received,
        "packets": packets,
        "duration_seconds": round(duration, 3),
        "label": config.label,
    }
    return flow


def generate_flows(
    rng: random.Random, benign_count: int, exfil_count: int, base_time: datetime
) -> List[dict[str, object]]:
    """Generate deterministic benign and exfiltration flows."""

    flows: List[dict[str, object]] = []
    for _ in range(benign_count):
        flows.append(_synthesize_flow(rng, BENIGN_CONFIG, base_time))
    for _ in range(exfil_count):
        flows.append(_synthesize_flow(rng, EXFIL_CONFIG, base_time))
    flows.sort(key=lambda record: record["timestamp"])  # chronological order
    return flows


def write_flows(path: Path, flows: Iterable[dict[str, object]]) -> None:
    """Write flow dictionaries to ``path`` as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in flows:
            serializable = dict(record)
            timestamp = serializable["timestamp"]
            if isinstance(timestamp, datetime):
                serializable["timestamp"] = timestamp.isoformat(timespec="seconds")
            writer.writerow(serializable)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    flows = generate_flows(rng, args.benign, args.exfil, base_time)
    write_flows(args.output, flows)
    print(f"Wrote {len(flows)} flows to {args.output}")


if __name__ == "__main__":
    main()
