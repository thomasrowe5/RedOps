"""Compute basic detection coverage metrics from events and detection hits.

This utility ingests two JSON documents: one describing the simulated attack
events (``events.json``) and another containing detection hits produced by the
analytics pipeline.  Using these data sets the script summarises the overall
coverage, tactic level detection rates, and produces a coarse mean time to
detect (MTTD) estimate.

Usage
-----

.. code-block:: bash

    python analysis/score.py path/to/events.json path/to/detections.json \
        --json-output results.json --markdown-output results.md

When the optional output arguments are omitted the script prints both the JSON
summary and the Markdown report to ``stdout``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, MutableMapping, Optional, Sequence


JsonValue = MutableMapping[str, "JsonValue"] | Sequence["JsonValue"] | str | int | float | bool | None


def load_json(path: Path) -> JsonValue:
    """Load a JSON document from ``path`` and return it as a Python object."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_event_list(data: JsonValue, container_key: str = "events") -> List[dict]:
    """Normalise ``data`` into a list of dictionaries representing events.

    Parameters
    ----------
    data:
        The JSON value parsed from disk.
    container_key:
        Optional key containing the list of events when ``data`` is a mapping.

    Returns
    -------
    list
        The normalised list of event dictionaries.
    """

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        if container_key in data and isinstance(data[container_key], list):
            return [item for item in data[container_key] if isinstance(item, dict)]

        # Elastic style documents might store the hits in ``data["hits"]["hits"]``.
        if "hits" in data:
            hits_obj = data["hits"]
            if isinstance(hits_obj, list):
                return [coerce_hit_to_event(hit) for hit in hits_obj if isinstance(hit, dict)]
            if isinstance(hits_obj, dict) and "hits" in hits_obj:
                inner_hits = hits_obj["hits"]
                if isinstance(inner_hits, list):
                    return [
                        coerce_hit_to_event(hit) for hit in inner_hits if isinstance(hit, dict)
                    ]

    return []


def coerce_hit_to_event(hit: MutableMapping[str, JsonValue]) -> dict:
    """Flatten a detection hit into a plain event dictionary."""

    if "_source" in hit and isinstance(hit["_source"], dict):
        return hit["_source"]  # type: ignore[return-value]
    return dict(hit)


def extract_first_matching(entry: MutableMapping[str, JsonValue], keys: Iterable[str]) -> Optional[str]:
    """Return the first truthy string value found in ``entry`` for ``keys``."""

    for key in keys:
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                return text
        if isinstance(value, list) and value:
            # Some schemas store tactics/techniques as a list.
            for item in value:
                if isinstance(item, (str, int, float)):
                    text = str(item).strip()
                    if text:
                        return text
    return None


def parse_timestamp(value: JsonValue) -> Optional[datetime]:
    """Parse ``value`` into an aware :class:`~datetime.datetime` if possible."""

    if value is None:
        return None

    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        # Replace ``Z`` with ``+00:00`` for ISO 8601 compatibility.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        for fmt in (
            None,  # Attempt ``datetime.fromisoformat`` first.
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                if fmt is None:
                    return datetime.fromisoformat(text)
                parsed = datetime.strptime(text, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


TECHNIQUE_KEYS = (
    "technique_id",
    "technique",
    "techniqueName",
    "mitre_technique",
    "attack_technique",
)

TACTIC_KEYS = (
    "tactic",
    "tactic_id",
    "tactic_name",
    "kill_chain_phase",
    "phase",
)

TIMESTAMP_KEYS = (
    "timestamp",
    "@timestamp",
    "event_time",
    "time",
    "observed",
    "ingested",
)


@dataclass
class DetectionMetrics:
    """Container for the aggregated detection metrics."""

    total_simulated_techniques: int
    detected_techniques: int
    undetected_techniques: int
    overall_detection_rate: float
    tactic_detection_rates: dict[str, float]
    mttd_seconds: Optional[float]


def compute_metrics(events: Sequence[dict], detections: Sequence[dict]) -> DetectionMetrics:
    """Compute detection coverage metrics for the provided events."""

    technique_to_tactics: defaultdict[str, set[str]] = defaultdict(set)
    techniques_per_tactic: defaultdict[str, set[str]] = defaultdict(set)
    event_techniques: set[str] = set()
    event_timestamps: list[datetime] = []

    for event in events:
        technique = extract_first_matching(event, TECHNIQUE_KEYS)
        if not technique:
            continue

        tactic = extract_first_matching(event, TACTIC_KEYS)
        timestamp = extract_first_matching(event, TIMESTAMP_KEYS)

        event_techniques.add(technique)
        if tactic:
            technique_to_tactics[technique].add(tactic)
            techniques_per_tactic[tactic].add(technique)

        if timestamp is not None:
            parsed_time = parse_timestamp(timestamp)
            if parsed_time is not None:
                event_timestamps.append(parsed_time)

    detected_techniques: set[str] = set()
    detected_per_tactic: defaultdict[str, set[str]] = defaultdict(set)
    detection_timestamps: list[datetime] = []

    for hit in detections:
        technique = extract_first_matching(hit, TECHNIQUE_KEYS)
        if not technique or technique not in event_techniques:
            continue

        detected_techniques.add(technique)
        mapped_tactics = technique_to_tactics.get(technique, set())
        if mapped_tactics:
            for tactic in mapped_tactics:
                detected_per_tactic[tactic].add(technique)
        else:
            # Fall back to tactics embedded in the detection document.
            tactic = extract_first_matching(hit, TACTIC_KEYS)
            if tactic:
                detected_per_tactic[tactic].add(technique)

        timestamp = extract_first_matching(hit, TIMESTAMP_KEYS)
        if timestamp is not None:
            parsed_time = parse_timestamp(timestamp)
            if parsed_time is not None:
                detection_timestamps.append(parsed_time)

    total_count = len(event_techniques)
    detected_count = len(detected_techniques)
    undetected_count = total_count - detected_count
    overall_rate = (detected_count / total_count) if total_count else 0.0

    tactic_rates: dict[str, float] = {}
    for tactic, techniques in sorted(techniques_per_tactic.items()):
        detected_for_tactic = len(detected_per_tactic.get(tactic, set()))
        total_for_tactic = len(techniques)
        rate = (detected_for_tactic / total_for_tactic) if total_for_tactic else 0.0
        tactic_rates[tactic] = rate

    mttd_seconds: Optional[float] = None
    if event_timestamps and detection_timestamps:
        first_event = min(event_timestamps)
        first_detection = min(detection_timestamps)
        mttd_seconds = (first_detection - first_event).total_seconds()

    return DetectionMetrics(
        total_simulated_techniques=total_count,
        detected_techniques=detected_count,
        undetected_techniques=max(undetected_count, 0),
        overall_detection_rate=overall_rate,
        tactic_detection_rates=tactic_rates,
        mttd_seconds=mttd_seconds,
    )


def format_metrics_as_json(metrics: DetectionMetrics) -> str:
    """Serialise ``metrics`` into a JSON string."""

    payload = {
        "total_simulated_techniques": metrics.total_simulated_techniques,
        "detected_techniques": metrics.detected_techniques,
        "undetected_techniques": metrics.undetected_techniques,
        "overall_detection_rate": metrics.overall_detection_rate,
        "tactic_detection_rates": metrics.tactic_detection_rates,
        "mttd_seconds": metrics.mttd_seconds,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def format_metrics_as_markdown(metrics: DetectionMetrics) -> str:
    """Render ``metrics`` as a concise Markdown report."""

    lines = ["## Detection Coverage Summary", ""]
    lines.append(f"- **Simulated techniques:** {metrics.total_simulated_techniques}")
    lines.append(f"- **Detected techniques:** {metrics.detected_techniques}")
    lines.append(f"- **Undetected techniques:** {metrics.undetected_techniques}")
    lines.append(
        f"- **Overall detection rate:** {metrics.overall_detection_rate:.1%}"
    )

    if metrics.mttd_seconds is not None:
        lines.append(
            f"- **Estimated MTTD:** {metrics.mttd_seconds:.2f} seconds"
        )
    else:
        lines.append("- **Estimated MTTD:** Not enough data")

    if metrics.tactic_detection_rates:
        lines.append("\n### Detection rate by tactic")
        for tactic, rate in sorted(metrics.tactic_detection_rates.items()):
            lines.append(f"- {tactic}: {rate:.1%}")

    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute detection coverage statistics from simulated events and detection hits."
        )
    )
    parser.add_argument(
        "events",
        type=Path,
        help="Path to the JSON file containing the simulated events.",
    )
    parser.add_argument(
        "detections",
        type=Path,
        help="Path to the JSON file containing detection hits.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write the JSON summary.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional path to write the Markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    events_data = load_json(args.events)
    detections_data = load_json(args.detections)

    events = ensure_event_list(events_data)
    detections = ensure_event_list(detections_data, container_key="detections")

    metrics = compute_metrics(events, detections)

    json_output = format_metrics_as_json(metrics)
    markdown_output = format_metrics_as_markdown(metrics)

    if args.json_output:
        args.json_output.write_text(json_output + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown_output + "\n", encoding="utf-8")

    if not args.json_output:
        print(json_output)
    if not args.markdown_output:
        if not args.json_output:
            print()
        print(markdown_output)


if __name__ == "__main__":
    main()

