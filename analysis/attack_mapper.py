"""Utility for constructing a simple ATT&CK technique graph from telemetry events.

This script ingests a JSON list of events and a YAML mapping of internal
technique keys to MITRE ATT&CK technique identifiers. It produces a structured
``attack_graph.json`` file that captures relationships between agents,
techniques, and timestamps observed in the event stream.

The generated output is intentionally compatible with downstream graphing or
visualisation tooling. If you prefer to explore the techniques with the MITRE
ATT&CK Navigator, export the resulting techniques list into the Navigator JSON
format following MITRE's documentation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import yaml


@dataclass(frozen=True)
class Node:
    """Representation of a graph node."""

    id: str
    type: str
    label: str


@dataclass(frozen=True)
class Edge:
    """Simple directed edge between two node identifiers."""

    source: str
    target: str
    relationship: str


@dataclass
class AttackGraph:
    """Container for the nodes and edges making up the attack graph."""

    nodes: List[Node]
    edges: List[Edge]

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }


def load_events(path: pathlib.Path) -> List[dict]:
    """Load the events JSON file as a list of dictionaries."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Events file must contain a JSON list of event objects.")

    return data


def load_mapping(path: pathlib.Path) -> Dict[str, str]:
    """Load the YAML mapping of internal technique keys to ATT&CK IDs."""

    with path.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}

    if not isinstance(mapping, dict):
        raise ValueError("Technique mapping must be a dictionary.")

    return {str(key): str(value) for key, value in mapping.items()}


def unique_nodes(nodes: Iterable[Node]) -> List[Node]:
    """Deduplicate nodes while preserving insertion order."""

    seen = set()
    deduped: List[Node] = []
    for node in nodes:
        if node.id in seen:
            continue
        seen.add(node.id)
        deduped.append(node)
    return deduped


def unique_edges(edges: Iterable[Edge]) -> List[Edge]:
    """Deduplicate edges while preserving insertion order."""

    seen = set()
    deduped: List[Edge] = []
    for edge in edges:
        key: Tuple[str, str, str] = (edge.source, edge.target, edge.relationship)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def build_attack_graph(events: List[dict], technique_map: Dict[str, str]) -> AttackGraph:
    """Construct the attack graph from raw events and technique mappings."""

    nodes: List[Node] = []
    edges: List[Edge] = []

    for event in events:
        agent_name = str(event.get("agent", "unknown-agent"))
        internal_technique = str(event.get("technique", "unknown-technique"))
        timestamp = str(event.get("timestamp", "unknown-timestamp"))

        mitre_id = technique_map.get(internal_technique, "UNMAPPED")

        agent_node = Node(id=f"agent:{agent_name}", type="agent", label=agent_name)
        technique_label = (
            f"{internal_technique} ({mitre_id})" if mitre_id != "UNMAPPED" else internal_technique
        )
        technique_node = Node(
            id=f"technique:{internal_technique}",
            type="technique",
            label=technique_label,
        )
        timestamp_node = Node(
            id=f"timestamp:{timestamp}",
            type="timestamp",
            label=timestamp,
        )

        nodes.extend([agent_node, technique_node, timestamp_node])

        edges.append(
            Edge(
                source=agent_node.id,
                target=technique_node.id,
                relationship="used",
            )
        )
        edges.append(
            Edge(
                source=technique_node.id,
                target=timestamp_node.id,
                relationship="observed_at",
            )
        )

    return AttackGraph(nodes=unique_nodes(nodes), edges=unique_edges(edges))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ATT&CK technique graph from events.")
    parser.add_argument(
        "events_file",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("events.json"),
        help="Path to the JSON events file (default: events.json)",
    )
    parser.add_argument(
        "--mapping",
        type=pathlib.Path,
        default=pathlib.Path("mapping/techniques.yml"),
        help="Path to the YAML mapping file (default: mapping/techniques.yml)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("attack_graph.json"),
        help="Path to write the generated attack graph JSON (default: attack_graph.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    events = load_events(args.events_file)
    technique_map = load_mapping(args.mapping)

    graph = build_attack_graph(events, technique_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(graph.to_dict(), handle, indent=2)


if __name__ == "__main__":
    main()
