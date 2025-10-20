"""Construct attack graphs from orchestrator events and compute graph metrics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple

import networkx as nx


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _NormalizedEvent:
    """Representation of an orchestrator event with normalized fields."""

    agent: str
    tactic: str
    technique: str
    timestamp: Optional[str]
    parsed_timestamp: Optional[datetime]


def _read_events(events_path: Path) -> List[Dict]:
    """Load an events file that may be JSON or JSON-lines formatted."""

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    content = events_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        loaded = json.loads(content)
        if isinstance(loaded, list):
            return loaded
        LOGGER.warning("Events file %s did not contain a list. Treating as empty.", events_path)
        return []
    except json.JSONDecodeError:
        events: List[Dict] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                LOGGER.warning("Skipping invalid JSON line in %s: %s", events_path, line)
        return events


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamps with graceful fallbacks."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (OSError, OverflowError, ValueError):
            return None

    text = str(value)
    if not text:
        return None
    # Handle timestamps that end with "Z" (UTC designator)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    for fmt in (None, "%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z"):
        try:
            if fmt is None:
                return datetime.fromisoformat(text)
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _normalize_event(event: MutableMapping) -> _NormalizedEvent:
    """Normalize a raw event dictionary to the fields used for graph building."""

    agent = event.get("agent") or event.get("agent_id") or "unknown-agent"
    tactic = event.get("tactic") or event.get("mitre_tactic") or "unknown-tactic"
    technique = event.get("technique") or event.get("mitre_technique") or "unknown-technique"
    timestamp = event.get("timestamp") or event.get("time")

    parsed_timestamp = _parse_timestamp(timestamp if timestamp is None else str(timestamp))

    return _NormalizedEvent(
        agent=str(agent),
        tactic=str(tactic),
        technique=str(technique),
        timestamp=str(timestamp) if timestamp is not None else None,
        parsed_timestamp=parsed_timestamp,
    )


def build_graph_from_events(events_path: Path | str, node_mode: str = "technique") -> nx.MultiDiGraph:
    """Create a directed multigraph from orchestrator events.

    Parameters
    ----------
    events_path:
        Path to the ``events.json`` file.
    node_mode:
        Determines which event field becomes the primary node label. Supported
        values are ``"technique"`` and ``"tactic"``.
    """

    if node_mode not in {"technique", "tactic"}:
        raise ValueError("node_mode must be either 'technique' or 'tactic'")

    path = Path(events_path)
    raw_events = _read_events(path)

    graph = nx.MultiDiGraph(node_mode=node_mode)

    events_by_agent: Dict[str, List[Tuple[int, _NormalizedEvent]]] = defaultdict(list)
    for index, raw_event in enumerate(raw_events):
        normalized = _normalize_event(raw_event)
        node_value = getattr(normalized, node_mode)

        if not graph.has_node(node_value):
            node_attributes = {
                "type": node_mode,
                "label": node_value,
                "techniques": {normalized.technique},
                "tactics": {normalized.tactic},
            }
            graph.add_node(node_value, **node_attributes)
        else:
            node_data = graph.nodes[node_value]
            node_data.setdefault("techniques", set()).add(normalized.technique)
            node_data.setdefault("tactics", set()).add(normalized.tactic)

        events_by_agent[normalized.agent].append((index, normalized))

    for agent, event_entries in events_by_agent.items():
        agent_node = f"agent:{agent}"
        if not graph.has_node(agent_node):
            graph.add_node(agent_node, type="agent", agent=agent, label=agent)

        # Sort events by timestamp (if available) while preserving original order for ties
        sorted_events = sorted(
            event_entries,
            key=lambda item: (
                item[1].parsed_timestamp if item[1].parsed_timestamp is not None else datetime.max,
                item[0],
            ),
        )

        # Create edges for chronological transitions per agent
        for (prev_index, prev_event), (curr_index, curr_event) in zip(sorted_events, sorted_events[1:]):
            source_node = getattr(prev_event, node_mode)
            target_node = getattr(curr_event, node_mode)

            edge_attributes = {
                "agent": agent,
                "source_event_index": prev_index,
                "target_event_index": curr_index,
                "source_timestamp": prev_event.timestamp,
                "target_timestamp": curr_event.timestamp,
            }
            graph.add_edge(source_node, target_node, **edge_attributes)

    # Convert set attributes to sorted lists for downstream serialization safety
    for node, data in graph.nodes(data=True):
        for field in ("techniques", "tactics"):
            if field in data and isinstance(data[field], set):
                data[field] = sorted(data[field])

    return graph


def _aggregate_weighted_digraph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """Aggregate a MultiDiGraph into a weighted DiGraph for metric calculations."""

    simple_graph = nx.DiGraph()
    simple_graph.add_nodes_from(graph.nodes(data=True))

    for source, target, edge_data in graph.edges(data=True):
        weight = float(edge_data.get("weight", 1.0))
        if simple_graph.has_edge(source, target):
            simple_graph[source][target]["weight"] += weight
        else:
            simple_graph.add_edge(source, target, weight=weight)

    return simple_graph


def compute_metrics(graph: nx.MultiDiGraph) -> Dict[str, Dict[str, float]]:
    """Compute centrality metrics for a graph."""

    if graph.number_of_nodes() == 0:
        return {
            "degree_centrality": {},
            "betweenness_centrality": {},
            "pagerank": {},
        }

    simple_graph = _aggregate_weighted_digraph(graph)

    if simple_graph.number_of_nodes() == 0:
        return {
            "degree_centrality": {},
            "betweenness_centrality": {},
            "pagerank": {},
        }

    degree_centrality = nx.degree_centrality(simple_graph)
    betweenness_centrality = nx.betweenness_centrality(simple_graph, weight="weight", normalized=True)
    pagerank = nx.pagerank(simple_graph, weight="weight")

    return {
        "degree_centrality": degree_centrality,
        "betweenness_centrality": betweenness_centrality,
        "pagerank": pagerank,
    }


def save_metrics(metrics: Dict[str, Dict[str, float]], out_json: Path | str, out_csv: Path | str) -> None:
    """Persist metrics to JSON and CSV files."""

    out_json_path = Path(out_json)
    out_csv_path = Path(out_csv)

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        metric: {node: float(value) for node, value in values.items()}
        for metric, values in metrics.items()
    }

    out_json_path.write_text(json.dumps(serializable_metrics, indent=2, sort_keys=True), encoding="utf-8")

    all_nodes = set()
    for values in serializable_metrics.values():
        all_nodes.update(values.keys())

    fieldnames = ["node", "degree_centrality", "betweenness_centrality", "pagerank"]

    with out_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for node in sorted(all_nodes):
            writer.writerow(
                {
                    "node": node,
                    "degree_centrality": serializable_metrics.get("degree_centrality", {}).get(node, 0.0),
                    "betweenness_centrality": serializable_metrics.get("betweenness_centrality", {}).get(
                        node, 0.0
                    ),
                    "pagerank": serializable_metrics.get("pagerank", {}).get(node, 0.0),
                }
            )


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the metrics CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        required=True,
        type=Path,
        help="Path to the events.json file produced by an orchestrator run.",
    )
    parser.add_argument(
        "--node-mode",
        default="technique",
        choices=("technique", "tactic"),
        help="Graph node representation (technique IDs or tactic names).",
    )
    parser.add_argument(
        "--out",
        dest="out_json",
        required=True,
        type=Path,
        help="Destination JSON file for the computed metrics.",
    )
    parser.add_argument(
        "--csv",
        dest="out_csv",
        required=True,
        type=Path,
        help="Destination CSV file for the computed metrics.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Entry point for the command line interface."""

    args = _parse_args(argv)

    graph = build_graph_from_events(args.events, node_mode=args.node_mode)
    metrics = compute_metrics(graph)
    save_metrics(metrics, args.out_json, args.out_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

