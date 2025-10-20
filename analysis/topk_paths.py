"""Compute top-k attack paths between entry and goal tactics from orchestrator events."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import networkx as nx

try:
    import yaml
except ImportError:  # pragma: no cover - fallback when PyYAML is unavailable.
    yaml = None  # type: ignore

from analysis.graph_metrics import build_graph_from_events

LOGGER = logging.getLogger(__name__)
ENTRY_TACTICS = {"reconnaissance", "initial-access"}
GOAL_TACTIC = "exfiltration"
DEFAULT_DETECTABILITY_PATH = Path("analysis/detectability.yml")

PathWithLength = Tuple[List[str], float]


def load_detectability_mapping(path: Optional[Path]) -> Dict[str, float]:
    """Load a mapping of technique identifiers to detectability scores."""

    if path is None:
        return {}
    if not path.exists():
        LOGGER.debug("Detectability mapping %s not found", path)
        return {}
    if yaml is None:
        LOGGER.warning(
            "PyYAML is not installed; cannot load detectability mapping from %s", path
        )
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging path
        LOGGER.warning("Failed to parse detectability mapping %s: %s", path, exc)
        return {}
    if not isinstance(data, Mapping):
        LOGGER.warning("Detectability mapping %s did not yield a dictionary", path)
        return {}

    if "techniques" in data and isinstance(data["techniques"], Mapping):
        data = data["techniques"]

    mapping: Dict[str, float] = {}
    for technique, value in data.items():
        try:
            mapping[str(technique)] = float(value)
        except (TypeError, ValueError):
            LOGGER.debug("Skipping non-numeric detectability value for %s", technique)
            continue
    return mapping


def find_entry_nodes(graph: nx.Graph) -> List[str]:
    """Return nodes whose associated tactic is reconnaissance or initial-access."""

    entries: List[str] = []
    for node, data in graph.nodes(data=True):
        if not isinstance(data, MutableMapping):
            continue
        node_type = data.get("type")
        if node_type not in {"technique", "tactic"}:
            continue
        tactics = data.get("tactics") or []
        for tactic in tactics:
            try:
                tactic_norm = str(tactic).strip().lower()
            except Exception:
                continue
            if tactic_norm in ENTRY_TACTICS:
                entries.append(node)
                break
    return entries


def find_goal_nodes(graph: nx.Graph) -> List[str]:
    """Return technique nodes associated with the exfiltration tactic."""

    goals: List[str] = []
    for node, data in graph.nodes(data=True):
        if not isinstance(data, MutableMapping):
            continue
        node_type = data.get("type")
        if node_type not in {"technique", "tactic"}:
            continue
        tactics = data.get("tactics") or []
        for tactic in tactics:
            try:
                tactic_norm = str(tactic).strip().lower()
            except Exception:
                continue
            if tactic_norm == GOAL_TACTIC:
                goals.append(node)
                break
    return goals


def edge_weight(
    graph: nx.Graph,
    source: str,
    target: str,
    detectability: Optional[Mapping[str, float]] = None,
) -> float:
    """Return the traversal weight for an edge with optional detectability adjustments."""

    base_weight = 1.0
    if not detectability:
        return base_weight

    node_data = graph.nodes.get(target, {})
    label = node_data.get("label", target)
    detect_value: Optional[float] = None

    if isinstance(label, str) and label in detectability:
        detect_value = detectability[label]
    else:
        techniques = node_data.get("techniques") or []
        for technique in techniques:
            if technique in detectability:
                detect_value = detectability[technique]
                break

    if detect_value is None:
        return base_weight

    try:
        detect_float = float(detect_value)
    except (TypeError, ValueError):
        return base_weight

    detect_float = max(0.0, detect_float)
    # Higher detectability should increase the path cost, encouraging the algorithm
    # to prefer stealthier (lower detectability) steps where alternatives exist.
    adjusted = base_weight + detect_float
    return max(adjusted, 0.0)


def _build_weighted_graph(
    graph: nx.MultiDiGraph, detectability: Optional[Mapping[str, float]] = None
) -> nx.DiGraph:
    """Aggregate a MultiDiGraph into a weighted DiGraph suitable for path search."""

    weighted = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        if isinstance(data, MutableMapping) and data.get("type") == "agent":
            continue
        weighted.add_node(node, **data)

    for source, target, data in graph.edges(data=True):
        if not weighted.has_node(source) or not weighted.has_node(target):
            continue
        weight = edge_weight(graph, source, target, detectability=detectability)
        if weighted.has_edge(source, target):
            existing = weighted[source][target]["weight"]
            weighted[source][target]["weight"] = min(existing, weight)
            weighted[source][target]["count"] = weighted[source][target].get("count", 1) + 1
        else:
            attributes = dict(data)
            attributes["weight"] = weight
            attributes["count"] = 1
            weighted.add_edge(source, target, **attributes)
    return weighted


def _path_total_weight(graph: nx.DiGraph, path: Sequence[str]) -> float:
    total = 0.0
    for source, target in zip(path, path[1:]):
        edge_data = graph.get_edge_data(source, target)
        if not edge_data:
            raise ValueError(f"Missing edge data for segment {source!r}->{target!r}")
        total += float(edge_data["weight"])
    return total


def k_shortest_paths(
    graph: nx.DiGraph,
    sources: Iterable[str],
    targets: Iterable[str],
    k: int = 10,
) -> List[PathWithLength]:
    """Return the top-k shortest paths across all source/target pairs."""

    if k <= 0:
        return []

    unique_sources = [node for node in dict.fromkeys(sources)]
    unique_targets = [node for node in dict.fromkeys(targets)]

    if not unique_sources or not unique_targets:
        return []

    collected: List[PathWithLength] = []
    for source in unique_sources:
        for target in unique_targets:
            if source == target:
                continue
            try:
                generator = nx.shortest_simple_paths(graph, source, target, weight="weight")
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue
            for path_index, path in enumerate(generator):
                if path_index >= k:
                    break
                length = _path_total_weight(graph, path)
                collected.append((list(path), length))

    collected.sort(key=lambda item: (item[1], len(item[0])))
    return collected[:k]


def bottleneck_nodes(paths: Iterable[PathWithLength], top_n: int = 10) -> List[Tuple[str, int]]:
    """Return nodes that appear most frequently across the provided paths."""

    if top_n <= 0:
        return []
    counter: Counter[str] = Counter()
    for path, _ in paths:
        counter.update(path)
    return counter.most_common(top_n)


def _serialize_paths(paths: Sequence[PathWithLength]) -> List[Dict[str, object]]:
    serialized: List[Dict[str, object]] = []
    for path, length in paths:
        serialized.append(
            {
                "nodes": path,
                "source": path[0] if path else None,
                "target": path[-1] if path else None,
                "length": length,
            }
        )
    return serialized


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def run_analysis(
    events_path: Path,
    output_path: Path,
    k: int = 10,
    bottleneck_path: Optional[Path] = None,
    detectability_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Execute the top-k path analysis pipeline and persist results to disk."""

    LOGGER.info("Building graph from %s", events_path)
    graph = build_graph_from_events(events_path, node_mode="technique")
    detectability_map = load_detectability_mapping(detectability_path)

    weighted_graph = _build_weighted_graph(graph, detectability_map)
    sources = find_entry_nodes(weighted_graph)
    targets = find_goal_nodes(weighted_graph)
    LOGGER.info("Identified %d entry nodes and %d goal nodes", len(sources), len(targets))

    paths = k_shortest_paths(weighted_graph, sources, targets, k=k)
    LOGGER.info("Computed %d paths", len(paths))

    results = {
        "events": str(events_path),
        "k": k,
        "sources": sources,
        "targets": targets,
        "paths": _serialize_paths(paths),
    }

    _write_json(output_path, results)

    if bottleneck_path is not None:
        top_nodes = bottleneck_nodes(paths, top_n=10)
        bottleneck_payload = {
            "events": str(events_path),
            "k": k,
            "bottlenecks": [
                {"node": node, "count": count} for node, count in top_nodes
            ],
        }
        _write_json(bottleneck_path, bottleneck_payload)

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute top-k attack paths from orchestrator events"
    )
    parser.add_argument("--events", required=True, help="Path to events.json file")
    parser.add_argument("--k", type=int, default=10, help="Number of paths to return")
    parser.add_argument("--out", required=True, help="Destination for path summary JSON")
    parser.add_argument(
        "--bottlenecks",
        help="Optional path to write bottleneck node summary JSON",
    )
    parser.add_argument(
        "--detectability",
        help="Optional path to detectability mapping YAML (default: analysis/detectability.yml)",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)"
    )
    return parser


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    events_path = Path(args.events)
    out_path = Path(args.out)
    bottleneck_path = Path(args.bottlenecks) if args.bottlenecks else None
    detectability_path = (
        Path(args.detectability)
        if args.detectability
        else DEFAULT_DETECTABILITY_PATH if DEFAULT_DETECTABILITY_PATH.exists()
        else None
    )

    run_analysis(
        events_path=events_path,
        output_path=out_path,
        k=int(args.k),
        bottleneck_path=bottleneck_path,
        detectability_path=detectability_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
