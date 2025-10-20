"""Render attack technique graphs with centrality-based styling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

import matplotlib.pyplot as plt
import networkx as nx

from analysis.graph_metrics import build_graph_from_events


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _aggregate_digraph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    simple_graph = nx.DiGraph()
    simple_graph.add_nodes_from(graph.nodes(data=True))

    for source, target, edge_data in graph.edges(data=True):
        weight = float(edge_data.get("weight", 1.0))
        if simple_graph.has_edge(source, target):
            simple_graph[source][target]["weight"] += weight
        else:
            simple_graph.add_edge(source, target, weight=weight)

    return simple_graph


def _load_metrics(metrics_path: Path) -> Dict[str, Dict[str, float]]:
    raw_metrics = _load_json(metrics_path)
    metrics: Dict[str, Dict[str, float]] = {}
    for metric_name, values in raw_metrics.items():
        metrics[metric_name] = {str(node): float(score) for node, score in values.items()}
    return metrics


def _load_bottlenecks(bottlenecks_path: Optional[Path]) -> Set[str]:
    if bottlenecks_path is None:
        return set()
    if not bottlenecks_path.exists():
        return set()
    data = _load_json(bottlenecks_path)
    bottlenecks = data.get("bottlenecks", [])
    nodes: Set[str] = set()
    if isinstance(bottlenecks, Iterable):
        for entry in bottlenecks:
            if isinstance(entry, Mapping) and "node" in entry:
                nodes.add(str(entry["node"]))
            elif isinstance(entry, Sequence) and entry:
                nodes.add(str(entry[0]))
    return nodes


def _determine_layout(graph: nx.DiGraph, layout: str) -> Dict[str, Sequence[float]]:
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout == "planar":
        try:
            return nx.planar_layout(graph)
        except nx.NetworkXException:
            return nx.spring_layout(graph, seed=42)
    # Default to spring layout with deterministic seed for reproducibility
    return nx.spring_layout(graph, seed=42)


def _compute_node_sizes(
    nodes: Iterable[str], pagerank: Mapping[str, float], scale: float, min_size: float = 200.0
) -> Dict[str, float]:
    if not pagerank:
        return {node: min_size for node in nodes}
    max_rank = max(pagerank.values()) or 1.0
    return {node: max(min_size, scale * (pagerank.get(node, 0.0) / max_rank)) for node in nodes}


def _select_labels(pagerank: Mapping[str, float], count: int) -> Set[str]:
    if count <= 0 or not pagerank:
        return set()
    ordered = sorted(pagerank.items(), key=lambda item: item[1], reverse=True)
    return {node for node, _ in ordered[:count]}


def visualize_graph(
    events_path: Path,
    metrics_path: Path,
    out_path: Path,
    bottlenecks_path: Optional[Path] = None,
    layout: str = "spring",
    label_count: int = 10,
    node_scale: float = 4000.0,
) -> None:
    graph = build_graph_from_events(events_path, node_mode="technique")
    simple_graph = _aggregate_digraph(graph)

    metrics = _load_metrics(metrics_path)
    pagerank = metrics.get("pagerank", {})
    betweenness = metrics.get("betweenness_centrality", {})
    bottleneck_nodes = _load_bottlenecks(bottlenecks_path)

    if simple_graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty. Ensure events file contains valid data.")

    nodes = list(simple_graph.nodes)
    positions = _determine_layout(simple_graph, layout)
    node_sizes = _compute_node_sizes(nodes, pagerank, scale=node_scale)
    labels = _select_labels(pagerank, label_count)

    betweenness_values = [betweenness.get(node, 0.0) for node in nodes]
    if betweenness_values:
        max_betweenness = max(betweenness_values) or 1.0
        color_map = {node: betweenness.get(node, 0.0) / max_betweenness for node in nodes}
    else:
        color_map = {node: 0.5 for node in nodes}

    highlighted = {node: (node in bottleneck_nodes) for node in nodes}
    edge_colors = ["#999999" for _ in simple_graph.edges]
    edge_widths = [simple_graph.edges[edge].get("weight", 1.0) for edge in simple_graph.edges]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(
        simple_graph,
        positions,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.5,
        arrows=True,
        arrowsize=12,
    )

    regular_nodes = [node for node, flag in highlighted.items() if not flag]
    highlighted_nodes = [node for node, flag in highlighted.items() if flag]

    nx.draw_networkx_nodes(
        simple_graph,
        positions,
        nodelist=regular_nodes,
        node_size=[node_sizes.get(node, 200.0) for node in regular_nodes],
        node_color=[color_map.get(node, 0.5) for node in regular_nodes],
        cmap=plt.cm.viridis,
        linewidths=1.0,
        edgecolors="#333333",
    )

    if highlighted_nodes:
        nx.draw_networkx_nodes(
            simple_graph,
            positions,
            nodelist=highlighted_nodes,
            node_size=[node_sizes.get(node, 300.0) for node in highlighted_nodes],
            node_color=[color_map.get(node, 0.5) for node in highlighted_nodes],
            cmap=plt.cm.viridis,
            linewidths=3.0,
            edgecolors="#ff7f0e",
        )

    label_mapping = {node: node for node in labels}
    nx.draw_networkx_labels(simple_graph, positions, labels=label_mapping, font_size=10)

    if betweenness_values:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array(betweenness_values)
        cbar = plt.colorbar(sm)
        cbar.set_label("Betweenness centrality (normalized)")

    plt.title("Technique Graph with Centrality Metrics")
    plt.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=200)
    plt.close()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True, help="Path to orchestrator events JSON")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to metrics JSON file")
    parser.add_argument("--bottlenecks", type=Path, help="Optional bottleneck summary JSON")
    parser.add_argument("--out", type=Path, required=True, help="Destination PNG path")
    parser.add_argument(
        "--layout",
        choices=("spring", "kamada_kawai", "planar"),
        default="spring",
        help="Graph layout algorithm to use",
    )
    parser.add_argument(
        "--label-count",
        type=int,
        default=10,
        help="Number of top PageRank nodes to label",
    )
    parser.add_argument(
        "--node-scale",
        type=float,
        default=4000.0,
        help="Scaling factor applied to PageRank when determining node sizes",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    visualize_graph(
        events_path=args.events,
        metrics_path=args.metrics,
        bottlenecks_path=args.bottlenecks,
        out_path=args.out,
        layout=args.layout,
        label_count=args.label_count,
        node_scale=args.node_scale,
    )


if __name__ == "__main__":
    main()
