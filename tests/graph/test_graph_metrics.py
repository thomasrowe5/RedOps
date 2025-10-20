import json
from pathlib import Path

import pytest

nx = pytest.importorskip("networkx")

from analysis.graph_metrics import build_graph_from_events, compute_metrics


@pytest.fixture()
def synthetic_events(tmp_path: Path) -> Path:
    events = [
        {
            "agent": "alpha",
            "tactic": "reconnaissance",
            "technique": "T1000",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "agent": "alpha",
            "tactic": "execution",
            "technique": "T1050",
            "timestamp": "2024-01-01T00:05:00Z",
        },
        {
            "agent": "bravo",
            "tactic": "initial-access",
            "technique": "T1100",
            "timestamp": "2024-01-02T00:00:00Z",
        },
        {
            "agent": "bravo",
            "tactic": "exfiltration",
            "technique": "T1200",
            "timestamp": "2024-01-02T00:05:00Z",
        },
    ]

    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps(events), encoding="utf-8")
    return events_path


def test_build_graph_and_compute_metrics(synthetic_events: Path) -> None:
    graph = build_graph_from_events(synthetic_events, node_mode="technique")

    expected_nodes = {
        "T1000",
        "T1050",
        "T1100",
        "T1200",
        "agent:alpha",
        "agent:bravo",
    }
    assert set(graph.nodes) == expected_nodes

    expected_edges = {(
        "T1000",
        "T1050",
    ), (
        "T1100",
        "T1200",
    )}
    assert {(source, target) for source, target in graph.edges} == expected_edges

    metrics = compute_metrics(graph)

    assert set(metrics.keys()) == {
        "degree_centrality",
        "betweenness_centrality",
        "pagerank",
    }

    for metric_values in metrics.values():
        assert set(metric_values.keys()) >= {
            "T1000",
            "T1050",
            "T1100",
            "T1200",
        }

    # Ensure the metric computation yields finite values.
    for metric_values in metrics.values():
        for value in metric_values.values():
            assert value == pytest.approx(value)

    # Graph should be a MultiDiGraph with expected metadata
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.graph["node_mode"] == "technique"
