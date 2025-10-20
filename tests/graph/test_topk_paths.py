import pytest

nx = pytest.importorskip("networkx")

from analysis.topk_paths import bottleneck_nodes, k_shortest_paths


def build_sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("A", "C", 5.0),
            ("A", "D", 2.0),
            ("D", "C", 2.0),
        ],
        weight="weight",
    )
    return graph


def test_k_shortest_paths_returns_expected_paths() -> None:
    graph = build_sample_graph()

    paths = k_shortest_paths(graph, sources=["A"], targets=["C"], k=3)

    assert paths == [
        (["A", "B", "C"], 2.0),
        (["A", "D", "C"], 4.0),
        (["A", "C"], 5.0),
    ]


def test_bottleneck_nodes_counts_frequency_across_paths() -> None:
    graph = build_sample_graph()
    paths = k_shortest_paths(graph, sources=["A"], targets=["C"], k=3)

    bottlenecks = bottleneck_nodes(paths, top_n=2)

    assert bottlenecks == [("A", 3), ("C", 3)]
