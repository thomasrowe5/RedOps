"""Generate a heatmap summarising detection metrics across detection policies.

This script scans JSON result files from the ``results`` directory, aggregates
``detection_rate`` and ``mttd`` metrics by policy, and renders the data as a
heatmap image saved to ``results/metrics_heatmap.png``.

Usage:
    python heatmap.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MetricAggregate:
    """Container for aggregating numeric metrics for a policy."""

    detection_rates: List[float]
    mttds: List[float]

    @property
    def detection_rate(self) -> Optional[float]:
        return _mean_or_none(self.detection_rates)

    @property
    def mttd(self) -> Optional[float]:
        return _mean_or_none(self.mttds)


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def _default_results_dir(script_path: Path) -> Path:
    """Resolve the default results directory relative to the repository root."""

    # The script lives in ``analysis/`` so the repository root is one level up.
    repo_root = script_path.resolve().parent.parent
    return repo_root / "results"


def _load_results(results_dir: Path) -> List[Mapping[str, object]]:
    """Load JSON objects from all files within ``results_dir``."""

    result_files = sorted(results_dir.glob("*.json"))
    results: List[Mapping[str, object]] = []

    for file_path in result_files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
            print(f"Skipping {file_path.name}: invalid JSON ({exc}).", file=sys.stderr)
            continue
        except OSError as exc:  # pragma: no cover - defensive branch
            print(f"Skipping {file_path.name}: unable to read file ({exc}).", file=sys.stderr)
            continue

        # Support either a single result object or an array of them per file.
        if isinstance(data, Mapping):
            results.append(data)
        elif isinstance(data, list):
            results.extend([item for item in data if isinstance(item, Mapping)])
        else:  # pragma: no cover - defensive branch
            print(
                f"Skipping {file_path.name}: unsupported JSON structure (expected object or list).",
                file=sys.stderr,
            )
    return results


def _extract_metric(record: Mapping[str, object], key: str) -> Optional[float]:
    """Attempt to retrieve a numeric metric from common JSON structures."""

    # Direct key (e.g., {"detection_rate": 0.8})
    value = record.get(key)
    if isinstance(value, (int, float)):
        return float(value)

    # Nested under "metrics" or "results"
    for container_key in ("metrics", "results", "summary"):
        container = record.get(container_key)
        if isinstance(container, Mapping):
            nested_value = container.get(key)
            if isinstance(nested_value, (int, float)):
                return float(nested_value)

    return None


def _extract_policy(record: Mapping[str, object]) -> Optional[str]:
    """Retrieve the detection policy identifier from a result record."""

    for key in ("policy", "detector", "strategy", "policy_name"):
        value = record.get(key)
        if isinstance(value, str):
            return value.lower()

    # Nested metadata (e.g., {"metadata": {"policy": "llm"}})
    for container_key in ("metadata", "config"):
        container = record.get(container_key)
        if isinstance(container, Mapping):
            for key in ("policy", "detector", "strategy", "policy_name"):
                value = container.get(key)
                if isinstance(value, str):
                    return value.lower()

    return None


def aggregate_metrics(records: Iterable[Mapping[str, object]]) -> Dict[str, MetricAggregate]:
    """Aggregate detection metrics by policy."""

    aggregates: MutableMapping[str, MetricAggregate] = {}

    for record in records:
        policy = _extract_policy(record)
        if not policy:
            continue

        detection_rate = _extract_metric(record, "detection_rate")
        mttd = _extract_metric(record, "mttd")

        if policy not in aggregates:
            aggregates[policy] = MetricAggregate(detection_rates=[], mttds=[])

        if detection_rate is not None:
            aggregates[policy].detection_rates.append(detection_rate)
        if mttd is not None:
            aggregates[policy].mttds.append(mttd)

    return dict(aggregates)


def build_heatmap_matrix(aggregates: Mapping[str, MetricAggregate], policies: List[str], metrics: List[str]) -> np.ndarray:
    """Create a 2D matrix populated with aggregated metric values."""

    matrix = np.full((len(policies), len(metrics)), np.nan, dtype=float)

    for row_index, policy in enumerate(policies):
        aggregate = aggregates.get(policy)
        if not aggregate:
            continue

        values = {
            "detection_rate": aggregate.detection_rate,
            "mttd": aggregate.mttd,
        }

        for col_index, metric in enumerate(metrics):
            value = values.get(metric)
            if value is not None:
                matrix[row_index, col_index] = value

    return matrix


def render_heatmap(matrix: np.ndarray, policies: List[str], metrics: List[str], output_path: Path) -> None:
    """Render and save the heatmap figure."""

    fig, ax = plt.subplots(figsize=(6, 4))

    # Use a masked array so NaNs are displayed distinctly.
    masked_matrix = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis
    cmap.set_bad(color="lightgrey")

    cax = ax.imshow(masked_matrix, interpolation="nearest", aspect="auto", cmap=cmap)
    fig.colorbar(cax, ax=ax, label="Metric Value")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([metric.replace("_", " ").title() for metric in metrics])
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([policy.title() for policy in policies])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Policy")
    ax.set_title("Detection Metrics Heatmap")

    # Annotate each cell with its value, if present.
    for row_index, policy in enumerate(policies):
        for col_index, metric in enumerate(metrics):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                continue
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color="white")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a metrics heatmap from result JSON files.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing JSON result files (defaults to <repo_root>/results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the generated heatmap image (defaults to <results_dir>/metrics_heatmap.png).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    script_path = Path(__file__)
    results_dir = args.results_dir or _default_results_dir(script_path)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    records = _load_results(results_dir)
    if not records:
        print(f"No JSON result records found in {results_dir}.", file=sys.stderr)
        return 1

    aggregates = aggregate_metrics(records)
    if not aggregates:
        print("No metrics available to plot after aggregation.", file=sys.stderr)
        return 1

    # Policies ordered explicitly for consistent presentation. Include any others found.
    policies = ["random", "rule", "llm"]
    extra_policies = sorted(policy for policy in aggregates if policy not in policies)
    policies.extend(extra_policies)

    metrics = ["detection_rate", "mttd"]

    matrix = build_heatmap_matrix(aggregates, policies, metrics)

    output_path = args.output or (results_dir / "metrics_heatmap.png")
    render_heatmap(matrix, policies, metrics, output_path)

    print(f"Heatmap saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
