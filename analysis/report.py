"""Assemble consolidated Markdown run reports from analysis artefacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

JsonMapping = MutableMapping[str, Any]


@dataclass(frozen=True)
class RunInputs:
    """Collection of artefacts discovered for a given run identifier."""

    score_path: Optional[Path]
    graph_metrics_path: Optional[Path]
    topk_path: Optional[Path]
    bottlenecks_path: Optional[Path]
    cosim_summary_path: Optional[Path]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Identifier used to name the result artefacts")
    parser.add_argument(
        "--results-dir",
        default=Path("results"),
        type=Path,
        help="Directory containing analysis JSON artefacts (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("docs/reports"),
        type=Path,
        help="Destination directory for generated Markdown reports",
    )
    parser.add_argument(
        "--top-n",
        default=5,
        type=int,
        help="Number of top-ranked nodes to display for each graph metric",
    )
    return parser.parse_args(argv)


def find_inputs(run_id: str, results_dir: Path) -> RunInputs:
    def existing(path: Path) -> Optional[Path]:
        return path if path.exists() else None

    return RunInputs(
        score_path=existing(results_dir / f"score_{run_id}.json"),
        graph_metrics_path=existing(results_dir / f"graph_metrics_{run_id}.json"),
        topk_path=existing(results_dir / f"topk_{run_id}.json"),
        bottlenecks_path=existing(results_dir / f"bottlenecks_{run_id}.json"),
        cosim_summary_path=existing(results_dir / f"cosim_summary_{run_id}.json"),
    )


def load_json(path: Optional[Path]) -> Optional[Any]:
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def ensure_mapping(value: Any) -> Optional[JsonMapping]:
    if isinstance(value, MutableMapping):
        return value
    return None


def format_optional(value: Any) -> str:
    if value is None:
        return "Not available"
    text = str(value).strip()
    return text if text else "Not available"


def format_percentage(value: Optional[float]) -> str:
    if value is None:
        return "Not available"
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "Not available"


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "Not available"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "Not available"
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    if total_seconds < 1:
        return f"{total_seconds:.3f} s"
    if total_seconds < 60:
        return f"{total_seconds:.2f} s"
    if total_seconds < 3600:
        minutes = total_seconds / 60.0
        return f"{minutes:.2f} min"
    hours = total_seconds / 3600.0
    return f"{hours:.2f} h"


def count_events_from_path(path_text: Any) -> Optional[int]:
    if path_text is None:
        return None
    if isinstance(path_text, Path):
        path = path_text
    else:
        try:
            path = Path(str(path_text))
        except (TypeError, ValueError):
            return None
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not content:
        return 0
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError:
        # Assume JSONL
        return sum(1 for line in content.splitlines() if line.strip())
    if isinstance(loaded, list):
        return len(loaded)
    return None


def extract_run_summary(
    score_data: Optional[JsonMapping],
    topk_data: Optional[JsonMapping],
    cosim_data: Optional[JsonMapping],
) -> Dict[str, Any]:
    events_observed: Optional[int] = None
    if cosim_data:
        metrics = ensure_mapping(cosim_data.get("metrics"))
        if metrics and isinstance(metrics.get("events_observed"), (int, float)):
            events_observed = int(metrics["events_observed"])
    if events_observed is None and topk_data:
        events_observed = count_events_from_path(topk_data.get("events"))

    red_policy = None
    blue_policy = None
    if cosim_data:
        red_info = ensure_mapping(cosim_data.get("red_agent"))
        blue_info = ensure_mapping(cosim_data.get("blue_agent"))
        if red_info:
            red_policy = red_info.get("policy") or red_info.get("rl_policy_path")
        if blue_info:
            blue_policy = blue_info.get("policy_config")

    if red_policy is not None and not isinstance(red_policy, str):
        red_policy = str(red_policy)
    if blue_policy is not None and not isinstance(blue_policy, str):
        blue_policy = str(blue_policy)

    total_techniques = None
    if score_data and isinstance(score_data.get("total_simulated_techniques"), (int, float)):
        total_techniques = int(score_data["total_simulated_techniques"])

    return {
        "events_observed": events_observed,
        "red_policy": red_policy,
        "blue_policy": blue_policy,
        "techniques_simulated": total_techniques,
    }


def extract_detection_metrics(
    score_data: Optional[JsonMapping],
    cosim_data: Optional[JsonMapping],
) -> Dict[str, Optional[float]]:
    detection_rate: Optional[float] = None
    mttd_seconds: Optional[float] = None

    if score_data:
        rate = score_data.get("overall_detection_rate")
        if isinstance(rate, (int, float)):
            detection_rate = float(rate)
        mttd = score_data.get("mttd_seconds")
        if isinstance(mttd, (int, float)):
            mttd_seconds = float(mttd)

    mttr_seconds: Optional[float] = None
    cosim_metrics = ensure_mapping(cosim_data.get("metrics")) if cosim_data else None
    if cosim_metrics:
        if detection_rate is None:
            rate = cosim_metrics.get("detections_observed")
            total = cosim_metrics.get("events_observed")
            if isinstance(rate, (int, float)) and isinstance(total, (int, float)) and total:
                detection_rate = float(rate) / float(total)
        if mttd_seconds is None:
            ttd = cosim_metrics.get("time_to_first_detection_seconds")
            if isinstance(ttd, (int, float)):
                mttd_seconds = float(ttd)
        mttr = cosim_metrics.get("time_to_first_response_seconds")
        if isinstance(mttr, (int, float)):
            mttr_seconds = float(mttr)

    return {
        "detection_rate": detection_rate,
        "mttd_seconds": mttd_seconds,
        "mttr_seconds": mttr_seconds,
    }


def format_metric_table(title: str, metric_data: Mapping[str, float], top_n: int) -> list[str]:
    lines = [f"#### {title}"]
    if not metric_data:
        lines.append("No data available.")
        lines.append("")
        return lines
    sorted_nodes = sorted(metric_data.items(), key=lambda item: item[1], reverse=True)[: max(top_n, 0)]
    lines.append("| Rank | Node | Score |")
    lines.append("| --- | --- | --- |")
    for index, (node, score) in enumerate(sorted_nodes, start=1):
        lines.append(f"| {index} | {node} | {score:.5f} |")
    lines.append("")
    return lines


def render_graph_metrics(graph_data: Optional[JsonMapping], top_n: int) -> list[str]:
    lines: list[str] = ["## Graph Centrality Highlights", ""]
    if not graph_data:
        lines.append("No graph metrics were produced for this run.")
        lines.append("")
        return lines

    degree = ensure_mapping(graph_data.get("degree_centrality")) if graph_data else None
    betweenness = ensure_mapping(graph_data.get("betweenness_centrality")) if graph_data else None
    pagerank = ensure_mapping(graph_data.get("pagerank")) if graph_data else None

    metric_sections: list[str] = []
    if degree:
        metric_sections.extend(format_metric_table("Degree centrality", degree, top_n))
    if betweenness:
        metric_sections.extend(format_metric_table("Betweenness centrality", betweenness, top_n))
    if pagerank:
        metric_sections.extend(format_metric_table("PageRank", pagerank, top_n))

    if not metric_sections:
        lines.append("Graph metrics file was empty.")
        lines.append("")
    else:
        lines.extend(metric_sections)
    return lines


def render_paths(paths_data: Optional[JsonMapping]) -> list[str]:
    lines: list[str] = ["## Top Attack Paths", ""]
    if not paths_data:
        lines.append("No path analysis artefact was found.")
        lines.append("")
        return lines

    paths = paths_data.get("paths") if isinstance(paths_data, MutableMapping) else None
    if not isinstance(paths, list) or not paths:
        lines.append("No attack paths were identified.")
        lines.append("")
        return lines

    for index, entry in enumerate(paths, start=1):
        if not isinstance(entry, MutableMapping):
            continue
        nodes = entry.get("nodes") if isinstance(entry.get("nodes"), list) else []
        node_labels = " → ".join(str(node) for node in nodes)
        length_value = entry.get("length")
        length_str = ""
        if isinstance(length_value, (int, float)):
            length_str = f" (weight {float(length_value):.2f})"
        lines.append(f"{index}. {node_labels}{length_str}")
    lines.append("")
    return lines


def render_bottlenecks(bottleneck_data: Optional[JsonMapping]) -> list[str]:
    lines: list[str] = ["## Path Bottlenecks", ""]
    if not bottleneck_data:
        lines.append("Bottleneck summary was not generated.")
        lines.append("")
        return lines

    entries = bottleneck_data.get("bottlenecks") if isinstance(bottleneck_data, MutableMapping) else None
    if not isinstance(entries, list) or not entries:
        lines.append("No bottlenecks were observed across the analysed paths.")
        lines.append("")
        return lines

    for item in entries:
        if not isinstance(item, MutableMapping):
            continue
        node = item.get("node", "unknown")
        count = item.get("count")
        count_str = str(int(count)) if isinstance(count, (int, float)) else "?"
        lines.append(f"- {node}: observed in {count_str} paths")
    lines.append("")
    return lines


def build_report(
    run_id: str,
    inputs: RunInputs,
    output_dir: Path,
    top_n: int,
) -> Path:
    score_data = ensure_mapping(load_json(inputs.score_path))
    graph_data = ensure_mapping(load_json(inputs.graph_metrics_path))
    topk_data = ensure_mapping(load_json(inputs.topk_path))
    bottleneck_data = ensure_mapping(load_json(inputs.bottlenecks_path))
    cosim_data = ensure_mapping(load_json(inputs.cosim_summary_path))

    summary = extract_run_summary(score_data, topk_data, cosim_data)
    detection_metrics = extract_detection_metrics(score_data, cosim_data)

    lines: list[str] = [f"# Run Report: {run_id}", ""]

    lines.append("## Run Summary")
    lines.append("")
    events_observed = summary.get("events_observed")
    if isinstance(events_observed, int):
        events_line = str(events_observed)
    elif isinstance(events_observed, float):
        events_line = str(int(events_observed))
    elif events_observed is None:
        events_line = "Not available"
    else:
        events_line = str(events_observed)
    lines.append(f"- **Events observed:** {events_line}")

    techniques = summary.get("techniques_simulated")
    if isinstance(techniques, int):
        tech_line = str(techniques)
    elif isinstance(techniques, float):
        tech_line = str(int(techniques))
    elif techniques is None:
        tech_line = "Not available"
    else:
        tech_line = str(techniques)
    lines.append(f"- **Simulated techniques:** {tech_line}")
    lines.append(f"- **Red team policy:** {format_optional(summary.get('red_policy'))}")
    lines.append(f"- **Blue team policy:** {format_optional(summary.get('blue_policy'))}")
    lines.append("")

    lines.append("## Detection Metrics")
    lines.append("")
    lines.append(f"- **Detection rate:** {format_percentage(detection_metrics.get('detection_rate'))}")
    lines.append(f"- **Mean time to detect (MTTD):** {format_seconds(detection_metrics.get('mttd_seconds'))}")
    lines.append(f"- **Mean time to respond (MTTR):** {format_seconds(detection_metrics.get('mttr_seconds'))}")
    lines.append("")

    lines.extend(render_graph_metrics(graph_data, top_n))
    lines.extend(render_paths(topk_data))
    lines.extend(render_bottlenecks(bottleneck_data))

    lines.append("---")
    lines.append("")
    lines.append(
        "**Ethics Reminder:** These simulated scenarios are for defensive readiness "
        "evaluation. Ensure findings are handled responsibly and align with "
        "applicable policies."
    )
    lines.append("")
    lines.append("**LAB USE ONLY – DO NOT DISTRIBUTE.**")
    lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{run_id}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    inputs = find_inputs(args.run_id, args.results_dir)
    build_report(args.run_id, inputs, args.output_dir, max(args.top_n, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
