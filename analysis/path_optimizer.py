"""Beam-search planning utilities for selecting action paths."""

from __future__ import annotations

import argparse
import dataclasses
import random
from typing import Iterable, List, Sequence, Tuple, Union


ALPHA: float = 0.9


ActionType = Union[dict, "_ActionLike"]


@dataclasses.dataclass(frozen=True)
class _ActionLike:
    """Protocol-like helper for static type checking."""

    name: str
    impact: float
    detectability: float


@dataclasses.dataclass
class _Node:
    plan: List[ActionType]
    score: float
    exposure: float


def _get_action_attr(action: ActionType, attribute: str) -> float:
    if isinstance(action, dict):
        value = action[attribute]
    else:
        value = getattr(action, attribute)
    return float(value)


def _action_value(action: ActionType, exposure: float) -> Tuple[float, float]:
    impact = _get_action_attr(action, "impact")
    detectability = _get_action_attr(action, "detectability")
    value = impact * (1.0 - exposure) - detectability * ALPHA
    new_exposure = 1.0 - (1.0 - exposure) * (1.0 - detectability)
    new_exposure = max(0.0, min(1.0, new_exposure))
    return value, new_exposure


def _plan_score(plan: Iterable[ActionType], exposure_start: float) -> Tuple[float, float]:
    score = 0.0
    exposure = float(exposure_start)
    for action in plan:
        value, exposure = _action_value(action, exposure)
        score += value
    return score, exposure


def plan_path(
    candidate_actions: Sequence[ActionType],
    exposure_start: float,
    beam_width: int = 6,
    max_depth: int = 6,
) -> List[ActionType]:
    """Return the best plan using beam search over the provided actions."""

    if max_depth <= 0 or not candidate_actions:
        return []

    initial = _Node(plan=[], score=0.0, exposure=float(exposure_start))
    beam: List[_Node] = [initial]

    for _ in range(max_depth):
        expanded: List[_Node] = []
        for node in beam:
            for action in candidate_actions:
                value, new_exposure = _action_value(action, node.exposure)
                expanded.append(
                    _Node(
                        plan=node.plan + [action],
                        score=node.score + value,
                        exposure=new_exposure,
                    )
                )
        if not expanded:
            break
        expanded.sort(key=lambda n: (n.score, -len(n.plan)), reverse=True)
        beam = expanded[: max(1, beam_width)]

    best_node = max(beam, key=lambda n: (n.score, -len(n.plan)))
    return best_node.plan


DEFAULT_ACTION_LIBRARY: Tuple[ActionType, ...] = (
    {"name": "Network Scanning", "impact": 1.0, "detectability": 0.4},
    {"name": "OSINT Profiling", "impact": 0.6, "detectability": 0.2},
    {"name": "Phishing Email", "impact": 1.5, "detectability": 0.6},
    {"name": "Exploit Web Vulnerability", "impact": 2.0, "detectability": 0.8},
    {"name": "Create Scheduled Task", "impact": 1.2, "detectability": 0.7},
    {"name": "Modify Startup Scripts", "impact": 1.1, "detectability": 0.5},
    {"name": "Pass-the-Hash", "impact": 1.8, "detectability": 0.9},
    {"name": "Remote Service Abuse", "impact": 1.6, "detectability": 0.7},
    {"name": "Archive and Compress Data", "impact": 1.4, "detectability": 0.6},
    {"name": "Exfiltrate Over HTTPS", "impact": 2.1, "detectability": 0.9},
)


def _format_plan(plan: Sequence[ActionType], exposure_start: float) -> Tuple[List[str], float]:
    score, _ = _plan_score(plan, exposure_start)
    names = []
    for action in plan[:5]:
        if isinstance(action, dict):
            names.append(action.get("name", "<unnamed>"))
        else:
            names.append(getattr(action, "name", "<unnamed>"))
    return names, score


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beam-search optimizer for action paths")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic ordering")
    parser.add_argument("--depth", type=int, default=6, help="Maximum depth of the plan")
    parser.add_argument("--beam", type=int, default=6, help="Beam width")
    parser.add_argument("--exposure", type=float, default=0.15, help="Starting exposure value")
    return parser


def _run_cli(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)

    actions = list(DEFAULT_ACTION_LIBRARY)
    random.shuffle(actions)

    plan = plan_path(actions, exposure_start=args.exposure, beam_width=args.beam, max_depth=args.depth)
    top_names, score = _format_plan(plan, args.exposure)

    print("Top-5 actions:")
    for index, name in enumerate(top_names, start=1):
        print(f"  {index}. {name}")
    print(f"Total score: {score:.3f}")


def _self_test() -> None:
    actions = (
        {"name": "High Impact", "impact": 2.0, "detectability": 0.2},
        {"name": "Low Impact", "impact": 1.0, "detectability": 0.1},
    )
    plan = plan_path(actions, exposure_start=0.0, beam_width=2, max_depth=2)
    assert len(plan) == 2, "Plan should contain two actions"
    assert all(
        (isinstance(step, dict) and step.get("name") == "High Impact") for step in plan
    ), "Beam search should prefer repeating the high impact action"


if __name__ == "__main__":
    _self_test()
    parser = _build_cli_parser()
    _run_cli(parser.parse_args())
