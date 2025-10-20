"""Reward model utilities for lab-only reinforcement learning experiments.

This module provides deterministic functions for computing rewards and
updating exposure levels based on event metadata. It avoids external
state and randomness so it can be tested reliably.
"""
from __future__ import annotations

from typing import Dict, Mapping


# Default impact scores for tactics when ``impact_est`` is not supplied.
_TACTIC_BASE_IMPACT: Mapping[str, float] = {
    "initial-access": 3.0,
    "execution": 2.5,
    "persistence": 2.0,
    "privilege-escalation": 2.5,
    "defense-evasion": 1.5,
    "credential-access": 2.5,
    "discovery": 1.0,
    "lateral-movement": 3.0,
    "collection": 1.5,
    "command-and-control": 2.0,
    "exfiltration": 3.5,
    "impact": 4.0,
}

# Default detectability scores for techniques. Values should range between 0 and 1.
_TECHNIQUE_DETECTABILITY: Mapping[str, float] = {
    "T1003": 0.7,
    "T1059": 0.5,
    "T1078": 0.6,
    "T1105": 0.4,
    "T1110": 0.8,
    "T1566": 0.9,
    "T1027": 0.3,
    "T1041": 0.5,
    "T1486": 0.6,
    "default": 0.5,
}


def tactic_to_base_impact(tactic: str) -> float:
    """Return the default impact score for a tactic.

    Unknown tactics fall back to the minimum configured impact value to
    discourage unclassified actions from receiving excessive reward.
    """

    if not tactic:
        return min(_TACTIC_BASE_IMPACT.values())
    return _TACTIC_BASE_IMPACT.get(tactic.lower(), min(_TACTIC_BASE_IMPACT.values()))


def technique_detectability(technique: str) -> float:
    """Return the detectability score for a technique.

    The value lies in ``[0, 1]`` and defaults to the configured
    ``"default"`` score when the technique is unknown or missing.
    """

    if not technique:
        return _TECHNIQUE_DETECTABILITY["default"]
    return _TECHNIQUE_DETECTABILITY.get(technique.upper(), _TECHNIQUE_DETECTABILITY["default"])


def reward_for_event(event: Dict[str, object], exposure_before: float) -> float:
    """Compute the reward for an event.

    Args:
        event: Metadata dictionary describing the event. Expected keys are
            ``tactic``, ``technique``, ``kind``, ``agent_id``, ``timestamp``,
            and optionally ``detect_flag`` (bool) and ``impact_est`` (float).
        exposure_before: Exposure value before the event occurred. Expected
            to lie between 0.0 and 1.0 but clamped if outside this range.

    Returns:
        A deterministic floating-point reward value.
    """

    tactic = str(event.get("tactic", ""))
    technique = str(event.get("technique", ""))

    impact_est = event.get("impact_est")
    base_reward = float(impact_est) if impact_est is not None else tactic_to_base_impact(tactic)

    detectability = technique_detectability(technique)
    detect_flag = bool(event.get("detect_flag", False))
    penalty = detectability * 2.0 if detect_flag else 0.0

    exposure_clamped = max(0.0, min(1.0, float(exposure_before)))
    exposure_cost = exposure_clamped * 0.5

    total_reward = base_reward - penalty - exposure_cost
    return total_reward


def update_exposure(exposure_before: float, event: Dict[str, object]) -> float:
    """Update exposure after handling an event.

    Exposure increases proportionally to the technique detectability. The
    value is capped at ``1.0`` to keep the metric bounded.
    """

    technique = str(event.get("technique", ""))
    detectability = technique_detectability(technique)
    new_exposure = float(exposure_before) + detectability * 0.2
    return max(0.0, min(1.0, new_exposure))


if __name__ == "__main__":
    # Simple deterministic tests for manual verification.
    seeded_events = [
        {
            "tactic": "initial-access",
            "technique": "T1566",
            "kind": "phishing",
            "agent_id": "agent-001",
            "timestamp": "2024-04-22T10:00:00Z",
            "detect_flag": True,
        },
        {
            "tactic": "lateral-movement",
            "technique": "T1105",
            "kind": "remote-service",
            "agent_id": "agent-002",
            "timestamp": "2024-04-22T10:05:00Z",
            "impact_est": 3.2,
            "detect_flag": False,
        },
        {
            "tactic": "collection",
            "technique": "T1041",
            "kind": "exfil",
            "agent_id": "agent-003",
            "timestamp": "2024-04-22T10:10:00Z",
            "impact_est": 2.8,
            "detect_flag": True,
        },
    ]

    exposure = 0.0
    for event in seeded_events:
        reward = reward_for_event(event, exposure)
        exposure = update_exposure(exposure, event)
        print(f"event={event['technique']} reward={reward:.3f} exposure={exposure:.3f}")
