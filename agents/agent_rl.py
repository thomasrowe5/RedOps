"""RL policy wrapper for executing simulated RedOps episodes."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx
import gym
from stable_baselines3 import PPO

LOGGER = logging.getLogger("redops.agent.rl")


class SafetyError(RuntimeError):
    """Raised when a safety constraint is violated."""


def _sanitize_orchestrator_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise SafetyError("Orchestrator URL must include http or https scheme")
    host = parsed.hostname or ""
    if host not in {"localhost"} and not host.startswith("127."):
        raise SafetyError("Unsafe orchestrator host; only localhost or 127.* are allowed")
    return raw_url.rstrip("/")


def _ensure_policy_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise SafetyError(f"Policy file not found: {path}")
    if path.is_dir():
        raise SafetyError(f"Policy path points to a directory: {path}")
    return path


def _build_event(
    agent_id: str,
    action: Any,
    info: Optional[Dict[str, Any]],
    default_note: str = "ppo-policy",
) -> Dict[str, Any]:
    event: Dict[str, Any] = {}
    if isinstance(info, dict):
        embedded = info.get("event")
        if isinstance(embedded, dict):
            event.update(embedded)
        else:
            if "tactic" in info:
                event["tactic"] = str(info["tactic"])
            if "technique" in info:
                event["technique"] = str(info["technique"])
            if "note" in info:
                event["note"] = str(info["note"])
    event.setdefault("tactic", "unknown")
    technique_default = f"action_{action}" if action is not None else "unknown"
    event.setdefault("technique", technique_default)
    event.setdefault("note", default_note)
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    event.setdefault("agent_id", agent_id)
    event["kind"] = "simulated"
    return event


def _post_event(client: httpx.Client, run_id: str, event: Dict[str, Any]) -> None:
    endpoint = f"/runs/{run_id}/events"
    try:
        response = client.post(endpoint, json=event)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        LOGGER.error("Failed to post event: %s", exc)


def _run_episode(
    env: gym.Env,
    model: PPO,
    client: httpx.Client,
    run_id: str,
    agent_id: str,
    episode_index: int,
) -> Tuple[float, int, Optional[float]]:
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, reset_info = reset_result
        if isinstance(reset_info, dict) and "exposure" in reset_info:
            try:
                final_exposure = float(reset_info["exposure"])
            except (TypeError, ValueError):
                LOGGER.debug("Exposure value not convertible to float during reset: %s", reset_info["exposure"])
        else:
            final_exposure = None
    else:
        obs = reset_result
        final_exposure = None
    done = False
    cumulative_reward = 0.0
    steps = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_result
        cumulative_reward += float(reward)
        steps += 1

        if isinstance(info, dict) and "exposure" in info:
            try:
                final_exposure = float(info["exposure"])
            except (TypeError, ValueError):
                LOGGER.debug("Exposure value not convertible to float: %s", info["exposure"])

        event_payload = _build_event(agent_id, action, info if isinstance(info, dict) else None)
        _post_event(client, run_id, event_payload)

        if done:
            LOGGER.info(
                "Episode %s completed in %s steps with reward %.2f",
                episode_index + 1,
                steps,
                cumulative_reward,
            )
            break

    return cumulative_reward, steps, final_exposure


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a trained PPO policy in the RedOps gym environment")
    parser.add_argument("--policy-path", required=True, help="Path to the PPO policy .zip file")
    parser.add_argument(
        "--env-url",
        default="http://localhost:8000",
        help="Base URL for the orchestrator hosting the environment (default: http://localhost:8000)",
    )
    parser.add_argument("--agent-id", required=True, help="Identifier for this agent instance")
    parser.add_argument("--run-id", required=True, help="Identifier for the orchestrated run")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to execute (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

        orchestrator_url = _sanitize_orchestrator_url(args.env_url)
        policy_path = _ensure_policy_path(args.policy_path)

        env = gym.make(
            "redops-v0",
            orchestrator=orchestrator_url,
            run_id=args.run_id,
            agent_id=args.agent_id,
        )
        model = PPO.load(str(policy_path), env=env)

        client = httpx.Client(base_url=orchestrator_url, timeout=10.0)

        try:
            for episode in range(max(args.num_episodes, 1)):
                reward, steps, exposure = _run_episode(env, model, client, args.run_id, args.agent_id, episode)
                exposure_display = "n/a" if exposure is None else f"{exposure:.4f}"
                print(
                    f"Episode {episode + 1}: reward={reward:.2f}, steps={steps}, final_exposure={exposure_display}"
                )
        finally:
            client.close()
            env.close()

        return 0
    except SafetyError as exc:
        LOGGER.error("Safety constraint violated: %s", exc)
        return 2
    except FileNotFoundError as exc:
        LOGGER.error("Required resource missing: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure path
        LOGGER.exception("Unexpected error during RL agent execution: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
