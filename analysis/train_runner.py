"""Utility for orchestrating multi-seed RedOps training experiments."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

import yaml

import train_loop


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_local_orchestrator(config: Dict[str, Any]) -> None:
    orchestrator_url = config.get("env", {}).get("orchestrator")
    if not orchestrator_url:
        raise ValueError("Configuration must define env.orchestrator URL.")
    if not train_loop.is_local_orchestrator(orchestrator_url):
        raise ValueError(
            "Aborting: orchestrator URL must be local (localhost/127.0.0.1/::1) for safety."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple seeded training sessions and aggregate metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/experiment_config.yaml"),
        help="Path to the experiment configuration YAML file.",
    )
    parser.add_argument(
        "--allow-long-run",
        action="store_true",
        help="Allow total timesteps to exceed the safety cap.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override the total timesteps for each training run.",
    )
    return parser.parse_args()


def compute_cumulative_metrics(evaluation: Dict[str, Any]) -> Dict[str, float]:
    episodes = evaluation.get("episodes", [])
    total_reward = float(sum(episode.get("reward", 0.0) for episode in episodes))
    total_steps = float(sum(episode.get("steps", 0) for episode in episodes))
    mean_reward = float(evaluation.get("mean_reward", 0.0))
    mean_steps = float(evaluation.get("mean_steps", 0.0))
    return {
        "total_reward": total_reward,
        "total_steps": total_steps,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
    }


def safe_stdev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(stdev(values))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    config_path = args.config
    config = load_config(config_path)

    ensure_local_orchestrator(config)

    rl_config = config.get("rl", {})
    experiment_config = config.get("experiment", {})

    base_seed = rl_config.get("seed")
    if base_seed is None:
        raise ValueError("Configuration must specify rl.seed for deterministic runs.")

    num_seeds = int(experiment_config.get("num_seeds", 1))
    if num_seeds <= 0:
        raise ValueError("experiment.num_seeds must be a positive integer.")

    seeds = [int(base_seed) + offset for offset in range(num_seeds)]

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_seed_results: List[Dict[str, Any]] = []

    logging.info("Starting experiment batch with seeds: %s", seeds)

    for seed in seeds:
        logging.info("Running training loop for seed %s", seed)
        model_path, stats_path, prefix = train_loop.train(
            config=config,
            seed=seed,
            allow_long_run=args.allow_long_run,
            override_timesteps=args.total_timesteps,
        )

        logging.info("Evaluating trained model for seed %s", seed)
        evaluation_path = train_loop.evaluate(
            model_path=model_path,
            stats_path=stats_path,
            env_config=config.get("env", {}),
            seed=seed,
            prefix=prefix,
        )

        with Path(evaluation_path).open("r", encoding="utf-8") as handle:
            evaluation_data = json.load(handle)

        metrics = compute_cumulative_metrics(evaluation_data)

        per_seed_results.append(
            {
                "seed": seed,
                "model_path": str(model_path),
                "stats_path": str(stats_path),
                "evaluation_path": str(evaluation_path),
                "prefix": prefix,
                "metrics": metrics,
                "evaluation": evaluation_data,
            }
        )

    cumulative_rewards = [item["metrics"]["total_reward"] for item in per_seed_results]
    cumulative_steps = [item["metrics"]["total_steps"] for item in per_seed_results]

    summary = {
        "cumulative_reward": {
            "mean": float(mean(cumulative_rewards)),
            "std": safe_stdev(cumulative_rewards),
        },
        "cumulative_steps": {
            "mean": float(mean(cumulative_steps)),
            "std": safe_stdev(cumulative_steps),
        },
    }

    json_output_path = results_dir / f"experiments_{timestamp}.json"
    json_payload = {
        "timestamp": timestamp,
        "config_path": str(config_path),
        "seeds": seeds,
        "runs": per_seed_results,
        "summary": summary,
    }

    with json_output_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)

    csv_output_path = results_dir / f"experiments_{timestamp}_summary.csv"
    with csv_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std"])
        writer.writerow(["cumulative_reward", summary["cumulative_reward"]["mean"], summary["cumulative_reward"]["std"]])
        writer.writerow(["cumulative_steps", summary["cumulative_steps"]["mean"], summary["cumulative_steps"]["std"]])

    logging.info("Experiment results saved to %s", json_output_path)
    logging.info("Summary CSV saved to %s", csv_output_path)


if __name__ == "__main__":
    main()
