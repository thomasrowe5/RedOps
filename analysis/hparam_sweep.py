"""Hyperparameter sweep utilities for PPO training on the RedOps environment."""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, Iterator, List, Mapping

# Enforce CPU-only execution before any torch dependencies are imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import yaml

import train_loop


DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "learning_rate": [3e-4, 1e-4, 5e-4],
    "n_steps": [1024, 2048, 4096],
    "batch_size": [32, 64, 128],
    "gamma": [0.95, 0.99, 0.995],
    "clip_range": [0.1, 0.2, 0.3],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform grid or random search over PPO hyperparameters."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/experiment_config.yaml"),
        help="Base experiment configuration file to clone for each run.",
    )
    parser.add_argument(
        "--search",
        choices=("grid", "random"),
        default="grid",
        help="Type of hyperparameter search to execute.",
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        default=None,
        help="Optional YAML/JSON file describing the hyperparameter search space.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of random configurations to evaluate (random search only).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds per configuration. Overrides experiment.num_seeds.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset to apply to the base seed for reproducibility of sweeps.",
    )
    parser.add_argument(
        "--sweep-seed",
        type=int,
        default=0,
        help="Seed used to sample configurations for random search.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override the total timesteps per run.",
    )
    parser.add_argument(
        "--allow-long-run",
        action="store_true",
        help="Permit runs that exceed the default safety timestep cap.",
    )
    parser.add_argument(
        "--baseline-random",
        type=float,
        default=0.0,
        help="Baseline reward for a random policy used in early stopping.",
    )
    parser.add_argument(
        "--baseline-delta",
        type=float,
        default=5.0,
        help="Tolerance below the baseline random reward before aborting a config.",
    )
    parser.add_argument(
        "--early-stop-steps",
        type=int,
        default=0,
        help="Minimum total timesteps before applying the early stop rule (0 disables).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to store sweep CSV summaries.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    return train_loop.load_config(path)


def load_search_space(path: Path | None) -> Dict[str, List[Any]]:
    if path is None:
        return copy.deepcopy(DEFAULT_SEARCH_SPACE)

    if not path.exists():
        raise FileNotFoundError(f"Search space file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)

    if not isinstance(data, Mapping):
        raise ValueError("Search space file must define a mapping of parameter -> values.")

    search_space: Dict[str, List[Any]] = {}
    for key, value in data.items():
        normalized = normalize_param_name(key)
        if isinstance(value, Mapping) and "values" in value:
            candidates = value["values"]
        else:
            candidates = value
        if not isinstance(candidates, Iterable) or isinstance(candidates, (str, bytes)):
            raise ValueError(
                f"Search space for '{key}' must be an iterable of candidate values."
            )
        search_space[normalized] = list(candidates)
    return search_space


def normalize_param_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized in {"lr", "learning_rate"}:
        return "learning_rate"
    if normalized in {"clip", "clip_range"}:
        return "clip_range"
    if normalized in {"nsteps", "n_steps"}:
        return "n_steps"
    if normalized in {"batch", "batch_size"}:
        return "batch_size"
    if normalized in {"gamma"}:
        return "gamma"
    raise ValueError(f"Unsupported hyperparameter name: {name}")


def iter_grid(search_space: Mapping[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    keys = list(search_space.keys())
    for combination in itertools.product(*(search_space[k] for k in keys)):
        yield {keys[idx]: combination[idx] for idx in range(len(keys))}


def iter_random(
    search_space: Mapping[str, List[Any]], num_samples: int, seed: int
) -> Iterator[Dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError("Number of random samples must be positive.")

    rng = random.Random(seed)
    keys = list(search_space.keys())
    for _ in range(num_samples):
        sample = {}
        for key in keys:
            candidates = search_space[key]
            if not candidates:
                raise ValueError(f"No candidate values provided for parameter '{key}'.")
            sample[key] = rng.choice(candidates)
        yield sample


def ensure_local_orchestrator(config: Mapping[str, Any]) -> None:
    orchestrator_url = config.get("env", {}).get("orchestrator")
    if not orchestrator_url:
        raise ValueError("Configuration must define env.orchestrator URL.")
    if not train_loop.is_local_orchestrator(orchestrator_url):
        raise ValueError(
            "Aborting: orchestrator URL must be local (localhost/127.0.0.1/::1) for safety."
        )


def cast_param_value(name: str, value: Any) -> Any:
    if name in {"n_steps", "batch_size"}:
        return int(value)
    if name in {"learning_rate", "gamma", "clip_range"}:
        return float(value)
    return value


def apply_hyperparameters(
    base_config: Mapping[str, Any], params: Mapping[str, Any]
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    rl_config = config.setdefault("rl", {})
    for name, raw_value in params.items():
        value = cast_param_value(name, raw_value)
        rl_config[name] = value
    return config


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def safe_stdev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(stdev(values))


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    rewards = [float(run["mean_reward"]) for run in runs]
    steps = [float(run["mean_steps"]) for run in runs]
    return {
        "reward_mean": safe_mean(rewards),
        "reward_std": safe_stdev(rewards),
        "steps_mean": safe_mean(steps),
        "steps_std": safe_stdev(steps),
    }


def evaluate_seed(
    config: Mapping[str, Any],
    seed: int,
    allow_long_run: bool,
    override_timesteps: int | None,
) -> Dict[str, Any]:
    model_path, stats_path, prefix = train_loop.train(
        config=config,
        seed=seed,
        allow_long_run=allow_long_run,
        override_timesteps=override_timesteps,
    )

    evaluation_path = train_loop.evaluate(
        model_path=model_path,
        stats_path=stats_path,
        env_config=config.get("env", {}),
        seed=seed,
        prefix=prefix,
    )

    with Path(evaluation_path).open("r", encoding="utf-8") as handle:
        evaluation_data = json.load(handle)

    return {
        "seed": seed,
        "mean_reward": float(evaluation_data.get("mean_reward", 0.0)),
        "median_reward": float(evaluation_data.get("median_reward", 0.0)),
        "mean_steps": float(evaluation_data.get("mean_steps", 0.0)),
        "evaluation_path": str(evaluation_path),
    }


def sweep_configs(args: argparse.Namespace) -> Path:
    base_config = load_config(args.config)
    ensure_local_orchestrator(base_config)

    rl_config = base_config.get("rl", {})
    experiment_config = base_config.get("experiment", {})

    base_seed = int(rl_config.get("seed", 0)) + int(args.seed_offset)
    num_seeds = int(args.seeds) if args.seeds is not None else int(experiment_config.get("num_seeds", 1))
    if num_seeds <= 0:
        raise ValueError("Number of seeds must be a positive integer.")

    total_timesteps = (
        int(args.total_timesteps)
        if args.total_timesteps is not None
        else int(rl_config.get("total_timesteps", 0))
    )
    if total_timesteps <= 0:
        raise ValueError("Total timesteps must be a positive integer.")

    search_space = load_search_space(args.search_space)

    if args.search == "grid":
        config_iterator = iter_grid(search_space)
    else:
        config_iterator = iter_random(search_space, args.num_samples, args.sweep_seed)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.results_dir / f"hparam_sweep_{timestamp}.csv"

    fieldnames = [
        "config_id",
        "search_type",
        "learning_rate",
        "n_steps",
        "batch_size",
        "gamma",
        "clip_range",
        "total_timesteps",
        "seeds_requested",
        "seeds_completed",
        "reward_mean",
        "reward_std",
        "steps_mean",
        "steps_std",
        "aborted",
        "abort_reason",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for config_id, params in enumerate(config_iterator, start=1):
            logging.info("Evaluating configuration %s: %s", config_id, params)
            per_seed_results: List[Dict[str, Any]] = []
            aborted = False
            abort_reason = ""

            config_template = apply_hyperparameters(base_config, params)
            config_template.setdefault("rl", {})["total_timesteps"] = total_timesteps

            for seed_offset in range(num_seeds):
                seed = base_seed + seed_offset
                run_config = copy.deepcopy(config_template)
                run_config.setdefault("rl", {})["seed"] = seed

                try:
                    result = evaluate_seed(
                        config=run_config,
                        seed=seed,
                        allow_long_run=args.allow_long_run,
                        override_timesteps=args.total_timesteps,
                    )
                except Exception as exc:  # pragma: no cover - defensive programming
                    logging.exception(
                        "Configuration %s failed for seed %s due to %s", config_id, seed, exc
                    )
                    aborted = True
                    abort_reason = f"failure: {exc}"
                    break

                per_seed_results.append(result)

                logging.info(
                    "Config %s seed %s -> mean_reward=%.3f mean_steps=%.2f",
                    config_id,
                    seed,
                    result["mean_reward"],
                    result["mean_steps"],
                )

                if args.early_stop_steps and total_timesteps >= args.early_stop_steps:
                    mean_reward = safe_mean([r["mean_reward"] for r in per_seed_results])
                    threshold = float(args.baseline_random) - float(args.baseline_delta)
                    if mean_reward < threshold:
                        aborted = True
                        abort_reason = (
                            "early_stop: mean_reward "
                            f"{mean_reward:.3f} < threshold {threshold:.3f}"
                        )
                        logging.info(
                            "Early stopping configuration %s after seed %s: %s",
                            config_id,
                            seed,
                            abort_reason,
                        )
                        break

            summary = summarize_runs(per_seed_results)

            row = {
                "config_id": config_id,
                "search_type": args.search,
                "learning_rate": cast_param_value(
                    "learning_rate", params.get("learning_rate", rl_config.get("learning_rate"))
                ),
                "n_steps": cast_param_value(
                    "n_steps", params.get("n_steps", rl_config.get("n_steps"))
                ),
                "batch_size": cast_param_value(
                    "batch_size", params.get("batch_size", rl_config.get("batch_size"))
                ),
                "gamma": cast_param_value("gamma", params.get("gamma", rl_config.get("gamma"))),
                "clip_range": cast_param_value(
                    "clip_range", params.get("clip_range", rl_config.get("clip_range"))
                ),
                "total_timesteps": total_timesteps,
                "seeds_requested": num_seeds,
                "seeds_completed": len(per_seed_results),
                "reward_mean": summary["reward_mean"],
                "reward_std": summary["reward_std"],
                "steps_mean": summary["steps_mean"],
                "steps_std": summary["steps_std"],
                "aborted": aborted,
                "abort_reason": abort_reason,
            }

            writer.writerow(row)
            handle.flush()

    logging.info("Sweep complete. Results written to %s", csv_path)
    return csv_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sweep_configs(args)


if __name__ == "__main__":
    main()
