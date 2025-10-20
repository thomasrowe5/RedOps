"""Train a PPO agent on the RedOps Gym environment.

Minimal dependencies: gym>=0.26, stable-baselines3>=2.0.0, torch, pyyaml.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SAFETY_TIMESTEP_CAP = 5_000_000
EVAL_EPISODES = 10


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def is_local_orchestrator(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"}


def make_env(env_config: Dict[str, Any], seed: int):
    def _init():
        env = gym.make(
            "redops-v0",
            orchestrator=env_config.get("orchestrator"),
            max_steps=env_config.get("max_steps"),
            run_prefix=env_config.get("run_prefix"),
        )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _init


def set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


def create_vec_env(env_config: Dict[str, Any], seed: int) -> VecNormalize:
    vec_env = DummyVecEnv([make_env(env_config, seed)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    vec_env.seed(seed)
    return vec_env


def create_eval_env(
    env_config: Dict[str, Any],
    seed: int,
    stats_path: Optional[Path] = None,
    reference_env: Optional[VecNormalize] = None,
) -> VecNormalize:
    base_env = DummyVecEnv([make_env(env_config, seed)])
    if stats_path and stats_path.exists():
        eval_env = VecNormalize.load(str(stats_path), base_env)
    else:
        eval_env = VecNormalize(base_env, training=False, norm_obs=True, norm_reward=False)
        if reference_env is not None:
            eval_env.obs_rms = reference_env.obs_rms
            eval_env.clip_obs = reference_env.clip_obs
            eval_env.clip_reward = reference_env.clip_reward
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.seed(seed)
    return eval_env


def train(
    config: Dict[str, Any],
    seed: int,
    allow_long_run: bool,
    override_timesteps: Optional[int],
) -> tuple[Path, Path, str]:
    env_config = config.get("env", {})
    rl_config = config.get("rl", {})
    logging_config = config.get("logging", {})

    orchestrator_url = env_config.get("orchestrator", "")
    if not is_local_orchestrator(orchestrator_url):
        raise ValueError(
            "Aborting: orchestrator URL must be local (localhost/127.0.0.1/::1) for safety."
        )

    total_timesteps = override_timesteps or rl_config.get("total_timesteps", 0)
    if total_timesteps <= 0:
        raise ValueError("Total timesteps must be a positive integer.")

    if total_timesteps > SAFETY_TIMESTEP_CAP and not allow_long_run:
        raise ValueError(
            f"Requested total_timesteps={total_timesteps} exceeds safety cap of {SAFETY_TIMESTEP_CAP}. "
            "Use --allow-long-run to override."
        )

    set_deterministic_seeds(seed)

    vec_env = create_vec_env(env_config, seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_root = Path(logging_config.get("checkpoint_dir", "models/rl"))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = f"ppo_redops_{timestamp}"

    checkpoint_callback = CheckpointCallback(
        save_freq=int(logging_config.get("save_interval", 5000)),
        save_path=str(checkpoint_root),
        name_prefix=checkpoint_prefix,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_stats_path = checkpoint_root / f"{checkpoint_prefix}_vecnormalize.pkl"
    eval_env_for_callback = create_eval_env(env_config, seed + 1, None, reference_env=vec_env)
    eval_callback = EvalCallback(
        eval_env_for_callback,
        best_model_save_path=str(checkpoint_root / f"{checkpoint_prefix}_best"),
        log_path=str(checkpoint_root / f"{checkpoint_prefix}_eval"),
        eval_freq=max(1, int(logging_config.get("save_interval", 5000))),
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(rl_config.get("learning_rate", 3e-4)),
        n_steps=int(rl_config.get("n_steps", 2048)),
        batch_size=int(rl_config.get("batch_size", 64)),
        n_epochs=int(rl_config.get("n_epochs", 10)),
        gamma=float(rl_config.get("gamma", 0.99)),
    )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    vec_env.save(str(eval_stats_path))
    vec_env.close()
    eval_env_for_callback.close()

    best_model_path = checkpoint_root / f"{checkpoint_prefix}_best" / "best_model.zip"
    return best_model_path, eval_stats_path, checkpoint_prefix


def evaluate(
    model_path: Path,
    stats_path: Path,
    env_config: Dict[str, Any],
    seed: int,
    prefix: str,
) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found at {model_path}")

    eval_env = create_eval_env(env_config, seed + 10_000, stats_path)
    model = PPO.load(str(model_path), env=eval_env)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for episode in range(EVAL_EPISODES):
        obs = eval_env.reset()
        done = False
        state = None
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            episode_reward += float(reward[0])
            episode_length += 1
            done = bool(dones[0])
            if done and "terminal_observation" in infos[0]:
                obs = infos[0]["terminal_observation"]

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    eval_env.close()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"evaluation_{prefix}_{timestamp}.json"

    summary = {
        "seed": seed,
        "timestamp": timestamp,
        "model_path": str(model_path),
        "stats_path": str(stats_path),
        "episodes": [
            {
                "episode": idx,
                "reward": reward,
                "steps": steps,
            }
            for idx, (reward, steps) in enumerate(zip(episode_rewards, episode_lengths), start=1)
        ],
        "mean_reward": float(np.mean(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "mean_steps": float(np.mean(episode_lengths)),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logging.info("Evaluation completed. Results written to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on RedOps gym environment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/experiment_config.yaml"),
        help="Path to experiment configuration YAML.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training seed from configuration.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total training timesteps.",
    )
    parser.add_argument(
        "--allow-long-run",
        action="store_true",
        help="Allow runs beyond the default safety timestep cap.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    config = load_config(args.config)

    config_seed = config.get("rl", {}).get("seed")
    seed = args.seed if args.seed is not None else config_seed
    if seed is None:
        raise ValueError("A deterministic seed must be provided either via config or CLI.")

    seed = int(seed)

    logging.info("Starting training with seed %s", seed)

    model_path, stats_path, prefix = train(
        config=config,
        seed=seed,
        allow_long_run=args.allow_long_run,
        override_timesteps=args.total_timesteps,
    )

    logging.info("Training complete. Best model at %s", model_path)

    evaluate(
        model_path=model_path,
        stats_path=stats_path,
        env_config=config.get("env", {}),
        seed=seed,
        prefix=prefix,
    )


if __name__ == "__main__":
    main()
