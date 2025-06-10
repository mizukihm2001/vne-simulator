# run_experiment.py

import argparse
import os
import copy
from datetime import datetime
import pandas as pd

from envs.poisson_vne_env import PoissonVNEEnv
from utils.config_loader import load_config
from utils.seed import set_seed
from utils.embedder_factory import get_embedder


def run_single_experiment(config: dict, run_id: int = 0) -> dict:
    seed = config["experiment"].get("seed", 42) + run_id
    set_seed(seed)
    config["experiment"]["seed"] = seed

    embedder_name = config["experiment"].get("embedder", "first_fit")
    embedder = get_embedder(embedder_name)
    config["embedder"] = embedder  # PoissonVNEEnv に渡す

    env = PoissonVNEEnv(copy.deepcopy(config))
    obs, info = env.reset()

    total_reward = 0.0
    success_count = 0
    steps = config["experiment"].get("max_steps", 30)
    log_records = []

    for step in range(1, steps + 1):
        obs, reward, done, truncated, info = env.step(action=0)
        total_reward += reward
        if info.get("success"):
            success_count += 1

        log_records.append(
            {
                "step": step,
                "reward": reward,
                "success": info.get("success"),
                "node_mapping": str(info.get("node_mapping", "")),
                "link_paths": str(info.get("link_paths", "")),
                "expires_at": info.get("expires_at", ""),
            }
        )

    os.makedirs("logs/poisson", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/poisson/run_{run_id}_{timestamp}.csv"
    pd.DataFrame(log_records).to_csv(log_path, index=False)

    return {
        "run_id": run_id,
        "seed": seed,
        "total_reward": total_reward,
        "success_count": success_count,
        "acceptance_rate": success_count / steps,
    }


def run_batch(config_path: str, repeat: int = 1) -> None:
    config = load_config(config_path)
    results = []

    for run_id in range(repeat):
        print(f"\n--- Running experiment {run_id + 1}/{repeat} ---")
        result = run_single_experiment(copy.deepcopy(config), run_id=run_id)
        results.append(result)

    df = pd.DataFrame(results)
    embedder = config["experiment"].get("embedder", "unknown")
    os.makedirs("results", exist_ok=True)
    result_path = f"results/summary_{embedder}.csv"
    df.to_csv(result_path, index=False)

    print("\n✅ Batch experiment finished.")
    print(df.describe())
    print(f"\n📁 Results saved to: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VNE experiment(s)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of repeated experiments"
    )
    args = parser.parse_args()
    run_batch(args.config, repeat=args.repeat)
