# run_experiment.py

import argparse
import os
from datetime import datetime

import pandas as pd

from envs.poisson_vne_env import PoissonVNEEnv
from utils.config_loader import load_config


def run(config_path: str) -> None:
    config = load_config(config_path)
    env = PoissonVNEEnv(config)

    obs, info = env.reset()
    print("PoissonVNEEnv initialized.")
    print(info["reset_info"])

    total_reward = 0.0
    success_count = 0
    steps = config["experiment"].get("max_steps", 30)
    log_records = []

    for step in range(1, steps + 1):
        obs, reward, done, truncated, info = env.step(action=0)
        total_reward += reward

        print(f"\n[Step {step}]")
        print(f"  Reward: {reward:.1f}")

        success = info.get("success")
        if success is True:
            success_count += 1
            print("  ✅ Success")
            print(f"  Node mapping: {info['node_mapping']}")
            print(f"  Link paths: {info['link_paths']}")
        elif success is False:
            print("  ❌ Failed to embed VNR.")
        else:
            print("  No VNR arrived this step.")

        env.render()

        log_records.append(
            {
                "step": step,
                "reward": reward,
                "success": success,
                "node_mapping": str(info.get("node_mapping", "")),
                "link_paths": str(info.get("link_paths", "")),
                "expires_at": info.get("expires_at", ""),
            }
        )

    # 保存先ディレクトリとファイル名
    os.makedirs("logs/poisson", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/poisson/run_{timestamp}.csv"
    pd.DataFrame(log_records).to_csv(log_path, index=False)

    print("\n--- Experiment Summary ---")
    print(f"Total reward: {total_reward}")
    print(f"Accepted VNRs: {success_count}/{steps}")
    print(f"Acceptance rate: {success_count / steps:.2f}")
    print(f"📁 Log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VNE experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    run(args.config)
