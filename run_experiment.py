# run_experiment.py

import argparse

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

    for step in range(1, steps + 1):
        obs, reward, done, truncated, info = env.step(action=0)
        total_reward += reward

        print(f"\n[Step {step}]")
        print(f"  Reward: {reward:.1f}")

        if info.get("success") is True:
            success_count += 1
            print("  ✅ Success")
            print(f"  Node mapping: {info['node_mapping']}")
            print(f"  Link paths: {info['link_paths']}")
        elif info.get("success") is False:
            print("  ❌ Failed to embed VNR.")
        else:
            print("  No VNR arrived this step.")

        env.render()

    print("\n--- Experiment Summary ---")
    print(f"Total reward: {total_reward}")
    print(f"Accepted VNRs: {success_count}/{steps}")
    print(f"Acceptance rate: {success_count / steps:.2f}")


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
