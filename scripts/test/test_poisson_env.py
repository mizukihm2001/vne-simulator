from envs.poisson_vne_env import PoissonVNEEnv
from utils.config_loader import load_config


def main():
    config = load_config("configs/default.yaml")
    env = PoissonVNEEnv(config)

    obs, info = env.reset()
    print("Environment initialized")
    print(info["reset_info"])

    total_reward = 0.0
    max_steps = 30
    success_count = 0

    for step in range(1, max_steps + 1):
        obs, reward, done, truncated, info = env.step(action=0)
        total_reward += reward

        print(f"\n[Step {step}]")
        print(f"  Reward: {reward}")
        if "success" in info:
            print(f"  Success: {info['success']}")
            if info["success"]:
                success_count += 1
                print(f"  Node mapping: {info['node_mapping']}")
                print(f"  Link paths: {info['link_paths']}")
                print(f"  Expires at: {info['expires_at']}")
        else:
            print("  No VNR arrived this step.")

        env.render()

    print("\nEpisode finished.")
    print(f"Total reward: {total_reward}")
    print(f"Success count: {success_count}/{max_steps}")


if __name__ == "__main__":
    main()
