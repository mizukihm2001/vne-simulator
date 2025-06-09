from envs.multi_vne_env import MultiVNEEnv
from utils.config_loader import load_config


def main():
    config = load_config("configs/default.yaml")
    env = MultiVNEEnv(config)

    obs, info = env.reset()
    print("Environment initialized")
    print(info["reset_info"])

    done = False
    total_reward = 0.0

    while not done:
        obs, reward, done, truncated, info = env.step(action=0)
        total_reward += reward

        print(f"\n[Step {info['step']}]")
        print(f"  Success: {info['success']}")
        print(f"  Node mapping: {info['node_mapping']}")
        print(f"  Link paths: {info['link_paths']}")
        print(f"  Cumulative reward: {total_reward}")

        env.render()

    print("\nEpisode finished.")
    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    main()
