from utils.config_loader import load_config
from envs.single_vne_env import SingleVNEEnv

config = load_config("configs/default.yaml")
env = SingleVNEEnv(config=config)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)

print("Config embedder:", config["embedder"])
print("Embedding success:", info["success"])
