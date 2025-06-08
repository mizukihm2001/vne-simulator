# envs/multi_vne_env.py

from typing import Any, Dict, List
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.substrate_generator import generate_substrate_network
from utils.vnr_generator import generate_virtual_network_request
from utils.evaluator import apply_embedding
from agents.random_embedder import RandomEmbedder


class MultiVNEEnv(gym.Env):
    """
    VNE environment that embeds multiple VNRs over time.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.substrate = None
        self.vnr_queue: List[Any] = []
        self.current_vnr = None
        self.current_step = 0
        self.embedder = RandomEmbedder()

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

    def reset(self, seed: int = None, options: Dict[str, Any] = None):
        super().reset(seed=seed)

        sn_config = self.config["substrate"]
        vnr_config = self.config["vnr"]
        episodes = self.config["experiment"]["episodes"]

        self.substrate = generate_substrate_network(sn_config)
        self.vnr_queue = [
            generate_virtual_network_request(vnr_config)
            for _ in range(episodes)
        ]
        self.current_vnr = self.vnr_queue.pop(0)
        self.current_step = 0

        self.state = np.random.rand(10).astype(np.float32)
        info = {"reset_info": "MultiVNEEnv initialized"}
        return self.state, info

    def step(self, action: int):
        success, node_map, link_paths = self.embedder.embed(
            self.substrate, self.current_vnr
        )

        if success:
            apply_embedding(
                self.substrate, self.current_vnr, node_map, link_paths
            )
            reward = 1.0
        else:
            reward = -1.0

        done = len(self.vnr_queue) == 0
        truncated = False

        # VNRを次に進める
        if not done:
            self.current_vnr = self.vnr_queue.pop(0)

        self.current_step += 1
        self.state = np.random.rand(10).astype(np.float32)

        info = {
            "step": self.current_step,
            "success": success,
            "node_mapping": node_map,
            "link_paths": link_paths,
        }
        return self.state, reward, done, truncated, info

    def render(self) -> None:
        print(
            f"Step: {self.current_step}, Remaining VNRs: {len(self.vnr_queue)}"
        )

    def close(self) -> None:
        pass
