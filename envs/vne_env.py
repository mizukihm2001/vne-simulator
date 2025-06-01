# envs/vne_env.py

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces

from utils.substrate_generator import generate_substrate_network


class VNEEnv(gym.Env):
    """
    Virtual Network Embedding Environment (Gymnasium-compatible).
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        self.config = config or {}

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self.state = None
        self.substrate = None  # ← SNを保持

    def reset(
        self,
        seed: int = None,
        options: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Substrateネットワークを生成
        self.substrate = generate_substrate_network()

        self.state = np.random.rand(10).astype(np.float32)
        info = {"reset_info": "Substrate generated"}
        return self.state, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = float(np.random.rand())
        terminated = False
        truncated = False

        self.state = np.random.rand(10).astype(np.float32)
        info = {
            "step_info": f"Dummy step with action {action}",
        }

        return self.state, reward, terminated, truncated, info

    def render(self) -> None:
        print("Substrate network overview:")
        print(f"- Nodes: {self.substrate.number_of_nodes()}")
        print(f"- Edges: {self.substrate.number_of_edges()}")

        print("Node CPU capacities:")
        for node, data in self.substrate.nodes(data=True):
            print(f"  Node {node}: CPU={data['cpu']}")

    def close(self) -> None:
        pass
