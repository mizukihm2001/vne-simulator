# envs/vne_env.py

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class VNEEnv(gym.Env):
    """
    Virtual Network Embedding Environment (Gymnasium-compatible).

    Dummy environment for structural testing.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        self.config = config or {}

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self.state = None

    def reset(
        self,
        seed: int = None,
        options: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.state = np.random.rand(10).astype(np.float32)
        info = {
            "reset_info": "Dummy reset",
        }
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
        print(f"Current state: {self.state}")

    def close(self) -> None:
        pass
