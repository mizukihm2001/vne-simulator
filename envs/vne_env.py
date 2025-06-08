# envs/vne_env.py

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.substrate_generator import generate_substrate_network
from utils.vnr_generator import generate_virtual_network_request
# 追加インポート
from agents.random_embedder import RandomEmbedder


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
        self.vnr = None  # ← VNRを保持する属性を追加
        self.embedder = RandomEmbedder()  # ← 埋め込み器を環境に組み込む

    def reset(
        self,
        seed: int = None,
        options: Dict[str, Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Substrateネットワークを生成
        self.substrate = generate_substrate_network()
        self.vnr = generate_virtual_network_request()  # ← ここでVNR生成

        self.state = np.random.rand(10).astype(np.float32)

        info = {"reset_info": "Substrate and VNR generated"}
        return self.state, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 埋め込み実行
        success, node_mapping, link_paths = self.embedder.embed(
            self.substrate, self.vnr
        )

        # 簡易な報酬設計：成功 +1.0 / 失敗 -1.0
        reward = 1.0 if success else -1.0
        terminated = False
        truncated = False

        # 状態更新（とりあえずランダム継続）
        self.state = np.random.rand(10).astype(np.float32)

        info = {
            "success": success,
            "node_mapping": node_mapping,
            "link_paths": link_paths,
        }

        return self.state, reward, terminated, truncated, info

    def render(self) -> None:
        print("Substrate network overview:")
        print(f"- Nodes: {self.substrate.number_of_nodes()}")
        print(f"- Edges: {self.substrate.number_of_edges()}")

        print("Node CPU capacities:")
        for node, data in self.substrate.nodes(data=True):
            print(f"  Node {node}: CPU={data['cpu']}")

        print("\nVNR overview:")
        print(f"- Nodes: {self.vnr.number_of_nodes()}")
        print(f"- Edges: {self.vnr.number_of_edges()}")
        print("Node CPU demands:")
        for node, data in self.vnr.nodes(data=True):
            print(f"  Node {node}: CPU={data['cpu']}")

    def close(self) -> None:
        pass
