# envs/poisson_vne_env.py

import heapq
import itertools
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from utils.evaluator import apply_embedding
from utils.substrate_generator import generate_substrate_network
from utils.vnr_generator import generate_virtual_network_request
from utils.embedder_factory import get_embedder


class PoissonVNEEnv(gym.Env):
    """
    VNE environment with Poisson arrival and duration-based departure.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        embedder_name = config["experiment"].get("embedder", "random")
        self.embedder = get_embedder(embedder_name)

        self.substrate: nx.Graph = None
        self.active_vnrs = []
        self.event_queue = []
        self.current_time = 0
        self.vnr_id_counter = itertools.count()

        self.arrival_rate = config["experiment"].get("arrival_rate", 1.0)
        self.duration_range = config["vnr"].get("duration_range", [5, 10])

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

    def reset(self, seed: int = None, options: Dict[str, Any] = None):
        super().reset(seed=seed)

        self.substrate = generate_substrate_network(self.config["substrate"])
        self.active_vnrs.clear()
        self.event_queue.clear()
        self.current_time = 0

        arrival = np.random.exponential(1.0 / self.arrival_rate)
        heapq.heappush(self.event_queue, (arrival, "arrival"))

        self.state = np.random.rand(10).astype(np.float32)
        return self.state, {"reset_info": "PoissonVNEEnv initialized"}

    def step(
            self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_time += 1
        reward = 0.0
        info = {}

        self._expire_vnrs()

        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            _, event_type = heapq.heappop(self.event_queue)
            if event_type == "arrival":
                vnr = generate_virtual_network_request(self.config["vnr"])
                vnr_id = next(self.vnr_id_counter)
                duration = np.random.randint(*self.duration_range)
                expire_at = self.current_time + duration

                success, node_map, link_paths = self.embedder.embed(
                    self.substrate, vnr
                )

                if success:
                    apply_embedding(
                        self.substrate, vnr, node_map, link_paths
                    )
                    self.active_vnrs.append(
                        (vnr_id, vnr, node_map, link_paths, expire_at)
                    )
                    reward = 1.0
                    info.update(
                        {
                            "success": True,
                            "vnr_id": vnr_id,
                            "node_mapping": node_map,
                            "link_paths": link_paths,
                            "expires_at": expire_at,
                        }
                    )
                else:
                    reward = -1.0
                    info.update({"success": False})

                next_arrival = (
                    self.current_time
                    + np.random.exponential(1.0 / self.arrival_rate)
                )
                heapq.heappush(self.event_queue, (next_arrival, "arrival"))
                break

        self.state = np.random.rand(10).astype(np.float32)
        done = False
        truncated = False
        return self.state, reward, done, truncated, info

    def _expire_vnrs(self) -> None:
        remaining = []
        for vnr_id, vnr, node_map, link_paths, expire_at in self.active_vnrs:
            if expire_at <= self.current_time:
                self._release_resources(vnr, node_map, link_paths)
            else:
                remaining.append(
                    (vnr_id, vnr, node_map, link_paths, expire_at)
                )
        self.active_vnrs = remaining

    def _release_resources(
        self,
        vnr: nx.Graph,
        node_map: Dict[int, int],
        link_paths: Dict[Tuple[int, int], List[int]],
    ) -> None:
        for vnode, snode in node_map.items():
            cpu = vnr.nodes[vnode]["cpu"]
            self.substrate.nodes[snode]["cpu"] += cpu

        for (u, v), path in link_paths.items():
            bw = vnr.edges[u, v]["bandwidth"]
            for i in range(len(path) - 1):
                u_, v_ = path[i], path[i + 1]
                if self.substrate.has_edge(u_, v_):
                    self.substrate.edges[u_, v_]["bandwidth"] += bw

    def render(self) -> None:
        print(
            f"Time: {self.current_time}, Active VNRs: {len(self.active_vnrs)}"
        )

    def close(self) -> None:
        pass
