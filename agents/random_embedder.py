# agents/random_embedder.py

import random
from typing import Tuple
import networkx as nx

from agents.base_embedder import BaseEmbedder


class RandomEmbedder(BaseEmbedder):
    """
    Random Node Mapping Embedder.

    For each VNR node, assign a random SN node that has sufficient CPU
    and is not already assigned.
    """

    def embed(
        self,
        substrate: nx.Graph,
        vnr: nx.Graph,
    ) -> Tuple[bool, dict]:
        mapping = {}
        used_snodes = set()

        for vnode in vnr.nodes:
            cpu_demand = vnr.nodes[vnode]["cpu"]

            # 候補SNノードを条件で絞り込み（未使用 + CPUが十分）
            candidates = [
                snode
                for snode in substrate.nodes
                if snode not in used_snodes
                and substrate.nodes[snode]["cpu"] >= cpu_demand
            ]

            if not candidates:
                return False, {}  # 埋め込み失敗

            chosen = random.choice(candidates)
            mapping[vnode] = chosen
            used_snodes.add(chosen)

        return True, mapping
