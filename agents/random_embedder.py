# agents/random_embedder.py

import random
from typing import Tuple, Dict
import networkx as nx

from agents.base_embedder import BaseEmbedder


class RandomEmbedder(BaseEmbedder):
    """
    Random Node & Link Mapping Embedder.
    """

    def embed(
        self,
        substrate: nx.Graph,
        vnr: nx.Graph,
    ) -> Tuple[bool, Dict[int, int], Dict[Tuple[int, int], list]]:
        node_mapping = {}
        used_snodes = set()

        # --- ノード埋め込み ---
        for vnode in vnr.nodes:
            cpu_demand = vnr.nodes[vnode]["cpu"]

            candidates = [
                snode
                for snode in substrate.nodes
                if snode not in used_snodes
                and substrate.nodes[snode]["cpu"] >= cpu_demand
            ]

            if not candidates:
                return False, {}, {}

            chosen = random.choice(candidates)
            node_mapping[vnode] = chosen
            used_snodes.add(chosen)

        # --- リンク埋め込み ---
        link_paths = {}

        for u, v in vnr.edges:
            sn_u = node_mapping[u]
            sn_v = node_mapping[v]
            bw_demand = vnr.edges[u, v]["bandwidth"]

            # 帯域条件を満たすエッジのみのサブグラフを作る
            G_sub = nx.Graph()
            for x, y, data in substrate.edges(data=True):
                if data["bandwidth"] >= bw_demand:
                    G_sub.add_edge(x, y)

            try:
                path = nx.shortest_path(G_sub, source=sn_u, target=sn_v)
                link_paths[(u, v)] = path
            except nx.NetworkXNoPath:
                return False, {}, {}

        return True, node_mapping, link_paths
