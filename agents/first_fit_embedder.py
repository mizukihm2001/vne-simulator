# agents/first_fit_embedder.py

from typing import Dict, Tuple, List
import networkx as nx

from agents.base_embedder import BaseEmbedder


class FirstFitEmbedder(BaseEmbedder):
    """
    First-Fit Embedder:
    Selects the first SN node/link that satisfies the resource requirements.
    """

    def embed(
        self,
        substrate: nx.Graph,
        vnr: nx.Graph,
    ) -> Tuple[bool, Dict[int, int], Dict[Tuple[int, int], List[int]]]:
        node_mapping = {}
        used_snodes = set()

        # --- ノード埋め込み ---
        for vnode in vnr.nodes:
            cpu_demand = vnr.nodes[vnode]["cpu"]

            for snode in substrate.nodes:
                if (
                    snode not in used_snodes
                    and substrate.nodes[snode]["cpu"] >= cpu_demand
                ):
                    node_mapping[vnode] = snode
                    used_snodes.add(snode)
                    break
            else:
                return False, {}, {}

        # --- リンク埋め込み ---
        link_paths = {}

        for u, v in vnr.edges:
            sn_u = node_mapping[u]
            sn_v = node_mapping[v]
            bw_demand = vnr.edges[u, v]["bandwidth"]

            # 帯域を満たすエッジのみで経路探索
            G_sub = nx.Graph()
            for x, y, data in substrate.edges(data=True):
                if data["bandwidth"] >= bw_demand:
                    G_sub.add_edge(x, y)

            try:
                if sn_u not in G_sub or sn_v not in G_sub:
                    return False, {}, {}

                path = nx.shortest_path(G_sub, source=sn_u, target=sn_v)
                link_paths[(u, v)] = path
            except nx.NetworkXNoPath:
                return False, {}, {}

        return True, node_mapping, link_paths
