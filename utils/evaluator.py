# utils/evaluator.py

import networkx as nx
from typing import Dict, Tuple, List


def apply_embedding(
    substrate: nx.Graph,
    vnr: nx.Graph,
    node_mapping: Dict[int, int],
    link_paths: Dict[Tuple[int, int], List[int]],
) -> None:
    """
    Reduce substrate resources based on a successful embedding.

    Args:
        substrate: The substrate network
        vnr: The virtual network request
        node_mapping: VNR node → SN node
        link_paths: VNR edge → SN path
    """
    # ノード資源を減算
    for vnode, snode in node_mapping.items():
        cpu_demand = vnr.nodes[vnode]["cpu"]
        substrate.nodes[snode]["cpu"] -= cpu_demand

    # リンク資源を減算
    for (u, v), path in link_paths.items():
        bw_demand = vnr.edges[u, v]["bandwidth"]
        for i in range(len(path) - 1):
            u_, v_ = path[i], path[i + 1]
            if substrate.has_edge(u_, v_):
                substrate.edges[u_, v_]["bandwidth"] -= bw_demand
