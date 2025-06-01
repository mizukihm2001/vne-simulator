# utils/vnr_generator.py

import networkx as nx
import numpy as np
from typing import List, Tuple


def generate_virtual_network_request(
    num_nodes: int = 3,
    edge_prob: float = 0.5,
    cpu_range: Tuple[int, int] = (10, 30),
    bandwidth_range: Tuple[int, int] = (10, 30),
) -> nx.Graph:
    """
    Generate a virtual network request with resource demands.

    Args:
        num_nodes: Number of nodes in the VNR
        edge_prob: Probability of edge creation
        cpu_range: Range of CPU demand per virtual node
        bandwidth_range: Range of bandwidth per virtual link

    Returns:
        A NetworkX graph representing a VNR
    """
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob)

    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob)

    for node in G.nodes:
        G.nodes[node]["cpu"] = np.random.randint(*cpu_range)

    for u, v in G.edges:
        G.edges[u, v]["bandwidth"] = np.random.randint(*bandwidth_range)

    return G
