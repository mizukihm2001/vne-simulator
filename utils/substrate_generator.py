# utils/substrate_generator.py

import networkx as nx
import numpy as np
from typing import Tuple


def generate_substrate_network(
    num_nodes: int = 10,
    edge_prob: float = 0.3,
    cpu_range: Tuple[int, int] = (50, 100),
    bandwidth_range: Tuple[int, int] = (50, 100),
) -> nx.Graph:
    """
    Generate a substrate network graph with CPU and bandwidth attributes.

    Args:
        num_nodes: Number of substrate nodes
        edge_prob: Probability of edge creation (for random graph)
        cpu_range: Range of CPU resources per node
        bandwidth_range: Range of bandwidth per link

    Returns:
        A NetworkX graph with node and edge attributes
    """
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, seed=42)

    for node in G.nodes:
        G.nodes[node]["cpu"] = np.random.randint(*cpu_range)

    for u, v in G.edges:
        G.edges[u, v]["bandwidth"] = np.random.randint(*bandwidth_range)

    return G
