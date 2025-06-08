# utils/vnr_generator.py

import networkx as nx
import numpy as np


def generate_virtual_network_request(config: dict) -> nx.Graph:
    num_nodes = config["num_nodes"]
    edge_prob = config["edge_prob"]
    cpu_range = tuple(config["cpu_range"])
    bandwidth_range = tuple(config["bandwidth_range"])

    while True:
        G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob)
        if nx.is_connected(G):
            break

    for node in G.nodes:
        G.nodes[node]["cpu"] = np.random.randint(*cpu_range)

    for u, v in G.edges:
        G.edges[u, v]["bandwidth"] = np.random.randint(*bandwidth_range)

    return G
