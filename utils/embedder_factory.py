# utils/embedder_factory.py

from agents.random_embedder import RandomEmbedder
from agents.first_fit_embedder import FirstFitEmbedder
from agents.greedy_embedder import GreedyEmbedder


def get_embedder(name: str):
    name = name.lower()
    if name == "random":
        return RandomEmbedder()
    elif name == "first_fit":
        return FirstFitEmbedder()
    elif name == "greedy":
        return GreedyEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {name}")
