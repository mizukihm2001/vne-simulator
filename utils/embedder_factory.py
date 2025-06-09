# utils/embedder_factory.py

from agents.random_embedder import RandomEmbedder
# FirstFitEmbedderやGreedyEmbedderはあとで実装予定


def get_embedder(name: str):
    return RandomEmbedder()
