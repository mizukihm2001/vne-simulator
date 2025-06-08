# agents/base_embedder.py

from abc import ABC, abstractmethod
import networkx as nx
from typing import Tuple


class BaseEmbedder(ABC):
    """
    Abstract base class for virtual network embedding algorithms.
    """

    @abstractmethod
    def embed(
        self,
        substrate: nx.Graph,
        vnr: nx.Graph,
    ) -> Tuple[bool, dict]:
        """
        Attempt to embed the VNR onto the substrate network.

        Args:
            substrate: The substrate network (with node/edge attributes)
            vnr: The virtual network request

        Returns:
            success (bool): Whether embedding succeeded
            mapping (dict): Node mapping (vnode -> snode), empty if failed
        """
        pass
