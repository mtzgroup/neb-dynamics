from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List

from neb_dynamics.nodes.node import NodeType


@dataclass
class Optimizer(ABC):
    """Base class for Optimizers"""

    @abstractmethod
    def take_step(
        self, coords: list[list[float]], mod_grads: list[list[float]]
    ) -> List[NodeType]:
        """Take a step in the optimization"""
        raise NotImplementedError()
