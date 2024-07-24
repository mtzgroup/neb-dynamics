from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Union, List
from neb_dynamics.nodes.node import Node
from neb_dynamics.chain import Chain
from neb_dynamics.engines import Engine
import numpy as np


@dataclass
class ThreeWellPotential(Engine):

    def _en_func(self, xy: np.array) -> float:
        """
        computes energy from xy point
        """
        x, y = xy
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def _grad_func(self, xy: np.array) -> NDArray:
        """
        computes gradient from xy point
        """
        x, y = xy
        dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
        return np.array([dx, dy])

    def compute_energies(self, chain: Union[Chain, List[Node]]) -> NDArray:
        if isinstance(chain, Chain):
            return np.array([self._en_func(xy) for xy in chain.coordinates])
        elif isinstance(chain, List):
            return np.array([self._en_func(xy.structure) for xy in chain])
        else:
            raise ValueError(f"Unsupported type {type(chain)}")

    def compute_gradients(self, chain: Union[Chain, List[Node]]) -> NDArray:
        if isinstance(chain, Chain):
            return np.array([self._grad_func(xy) for xy in chain.coordinates])
        elif isinstance(chain, List):
            return np.array([self._grad_func(xy.structure) for xy in chain])
        else:
            raise ValueError(f"Unsupported type {type(chain)}")
