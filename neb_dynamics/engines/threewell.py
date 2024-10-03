from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Union, List
from neb_dynamics.nodes.node import Node
from neb_dynamics.chain import Chain
from neb_dynamics.engines import Engine
from neb_dynamics.nodes.node import XYNode
from neb_dynamics.dynamics.chainbiaser import ChainBiaser


import numpy as np


@dataclass
class ThreeWellPotential(Engine):
    biaser: ChainBiaser = None

    def _en_func(self, xy: np.array) -> float:
        """
        computes energy from xy point
        """
        x, y = xy
        ene = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
        if self.biaser:
            dist = self.biaser.compute_min_dist_to_ref(
                node=XYNode(structure=xy),
                dist_func=self.biaser.compute_euclidean_distance,
            )
            ene += self.biaser.energy_gaussian_bias(distance=dist)
        return ene

    def _grad_func(self, xy: np.array) -> NDArray:
        """
        computes gradient from xy point
        """
        x, y = xy
        dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
        grad = np.array([dx, dy])
        if self.biaser:
            g_bias = self.biaser.gradient_node_bias(node=XYNode(structure=xy))
            grad += g_bias
        return grad

    def _compute_ene_grads(self, chain: Union[Chain, List[Node]]):
        if isinstance(chain, Chain):
            ene_grad_tuple = [
                (self._en_func(xy), self._grad_func(xy)) for xy in chain.coordinates
            ]
        elif isinstance(chain, List):
            ene_grad_tuple = [
                (self._en_func(xy.structure), self._grad_func(xy.structure))
                for xy in chain
            ]
        else:
            raise ValueError(f"Unsupported type {type(chain)}")

        return ene_grad_tuple

    def compute_energies(self, chain: Union[Chain, List[Node]]) -> NDArray:

        ene_grad_tuple = self._compute_ene_grads(chain)
        for node, tup in zip(chain, ene_grad_tuple):
            node._cached_energy = tup[0]
            node._cached_gradient = tup[1]

        return np.array([t[0] for t in ene_grad_tuple])

    def compute_gradients(self, chain: Union[Chain, List[Node]]) -> NDArray:
        ene_grad_tuple = self._compute_ene_grads(chain)
        for node, tup in zip(chain, ene_grad_tuple):
            node._cached_energy = tup[0]
            node._cached_gradient = tup[1]

        return np.array([t[1] for t in ene_grad_tuple])
