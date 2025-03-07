from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Union, List
from neb_dynamics import Node, Chain
from neb_dynamics.engines.engine import Engine
import numpy as np


@dataclass
class FlowerPotential(Engine):

    def _en_func(self, xy: np.array) -> float:
        """
        computes energy from xy point
        """
        x, y = xy

        return (
            (1.0 / 20.0)
            * ((1 * (x**2 + y**2) - 6 * np.sqrt(x**2 + y**2)) ** 2 + 30)
            * -1
            * np.abs(0.4 * np.cos(6 * np.arctan(x / y)) + 1)
        )

    def _grad_func(self, xy: np.array) -> NDArray:
        """
        computes gradient from xy point
        """
        x, y = xy
        x2y2 = x**2 + y**2

        cos_term = 0.4 * np.cos(6 * np.arctan(x / y)) + 1
        # d/dx
        Ax = 0.12 * ((-6 * np.sqrt(x2y2) + x2y2) ** 2 + 30)

        Bx = np.sin(6 * np.arctan(x / y)) * (cos_term)

        Cx = y * (x**2 / y**2 + 1) * np.abs(cos_term)

        Dx = (
            (1 / 10)
            * (2 * x - (6 * x / np.sqrt(x2y2)))
            * (-6 * np.sqrt(x2y2) + x2y2)
            * np.abs(cos_term)
        )

        dx = (Ax * Bx) / Cx - Dx

        # d/dy
        Ay = (
            (-1 / 10)
            * (2 * y - 6 * y / (np.sqrt(x2y2)))
            * (-6 * np.sqrt(x2y2) + x2y2)
            * (np.abs(cos_term))
        )

        By = (
            0.12
            * x
            * ((-6 * np.sqrt(x2y2) + x2y2) ** 2 + 30)
            * np.sin(6 * np.arctan(x / y))
        )

        Cy = cos_term

        Dy = y**2 * (x**2 / y**2 + 1) * np.abs(cos_term)

        dy = Ay - (By * Cy) / Dy

        return np.array([dx, dy])

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
