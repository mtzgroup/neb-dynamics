from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Union, List
from neb_dynamics import Node, Chain
from neb_dynamics.engines.engine import Engine
from neb_dynamics.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
import numpy as np


@dataclass
class LEPS(Engine):

    def coulomb(self, r, d, r0, alpha):
        return (d / 2) * (
            (3 / 2) * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0))
        )

    def _en_func(self, node):
        a = 0.05
        b = 0.30
        c = 0.05
        d_ab = 4.746
        d_bc = 4.746
        d_ac = 3.445
        r0 = 0.742
        alpha = 1.942
        inp = node
        r_ab, r_bc = inp

        Q_AB = self.coulomb(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
        Q_BC = self.coulomb(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
        Q_AC = self.coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        J_AB = self.exchange(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
        J_BC = self.exchange(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
        J_AC = self.exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        result_Qs = (Q_AB / (1 + a)) + (Q_BC / (1 + b)) + (Q_AC / (1 + c))
        result_Js_1 = (
            ((J_AB**2) / ((1 + a) ** 2))
            + ((J_BC**2) / ((1 + b) ** 2))
            + ((J_AC**2) / ((1 + c) ** 2))
        )
        result_Js_2 = (
            ((J_AB * J_BC) / ((1 + a) * (1 + b)))
            + ((J_AC * J_BC) / ((1 + c) * (1 + b)))
            + ((J_AB * J_AC) / ((1 + a) * (1 + c)))
        )
        result_Js = result_Js_1 - result_Js_2

        result = result_Qs - (result_Js) ** (1 / 2)
        return result

    def grad_x(self, node):
        a = 0.05
        b = 0.30
        c = 0.05
        d_ab = 4.746
        # d_bc = 4.746
        d_ac = 3.445
        r0 = 0.742
        alpha = 1.942

        inp = node
        r_ab, r_bc = inp

        ealpha_x = np.exp(alpha * (r0 - r_ab))
        neg_ealpha_x = np.exp(alpha * (r_ab - r0))
        ealpha_y = np.exp(alpha * (r0 - r_bc))
        neg_ealpha_y = np.exp(alpha * (r_bc - r0))

        e2alpha_x = np.exp(2 * alpha * (r0 - r_ab))
        e2alpha_y = np.exp(2 * alpha * (r0 - r_bc))

        aDenom = 1 / (1 + a)
        bDenom = 1 / (1 + b)

        # Qconst = Node2D_LEPS.coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
        Jconst = self.exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        cDenom = 1 / (1 + c)

        d = d_ab

        dx = (
            0.25
            * aDenom**2
            * alpha
            * d
            * ealpha_x
            * (
                -2 * (1 + a) * (-1 + 3 * ealpha_x)
                + (
                    (-3 + ealpha_x)
                    * (
                        2 * d * ealpha_x * (-6 + ealpha_x)
                        - (1 + a) * d * ealpha_y * (-6 + ealpha_y) * bDenom
                        - 4 * (1 + a) * Jconst * cDenom
                    )
                )
                / (
                    np.sqrt(
                        (
                            ((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2)
                            + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2
                            - d**2
                            * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc))
                            * (-1 + 6 * neg_ealpha_x)
                            * (-1 + 6 * neg_ealpha_y)
                            * aDenom
                            * bDenom
                        )
                        - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom
                        - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom)
                        + 16 * Jconst**2 * cDenom**2
                    )
                )
            )
        )

        return dx

    def exchange(self, r, d, r0, alpha):
        return (d / 4) * (np.exp(-2 * alpha * (r - r0)) - 6 * np.exp(-alpha * (r - r0)))

    def grad_y(self, node):
        a = 0.05
        b = 0.30
        c = 0.05
        # d_ab = 4.746
        d_bc = 4.746
        d_ac = 3.445
        r0 = 0.742
        alpha = 1.942
        r_ab, r_bc = node

        ealpha_x = np.exp(alpha * (r0 - r_ab))
        neg_ealpha_x = np.exp(alpha * (r_ab - r0))
        ealpha_y = np.exp(alpha * (r0 - r_bc))
        neg_ealpha_y = np.exp(alpha * (r_bc - r0))

        e2alpha_x = np.exp(2 * alpha * (r0 - r_ab))
        e2alpha_y = np.exp(2 * alpha * (r0 - r_bc))

        aDenom = 1 / (1 + a)
        bDenom = 1 / (1 + b)

        # Qconst = Node2D_LEPS.coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
        Jconst = self.exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        cDenom = 1 / (1 + c)

        d = d_bc

        dy = (
            0.25
            * bDenom**2
            * alpha
            * d
            * ealpha_y
            * (
                -2 * (1 + b) * (-1 + 3 * ealpha_y)
                + (
                    (-3 + ealpha_y)
                    * (
                        2 * d * ealpha_y * (-6 + ealpha_y)
                        - (1 + b) * d * ealpha_x * (-6 + ealpha_x) * aDenom
                        - 4 * (1 + b) * Jconst * cDenom
                    )
                )
                / (
                    np.sqrt(
                        (
                            ((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2)
                            + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2
                            - d**2
                            * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc))
                            * (-1 + 6 * neg_ealpha_x)
                            * (-1 + 6 * neg_ealpha_y)
                            * aDenom
                            * bDenom
                        )
                        - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom
                        - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom)
                        + 16 * Jconst**2 * cDenom**2
                    )
                )
            )
        )

        return dy

    def dQ_dr(d, alpha, r, r0):
        return (d / 2) * (
            (3 / 2) * (-2 * alpha * np.exp(-2 * alpha * (r - r0)))
            + alpha * np.exp(-alpha * (r - r0))
        )

    def dJ_dr(d, alpha, r, r0):
        return (d / 4) * (
            np.exp(-2 * alpha * (r - r0)) * (-2 * alpha)
            + 6 * alpha * np.exp(-alpha * (r - r0))
        )

    def _grad_func(self, xy: np.array) -> NDArray:
        """
        computes gradient from xy point
        """

        return np.array([self.grad_x(xy), self.grad_y(xy)])

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
