from __future__ import annotations


from dataclasses import dataclass
from neb_dynamics.optimizer import Optimizer
from neb_dynamics.optimizers import ALS
from chain import Chain
import numpy as np


@dataclass
class Linesearch(Optimizer):
    als_max_steps: int = 3
    step_size: float = 1.0
    min_step_size: float = 0.33
    alpha: float = 0.01
    beta: float = None

    activation_tol: float = 0.1

    def __post_init__(self):
        if self.beta is None:
            beta = (self.min_step_size / self.step_size) ** (1 / self.als_max_steps)
            self.beta = beta

    def optimize_step(self, chain, chain_gradients):

        if np.linalg.norm(chain_gradients) < self.activation_tol:

            disp = ALS.ArmijoLineSearch(
                chain=chain,
                t=self.step_size,
                alpha=self.alpha,
                beta=self.beta,
                grad=chain_gradients,
                max_steps=self.als_max_steps,
            )
        else:

            disp = self.min_step_size

        scaling = 1
        if np.linalg.norm(chain_gradients * disp) > self.step_size * len(
            chain
        ):  # if step size is too large
            scaling = (
                (1 / (np.linalg.norm(chain_gradients * disp)))
                * self.step_size
                * len(chain)
            )

        new_chain_coordinates = chain.coordinates - (chain_gradients * disp * scaling)
        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = Chain(new_nodes, parameters=chain.parameters)
        assert all([node.energy is not None for node in new_chain.nodes])

        return new_chain
