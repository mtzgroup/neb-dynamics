from __future__ import annotations

from dataclasses import dataclass
from neb_dynamics.optimizers.optimizer import Optimizer


import numpy as np


@dataclass
class ConjugateGradient(Optimizer):
    timestep: float = 0.5

    def __post_init__(self):
        self.g_old = None

    def optimize_step(self, chain, chain_gradients):
        """
        Performs the Conjugate Gradient method to minimize a function.

        Args:
            f_grad: Function that returns the gradient of the objective function.
            x0: Initial guess for the solution.
            tol: Tolerance for convergence.
            max_iter: Maximum number of iterations.

        Returns:
            x: Approximate minimizer of the objective function.
        """

        alpha = self.timestep
        for i, (node, grad) in enumerate(zip(chain.nodes, chain_gradients)):
            if node.converged:
                chain_gradients[i] = np.zeros_like(grad)

        g_new = chain_gradients.flatten()
        p = -chain_gradients.flatten()

        if self.g_old is not None:
            g = self.g_old.flatten().copy()
            # Fletcher-Reeves formula
            # beta = np.dot(g_new, g_new) / np.dot(g, g)
            beta = max(0, np.dot(g_new, g - g_new) /
                       np.dot(g, g))  # Polak-Ribiere formula
            # print("BETA---->", beta)
            p = -g_new + beta * p

            g = g_new.copy()
            self.g_old = g
        else:
            self.g_old = g_new.reshape(chain_gradients.shape).copy()

        p = p.reshape(chain_gradients.shape)
        new_chain_coordinates = chain.coordinates + alpha * p
        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = chain.model_copy(
            update={"nodes": new_nodes, "parameters": chain.parameters})

        return new_chain
