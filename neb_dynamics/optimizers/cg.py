from __future__ import annotations

from dataclasses import dataclass
from neb_dynamics.optimizers.optimizer import Optimizer
from neb_dynamics.errors import ElectronicStructureError


import numpy as np


@dataclass
class ConjugateGradient(Optimizer):
    timestep: float = 0.5

    def __post_init__(self):
        self.g_old = None

    def optimize_step(self, chain, chain_gradients):
        from neb_dynamics.chain import Chain
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

        # Perform line search to find optimal step size (alpha)
        alpha = self.timestep
        gradient_failed = True
        nretries = 0
        for i, (node, grad) in enumerate(zip(chain.nodes, chain_gradients)):
            if node.converged:
                chain_gradients[i] = np.zeros_like(grad)

        while gradient_failed and nretries < 10:
            try:
                g_new = chain_gradients.flatten()
                p = -chain_gradients.flatten()

                if self.g_old is not None:
                    g = self.g_old.flatten().copy()
                    # Fletcher-Reeves formula
                    beta = np.dot(g_new, g_new) / np.dot(g, g)
                    # beta = np.dot(g_new, g_new - g) / np.dot(g, g)  # Polak-Ribiere formula
                    p = -g_new + beta * p

                    g = g_new
                else:
                    self.g_old = g_new.reshape(chain_gradients.shape).copy()

                p = p.reshape(chain_gradients.shape)
                new_chain_coordinates = chain.coordinates + alpha * p
                new_nodes = []
                for node, new_coords in zip(chain.nodes, new_chain_coordinates):

                    new_nodes.append(node.update_coords(new_coords))

                new_chain = Chain(new_nodes, parameters=chain.parameters)
                gradient_failed = False

            except Exception as e:
                nretries += 1
                alpha *= .5

        return new_chain