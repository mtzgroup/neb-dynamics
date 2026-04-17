from __future__ import annotations

from dataclasses import dataclass
from neb_dynamics.optimizers.optimizer import Optimizer


import numpy as np


@dataclass
class ConjugateGradient(Optimizer):
    timestep: float = 0.5
    max_step_norm: float = 1.0
    step_up: float = 1.2
    step_down: float = 0.5
    min_timestep: float | None = None
    max_timestep: float | None = None

    def __post_init__(self):
        self.g_old = None
        self.p_old = None
        self.orig_timestep = self.timestep
        if self.min_timestep is None:
            self.min_timestep = max(1e-6, 1e-3 * self.orig_timestep)
        if self.max_timestep is None:
            self.max_timestep = max(self.orig_timestep, 4.0 * self.orig_timestep)

    def reset(self):
        self.g_old = None
        self.p_old = None
        self.timestep = self.orig_timestep

    def update_timestep(self, new_timestep: float) -> None:
        self.timestep = new_timestep

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
        converged_mask = np.array([node.converged for node in chain.nodes], dtype=bool)
        for i, (node, grad) in enumerate(zip(chain.nodes, chain_gradients)):
            if node.converged:
                chain_gradients[i] = np.zeros_like(grad)

        g_new = chain_gradients.flatten()
        p = -chain_gradients.flatten()

        if self.g_old is not None:
            g = self.g_old.flatten().copy()
            # Fletcher-Reeves formula
            # beta = np.dot(g_new, g_new) / np.dot(g, g)
            if g_new.shape != g.shape:
                print("Warning: Gradient shapes do not match. Resetting the optimizer.")
                self.reset()
                return self.optimize_step(chain, chain_gradients)
            denom = np.dot(g, g)
            if denom <= 1e-16:
                beta = 0.0
            else:
                beta = max(0.0, np.dot(g_new, g_new - g) / denom)  # Polak-Ribiere+

            prev_dir = self.p_old if self.p_old is not None else -g
            p = -g_new + beta * prev_dir
            # Guard against non-descent CG directions by restarting with steepest descent.
            if np.dot(p, g_new) >= 0.0:
                p = -g_new
            self.g_old = g_new.reshape(chain_gradients.shape).copy()
        else:
            self.g_old = g_new.reshape(chain_gradients.shape).copy()
            p = -g_new

        self.p_old = p.copy()

        p = p.reshape(chain_gradients.shape)
        if converged_mask.any():
            p[converged_mask] = 0.0

        step = alpha * p
        if self.max_step_norm is not None and self.max_step_norm > 0.0:
            flat_step = step.reshape(step.shape[0], -1)
            step_norms = np.linalg.norm(flat_step, axis=1)
            large_steps = step_norms > self.max_step_norm
            if np.any(large_steps):
                scale = (self.max_step_norm / (step_norms[large_steps] + 1e-16)).reshape(-1, 1)
                flat_step[large_steps] = flat_step[large_steps] * scale
                step = flat_step.reshape(step.shape)

        new_chain_coordinates = chain.coordinates + step
        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = chain.model_copy(
            update={"nodes": new_nodes, "parameters": chain.parameters})

        return new_chain
