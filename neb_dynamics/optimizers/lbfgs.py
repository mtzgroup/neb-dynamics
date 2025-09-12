from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np
from neb_dynamics.optimizers.optimizer import Optimizer


@dataclass
class LBFGS(Optimizer):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) optimizer.
    This implementation stores a limited number of past gradients and position updates
    to approximate the inverse Hessian, leading to faster convergence than simple
    gradient descent for many problems.
    """
    history_size: int = 10

    def __post_init__(self):
        self.s_history = deque(maxlen=self.history_size)
        self.y_history = deque(maxlen=self.history_size)
        self.g_old = None
        self.x_old = None

    def optimize_step(self, chain, chain_gradients):
        """
        Performs an L-BFGS optimization step.
        Args:
            chain: The current chain of nodes.
            chain_gradients: The gradients for each node in the chain.
        Returns:
            A new chain with updated coordinates.
        """
        # Set gradients for converged nodes to zero
        for i, (node, grad) in enumerate(zip(chain.nodes, chain_gradients)):
            if node.converged:
                chain_gradients[i] = np.zeros_like(grad)
        g_new = chain_gradients.flatten()
        # Handle the very first step
        if self.g_old is None:
            new_chain_coordinates = chain.coordinates - \
                g_new.reshape(chain_gradients.shape)
            self.x_old = new_chain_coordinates.flatten()
            self.g_old = g_new
            new_nodes = [node.update_coords(new_coords) for node, new_coords in zip(
                chain.nodes, new_chain_coordinates)]
            return chain.model_copy(update={"nodes": new_nodes, "parameters": chain.parameters})
        # Calculate s_k and y_k and store in history
        s_k = chain.coordinates.flatten() - self.x_old
        y_k = g_new - self.g_old
        self.s_history.append(s_k)
        self.y_history.append(y_k)
        # Perform two-loop recursion to compute the search direction p
        q = g_new.copy()
        alphas = []
        for i in range(len(self.s_history) - 1, -1, -1):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i_denominator = np.dot(y_i, s_i)
            # Check for a near-zero denominator to prevent NaNs
            if rho_i_denominator < 1e-8:
                rho_i = 0.0
            else:
                rho_i = 1.0 / rho_i_denominator
            alpha_i = rho_i * np.dot(s_i, q)
            alphas.append(alpha_i)
            q = q - alpha_i * y_i
        gamma_k = 1.0
        if len(self.y_history) > 0:
            y_last = self.y_history[-1]
            s_last = self.s_history[-1]
            # Check for near-zero denominator to prevent NaNs
            y_last_dot = np.dot(y_last, y_last)
            if y_last_dot > 1e-8:
                gamma_k = np.dot(s_last, y_last) / y_last_dot
        r = gamma_k * q
        for i in range(len(self.s_history)):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i_denominator = np.dot(y_i, s_i)
            if rho_i_denominator < 1e-8:
                rho_i = 0.0
            else:
                rho_i = 1.0 / rho_i_denominator
            beta = rho_i * np.dot(y_i, r)
            r = r + s_i * (alphas[len(alphas) - 1 - i] - beta)
        p = -r.reshape(chain_gradients.shape)
        # Apply the calculated search direction to update coordinates
        new_chain_coordinates = chain.coordinates + p
        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):
            new_nodes.append(node.update_coords(new_coords))
        new_chain = chain.model_copy(
            update={"nodes": new_nodes, "parameters": chain.parameters})
        # Update old position and gradient for the next iteration
        self.x_old = new_chain_coordinates.flatten()
        self.g_old = g_new
        return new_chain
