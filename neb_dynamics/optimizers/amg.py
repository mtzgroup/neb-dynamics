from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from neb_dynamics.optimizers.optimizer import Optimizer


@dataclass
class AdaptiveMomentumGradient(Optimizer):
    """
    Adam-like optimizer with NEB-oriented stabilizers:
    - automatic timestep adaptation from gradient correlation
    - global step clipping (trust-region style)
    """

    timestep: float = 0.10
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_step_norm: float = 1.0
    min_timestep: float = 1e-4
    max_timestep: float = 1.0
    step_up: float = 1.10
    step_down: float = 0.6

    def __post_init__(self):
        self.m = None
        self.v = None
        self.g_old = None
        self.t = 0

    def reset(self):
        self.m = None
        self.v = None
        self.g_old = None
        self.t = 0

    def _adapt_timestep(self, g_flat: np.ndarray) -> None:
        if self.g_old is None:
            return
        g_prev = self.g_old.flatten()
        denom = np.linalg.norm(g_prev) * np.linalg.norm(g_flat)
        if denom <= 1e-16:
            return
        corr = float(np.dot(g_prev, g_flat) / denom)
        if corr < -0.1:
            self.timestep *= self.step_down
        elif corr > 0.8:
            self.timestep *= self.step_up
        self.timestep = max(self.min_timestep, min(self.max_timestep, self.timestep))

    def optimize_step(self, chain, chain_gradients):
        g = np.array(chain_gradients, dtype=float, copy=True)
        converged_mask = np.array([node.converged for node in chain.nodes], dtype=bool)
        if converged_mask.any():
            g[converged_mask] = 0.0

        if self.m is None or self.v is None or self.m.shape != g.shape:
            self.reset()
            self.m = np.zeros_like(g, dtype=float)
            self.v = np.zeros_like(g, dtype=float)

        g_flat = g.flatten()
        self._adapt_timestep(g_flat=g_flat)

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g**2)
        if converged_mask.any():
            self.m[converged_mask] = 0.0
            self.v[converged_mask] = 0.0
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)

        step = self.timestep * m_hat / (np.sqrt(v_hat) + self.epsilon)
        if converged_mask.any():
            step[converged_mask] = 0.0
        step_norm = np.linalg.norm(step)
        if step_norm > self.max_step_norm and step_norm > 1e-16:
            step *= self.max_step_norm / step_norm

        new_chain_coordinates = chain.coordinates - step
        new_nodes = [
            node.update_coords(new_coords)
            for node, new_coords in zip(chain.nodes, new_chain_coordinates)
        ]
        self.g_old = g.copy()
        return chain.model_copy(update={"nodes": new_nodes, "parameters": chain.parameters})
