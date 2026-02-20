from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from neb_dynamics.optimizers.optimizer import Optimizer


@dataclass
class AdamOptimizer(Optimizer):
    timestep: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self):
        self.m = None
        self.v = None
        self.t = 0

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def optimize_step(self, chain, chain_gradients):
        for i, (node, grad) in enumerate(zip(chain.nodes, chain_gradients)):
            if node.converged:
                chain_gradients[i] = np.zeros_like(grad)

        g = chain_gradients.astype(float)
        if self.m is None or self.v is None or self.m.shape != g.shape:
            self.reset()
            self.m = np.zeros_like(g, dtype=float)
            self.v = np.zeros_like(g, dtype=float)

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g**2)

        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)

        step = self.timestep * m_hat / (np.sqrt(v_hat) + self.epsilon)
        new_chain_coordinates = chain.coordinates - step
        new_nodes = [node.update_coords(new_coords) for node, new_coords in zip(chain.nodes, new_chain_coordinates)]

        return chain.model_copy(update={"nodes": new_nodes, "parameters": chain.parameters})
