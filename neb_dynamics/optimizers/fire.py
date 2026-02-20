from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from neb_dynamics.optimizers.optimizer import Optimizer


@dataclass
class FIREOptimizer(Optimizer):
    """
    Fast Inertial Relaxation Engine (FIRE) optimizer.
    Reference: Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006).
    """

    timestep: float = 0.1
    dt_max: float = 1.0
    finc: float = 1.1
    fdec: float = 0.5
    alpha_start: float = 0.1
    falpha: float = 0.99
    n_min: int = 5
    max_step_norm: float = 1.0

    def __post_init__(self):
        self.velocity = None
        self.alpha = self.alpha_start
        self.n_positive = 0

    def reset(self):
        self.velocity = None
        self.alpha = self.alpha_start
        self.n_positive = 0

    def optimize_step(self, chain, chain_gradients):
        grads = chain_gradients.astype(float)
        for i, (node, grad) in enumerate(zip(chain.nodes, grads)):
            if node.converged:
                grads[i] = np.zeros_like(grad)

        force = -grads
        if self.velocity is None or self.velocity.shape != force.shape:
            self.velocity = np.zeros_like(force)
            self.alpha = self.alpha_start
            self.n_positive = 0

        # Velocity Verlet-like force kick.
        self.velocity = self.velocity + self.timestep * force
        p = float(np.dot(self.velocity.flatten(), force.flatten()))

        if p > 0.0:
            self.n_positive += 1
            if self.n_positive > self.n_min:
                self.timestep = min(self.timestep * self.finc, self.dt_max)
                self.alpha *= self.falpha
        else:
            self.n_positive = 0
            self.timestep *= self.fdec
            self.velocity[:] = 0.0
            self.alpha = self.alpha_start

        v_norm = np.linalg.norm(self.velocity)
        f_norm = np.linalg.norm(force)
        if v_norm > 1e-16 and f_norm > 1e-16:
            self.velocity = (1.0 - self.alpha) * self.velocity + self.alpha * force * (v_norm / f_norm)

        step = self.timestep * self.velocity
        step_norm = np.linalg.norm(step)
        if step_norm > self.max_step_norm and step_norm > 1e-16:
            step *= self.max_step_norm / step_norm

        new_chain_coordinates = chain.coordinates + step
        new_nodes = [node.update_coords(new_coords) for node, new_coords in zip(chain.nodes, new_chain_coordinates)]
        new_chain = chain.model_copy(update={"nodes": new_nodes, "parameters": chain.parameters})
        new_chain.velocity = step.tolist()
        return new_chain
