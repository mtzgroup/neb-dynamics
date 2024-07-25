from __future__ import annotations


from dataclasses import dataclass, field
from typing import Callable
from neb_dynamics.optimizer import Optimizer
from neb_dynamics.chain import Chain
from neb_dynamics.errors import ElectronicStructureError

import numpy as np


@dataclass
class VelocityProjectedOptimizer(Optimizer):
    timestep: float = 1.0
    activation_tol: float = 0.1
    zero_vel_count_thre: int = (
        5  # threshold for decreasing timestep if repeated velocity resetting
    )

    def __post_init__(self):
        self.zero_vel_count = 0

    def copy(self):
        return VelocityProjectedOptimizer(
            timestep=self.timestep, activation_tol=self.activation_tol
        )

    def optimize_step(self, chain, chain_gradients):
        prev_velocity = chain.velocity
        new_force = -(chain_gradients)
        new_force_unit = new_force / np.linalg.norm(new_force)
        timestep = self.timestep  # self.step_size_per_atom*atomn*len(chain)

        new_chain_gradients_fails = True
        retry_count_max = 5
        retry_count = 0
        while new_chain_gradients_fails and retry_count < retry_count_max:
            # try:

            if np.amax(np.abs(new_force)) < self.activation_tol:
                orig_shape = new_force.shape
                prev_velocity_flat = prev_velocity.flatten()
                # print(f"{prev_velocity=}\n{new_force_unit=}")
                projection = np.dot(prev_velocity_flat, new_force_unit.flatten())

                vj_flat = projection * new_force_unit.flatten()
                vj = vj_flat.reshape(orig_shape)

                vel_unit = prev_velocity / np.linalg.norm(prev_velocity)
                corr = np.dot(vel_unit.flatten(), new_force_unit.flatten())
                # print(f'\n/////{projection=} {corr=}////')
                if projection < 0:
                    # print(f"\nproj={projection} Reset!")
                    vj = np.zeros_like(new_force)
                    self.zero_vel_count += 1

                elif np.isclose(corr, 1.0):
                    pass
                else:
                    self.zero_vel_count = 0

                if self.zero_vel_count >= self.zero_vel_count_thre:
                    print(
                        "\nVPO: Step size causing oscillations. Shrinking by 50%\n"
                    )
                    self.timestep *= 0.5

                # print(f"\nzero_vel_count={self.zero_vel_count}\n")
                force = timestep * vj

            else:
                vj = np.zeros_like(new_force)
                force = timestep * new_force

            scaling = 1
            # if np.linalg.norm(force) > max_disp: # if step size is too large
            #     scaling = (1/(np.linalg.norm(force)))*max_disp

            new_chain_coordinates = chain.coordinates + force * scaling
            new_nodes = []
            for node, new_coords in zip(chain.nodes, new_chain_coordinates):

                new_nodes.append(node.update_coords(new_coords))

            new_chain = Chain(new_nodes, parameters=chain.parameters)

            # if np.linalg.norm(new_force) < self.activation_tol:
            new_vel = vj + timestep * (-(chain_gradients))
            # else:
            #     new_vel = vj + timestep*(-(chain_gradients))
            new_chain.velocity = new_vel
            new_chain_gradients_fails = False

            # except Exception:
            #     print(
            #         "VPO: Gradients failed with displacement. Resetting velocity, shrinking by 50%"
            #     )
            #     prev_velocity = new_force_unit  # has to be the gradient of the geometry so the projection is 1 and we just take a steepest descent. If it is 0, we will get stuck ehre forever.
            #     timestep *= 0.5
            #     retry_count += 1

        # if new_chain_gradients_fails:
        #     raise ElectronicStructureError(msg="Electronic structure of chain failed.")
        return new_chain
