from __future__ import annotations

import sys
from dataclasses import dataclass, field

# from hashlib import new
from pathlib import Path

import numpy as np
from retropaths.abinitio.trajectory import Trajectory
from scipy.signal import argrelextrema

from neb_dynamics.Chain import Chain
from neb_dynamics.helper_functions import pairwise
from neb_dynamics.Node import Node
from neb_dynamics import ALS
from neb_dynamics.Inputs import NEBInputs

import matplotlib.pyplot as plt


@dataclass
class NoneConvergedException(Exception):
    trajectory: list[Chain]
    msg: str
    obj: NEB


@dataclass
class NEB:
    initial_chain: Chain
    parameters: NEBInputs

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)

    def do_velvel(self, chain: Chain):
        max_force_on_node = max([np.linalg.norm(grad) for grad in chain.gradients])
        return max_force_on_node < self.parameters.vv_force_thre

    def _reset_node_convergence(self, chain):
        for node in chain:
            node.converged = False

    def set_climbing_nodes(self, chain: Chain):
        # reset node convergence
        self._reset_node_convergence(chain=chain)

        inds_maxima = argrelextrema(chain.energies, np.greater, order=2)[0]
        if self.parameters.v > 1:
            print(f"----->Setting {len(inds_maxima)} nodes to climb")

        for ind in inds_maxima:
            chain[ind].do_climb = True

    def optimize_chain(self):
        nsteps = 1
        chain_previous = self.initial_chain.copy()

        while nsteps < self.parameters.max_steps + 1:
            max_grad_val = chain_previous.get_maximum_grad_magnitude()
            if max_grad_val <= 3 * self.parameters.grad_thre and self.parameters.climb:
                self.set_climbing_nodes(chain=chain_previous)
                self.parameters.climb = False  # no need to set climbing nodes again

            new_chain = self.update_chain(chain=chain_previous)
            if self.parameters.v:
                print(
                    f"step {nsteps} // max |gradient| {max_grad_val}{' '*20}", end="\r"
                )
            sys.stdout.flush()

            self.chain_trajectory.append(new_chain)
            self.gradient_trajectory.append(new_chain.gradients)

            if self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
                if self.parameters.v:
                    print("\nChain converged!")

                self.optimized = new_chain
                return
            chain_previous = new_chain.copy()

            nsteps += 1

        new_chain = self.update_chain(chain=chain_previous)
        if not self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
            raise NoneConvergedException(
                trajectory=self.chain_trajectory,
                msg=f"\nChain did not converge at step {nsteps}",
                obj=self,
            )

    def get_chain_velocity(self, chain: Chain) -> np.array:
        prev_velocity = chain.parameters.velocity

        step = (
            self.parameters.grad_thre / 10
        )  # make the step size rel. to threshold we want

        new_force = -(chain.gradients) * step

        directions = prev_velocity * new_force
        prev_velocity[
            directions < 0
        ] = 0  # zero the velocities for which we overshot the minima

        new_velocity = prev_velocity + new_force
        return new_velocity

    def update_chain(self, chain: Chain) -> Chain:

        do_vv = self.do_velvel(chain=chain)

        if do_vv:
            velocity = self.get_chain_velocity(chain=chain)
            new_chain_coordinates = chain.coordinates + velocity

        else:
            disp = ALS.ArmijoLineSearch(
                chain=chain,
                t=chain.parameters.step_size,
                alpha=0.01,
                beta=0.5,
                grad=chain.gradients,
            )
            new_chain_coordinates = chain.coordinates - chain.gradients * disp

        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = Chain(new_nodes, parameters=chain.parameters)
        if do_vv:
            new_chain.velocity = velocity

        return new_chain

    def _update_node_convergence(self, chain: Chain, indices: np.array) -> None:
        for i, node in enumerate(chain):
            if i in indices:
                node.converged = True
            else:
                node.converged = False

    def _check_en_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
        differences = np.abs(chain_new.energies - chain_prev.energies)
        indices_converged = np.where(differences < self.parameters.en_thre)

        return indices_converged[0], differences

    def _check_grad_converged(self, chain: Chain) -> bool:
        # delta_grad = np.abs(chain_prev.gradients - chain_new.gradients)
        # mag_grad = np.array([np.linalg.norm(grad) for grad in chain_new.gradients])
        bools = []
        max_grad_components = []
        gradients = chain.gradients
        for grad in gradients:
            max_grad = np.amax(grad)
            max_grad_components.append(max_grad)
            # print(f'{max_grad=} < {self.parameters.grad_thre=}')
            bools.append(max_grad < self.parameters.grad_thre)

        # grad_converged = np.where(delta_grad < self.parameters.grad_thre)
        # mag_converged = np.where(mag_grad < self.mag_grad_thre)
        # mag_converged = mag_grad < self.mag_grad_thre

        # return delta_converged[0], mag_converged[0], delta_grad, mag_grad
        # return delta_converged[0], delta_grad
        return np.where(bools), max_grad_components

    def _check_rms_grad_converged(self, chain: Chain):
        bools = []
        rms_grads = []
        grads = chain.gradients
        for grad in grads:
            rms_gradient = np.sqrt(np.mean(np.square(grad)))
            rms_grads.append(rms_gradient)
            rms_grad_converged = rms_gradient < self.parameters.rms_grad_thre
            bools.append(rms_grad_converged)

        return np.where(bools), rms_grads

    def _chain_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:

        rms_grad_conv_ind, max_rms_grads = self._check_rms_grad_converged(chain_new)
        en_converged_indices, en_deltas = self._check_en_converged(
            chain_prev=chain_prev, chain_new=chain_new
        )

        grad_conv_ind, max_grad_components = self._check_grad_converged(chain=chain_new)

        converged_nodes_indices = np.intersect1d(
            en_converged_indices, rms_grad_conv_ind
        )
        converged_nodes_indices = np.intersect1d(converged_nodes_indices, grad_conv_ind)

        # [print(f"\t\tnode{i} | ∆E : {en_deltas[i]} | Max(∆Grad) : { np.amax(grad_deltas[i])} | |Grad| : {mag_grad_deltas[i]} | Converged? : {chain_new.nodes[i].converged}") for i in range(len(chain_new))]
        if self.parameters.v > 1:
            [
                print(
                    f"\t\tnode{i} | ∆E : {en_deltas[i]} | Max(RMS Grad): {max_rms_grads[i]} | Max(Grad components): {max_grad_components[i]} | Converged? : {chain_new.nodes[i].converged}"
                )
                for i in range(len(chain_new))
            ]
        if self.parameters.v > 1:
            print(f"\t{len(converged_nodes_indices)} nodes have converged")

        self._update_node_convergence(chain=chain_new, indices=converged_nodes_indices)
        # return len(converged_node_indices) == len(chain_new.nodes)

        return len(converged_nodes_indices) == len(chain_new)

        # return np.all(
        #     [np.linalg.norm(grad) <= self.parameters.grad_thre for grad in chain_new.gradients]
        # )

        # return en_bool and grad_bool

    def remove_chain_folding(self, chain: Chain) -> Chain:
        not_converged = True
        count = 0
        points_removed = []
        while not_converged:
            if self.parameters.v > 1:
                print(f"anti-folding: on count {count}...")
            new_chain = []
            new_chain.append(chain[0])

            for prev_node, current_node, next_node in chain.iter_triplets():
                vec1 = current_node.coords - prev_node.coords
                vec2 = next_node.coords - current_node.coords

                if np.all(current_node.dot_function(vec1, vec2)) > 0:
                    new_chain.append(current_node)
                else:
                    points_removed.append(current_node)

            new_chain.append(chain[-1])
            new_chain = Chain(nodes=new_chain, parameters=chain.parameters)
            if self._check_dot_product_converged(new_chain):
                not_converged = False
            chain = new_chain.copy()
            count += 1

        return chain

    def _check_dot_product_converged(self, chain: Chain) -> bool:
        dps = []
        for prev_node, current_node, next_node in chain.iter_triplets():
            vec1 = current_node.coords - prev_node.coords
            vec2 = next_node.coords - current_node.coords
            dps.append(current_node.dot_function(vec1, vec2) > 0)

        return all(dps)

    def redistribute_chain(self, chain: Chain, requested_length_of_chain: int) -> Chain:
        # if len(chain) < requested_length_of_chain:
        #     fixed_chain = chain.copy()
        #     [
        #         fixed_chain.nodes.insert(1, fixed_chain[1])
        #         for _ in range(requested_length_of_chain - len(chain))
        #     ]
        #     chain = fixed_chain

        direction = np.array(
            [
                next_node.coords - current_node.coords
                for current_node, next_node in pairwise(chain)
            ]
        )
        distances = np.linalg.norm(direction, axis=1)
        tot_dist = np.sum(distances)
        cumsum = np.cumsum(distances)  # cumulative sum
        cumsum = np.insert(cumsum, 0, 0)

        distributed_chain = []
        for num in np.linspace(0, tot_dist, len(chain)):
            new_node = self.redistribution_helper(num=num, cum=cumsum, chain=chain)

            distributed_chain.append(new_node)

        distributed_chain[0] = chain[0]
        distributed_chain[-1] = chain[-1]

        return Chain(distributed_chain, parameters=chain)

    def redistribution_helper(self, num, cum, chain: Chain) -> Node:
        """
        num: the distance from first node to return output point to
        cum: cumulative sums
        new_chain: chain that we are considering

        """

        for ii, ((cum_sum_init, node_start), (cum_sum_end, node_end)) in enumerate(
            pairwise(zip(cum, chain))
        ):

            if cum_sum_init <= num < cum_sum_end:
                direction = node_end.coords - node_start.coords
                percentage = (num - cum_sum_init) / (cum_sum_end - cum_sum_init)

                new_node = node_start.copy()
                new_coords = node_start.coords + (direction * percentage)
                new_node = new_node.update_coords(new_coords)

                return new_node

    def write_to_disk(self, fp: Path, write_history=False):
        out_traj = self.chain_trajectory[-1].to_trajectory()
        out_traj.write_trajectory(fp)

        if write_history:
            out_folder = fp.resolve().parent / fp.stem + "_history"
            if not out_folder.exists():
                out_folder.mkdir()

            for i, chain in enumerate(self.chain_trajectory):
                traj = chain.to_trajectory()
                traj.write_trajectory(out_folder / f"traj_{i}.xyz")

    def plot_opt_history(self):

        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))

        for i, chain in enumerate(self.chain_trajectory):
            if i == len(self.chain_trajectory) - 1:
                plt.plot(chain.integrated_path_length, chain.energies, "o-", alpha=1)
            else:
                plt.plot(
                    chain.integrated_path_length,
                    chain.energies,
                    "o-",
                    alpha=0.1,
                    color="gray",
                )

        plt.xlabel("Integrated path length", fontsize=fs)

        plt.ylabel("Energy (kcal/mol)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.show()
