from dataclasses import dataclass
from typing import Callable

import numpy as np

from ALS import ArmijoLineSearch
from helper_functions import pairwise


@dataclass
class Node:
    coords: np.array
    grad_func: Callable[[np.array], np.array]
    en_func: Callable[[np.array], float]

    @property
    def energy(self):
        return self.en_func(self.coords)

    @property
    def gradient(self):
        return self.grad_func(self.coords)

    def displacement(self, grad):
        phi0 = self.energy
        dr, _ = ArmijoLineSearch(
            f=self.en_func,
            xk=self.coords,
            pk=-1 * grad,
            gfk=grad,
            phi0=phi0,
        )
        return dr


@dataclass
class Chain:
    nodes: list[Node]
    k: float


    def __getitem__(self, index):
        return self.nodes.__getitem__(index)

    def copy(self):
        raise NotImplementedError

    def iter_triplets(self):
        raise NotImplementedError

    @property
    def energies(self):
        return [node.energy() for node in self.nodes]

    @property
    def gradients(self):
        grads = []
        for a, b, c in self.iter_triplets:
            grad = self.spring_grad_neb((a, b, c))
            grads.append(grad)
        return grads

    @property
    def displacements(self):
        return [node.displacement(grad) for node, grad in zip(self.nodes, self.gradients)]

    def _create_tangent_path(self, prev_node: Node, current_node: Node, next_node: Node):
        en_2 = next_node.energy
        en_1 = current_node.energy
        en_0 = prev_node.energy

        if en_2 > en_1 and en_1 > en_0:
            return next_node.coords - current_node.coords
        elif en_2 < en_1 and en_1 < en_2:
            return current_node.coords - prev_node.coords

        else:
            deltaV_max = max(np.abs(en_2 - en_1), np.abs(en_0 - en_1))
            deltaV_min = min(np.abs(en_2 - en_1), np.abs(en_0 - en_1))

            if en_2 > en_0:
                tan_vec = (next_node.coords - current_node.coords) * deltaV_max + (current_node.coords - prev_node.coords) * deltaV_min
            elif en_2 < en_0:
                tan_vec = (next_node.coords - current_node.coords) * deltaV_min + (current_node.coords - prev_node.coords) * deltaV_max
            return tan_vec

    def spring_grad_neb(self, prev_node: Node, current_node: Node, next_node: Node):
        vec_tan_path = self._create_tangent_path(prev_node, current_node, next_node)
        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        pe_grad = current_node.gradient
        pe_grad_nudged_const = np.dot(pe_grad, unit_tan_path)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path

        grads_neighs = []
        force_springs = []

        force_spring = -self.k * (np.abs(next_node.coords - current_node.coords) - np.abs(current_node.coords - prev_node.coords))

        direction = np.dot((next_node.coords - current_node.coords), force_spring)
        if direction < 0:
            force_spring *= -1

        force_springs.append(force_spring)

        force_spring_nudged_const = np.dot(force_spring, unit_tan_path)
        force_spring_nudged = force_spring - force_spring_nudged_const * unit_tan_path

        grads_neighs.append(force_spring_nudged)

        tot_grads_neighs = np.sum(grads_neighs, axis=0)

        # ANTI-KINK FORCE
        force_springs = np.sum(force_springs, axis=0)

        vec_2_to_1 = next_node.coords - current_node.coords
        vec_1_to_0 = current_node.coords - prev_node.coords
        cos_phi = np.dot(vec_2_to_1, vec_1_to_0) / (np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0))

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))

        proj_force_springs = force_springs - np.dot(force_springs, unit_tan_path) * unit_tan_path

        return (pe_grad_nudged - tot_grads_neighs) + f_phi * (proj_force_springs)


@dataclass
class NEB:
    initial_chain: Chain
    redistribute: bool = True
    en_thre: float = 0.001
    grad_thre: float = 0.001
    max_steps: float = 1000

    def optimize_chain(self):
        chain_traj = []
        nsteps = 0
        chain_previous = self.initial_chain.copy()

        while nsteps < self.max_steps:
            new_chain = self.update_chain(chain=chain_previous)
            chain_traj.append(new_chain)

            if self._chain_converged(chain_previous=chain_previous, new_chain=new_chain):
                print(f"Chain converged!\n{new_chain=}")
                return new_chain, chain_traj

            chain_previous = new_chain.copy()
            nsteps += 1

        print(f"Chain did not converge at step {nsteps}")
        return new_chain, chain_traj

    def update_chain(self, chain):
        return chain + chain.gradients * chain.displacements

    def _check_en_converged(self, chain_prev: Chain, chain_new: Chain):
        differences = np.abs(chain_new.energies - chain_prev.energies)
        return np.all(differences < self.en_thre)

    def _check_grad_converged(self, chain_prev: Chain, chain_new: Chain):
        delta_grad = np.abs(chain_prev.gradients - chain_new.gradients)
        return np.all(delta_grad < self.grad_thre)

    def _chain_converged(self, chain_prev: Chain, chain_new: Chain):
        return self._check_en_converged(chain_prev=chain_prev, chain_new=chain_new) and \
            self._check_grad_converged(chain_prev=chain_prev, chain_new=chain_new)

    def remove_chain_folding(self, chain: Chain):
        not_converged = True
        count = 0
        points_removed = []
        while not_converged:
            # print(f"on count {count}...")
            new_chain = []
            new_chain.append(chain[0])
            for prev_node, current_node, next_node in chain.iter_triplets():
                vec1 = current_node.coords - prev_node.coords
                vec2 = next_node.coords - current_node.coords

                if np.dot(vec1, vec2) > 0:
                    new_chain.append(current_node)
                else:
                    points_removed.append(current_node)
            
            new_chain = np.array(new_chain)
            if self._check_dot_product_converged(new_chain):
                not_converged = False
            chain = new_chain.copy()
            count += 1
        return chain

    def _check_dot_product_converged(self, chain: Chain):
        dps = []
        for prev_node, current_node, next_node in chain.iter_triplets():
            vec1 = current_node.coords - prev_node.coords
            vec2 = next_node.coords - current_node.coords
            dps.append(np.dot(vec1, vec2) > 0)

        return all(dps)

@dataclass
class neb:
    def optimize_chain(self, chain, grad_func, en_func, k, redistribute=True, en_thre=0.001, grad_thre=0.001, max_steps=1000):

        chain_traj = []
        nsteps = 0

        chain_previous = chain.copy()

        while nsteps < max_steps:
            new_chain = self.update_chain(chain=chain_previous, k=k, en_func=en_func, grad_func=grad_func, redistribute=redistribute)

            chain_traj.append(new_chain)

            if self._chain_converged(
                chain_previous=chain_previous,
                new_chain=new_chain,
                en_func=en_func,
                en_thre=en_thre,
                grad_func=grad_func,
                grad_thre=grad_thre,
            ):
                print("Chain converged!")
                print(f"{new_chain=}")
                return new_chain, chain_traj

            chain_previous = new_chain.copy()
            nsteps += 1

        print("Chain did not converge...")
        return new_chain, chain_traj

    def update_chain(self, chain, k, en_func, grad_func, redistribute):

        chain_copy = np.zeros_like(chain)
        chain_copy[0] = chain[0]
        chain_copy[-1] = chain[-1]

        for i in range(1, len(chain) - 1):
            view = chain[i - 1: i + 2]

            grad = self.spring_grad_neb(
                view,
                k=k,
                # ideal_distance=ideal_dist,
                grad_func=grad_func,
                en_func=en_func,
            )

            # dr = 0.01

            dr, _ = ArmijoLineSearch(
                f=en_func,
                xk=chain[i],
                gfk=grad,
                phi0=en_func(chain[i]),
                alpha0=0.01,
                pk=-1 * grad,
            )

            p_new = chain[i] - grad * dr

            chain_copy[i] = p_new

        return chain_copy

    def _create_tangent_path(self, view, en_func):
        en_2 = en_func(view[2])
        en_1 = en_func(view[1])
        en_0 = en_func(view[0])

        if en_2 > en_1 and en_1 > en_0:
            return view[2] - view[1]
        elif en_2 < en_1 and en_1 < en_2:
            return view[1] - view[0]

        else:
            deltaV_max = max(np.abs(en_2 - en_1), np.abs(en_0 - en_1))
            deltaV_min = min(np.abs(en_2 - en_1), np.abs(en_0 - en_1))

            if en_2 > en_0:
                tan_vec = (view[2] - view[1]) * deltaV_max + (view[1] - view[0]) * deltaV_min
            elif en_2 < en_0:
                tan_vec = (view[2] - view[1]) * deltaV_min + (view[1] - view[0]) * deltaV_max
            return tan_vec

    def spring_grad_neb(self, view, grad_func, k, en_func):
        vec_tan_path = self._create_tangent_path(view, en_func=en_func)
        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        pe_grad = grad_func(view[1])
        pe_grad_nudged_const = np.dot(pe_grad, unit_tan_path)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path

        grads_neighs = []
        force_springs = []

        force_spring = -k * (np.abs(view[2] - view[1]) - np.abs(view[1] - view[0]))

        direction = np.dot((view[2] - view[1]), force_spring)
        if direction < 0:
            force_spring *= -1

        force_springs.append(force_spring)

        force_spring_nudged_const = np.dot(force_spring, unit_tan_path)
        force_spring_nudged = force_spring - force_spring_nudged_const * unit_tan_path

        grads_neighs.append(force_spring_nudged)

        tot_grads_neighs = np.sum(grads_neighs, axis=0)

        # ANTI-KINK FORCE

        force_springs = np.sum(force_springs, axis=0)

        vec_2_to_1 = view[2] - view[1]
        vec_1_to_0 = view[1] - view[0]
        cos_phi = np.dot(vec_2_to_1, vec_1_to_0) / (np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0))

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))

        proj_force_springs = force_springs - np.dot(force_springs, unit_tan_path) * unit_tan_path

        return (pe_grad_nudged - tot_grads_neighs) + f_phi * (proj_force_springs)

    def _check_en_converged(self, chain_prev, chain_new, en_func, en_thre):
        for i in range(1, len(chain_prev) - 1):
            node_prev = chain_prev[i]
            node_new = chain_new[i]

            delta_e = np.abs(en_func(node_new) - en_func(node_prev))
            if delta_e > en_thre:
                return False
        return True

    def _check_grad_converged(self, chain_prev, chain_new, grad_func, grad_thre):
        for i in range(1, len(chain_prev) - 1):
            node_prev = chain_prev[i]
            node_new = chain_new[i]

            delta_grad = np.abs(grad_func(node_new) - grad_func(node_prev))

            if True in delta_grad > grad_thre:
                return False
        return True

    def _chain_converged(self, chain_previous, new_chain, en_func, en_thre, grad_func, grad_thre):

        en_converged = self._check_en_converged(
            chain_prev=chain_previous,
            chain_new=new_chain,
            en_func=en_func,
            en_thre=en_thre,
        )
        grad_converged = self._check_grad_converged(
            chain_prev=chain_previous,
            chain_new=new_chain,
            grad_func=grad_func,
            grad_thre=grad_thre,
        )

        return en_converged and grad_converged

    def remove_chain_folding(self, chain):
        not_converged = True
        count = 0
        points_removed = []
        while not_converged:
            # print(f"on count {count}...")
            new_chain = []
            for i in range(len(chain)):
                if i == 0 or i == len(chain) - 1:
                    new_chain.append(chain[i])
                    continue

                vec1, vec2 = self._get_vectors(chain, i)
                if np.dot(vec1, vec2) > 0:
                    new_chain.append(chain[i])
                else:
                    points_removed.append(chain[i])

            new_chain = np.array(new_chain)
            if self._check_dot_product_converged(new_chain):
                not_converged = False
            chain = new_chain.copy()
            count += 1
        return chain

    def redistribute_chain(self, chain):
        direction = np.array([b - a for a, b in pairwise(chain)])
        distances = np.linalg.norm(direction, axis=1)
        tot_dist = np.sum(distances)
        cumsum = np.cumsum(distances)  # cumulative sum
        cumsum = np.insert(cumsum, 0, 0)

        distributed_chain = []
        for num in np.linspace(0, tot_dist, len(chain)):
            foobar = self.redistribution_helper(num=num, cum=cumsum, new_chain=chain)
            # print(num, foobar)
            distributed_chain.append(foobar)

        distributed_chain[0] = chain[0]
        distributed_chain[-1] = chain[-1]

        return np.array(distributed_chain)

    def redistribution_helper(self, num, cum, new_chain):
        """
        num: the distance from first node to return output point to
        cum: cumulative sums
        new_chain: chain that we are considering

        """

        for ii, ((cum_sum_init, point_start), (cum_sum_end, point_end)) in enumerate(pairwise(zip(cum, new_chain))):

            if cum_sum_init < num < cum_sum_end:
                direction = point_end - point_start
                percentage = (num - cum_sum_init) / (cum_sum_end - cum_sum_init)
                point = point_start + (direction * percentage)

                return point

    def _get_vectors(self, chain, i):
        view = chain[i - 1: i + 2]
        vec1 = view[1] - view[0]
        vec2 = view[2] - view[1]
        return vec1, vec2

    def _check_dot_product_converged(self, chain):
        dps = []
        for i in range(1, len(chain) - 1):
            vec1, vec2 = self._get_vectors(chain, i)
            dps.append(np.dot(vec1, vec2) > 0)

        return all(dps)
