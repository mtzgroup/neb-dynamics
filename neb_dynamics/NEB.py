from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import List

import numpy as np
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics.helper_functions import pairwise
from neb_dynamics.tdstructure import TDStructure

ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROMS = 1 / ANGSTROM_TO_BOHR


@dataclass
class NoneConvergedException(Exception):
    trajectory: list[Chain]
    msg: str

    obj: NEB


@dataclass
class AlessioError(Exception):
    message: str


class Node(ABC):
    @abstractmethod
    def en_func(coords):
        ...

    @abstractmethod
    def grad_func(coords):
        ...

    @property
    @abstractmethod
    def energy(self):
        ...

    @abstractmethod
    def copy(self):
        ...

    @property
    @abstractmethod
    def gradient(self):
        ...

    @property
    @abstractmethod
    def coords(self):
        ...

    @abstractmethod
    def dot_function(self, other):
        ...

    @abstractmethod
    def displacement(self, grad):
        ...

    @abstractmethod
    def update_coords(self, coords):
        ...


@dataclass
class Node2D(Node):
    pair_of_coordinates: np.array

    @property
    def coords(self):
        return self.pair_of_coordinates

    @staticmethod
    def en_func(node: Node2D):
        x, y = node.coords
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    @staticmethod
    def grad_func(node: Node2D):
        x, y = node.coords
        dx = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
        dy = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
        return np.array([dx, dy])

    @property
    def energy(self) -> float:
        return self.en_func(self)

    @property
    def gradient(self) -> np.array:
        return self.grad_func(self)

    @staticmethod
    def dot_function(self, other: Node2D) -> float:
        return np.dot(self, other)

    def displacement(self, grad):
        from neb_dynamics import ALS

        dr = ALS.ArmijoLineSearch(
            node=self.copy()
        )
        return dr

    def copy(self):
        return Node2D(self.pair_of_coordinates)

    def update_coords(self, coords: np.array):
        self.pair_of_coordinates = coords
        return Node2D(self.pair_of_coordinates)


@dataclass
class Node3D(Node):
    tdstructure: TDStructure

    @property
    def coords(self):
        return self.tdstructure.coords

    @cached_property
    def _create_calculation_object(self):
        # print("_create_calc obj ", type(self.tdstructure))
        coords = self.tdstructure.coords_bohr
        atomic_numbers = self.tdstructure.atomic_numbers
        calc = Calculator(
            get_method("GFN2-xTB"),
            numbers=np.array(atomic_numbers),
            positions=coords,
            charge=self.tdstructure.charge,
            uhf=self.tdstructure.spinmult - 1,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()
        return res

    @property
    def energy(self):
        return self._create_calculation_object.get_energy()

    # i want to cache the result of this but idk how caching works
    def run_xtb_calc(tdstruct):
        # print("runxtb calc", type(tdstruct))
        coords = tdstruct.coords_bohr
        atomic_numbers = tdstruct.atomic_numbers
        calc = Calculator(
            get_method("GFN2-xTB"),
            numbers=np.array(atomic_numbers),
            positions=coords,
            charge=tdstruct.charge,
            uhf=tdstruct.spinmult - 1,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()
        return res

    @staticmethod
    def grad_func(node: Node3D):
        tdstruct = node.tdstructure
        res = Node3D.run_xtb_calc(tdstruct)
        return res.get_gradient()

    @staticmethod
    def en_func(node: Node3D):
        tdstruct = node.tdstructure
        res = Node3D.run_xtb_calc(tdstruct)
        return res.get_energy()

    @property
    def gradient(self):
        return self._create_calculation_object.get_gradient()

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
        # return np.tensordot(first, second, axes=2)

        return np.sum(first * second, axis=1).reshape(-1, 1)

    def displacement(self, grad):
        from neb_dynamics import ALS_xtb

        # print(f"{grad.shape=}")
        dr = ALS_xtb.ArmijoLineSearch(
            node=self, grad=grad, t=1, alpha=0.3, beta=0.8, f=self.en_func
        )
        return dr

    def update_coords(self, coords: np.array) -> None:
        self.tdstructure.update_coords(coords=coords)
        return Node3D(self.tdstructure)

    def copy(self):
        return Node3D(tdstructure=self.tdstructure.copy())


@dataclass
class Chain:
    nodes: List[Node]
    k: float

    def __getitem__(self, index):
        return self.nodes.__getitem__(index)

    def copy(self):
        list_of_nodes = [node.copy() for node in self.nodes]
        chain_copy = Chain(nodes=list_of_nodes, k=self.k)
        return chain_copy

    def iter_triplets(self):
        for i in range(1, len(self.nodes) - 1):
            yield self.nodes[i - 1 : i + 2]

    @classmethod
    def from_list_of_coords(cls, k, list_of_coords: List, node_class: Node) -> Chain:
        nodes = [node_class(point) for point in list_of_coords]
        return cls(nodes=nodes, k=k)

    @property
    def energies(self) -> np.array:
        return np.array([node.energy for node in self.nodes])

    @property
    def gradients(self) -> np.array:
        grads = []
        for prev_node, current_node, next_node in self.iter_triplets():
            grad = self.spring_grad_neb(prev_node, current_node, next_node)
            grads.append(grad)

        zero = np.zeros_like(grad)
        grads.insert(0, zero)
        grads.append(zero)

        return np.array(grads)

    @property
    def coordinates(self) -> np.array:

        return np.array([node.coords for node in self.nodes])

    @property
    def displacements(self):

        grads = self.gradients
        correct_dimensions = [1 if i > 0 else -1 for i, _ in enumerate(grads.shape)]
        return np.array(
            [node.displacement(grad) for node, grad in zip(self.nodes, grads)]
        ).reshape(*correct_dimensions)

    def _create_tangent_path(
        self, prev_node: Node, current_node: Node, next_node: Node
    ):
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
                tan_vec = (next_node.coords - current_node.coords) * deltaV_max + (
                    current_node.coords - prev_node.coords
                ) * deltaV_min
            elif en_2 < en_0:
                tan_vec = (next_node.coords - current_node.coords) * deltaV_min + (
                    current_node.coords - prev_node.coords
                ) * deltaV_max

            else:
                raise ValueError("Something catastrophic happened. Check chain traj.")

            return tan_vec

    def spring_grad_neb(self, prev_node: Node, current_node: Node, next_node: Node):
        vec_tan_path = self._create_tangent_path(prev_node, current_node, next_node)
        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        pe_grad = current_node.gradient
        pe_grad_nudged_const = current_node.dot_function(pe_grad, unit_tan_path)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path

        grads_neighs = []
        force_springs = []

        force_spring = -self.k * (
            np.abs(next_node.coords - current_node.coords)
            - np.abs(current_node.coords - prev_node.coords)
        )

        direction = np.sum(
            current_node.dot_function(
                (next_node.coords - current_node.coords), force_spring
            )
        )
        if direction < 0:
            force_spring *= -1

        force_springs.append(force_spring)

        force_spring_nudged_const = current_node.dot_function(
            force_spring, unit_tan_path
        )
        force_spring_nudged = force_spring - force_spring_nudged_const * unit_tan_path

        grads_neighs.append(force_spring_nudged)

        tot_grads_neighs = np.sum(grads_neighs, axis=0)

        # ANTI-KINK FORCE
        force_springs = np.sum(force_springs, axis=0)

        vec_2_to_1 = next_node.coords - current_node.coords
        vec_1_to_0 = current_node.coords - prev_node.coords
        cos_phi = current_node.dot_function(vec_2_to_1, vec_1_to_0) / (
            np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0)
        )

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))

        proj_force_springs = (
            force_springs
            - current_node.dot_function(force_springs, unit_tan_path) * unit_tan_path
        )

        return (pe_grad_nudged - tot_grads_neighs) + f_phi * (proj_force_springs)


@dataclass
class NEB:
    initial_chain: Chain
    redistribute: bool = True
    en_thre: float = 0.001
    grad_thre: float = 0.001
    max_steps: float = 1000

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)

    def optimize_chain(self):
        
        nsteps = 0
        chain_previous = self.initial_chain.copy()

        while nsteps < self.max_steps:
            new_chain = self.update_chain(chain=chain_previous)
            print(f"step {nsteps}")
            
            self.chain_trajectory.append(new_chain)

            if self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
                print(f"Chain converged!\n{new_chain=}")
                self.optimized = new_chain
                self.chain_trajectory = self.chain_trajectory
                break
            chain_previous = new_chain.copy()
            nsteps += 1

        if not self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
            self.chain_trajectory = self.chain_trajectory
            raise NoneConvergedException(
                trajectory=self.chain_trajectory, msg=f"Chain did not converge at step {nsteps}", obj=self
            )

    def update_chain(self, chain: Chain) -> Chain:
        new_chain_coordinates = (
            chain.coordinates - chain.gradients * chain.displacements
        )
        new_chain = chain.copy()
        for node, new_coords in zip(new_chain.nodes, new_chain_coordinates):
            node.update_coords(new_coords)
        return new_chain

    def _check_en_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
        differences = np.abs(chain_new.energies - chain_prev.energies)
        return np.all(differences < self.en_thre)

    def _check_grad_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
        delta_grad = np.abs(chain_prev.gradients - chain_new.gradients)
        mag_grad = np.array([np.linalg.norm(grad) for grad in chain_new.gradients])
        return np.all(delta_grad < self.grad_thre) and np.all(mag_grad < self.grad_thre)

    def _chain_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
        return self._check_en_converged(
            chain_prev=chain_prev, chain_new=chain_new
        ) and self._check_grad_converged(chain_prev=chain_prev, chain_new=chain_new)

    def remove_chain_folding(self, chain: Chain) -> Chain:
        not_converged = True
        count = 0
        points_removed = []
        while not_converged:
            print(f"on count {count}...")
            new_chain = []
            new_chain.append(chain[0])

            for prev_node, current_node, next_node in chain.iter_triplets():
                vec1 = current_node.coords - prev_node.coords
                vec2 = next_node.coords - current_node.coords

                if current_node.dot_function(vec1, vec2) > 0:
                    new_chain.append(current_node)
                else:
                    points_removed.append(current_node)

            new_chain = np.array(new_chain)
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
            dps.append(current_node.dot_func(vec1, vec2) > 0)

        return all(dps)

    def redistribute_chain(self, chain) -> Chain:
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

        return Chain(distributed_chain, k=chain.k)

    def redistribution_helper(self, num, cum, chain) -> Node:
        """
        num: the distance from first node to return output point to
        cum: cumulative sums
        new_chain: chain that we are considering

        """

        for ii, ((cum_sum_init, node_start), (cum_sum_end, node_end)) in enumerate(
            pairwise(zip(cum, chain))
        ):

            if cum_sum_init < num < cum_sum_end:
                direction = node_end.coords - node_start.coords
                percentage = (num - cum_sum_init) / (cum_sum_end - cum_sum_init)

                new_node = node_start.update_coords(vector=direction * percentage)

                new_coords = node_start.coords + (direction * percentage)
                new_node = Node(
                    coords=new_coords,
                    grad_func=node_start.grad_func,
                    en_func=node_start.en_func,
                    dot_func=node_start.dot_func,
                )

                return new_node


# @dataclass
# class neb:
#     def optimize_chain(self, chain, grad_func, en_func, k, redistribute=True, en_thre=0.001, grad_thre=0.001, max_steps=1000):

#         chain_traj = []
#         nsteps = 0

#         chain_previous = chain.copy()

#         while nsteps < max_steps:
#             new_chain = self.update_chain(chain=chain_previous, k=k, en_func=en_func, grad_func=grad_func, redistribute=redistribute)

#             chain_traj.append(new_chain)

#             if self._chain_converged(
#                 chain_previous=chain_previous,
#                 new_chain=new_chain,
#                 en_func=en_func,
#                 en_thre=en_thre,
#                 grad_func=grad_func,
#                 grad_thre=grad_thre,
#             ):
#                 print("Chain converged!")
#                 print(f"{new_chain=}")
#                 return new_chain, chain_traj

#             chain_previous = new_chain.copy()
#             nsteps += 1

#         print("Chain did not converge...")
#         return new_chain, chain_traj

#     def update_chain(self, chain, k, en_func, grad_func, redistribute):

#         chain_copy = np.zeros_like(chain)
#         chain_copy[0] = chain[0]
#         chain_copy[-1] = chain[-1]

#         for i in range(1, len(chain) - 1):
#             view = chain[i - 1: i + 2]

#             grad = self.spring_grad_neb(
#                 view,
#                 k=k,
#                 # ideal_distance=ideal_dist,
#                 grad_func=grad_func,
#                 en_func=en_func,
#             )

#             # dr = 0.01

#             dr, _ = ArmijoLineSearch(
#                 f=en_func,
#                 xk=chain[i],
#                 gfk=grad,
#                 phi0=en_func(chain[i]),
#                 alpha0=0.01,
#                 pk=-1 * grad,
#             )

#             p_new = chain[i] - grad * dr

#             chain_copy[i] = p_new

#         return chain_copy

#     def _create_tangent_path(self, view, en_func):
#         en_2 = en_func(view[2])
#         en_1 = en_func(view[1])
#         en_0 = en_func(view[0])

#         if en_2 > en_1 and en_1 > en_0:
#             return view[2] - view[1]
#         elif en_2 < en_1 and en_1 < en_2:
#             return view[1] - view[0]

#         else:
#             deltaV_max = max(np.abs(en_2 - en_1), np.abs(en_0 - en_1))
#             deltaV_min = min(np.abs(en_2 - en_1), np.abs(en_0 - en_1))

#             if en_2 > en_0:
#                 tan_vec = (view[2] - view[1]) * deltaV_max + (view[1] - view[0]) * deltaV_min
#             elif en_2 < en_0:
#                 tan_vec = (view[2] - view[1]) * deltaV_min + (view[1] - view[0]) * deltaV_max
#             return tan_vec

#     def spring_grad_neb(self, view, grad_func, k, en_func):
#         vec_tan_path = self._create_tangent_path(view, en_func=en_func)
#         unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

#         pe_grad = grad_func(view[1])
#         pe_grad_nudged_const = np.dot(pe_grad, unit_tan_path)
#         pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path

#         grads_neighs = []
#         force_springs = []

#         force_spring = -k * (np.abs(view[2] - view[1]) - np.abs(view[1] - view[0]))

#         direction = np.dot((view[2] - view[1]), force_spring)
#         if direction < 0:
#             force_spring *= -1

#         force_springs.append(force_spring)

#         force_spring_nudged_const = np.dot(force_spring, unit_tan_path)
#         force_spring_nudged = force_spring - force_spring_nudged_const * unit_tan_path

#         grads_neighs.append(force_spring_nudged)

#         tot_grads_neighs = np.sum(grads_neighs, axis=0)

#         # ANTI-KINK FORCE

#         force_springs = np.sum(force_springs, axis=0)

#         vec_2_to_1 = view[2] - view[1]
#         vec_1_to_0 = view[1] - view[0]
#         cos_phi = np.dot(vec_2_to_1, vec_1_to_0) / (np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0))

#         f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))

#         proj_force_springs = force_springs - np.dot(force_springs, unit_tan_path) * unit_tan_path

#         return (pe_grad_nudged - tot_grads_neighs) + f_phi * (proj_force_springs)

#     def _check_en_converged(self, chain_prev, chain_new, en_func, en_thre):
#         for i in range(1, len(chain_prev) - 1):
#             node_prev = chain_prev[i]
#             node_new = chain_new[i]

#             delta_e = np.abs(en_func(node_new) - en_func(node_prev))
#             if delta_e > en_thre:
#                 return False
#         return True

#     def _check_grad_converged(self, chain_prev, chain_new, grad_func, grad_thre):
#         for i in range(1, len(chain_prev) - 1):
#             node_prev = chain_prev[i]
#             node_new = chain_new[i]

#             delta_grad = np.abs(grad_func(node_new) - grad_func(node_prev))

#             if True in delta_grad > grad_thre:
#                 return False
#         return True

#     def _chain_converged(self, chain_previous, new_chain, en_func, en_thre, grad_func, grad_thre):

#         en_converged = self._check_en_converged(
#             chain_prev=chain_previous,
#             chain_new=new_chain,
#             en_func=en_func,
#             en_thre=en_thre,
#         )
#         grad_converged = self._check_grad_converged(
#             chain_prev=chain_previous,
#             chain_new=new_chain,
#             grad_func=grad_func,
#             grad_thre=grad_thre,
#         )

#         return en_converged and grad_converged

#     def remove_chain_folding(self, chain):
#         not_converged = True
#         count = 0
#         points_removed = []
#         while not_converged:
#             # print(f"on count {count}...")
#             new_chain = []
#             for i in range(len(chain)):
#                 if i == 0 or i == len(chain) - 1:
#                     new_chain.append(chain[i])
#                     continue

#                 vec1, vec2 = self._get_vectors(chain, i)
#                 if np.dot(vec1, vec2) > 0:
#                     new_chain.append(chain[i])
#                 else:
#                     points_removed.append(chain[i])

#             new_chain = np.array(new_chain)
#             if self._check_dot_product_converged(new_chain):
#                 not_converged = False
#             chain = new_chain.copy()
#             count += 1
#         return chain

#     def redistribute_chain(self, chain):
#         direction = np.array([next_node.coords - current_node.coords for current_node, next_node in pairwise(chain)])
#         distances = np.linalg.norm(direction, axis=1)
#         tot_dist = np.sum(distances)
#         cumsum = np.cumsum(distances)  # cumulative sum
#         cumsum = np.insert(cumsum, 0, 0)

#         distributed_chain = []
#         for num in np.linspace(0, tot_dist, len(chain)):
#             foobar = self.redistribution_helper(num=num, cum=cumsum, chain=chain)
#             distributed_chain.append(foobar)

#         distributed_chain[0] = chain[0]
#         distributed_chain[-1] = chain[-1]

#         return np.array(distributed_chain)

#     def redistribution_helper(self, num, cum, chain):
#         """
#         num: the distance from first node to return output point to
#         cum: cumulative sums
#         new_chain: chain that we are considering

#         """

#         for ii, ((cum_sum_init, point_start), (cum_sum_end, point_end)) in enumerate(pairwise(zip(cum, chain))):

#             if cum_sum_init < num < cum_sum_end:
#                 direction = point_end - point_start
#                 percentage = (num - cum_sum_init) / (cum_sum_end - cum_sum_init)
#                 point = point_start + (direction * percentage)

#                 return point

#     def _get_vectors(self, chain, i):
#         view = chain[i - 1: i + 2]
#         vec1 = view[1] - view[0]
#         vec2 = view[2] - view[1]
#         return vec1, vec2

#     def _check_dot_product_converged(self, chain):
#         dps = []
#         for i in range(1, len(chain) - 1):
#             vec1, vec2 = self._get_vectors(chain, i)
#             dps.append(np.dot(vec1, vec2) > 0)

#         return all(dps)
