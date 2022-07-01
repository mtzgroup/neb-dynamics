from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import List, Union

import numpy as np
from scipy.signal import argrelextrema
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics.helper_functions import pairwise
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory


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

    @property
    @abstractmethod
    def do_climb(self):
        ...

    @abstractmethod
    def dot_function(self, other):
        ...

    @abstractmethod
    def update_coords(self, coords):
        ...


@dataclass
class Node2D(Node):
    pair_of_coordinates: np.array
    converged: bool = False
    do_climb: bool = False

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

    def copy(self):
        return Node2D(
            pair_of_coordinates=self.pair_of_coordinates,
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array):
        new_node = self.copy()
        new_node.pair_of_coordinates = coords
        return new_node


@dataclass
class Node2D_2(Node):
    pair_of_coordinates: np.array
    converged: bool = False
    do_climb: bool = False

    @property
    def coords(self):
        return self.pair_of_coordinates

    @staticmethod
    def en_func(node):
        x, y = node.coords

        A = np.cos(np.pi * x)
        B = np.cos(np.pi * y)
        C = np.pi * np.exp(-np.pi * x ** 2)
        D = (np.exp(-np.pi * (y - 0.8) ** 2)) + np.exp(-np.pi * (y + 0.8) ** 2)
        return A + B + C * D

    @staticmethod
    def grad_func(node):
        x, y = node.coords
        A_x = (-2 * np.pi ** 2) * (np.exp(-np.pi * (x ** 2)) * x)
        B_x = np.exp(-np.pi * ((y - 0.8) ** 2)) + np.exp(-np.pi * (y + 0.8) ** 2)
        C_x = -np.pi * np.sin(np.pi * x)

        dx = A_x * B_x + C_x

        A_y = np.pi * np.exp(-np.pi * x ** 2)
        B_y = (-2 * np.pi * np.exp(-np.pi * (y - 0.8) ** 2)) * (y - 0.8)
        C_y = -2 * np.pi * np.exp(-np.pi * (y + 0.8) ** 2) * (y + 0.8)
        D_y = -np.pi * np.sin(np.pi * y)

        dy = A_y * (B_y + C_y) + D_y

        return np.array([dx, dy])

    @property
    def energy(self) -> float:
        return self.en_func(self)

    @property
    def gradient(self) -> np.array:
        return self.grad_func(self)

    @staticmethod
    def dot_function(self, other) -> float:
        return np.dot(self, other)

    def copy(self):
        return Node2D_2(
            pair_of_coordinates=self.pair_of_coordinates,
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array):
        new_node = self.copy()
        new_node.pair_of_coordinates = coords
        return new_node

@dataclass
class Node2D_ITM(Node):
    pair_of_coordinates: np.array
    converged: bool = False
    do_climb: bool = False

    @property
    def coords(self):
        return self.pair_of_coordinates

    @staticmethod
    def en_func(node: Node2D_ITM):
        x, y = node.coords
        Ax = 1
        Ay = 1
        return -1*(Ax*np.cos(2*np.pi*x) + Ay*np.cos(2*np.pi*y))

        

    @staticmethod
    def grad_func(node: Node2D_ITM):
        x, y = node.coords
        Ax = 1
        Ay = 1
        dx = 2*Ax*np.pi*np.sin(2*np.pi*x)
        dy = 2*Ay*np.pi*np.sin(2*np.pi*y)
        return np.array([dx, dy])

    @property
    def energy(self) -> float:
        return self.en_func(self)

    @property
    def gradient(self) -> np.array:
        return self.grad_func(self)

    @staticmethod
    def dot_function(self, other: Node2D_ITM) -> float:
        return np.dot(self, other)

    def copy(self):
        return Node2D_ITM(
            pair_of_coordinates=self.pair_of_coordinates,
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array):
        new_node = self.copy()
        new_node.pair_of_coordinates = coords
        return new_node

@dataclass
class Node3D(Node):
    tdstructure: TDStructure
    converged: bool = False
    do_climb: bool = False

    @property
    def coords(self):
        return self.tdstructure.coords

    @staticmethod
    def en_func(node: Node3D):
        res = Node3D.run_xtb_calc(node.tdstructure)
        return res.get_energy()

    @staticmethod
    def grad_func(node: Node3D):
        res = Node3D.run_xtb_calc(node.tdstructure)
        return res.get_gradient()

    @cached_property
    def energy(self):
        return Node3D.run_xtb_calc(self.tdstructure).get_energy()

    @cached_property
    def gradient(self):
        return Node3D.run_xtb_calc(self.tdstructure).get_gradient()

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
        return np.sum(first * second, axis=1).reshape(-1, 1)

    # i want to cache the result of this but idk how caching works
    def run_xtb_calc(tdstruct: TDStructure):
        atomic_numbers = tdstruct.atomic_numbers
        calc = Calculator(
            get_method("GFN2-xTB"),
            numbers=np.array(atomic_numbers),
            positions=tdstruct.coords,
            charge=tdstruct.charge,
            uhf=tdstruct.spinmult - 1,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()
        return res

    def copy(self):
        return Node3D(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct.update_coords(coords=coords)
        return Node3D(
            tdstructure=copy_tdstruct, converged=self.converged, do_climb=self.do_climb
        )


@dataclass
class Chain:
    nodes: List[Node]
    k: Union[List[float], float]
    delta_k: float
    step_size: float

    velocity: np.array

    def neighs_grad_func(
        self, prev_node: Node, current_node: Node, next_node: Node
    ):

        vec_tan_path = self._create_tangent_path(
            prev_node=prev_node, current_node=current_node, next_node=next_node
        )
        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        if not current_node.do_climb:
            pe_grads_nudged = self.get_pe_grad_nudged(
                current_node=current_node, unit_tan_path=unit_tan_path
            )
            spring_forces_nudged_no_k = self.get_force_spring_nudged_no_k(
                prev_node=prev_node,
                current_node=current_node,
                next_node=next_node,
                unit_tan_path=unit_tan_path,
            )

        elif current_node.do_climb:

            pe_grad = current_node.gradient
            pe_along_path_const = current_node.dot_function(pe_grad, unit_tan_path)
            pe_along_path = pe_along_path_const * unit_tan_path

            climbing_grad = 2 * pe_along_path

            pe_grads_nudged = pe_grad - climbing_grad

            zero = np.zeros_like(pe_grad)
            spring_forces_nudged_no_k = zero
            

        else:
            raise ValueError(
                f"current_node.do_climb is not a boolean: {current_node.do_climb=}"
            )

        return pe_grads_nudged, spring_forces_nudged_no_k #, anti_kinking_grads

    def _k_between_nodes(
        self, node0: Node, node1: Node, e_ref: float, k_max: float, e_max: float
    ):
        e_i = max(node1.energy, node0.energy)
        if e_i > e_ref:
            new_k = k_max - self.delta_k * ((e_max - e_i) / (e_max - e_ref))
        elif e_i <= e_ref:
            new_k = k_max - self.delta_k
        return new_k

    def _choose_k(self, k_vals: np.array, springs_forces: np.array):

        bools = (
            springs_forces >= 0
        )  # list of [..1, 0...] so will give index of left or right spring constant
        chosen_ks = []

        for k_pair, ind_choice in zip(k_vals, bools):
            ind_choice = int(ind_choice)

            chosen_ks.append(k_pair[ind_choice])

        return np.array(chosen_ks)

    def compute_k(self, spring_forces, ref_grad):
        new_ks = []
        k_max = max(self.k) if hasattr(self.k, "__iter__") else self.k
        e_ref = max(self.nodes[0].energy, self.nodes[-1].energy)
        e_max = max(self.energies)

        dir_springs_forces = []
        for spring_force_on_node, (prev_node, current_node, next_node) in zip(
            spring_forces, self.iter_triplets()
        ):
            tan_vec = self._create_tangent_path(
                prev_node=prev_node, current_node=current_node, next_node=next_node
            )
            unit_tan = tan_vec / np.linalg.norm(tan_vec)

            k01 = self._k_between_nodes(
                node0=prev_node,
                node1=current_node,
                e_ref=e_ref,
                k_max=k_max,
                e_max=e_max,
            )

            k10 = self._k_between_nodes(
                node0=current_node,
                node1=next_node,
                e_ref=e_ref,
                k_max=k_max,
                e_max=e_max,
            )

            new_ks.append([k01, k10])
            dir_springs_forces.append(
                np.sum(current_node.dot_function(unit_tan, spring_force_on_node))
            )

        new_ks = np.array(new_ks)
        # print(f"\t\tk_vals={new_ks}")
        dir_springs_forces = np.array(dir_springs_forces)

        ks_sub = self._choose_k(k_vals=new_ks, springs_forces=dir_springs_forces)

        # reshape k array
        correct_dimensions = [1 if i > 0 else -1 for i, _ in enumerate(ref_grad.shape)]
        new_ks = ks_sub.reshape(*correct_dimensions)

        self.k = new_ks

    def __getitem__(self, index):
        return self.nodes.__getitem__(index)

    def __len__(self):
        return len(self.nodes)

    def copy(self):
        list_of_nodes = [node.copy() for node in self.nodes]
        chain_copy = Chain(
            nodes=list_of_nodes,
            k=self.k,
            delta_k=self.delta_k,
            step_size=self.step_size,
            velocity=self.velocity,
        )
        return chain_copy

    def iter_triplets(self):
        for i in range(1, len(self.nodes) - 1):
            yield self.nodes[i - 1 : i + 2]

    @classmethod
    def from_traj(cls, traj, k, delta_k, step_size, node_class, velocity=None):
        if velocity == None:
            velocity = np.zeros_like([struct.coords for struct in traj])
        nodes = [node_class(s) for s in traj]
        return Chain(
            nodes, k=k, delta_k=delta_k, step_size=step_size, velocity=velocity
        )

    @classmethod
    def from_list_of_coords(
        cls,
        k,
        list_of_coords: List,
        node_class: Node,
        delta_k: float,
        step_size: float,
        velocity=None,
    ) -> Chain:

        if velocity == None:
            velocity = np.zeros_like([c for c in list_of_coords])
        nodes = [node_class(point) for point in list_of_coords]
        return cls(
            nodes=nodes, k=k, delta_k=delta_k, step_size=step_size, velocity=velocity
        )

    @cached_property
    def energies(self) -> np.array:
        return np.array([node.energy for node in self.nodes])

    @cached_property
    def gradients(self) -> np.array:
        pe_grads_nudged = []
        spring_forces_nudged_no_k = []
        # anti_kinking_grads = []
        for prev_node, current_node, next_node in self.iter_triplets():
            (
                pe_grad_nudged,
                spring_force_nudged_no_k) = self.neighs_grad_func(
                prev_node=prev_node, current_node=current_node, next_node=next_node
            )


            pe_grads_nudged.append(pe_grad_nudged)
            spring_forces_nudged_no_k.append(spring_force_nudged_no_k)
            # anti_kinking_grads.append(anti_kinking_grad)

        pe_grads_nudged = np.array(pe_grads_nudged)
        spring_forces_nudged_no_k = np.array(spring_forces_nudged_no_k)
        # anti_kinking_grads = np.array(anti_kinking_grads)

        self.compute_k(spring_forces=spring_forces_nudged_no_k, ref_grad=pe_grads_nudged)
        
        grads = (pe_grads_nudged - self.k * spring_forces_nudged_no_k) #+ self.k * anti_kinking_grads

        zero = np.zeros_like(grads[0])
        grads = np.insert(grads, 0, zero, axis=0)
        grads = np.insert(grads, len(grads), zero, axis=0)

        # remove rotations and translations
        if grads.shape[1] >= 3:  # if we have at least 3 atoms
            grads[:, 0, :] = 0  # this atom cannot move
            grads[:, 1, :2] = 0  # this atom can only move in a line
            grads[:, 2, :1] = 0  # this atom can only move in a plane

        return grads

    @property
    def unit_tangents(self):
        tan_list = []
        for prev_node, current_node, next_node in self.iter_triplets():
            tan_vec = self._create_tangent_path(
                prev_node=prev_node, current_node=current_node, next_node=next_node
            )
            unit_tan = tan_vec / np.linalg.norm(tan_vec)
            tan_list.append(unit_tan)

        return tan_list

    @property
    def coordinates(self) -> np.array:

        return np.array([node.coords for node in self.nodes])

    @cached_property
    def displacements(self):

        grads = self.gradients

        correct_dimensions = [1 if i > 0 else -1 for i, _ in enumerate(grads.shape)]
        disps = []

        for grad, (prev_node, current_node, next_node) in zip(
            grads, self.iter_triplets()
        ):
            disp = self.node_displacement(
                current_node=current_node,
                prev_node=prev_node,
                next_node=next_node,
                grad=grad,
            )
            disps.append(disp)

        disps = np.array(disps).reshape(*correct_dimensions)
        return disp

    def node_displacement(
        self, current_node: Node, prev_node: Node, next_node: Node, grad: np.array
    ):
        from neb_dynamics import ALS

        if not current_node.converged:
            dr = ALS.ArmijoLineSearch(
                node=current_node,
                t=self.step_size,
                grad=grad,
                next_node=next_node,
                prev_node=prev_node,
                grad_func=self.neighs_grad_func,
                beta=0.5,
                f=current_node.en_func,
                alpha=0.01,
                k=max(self.k),
            )
            return dr
        else:
            return 0.0

    def _create_tangent_path(
        self, prev_node: Node, current_node: Node, next_node: Node
    ):
        en_2 = next_node.energy
        en_1 = current_node.energy
        en_0 = prev_node.energy
        if en_2 > en_1 and en_1 > en_0:
            return next_node.coords - current_node.coords
        elif en_2 < en_1 and en_1 < en_0:
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
                return next_node.coords - current_node.coords
                # raise ValueError(
                #     f"Energies adjacent to current node are identical. {en_2=} {en_0=}"
                # )

            return tan_vec

    def _get_nudged_pe_grad(self, node, unit_tangent):
        pe_grad = node.gradient
        pe_grad_nudged_const = node.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent

        return pe_grad_nudged

    def _get_anti_kink_switch_func(
        self, prev_node, current_node, next_node
    ):
        # ANTI-KINK FORCE
        vec_2_to_1 = next_node.coords - current_node.coords
        vec_1_to_0 = current_node.coords - prev_node.coords
        cos_phi = current_node.dot_function(vec_2_to_1, vec_1_to_0) / (
            np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0)
        )

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))
        return f_phi

    def get_pe_grad_nudged(self, current_node: Node, unit_tan_path):

        pe_grad_nudged = self._get_nudged_pe_grad(
            node=current_node, unit_tangent=unit_tan_path
        )
        return pe_grad_nudged

    def get_force_spring_nudged_no_k(
        self,
        prev_node: Node,
        current_node: Node,
        next_node: Node,
        unit_tan_path: np.array,
    ):
        force_spring = np.linalg.norm(next_node.coords - current_node.coords) - np.linalg.norm(current_node.coords - prev_node.coords)
        return force_spring*unit_tan_path

    # def get_anti_kinking_grad(
    #     self,
    #     prev_node: Node,
    #     current_node: Node,
    #     next_node: Node,
    #     unit_tan_path: np.array,
    # ):
    #     f_phi = self._get_anti_kink_switch_func(
    #         prev_node=prev_node,
    #         current_node=current_node,
    #         next_node=next_node
    #     )
        
    #     force_spring = self.get_force_spring_no_k(
    #         prev_node=prev_node, current_node=current_node, next_node=next_node
    #     )
    #     force_spring_nudged = self.get_force_spring_nudged_no_k(
    #         prev_node=prev_node,
    #         current_node=current_node,
    #         next_node=next_node,
    #         unit_tan_path=unit_tan_path,
    #     )

    #     anti_kinking_grad = -1 * f_phi *  (force_spring - force_spring_nudged)
    #     if np.abs(f_phi - 1) <= 0.1:
    #          print(f"\t\t\t{anti_kinking_grad=}")
        
        
    #     return anti_kinking_grad*0


@dataclass
class NEB:
    initial_chain: Chain

    redistribute: bool = False
    remove_folding: bool = False
    climb: bool = False
    en_thre: float = 0.001
    grad_thre_per_atom: float = 0.001
    mag_grad_thre: float = 0.01
    max_steps: float = 1000

    vv_force_thre: float = 1.0

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)

    @property
    def grad_thre(self):
        n_atoms = self.initial_chain[0].coords.shape[0]
        return self.grad_thre_per_atom*(n_atoms)


    def do_velvel(self, chain: Chain):
        max_force_on_node = max([np.linalg.norm(grad) for grad in chain.gradients])
        return max_force_on_node < self.vv_force_thre

    def set_climbing_nodes(self, chain: Chain):
        # reset node convergence
        for node in chain:
            node.converged = False

        inds_maxima = argrelextrema(chain.energies, np.greater, order=2)[0]
        print(f"----->Setting {len(inds_maxima)} nodes to climb")

        for ind in inds_maxima:
            chain[ind].do_climb = True

    def optimize_chain(self):
        nsteps = 1
        chain_previous = self.initial_chain.copy()

        while nsteps < self.max_steps + 1:

            max_grad_val = np.max([np.linalg.norm(grad) for grad in chain_previous.gradients])
            if max_grad_val<= 2*self.grad_thre and self.climb: 
                self.set_climbing_nodes(chain=chain_previous)
                self.climb = False

            new_chain = self.update_chain(chain=chain_previous)
            print(
                f"step {nsteps} // max |gradient| {np.max([np.linalg.norm(grad) for grad in new_chain.gradients])}"
            )

            self.chain_trajectory.append(new_chain)
            self.gradient_trajectory.append(new_chain.gradients)

            if self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
                print(f"Chain converged!")
                original_chain_len = len(new_chain)

                if self.remove_folding:
                    new_chain = self.remove_chain_folding(chain=new_chain.copy())
                    self.chain_trajectory.append(new_chain)

                if self.redistribute:

                    new_chain = self.redistribute_chain(
                        chain=new_chain.copy(),
                        requested_length_of_chain=original_chain_len,
                    )
                    self.chain_trajectory.append(new_chain)

                if self.climb:
                    climbing_chain = new_chain.copy()
                    new_chain = self.climb_chain(chain=climbing_chain.copy())

                self.optimized = new_chain
                return
            chain_previous = new_chain.copy()
            nsteps += 1

        new_chain = self.update_chain(chain=chain_previous)
        if not self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
            raise NoneConvergedException(
                trajectory=self.chain_trajectory,
                msg=f"Chain did not converge at step {nsteps}",
                obj=self,
            )

    def get_chain_velocity(self, chain: Chain) -> np.array:
        prev_velocity = chain.velocity

        step = self.grad_thre/10 # make the step size rel. to threshold we want

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
            new_chain_coordinates = (
                chain.coordinates - chain.gradients * chain.displacements
            )

        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = Chain(
            new_nodes,
            k=chain.k,
            delta_k=chain.delta_k,
            step_size=chain.step_size,
            velocity=chain.velocity,
        )
        if do_vv:
            new_chain.velocity = velocity

        return new_chain

    def _update_node_convergence(self, chain: Chain, indices: np.array) -> None:
        for ind in indices:
            node = chain[ind]
            node.converged = True

    def _check_en_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
        differences = np.abs(chain_new.energies - chain_prev.energies)

        indices_converged = np.where(differences < self.en_thre)

        return np.all(differences < self.en_thre), indices_converged[0]

    def _check_grad_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
        delta_grad = np.abs(chain_prev.gradients - chain_new.gradients)
        mag_grad = np.array([np.linalg.norm(grad) for grad in chain_new.gradients])

        delta_converged = np.where(delta_grad < self.grad_thre)
        mag_converged = np.where(mag_grad < self.mag_grad_thre)

        return (
            np.all(delta_grad < self.grad_thre)
            and np.all(mag_grad < self.mag_grad_thre),
            np.intersect1d(delta_converged[0], mag_converged[0]),
        )

    def _chain_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
        # en_bool, en_converged_indices = self._check_en_converged(
        #     chain_prev=chain_prev, chain_new=chain_new
        # )

        # grad_bool, grad_converged_indices = self._check_grad_converged(
        #     chain_prev=chain_prev, chain_new=chain_new
        # )

        # converged_node_indices = np.intersect1d(
        #     en_converged_indices, grad_converged_indices
        # )

        converged_nodes_indices = [
            i
            for i, bool in enumerate(
                np.linalg.norm(grad) <= self.grad_thre for grad in chain_new.gradients
            )
            if bool
        ]
        print(f"\t{len(converged_nodes_indices)} nodes have converged")
        # print(f"\t{converged_nodes_indices} nodes have converged")

        # self._update_node_convergence(chain=chain_new, indices=converged_nodes_indices)
        # return len(converged_node_indices) == len(chain_new.nodes)

        return np.all(
            [np.linalg.norm(grad) <= self.grad_thre for grad in chain_new.gradients]
        )

        # return en_bool and grad_bool

    def remove_chain_folding(self, chain: Chain) -> Chain:
        not_converged = True
        count = 0
        points_removed = []
        while not_converged:
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
            new_chain = Chain(nodes=new_chain, k=chain.k)
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

        return Chain(distributed_chain, k=chain.k)

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

    def write_to_disk(self, fp: Path):
        out_traj = Trajectory([node.tdstructure for node in self.optimized.nodes])
        out_traj.write_trajectory(fp)
