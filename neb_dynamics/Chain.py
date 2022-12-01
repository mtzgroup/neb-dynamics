from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from retropaths.abinitio.trajectory import Trajectory

from neb_dynamics.helper_functions import quaternionrmsd
from neb_dynamics.Node import Node
from neb_dynamics.Node3D import Node3D
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

import multiprocessing as mp


@dataclass
class Chain:
    nodes: List[Node]
    k: Union[List[float], float]
    delta_k: float = 0
    step_size: float = 1
    velocity: np.array = np.zeros(1)
    node_class: Node = Node3D

    @classmethod
    def from_xyz(cls, fp: Path, k=0.1, delta_k=0, step_size=1, velocity=np.zeros(1), node_class=Node3D):
        traj = Trajectory.from_xyz(fp)
        chain = cls.from_traj(traj, k=k, delta_k=delta_k, step_size=step_size, velocity=velocity, node_class=node_class)
        return chain

    @property
    def integrated_path_length(self):

        cum_sums = [0]

        int_path_len = [0]
        for i, frame in enumerate(self.nodes):
            if i == len(self.nodes) - 1:
                continue
            next_frame = self.nodes[i + 1]
            dist_vec = next_frame.coords - frame.coords
            cum_sums.append(cum_sums[-1] + np.linalg.norm(dist_vec))
            # proj = (frame.dot_function(dist_vec, endpoint_vec) / frame.dot_function(endpoint_vec, endpoint_vec))*endpoint_vec

            # proj_dist = np.linalg.norm(proj)
            # int_path_len.append(int_path_len[-1]+proj_dist)
        cum_sums = np.array(cum_sums)
        int_path_len = cum_sums / cum_sums[-1]
        return np.array(int_path_len)

    def _k_between_nodes(self, node0: Node, node1: Node, e_ref: float, k_max: float, e_max: float):
        e_i = max(node1.energy, node0.energy)
        if e_i > e_ref:
            new_k = k_max - self.delta_k * ((e_max - e_i) / (e_max - e_ref))
        elif e_i <= e_ref:
            new_k = k_max - self.delta_k
        return new_k

    def plot_chain(self):
        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))
        plt.plot(self.integrated_path_length, (self.energies - self.energies[0]) * 627.5, "o--", label="neb")
        plt.ylabel("Energy (kcal/mol)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.show()

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

    def iter_triplets(self) -> list[list[Node]]:
        for i in range(1, len(self.nodes) - 1):
            yield self.nodes[i - 1: i + 2]

    @classmethod
    def from_traj(cls, traj, k, delta_k, step_size, node_class, velocity=None):
        if velocity is None:
            velocity = np.zeros_like([struct.coords for struct in traj])
        nodes = [node_class(s) for s in traj]
        return Chain(nodes, k=k, delta_k=delta_k, step_size=step_size, velocity=velocity)

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

        if velocity is None:
            velocity = np.zeros_like([c for c in list_of_coords])
        nodes = [node_class(point) for point in list_of_coords]
        return cls(nodes=nodes, k=k, delta_k=delta_k, step_size=step_size, velocity=velocity)

    @property
    def path_distances(self):
        dist = []
        for i in range(len(self.nodes)):
            if i == 0:
                continue
            start = self.nodes[i - 1]
            end = self.nodes[i]

            dist.append(quaternionrmsd(start.coords, end.coords))

        return np.array(dist)

    @cached_property
    def work(self) -> float:
        ens = self.energies
        ens -= ens[0]

        works = np.abs(ens[1:] * self.path_distances)
        tot_work = works.sum()
        return tot_work

    @cached_property
    def energies(self) -> np.array:
        return np.array([node.energy for node in self.nodes])

    def neighs_grad_func(self, prev_node: Node, current_node: Node, next_node: Node):

        vec_tan_path = self._create_tangent_path(prev_node=prev_node, current_node=current_node, next_node=next_node)
        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        pe_grad = current_node.gradient

        if not current_node.do_climb:
            pe_grads_nudged = current_node.get_nudged_pe_grad(unit_tan_path, gradient=pe_grad)
            spring_forces_nudged = self.get_force_spring_nudged(
                prev_node=prev_node,
                current_node=current_node,
                next_node=next_node,
                unit_tan_path=unit_tan_path,
            )

        elif current_node.do_climb:

            pe_along_path_const = current_node.dot_function(pe_grad, unit_tan_path)
            pe_along_path = pe_along_path_const * unit_tan_path

            climbing_grad = 2 * pe_along_path

            pe_grads_nudged = pe_grad - climbing_grad

            zero = np.zeros_like(pe_grad)
            spring_forces_nudged = zero
        else:
            raise ValueError(f"current_node.do_climb is not a boolean: {current_node.do_climb=}")

        return pe_grads_nudged, spring_forces_nudged  # , anti_kinking_grads

    def pe_grads_spring_forces_nudged(self):
        pe_grads_nudged = []
        spring_forces_nudged = []
        # anti_kinking_grads = []
        for prev_node, current_node, next_node in self.iter_triplets():
            pe_grad_nudged, spring_force_nudged = self.neighs_grad_func(
                prev_node=prev_node,
                current_node=current_node,
                next_node=next_node,
            )

            # anti_kinking_grads.append(anti_kinking_grad)
            if not current_node.converged:
                pe_grads_nudged.append(pe_grad_nudged)
                spring_forces_nudged.append(spring_force_nudged)
            else:
                zero = np.zeros_like(pe_grad_nudged)
                pe_grads_nudged.append(zero)
                spring_forces_nudged.append(zero)

        pe_grads_nudged = np.array(pe_grads_nudged)
        spring_forces_nudged = np.array(spring_forces_nudged)
        return pe_grads_nudged, spring_forces_nudged

    def get_maximum_grad_magnitude(self):
        return np.max([np.linalg.norm(grad) for grad in self.gradients])

    @staticmethod
    def calc_xtb_ene_grad_from_input_tuple(tuple):
        atomic_numbers, coords_bohr, charge, spinmult = tuple

        calc = Calculator(
            get_method("GFN2-xTB"),
            numbers=np.array(atomic_numbers),
            positions=coords_bohr,
            charge=charge,
            uhf=spinmult - 1,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()

        return res.get_energy(), res.get_gradient()*BOHR_TO_ANGSTROMS

    def calculate_energy_and_gradients_parallel(self):
        iterator = ((n.tdstructure.atomic_numbers,  n.tdstructure.coords_bohr, n.tdstructure.charge, n.tdstructure.spinmult) for n in self.nodes)
        with mp.Pool() as p:
            ene_gradients = p.map(self.calc_xtb_ene_grad_from_input_tuple, iterator)
        return ene_gradients

    @cached_property
    def gradients(self) -> np.array:
        energy_gradient_tuples = self.calculate_energy_and_gradients_parallel()
        for (ene, grad), node in zip(energy_gradient_tuples, self.nodes):
            node._cached_energy = ene
            node._cached_gradient = grad
        pe_grads_nudged, spring_forces_nudged = self.pe_grads_spring_forces_nudged()

        # anti_kinking_grads = np.array(anti_kinking_grads)

        grads = pe_grads_nudged - spring_forces_nudged  # + self.k * anti_kinking_grads

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
            tan_vec = self._create_tangent_path(prev_node=prev_node, current_node=current_node, next_node=next_node)
            unit_tan = tan_vec / np.linalg.norm(tan_vec)
            tan_list.append(unit_tan)

        return tan_list

    @property
    def coordinates(self) -> np.array:

        return np.array([node.coords for node in self.nodes])

    def _create_tangent_path(self, prev_node: Node, current_node: Node, next_node: Node):
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

            tau_plus = next_node.coords - current_node.coords
            tau_minus = current_node.coords - prev_node.coords
            if en_2 > en_0:
                tan_vec = deltaV_max * tau_plus + deltaV_min * tau_minus
            elif en_2 < en_0:
                tan_vec = deltaV_min * tau_plus + deltaV_max * tau_minus

            else:
                return 0.5 * (tau_minus + tau_plus)
                # raise ValueError(
                #     f"Energies adjacent to current node are identical. {en_2=} {en_0=}"
                # )

            return tan_vec

    def _get_anti_kink_switch_func(self, prev_node, current_node, next_node):
        # ANTI-KINK FORCE
        vec_2_to_1 = next_node.coords - current_node.coords
        vec_1_to_0 = current_node.coords - prev_node.coords
        cos_phi = current_node.dot_function(vec_2_to_1, vec_1_to_0) / (np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0))

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))
        return f_phi

    def get_force_spring_nudged(
        self,
        prev_node: Node,
        current_node: Node,
        next_node: Node,
        unit_tan_path: np.array,
    ):

        k_max = max(self.k) if hasattr(self.k, "__iter__") else self.k
        e_ref = max(self.nodes[0].energy, self.nodes[-1].energy)
        e_max = max(self.energies)
        # print(f"***{e_max=}//{e_ref=}//{k_max=}")

        k01 = self._k_between_nodes(
            node0=prev_node,
            node1=current_node,
            e_ref=e_ref,
            k_max=k_max,
            e_max=e_max,
        )

        k12 = self._k_between_nodes(
            node0=current_node,
            node1=next_node,
            e_ref=e_ref,
            k_max=k_max,
            e_max=e_max,
        )

        # print(f"***{k12=} // {k01=}")

        force_spring = k12 * np.linalg.norm(next_node.coords - current_node.coords) - k01 * np.linalg.norm(current_node.coords - prev_node.coords)
        return force_spring * unit_tan_path
