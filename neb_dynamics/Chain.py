from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from retropaths.abinitio.trajectory import Trajectory

from neb_dynamics.Node import Node
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.Inputs import ChainInputs
from neb_dynamics.helper_functions import RMSD, get_mass, _get_ind_minima, _get_ind_maxima, linear_distance, qRMSD_distance, pairwise, get_nudged_pe_grad


from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method
from pathlib import Path

import scipy


@dataclass
class Chain:
    nodes: List[Node]
    parameters: ChainInputs
    
    def __post_init__(self):
        if not hasattr(self.parameters, "velocity"):
            self._zero_velocity()
            
            
    def _zero_velocity(self):
        if self[0].is_a_molecule:
            self.parameters.velocity = np.zeros(shape=(len(self.nodes), len(self.nodes[0].coords), 3))
        else:
            self.parameters.velocity = np.zeros(shape=(len(self.nodes), len(self.nodes[0].coords)))

    @property
    def n_atoms(self):
        return self.coordinates[0].shape[0]

    @classmethod
    def from_xyz(cls, fp: Path, parameters: ChainInputs):
        if isinstance(fp, str): fp = Path(fp)
        traj = Trajectory.from_xyz(fp)
        chain = cls.from_traj(traj, parameters=parameters)
        energies_fp = fp.parent / Path(str(fp.stem)+".energies")
        grad_path = fp.parent / Path(str(fp.stem)+".gradients")
        grad_shape_path = fp.parent / "grad_shapes.txt"
        
        if energies_fp.exists() and grad_path.exists() and grad_shape_path.exists():
            energies = np.loadtxt(energies_fp)
            gradients_flat = np.loadtxt(grad_path)
            gradients_shape = np.loadtxt(grad_shape_path,dtype=int)
            
            gradients = gradients_flat.reshape(gradients_shape)
            
            for node,(ene, grad) in zip(chain.nodes, zip(energies, gradients)):
                node._cached_energy = ene
                node._cached_gradient = grad
        return chain
    
    @classmethod
    def from_list_of_chains(cls, list_of_chains, parameters):
        nodes = []
        for chain in list_of_chains:
            nodes.extend(chain.nodes)
        return cls(nodes=nodes, parameters=parameters)


    def _distance_to_chain(self, other_chain: Chain):
        chain1 = self
        chain2 = other_chain

        distances = []
        

        for node1, node2 in zip(chain1.nodes, chain2.nodes):
            if node1.coords.shape[0] > 2:
                dist,_ = RMSD(node1.coords, node2.coords)
            else:
                dist = np.linalg.norm(node1.coords - node2.coords)
            distances.append(dist)

        return sum(distances) / len(chain1)
    
    def _tangent_correlations(self, other_chain: Chain):
        chain1_vec = np.array(self.unit_tangents).flatten()
        chain2_vec = np.array(other_chain.unit_tangents).flatten()
        projector = np.dot(chain1_vec, chain2_vec)
        normalization = np.dot(chain1_vec, chain1_vec)
        
        return projector / normalization
    
    def _gradient_correlation(self, other_chain: Chain):
        
        chain1_vec = np.array(self.gradients).flatten()
        chain1_vec = chain1_vec / np.linalg.norm(chain1_vec)
        
        chain2_vec = np.array(other_chain.gradients).flatten()
        chain2_vec = chain2_vec / np.linalg.norm(chain2_vec)
        
        projector = np.dot(chain1_vec, chain2_vec)
        normalization = np.dot(chain1_vec, chain1_vec)
        
        return projector / normalization
    
    def _gradient_delta_mags(self, other_chain: Chain):
        
        chain1_vec = np.array(self.gradients).flatten()
        chain2_vec = np.array(other_chain.gradients ).flatten()
        diff = np.linalg.norm(chain2_vec - chain1_vec)
        normalization = self.n_atoms * len(self.nodes)
        
        return diff / normalization


    def _get_mass_weighed_coords(self):
        traj = self.to_trajectory()
        coords = traj.coords
        weights = np.array([np.sqrt(get_mass(s)) for s in traj.symbols]) 
        weights = weights  / sum(weights)
        mass_weighed_coords = coords  * weights.reshape(-1,1)
        return mass_weighed_coords


    @property
    def _path_len_coords(self):
        if self.nodes[0].is_a_molecule:
            coords = self._get_mass_weighed_coords()
        else:
            coords = self.coordinates
        return coords
    
    def _path_len_dist_func(self, coords1, coords2):
        if self.nodes[0].is_a_molecule:
            return qRMSD_distance(coords1, coords2)
        else:
            return linear_distance(coords1, coords2)

    @property
    def integrated_path_length(self):
        coords = self._path_len_coords
        cum_sums = [0]
        int_path_len = [0]
        for i, frame_coords in enumerate(coords):
            if i == len(coords) - 1:
                continue
            next_frame = coords[i + 1]
            distance = self._path_len_dist_func(frame_coords, next_frame)
            cum_sums.append(cum_sums[-1] + distance)

        cum_sums = np.array(cum_sums)
        int_path_len = cum_sums / cum_sums[-1]
        return np.array(int_path_len)

    def _k_between_nodes(
        self, node0: Node, node1: Node, e_ref: float, k_max: float, e_max: float
    ):
        e_i = max(node1.energy, node0.energy)
        if e_i > e_ref:
            new_k = k_max - self.parameters.delta_k * ((e_max - e_i) / (e_max - e_ref))
        elif e_i <= e_ref:
            new_k = k_max - self.parameters.delta_k
        return new_k

    def plot_chain(self):
        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))
        plt.plot(
            self.integrated_path_length,
            (self.energies - self.energies[0]) * 627.5,
            "o--",
            label="neb",
        )
        plt.ylabel("Energy (kcal/mol)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.show()

    def __getitem__(self, index):
        return self.nodes.__getitem__(index)

    def __len__(self):
        return len(self.nodes)

    def insert(self, index, node):
        self.nodes.insert(index, node)

    def append(self, node):
        self.nodes.append(node)

    def copy(self):
        list_of_nodes = [node.copy() for node in self.nodes]
        chain_copy = Chain(nodes=list_of_nodes, parameters=self.parameters)
        return chain_copy

    def iter_triplets(self) -> list[list[Node]]:
        for i in range(1, len(self.nodes) - 1):
            yield self.nodes[i - 1 : i + 2]

    @classmethod
    def from_traj(cls, traj: Trajectory, parameters: ChainInputs):
        nodes = [parameters.node_class(s) for s in traj]
        return Chain(nodes, parameters=parameters)

    @classmethod
    def from_list_of_coords(
        cls, list_of_coords: List, parameters: ChainInputs
    ) -> Chain:
        nodes = [parameters.node_class(point) for point in list_of_coords]
        return cls(nodes=nodes, parameters=parameters)

    @property
    def path_distances(self):
        dist = []
        for i in range(len(self.nodes)):
            if i == 0:
                continue
            start = self.nodes[i - 1]
            end = self.nodes[i]

            dist.append(self.quaternionrmsd(start.coords, end.coords))

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
    
    
    @property
    def energies_kcalmol(self) -> np.array:
        return (self.energies - self.energies[0])*627.5

    def neighs_grad_func(self, prev_node: Node, current_node: Node, next_node: Node):

        vec_tan_path = self._create_tangent_path(
            prev_node=prev_node, current_node=current_node, next_node=next_node
        )
        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        pe_grad = current_node.gradient
        
        # remove rotations and translations
        if prev_node.is_a_molecule:
            if pe_grad.shape[1] >= 3:  # if we have at least 3 atoms
                pe_grad[0, :] = 0  # this atom cannot move
                pe_grad[1, :2] = 0  # this atom can only move in a line
                pe_grad[2, :1] = 0  # this atom can only move in a plane
        
        
        

        if not current_node.do_climb:
            pe_grads_nudged = current_node.get_nudged_pe_grad(
                unit_tan_path, gradient=pe_grad
            )
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
            raise ValueError(
                f"current_node.do_climb is not a boolean: {current_node.do_climb=}"
            )

        return pe_grads_nudged, spring_forces_nudged 

    def pe_grads_spring_forces_nudged(self):
        pe_grads_nudged = []
        spring_forces_nudged = []
        
        for prev_node, current_node, next_node in self.iter_triplets():
            pe_grad_nudged, spring_force_nudged = self.neighs_grad_func(
                prev_node=prev_node,
                current_node=current_node,
                next_node=next_node,
            )
            if self.parameters.node_freezing:
                if not current_node.converged:
                    pe_grads_nudged.append(pe_grad_nudged)
                    spring_forces_nudged.append(spring_force_nudged)
                else:
                    zero = np.zeros_like(pe_grad_nudged)
                    pe_grads_nudged.append(zero)
                    spring_forces_nudged.append(zero)
            else:
                pe_grads_nudged.append(pe_grad_nudged)
                spring_forces_nudged.append(spring_force_nudged)

        pe_grads_nudged = np.array(pe_grads_nudged)
        spring_forces_nudged = np.array(spring_forces_nudged)
        return pe_grads_nudged, spring_forces_nudged

    def get_maximum_grad_magnitude(self):
        return np.max([np.amax(np.abs(grad)) for grad in self.gradients])
    
    def get_maximum_gperp(self):
        gperp, gspring = self.pe_grads_spring_forces_nudged()
        max_gperps = []
        for gp, node in zip(gperp, self):            
            if not node.converged:
                max_gperps.append(np.amax(np.abs(gp)))
        return np.max(max_gperps)
    
    def get_maximum_rms_grad(self):
        return np.max([np.sqrt(np.mean(np.square(grad.flatten())) / len(grad.flatten())) for grad in self.gradients])

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

        return res.get_energy(), res.get_gradient() * BOHR_TO_ANGSTROMS


    # @cached_property
    @property
    def gradients(self) -> np.array:
        
        all_grads = [node._cached_gradient for node in self.nodes]
        
        if not np.all([g is not None for g in all_grads]):
            if self.parameters.do_parallel:
                energy_gradient_tuples = self.parameters.node_class.calculate_energy_and_gradients_parallel(chain=self)
            else:
                energies = [node.energy for node in self.nodes]
                gradients = [node.gradient for node in self.nodes]
                energy_gradient_tuples = list(zip(energies, gradients))

            for (ene, grad), node in zip(energy_gradient_tuples, self.nodes):
                node._cached_energy = ene
                node._cached_gradient = grad
        
        pe_grads_nudged, spring_forces_nudged = self.pe_grads_spring_forces_nudged()

        grads = (
            pe_grads_nudged - spring_forces_nudged
        )  # + self.parameters.k * anti_kinking_grads

        # add chain bias if relevant
        if self.parameters.do_chain_biasing:
            tans = self.unit_tangents
            
            # cb = ChainBiaser(reference_chain=n_ref.optimized, amplitude=self.parameters.amp, sigma=self.parameters.sig, distance_func=self.parameters.distance_func)
            bias_grads = self.parameters.cb.grad_chain_bias(self)
            proj_grads = np.array([get_nudged_pe_grad(tan, grad) for tan,grad in zip(tans, bias_grads)])
            
            grads += proj_grads

        zero = np.zeros_like(grads[0])
        grads = np.insert(grads, 0, zero, axis=0)
        grads = np.insert(grads, len(grads), zero, axis=0)

        # # remove rotations and translations
        # if grads.shape[1] >= 3:  # if we have at least 3 atoms
        #     grads[:, 0, :] = 0  # this atom cannot move
        #     grads[:, 1, :2] = 0  # this atom can only move in a line
        #     grads[:, 2, :1] = 0  # this atom can only move in a plane


        # zero all nodes that have converged 
        for (i, grad), node in zip(enumerate(grads), self.nodes):
            if node.converged:
                grads[i] = grad*0

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
        cos_phi = current_node.dot_function(vec_2_to_1, vec_1_to_0) / (
            np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0)
        )

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))
        return f_phi

    def get_force_spring_nudged(
        self,
        prev_node: Node,
        current_node: Node,
        next_node: Node,
        unit_tan_path: np.array,
    ):

        k_max = (
            max(self.parameters.k)
            if hasattr(self.parameters.k, "__iter__")
            else self.parameters.k
        )
        e_ref = max(self.nodes[0].energy, self.nodes[len(self.nodes)-1].energy)
        e_max = max(self.energies)

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

        force_spring = k12 * np.linalg.norm(
            next_node.coords - current_node.coords
        ) - k01 * np.linalg.norm(current_node.coords - prev_node.coords)
        return force_spring * unit_tan_path

    def to_trajectory(self):
        t = Trajectory([n.tdstructure for n in self.nodes])
        return t
    
    def is_elem_step(self):
        chain = self.copy()
        if len(self) <= 1:
            return True

        conditions = {}
        is_concave = self._chain_is_concave()
        conditions['concavity'] = is_concave
        if not is_concave:
            return False, "minima"
        
        
        r,p = self._approx_irc()
        
        cases = [
            r.is_identical(chain[0]) and p.is_identical(chain[-1]), # best case
            r.is_identical(chain[0]) and p.is_identical(chain[0]), # both collapsed to reactants
            r.is_identical(chain[-1]) and p.is_identical(chain[-1]) # both collapsed to products
        ]
        
        minimizing_gives_endpoints = any(cases) 
        conditions['irc'] = minimizing_gives_endpoints

        split_method = self._select_split_method(conditions)
        elem_step = True if split_method is None else False
        return elem_step, split_method

    def _chain_is_concave(self):
        ind_minima = _get_ind_minima(self)
        minima_present =  len(ind_minima) != 0
        if minima_present:
            minimas_is_r_or_p = []
            for i in ind_minima:
                opt =  self[i].do_geometry_optimization()
                minimas_is_r_or_p.append(
                    opt.is_identical(self[0]) or opt.is_identical(self[-1]) 
                    )
            
            print(f"\n{minimas_is_r_or_p=}\n")
            return all(minimas_is_r_or_p)
            
        else:
            return True

    def _approx_irc(self, index=None):
        chain = self.copy()
        if index is None:
            arg_max = np.argmax(chain.energies)
        else:
            arg_max = index
            
        if arg_max == len(chain)-1 or arg_max == 0: # monotonically changing function, 
            return chain[0], chain[len(chain)-1]

        candidate_r = chain[arg_max - 1]
        candidate_p = chain[arg_max + 1]
        r = candidate_r.do_geometry_optimization()
        p = candidate_p.do_geometry_optimization()
        return r, p

    def _select_split_method(self, conditions: dict):
        all_conditions_met = all([val for key,val in conditions.items()])
        if all_conditions_met: 
            return None

        if conditions['concavity'] is False: # prioritize the minima condition
            return 'minima'
        elif conditions['irc'] is False:
            return 'maxima'
        
        
    def write_ene_info_to_disk(self, fp):
        ene_path = fp.parent / Path(str(fp.stem)+".energies")
        grad_path = fp.parent / Path(str(fp.stem)+".gradients")
        grad_shape_path = fp.parent / "grad_shapes.txt"
        
        np.savetxt(ene_path, self.energies)
        # np.savetxt(grad_path, self.gradients.flatten())
        np.savetxt(grad_path, np.array([node.gradient for node in self.nodes]).flatten())
        np.savetxt(grad_shape_path, self.gradients.shape)
        
    def write_to_disk(self, fp: Path):
        if isinstance(fp, str):
            fp = Path(fp)
        
        if self.nodes[0].is_a_molecule:
            traj = self.to_trajectory()
            traj.write_trajectory(fp)
            
            self.write_ene_info_to_disk(fp)
            
        else:
            raise NotImplementedError("Cannot write 2D chains yet.")

    def get_ts_guess(self):
        ind_ts_guess = self.energies.argmax()
        return self[ind_ts_guess].tdstructure
    
    def get_eA_chain(self):
        eA = max(self.energies_kcalmol)
        return eA
