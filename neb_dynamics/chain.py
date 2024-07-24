from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np


# from retropaths.abinitio.trajectory import Trajectory
# from neb_dynamics.trajectory import Trajectory
# from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.errors import EnergiesNotComputedError, GradientsNotComputedError
from neb_dynamics.geodesic_interpolation.fileio import write_xyz


from neb_dynamics.nodes.node import Node, StructureNode
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.fakeoutputs import FakeQCIOResults, FakeQCIOOutput
from neb_dynamics.helper_functions import (
    linear_distance,
    qRMSD_distance,
)


@dataclass
class Chain:
    nodes: List[Node]
    parameters: ChainInputs = ChainInputs()

    _cached_chain_bias: np.array = None

    def __post_init__(self):
        if not hasattr(self, "velocity"):
            self._zero_velocity()

    @property
    def n_atoms(self) -> int:
        """
        return number of atoms in system
        """
        return self.coordinates[0].shape[0]

    @classmethod
    def from_xyz(cls, fp: Union[Path, str], parameters: ChainInputs,
                 charge: int = 0, spinmult: int = 1) -> Chain:
        """
        Reads in a chain from an xyz file containing a list of structures.
        """
        from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file

        fp = Path(fp)

        nodes = [StructureNode(structure=struct) for struct in read_multiple_structure_from_file(
            fp, charge=charge, spinmult=spinmult)]
        chain = cls(nodes=nodes, parameters=parameters)

        energies_fp = fp.parent / Path(str(fp.stem) + ".energies")
        grad_path = fp.parent / Path(str(fp.stem) + ".gradients")
        grad_shape_path = fp.parent / Path(str(fp.stem) + "_grad_shapes.txt")
        grad_shape_path_old = fp.parent / "grad_shapes.txt"

        if grad_shape_path_old.exists() and not grad_shape_path.exists():
            grad_shape_path = grad_shape_path_old

        if energies_fp.exists() and grad_path.exists() and grad_shape_path.exists():
            energies = np.loadtxt(energies_fp)
            gradients_flat = np.loadtxt(grad_path)
            gradients_shape = np.loadtxt(grad_shape_path, dtype=int)

            gradients = gradients_flat.reshape(gradients_shape)

            for node, (ene, grad) in zip(chain.nodes, zip(energies, gradients)):
                fake_res = FakeQCIOResults(energy=ene, gradient=grad)
                fake_output = FakeQCIOOutput(results=fake_res)
                node._cached_result = fake_output
                node._cached_energy = ene
                node._cached_gradient = grad
        return chain

    @classmethod
    def from_list_of_chains(cls, list_of_chains, parameters) -> Chain:
        """
        Joins a list of Chains into a single chain
        """
        nodes = []
        for chain in list_of_chains:
            nodes.extend(chain.nodes)
        return cls(nodes=nodes, parameters=parameters)

    @property
    def _path_len_coords(self) -> np.array:
        import neb_dynamics.chainhelpers as ch
        if self.nodes[0].has_molecular_graph:
            coords = ch._get_mass_weighed_coords(self)
        else:
            coords = self.coordinates
        return coords

    def _path_len_dist_func(self, coords1, coords2):
        if self.nodes[0].has_molecular_graph:
            return qRMSD_distance(coords1, coords2)
        else:
            return linear_distance(coords1, coords2)

    @property
    def integrated_path_length(self) -> np.array:
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

    @property
    def path_length(self) -> np.array:
        coords = self._path_len_coords
        cum_sums = [0]
        for i, frame_coords in enumerate(coords):
            if i == len(coords) - 1:
                continue
            next_frame = coords[i + 1]
            distance = self._path_len_dist_func(frame_coords, next_frame)
            cum_sums.append(cum_sums[-1] + distance)

        cum_sums = np.array(cum_sums)
        path_len = cum_sums
        return np.array(path_len)

    def plot_chain(self, norm_path=True):
        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))

        if norm_path:
            path_len = self.integrated_path_length
        else:
            path_len = self.path_length

        plt.plot(
            path_len,
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

    def copy(self) -> Chain:
        list_of_nodes = [node.copy() for node in self.nodes]
        chain_copy = Chain(nodes=list_of_nodes,
                           parameters=self.parameters.copy())
        return chain_copy

    @property
    def _energies_already_computed(self) -> bool:
        all_ens = [node.energy for node in self.nodes]
        return all([val is not None for val in all_ens])

    @property
    def energies(self) -> np.array:
        if not self._energies_already_computed:
            raise EnergiesNotComputedError(
                msg="Energies have not been computed.")

        out_ens = np.array([node.energy for node in self.nodes])

        assert all(
            [en is not None for en in out_ens]
        ), f"Ens: Chain contains images with energies that did not converge: {out_ens}"
        return out_ens

    @property
    def energies_kcalmol(self) -> np.array:
        return (self.energies - self.energies[0]) * 627.5

    @property
    def _grads_already_computed(self) -> bool:
        all_grads = [node.gradient for node in self.nodes]
        return np.all([g is not None for g in all_grads])

    @property
    def gradients(self) -> np.array:
        if not self._grads_already_computed:
            raise GradientsNotComputedError(
                msg="Gradients have note been computed")
        grads = [node.gradient for node in self.nodes]
        return grads

    @property
    def rms_gradients(self):
        grads = self.gradients
        rms_grads = []
        for grad in grads:
            rms_gradient = np.sqrt(sum(np.square(grad.flatten())) / len(grad))
            rms_grads.append(rms_gradient)
        return np.array(rms_grads)

    @property
    def ts_triplet_gspring_infnorm(self):
        import chainhelpers as ch
        ind_ts = self.energies[1:-1].argmax()

        _, gsprings = ch.pe_grads_spring_forces_nudged(self)

        if ind_ts == 0:
            triplet = gsprings[0:2]
        elif ind_ts == len(self)-1:
            triplet = gsprings[ind_ts-1:]
        else:
            triplet = gsprings[ind_ts-1:ind_ts+2]
        infnorms = [np.amax(abs(gspr)) for gspr in triplet]
        return max(infnorms)

    @property
    def rms_gperps(self):
        # imported here to avoid circular imports
        import neb_dynamics.chainhelpers as ch
        grads = ch.get_g_perps(self)
        rms_grads = []
        for grad in grads:
            rms_gradient = np.sqrt(sum(np.square(grad.flatten())) / len(grad))
            rms_grads.append(rms_gradient)
        return np.array(rms_grads)

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

    @property
    def symbols(self) -> List[str]:
        """
        returns the system symbols, if Node objects have
        attribute `symbols`. otherwise will raise an error.
        """
        if hasattr(self.nodes[0], 'symbols'):
            return self.nodes[0].symbols
        else:
            raise TypeError(
                f"Node object {self.nodes[0]} does not have `symbols` attribute.")

    @property
    def energies_are_monotonic(self):
        arg_max = self.energies.argmax()
        return arg_max == len(self) - 1 or arg_max == 0

    def write_ene_info_to_disk(self, fp):
        ene_path = fp.parent / Path(str(fp.stem) + ".energies")
        np.savetxt(ene_path, self.energies)

    def write_grad_info_to_disk(self, fp):
        grad_path = fp.parent / Path(str(fp.stem) + ".gradients")
        grad_shape_path = fp.parent / Path(str(fp.stem) + "_grad_shapes.txt")
        np.savetxt(
            grad_path, np.array(
                [node.gradient for node in self.nodes]).flatten()
        )
        np.savetxt(grad_shape_path, np.array(self.gradients).shape)

    def write_to_disk(self, fp: Path):
        fp = Path(fp)
        xyz_arr = self.coordinates*BOHR_TO_ANGSTROMS
        symbs = self.symbols
        write_xyz(filename=fp, atoms=symbs, coords=xyz_arr)

        if self._energies_already_computed:
            self.write_ene_info_to_disk(fp)
        if self._grads_already_computed:
            self.write_grad_info_to_disk(fp)

    def get_ts_node(self) -> Node:
        """
        return the node corresponding to the transition state guess.
        """
        ind_ts_guess = self.energies.argmax()
        return self[ind_ts_guess]

    def get_eA_chain(self):
        eA = max(self.energies_kcalmol)
        return eA

    def _zero_velocity(self):
        self.velocity = np.zeros_like(a=self.coordinates)
