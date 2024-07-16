from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import qcop
from chemcloud.client import CCClient
from qcio import ProgramInput, ProgramOutput, Structure

from neb_dynamics.elements import atomic_number_to_symbol
from neb_dynamics.nodes.Node import Node
from neb_dynamics.tdstructure import TDStructure

AVAIL_COMPUTE_METHODS = ['qcop', 'chemcloud']


@dataclass
class Node3D_QCIO(Node):
    """
    Node object that depends on QCIO data structure.
    """

    # qcio objects
    structure: Structure
    prog_inp: ProgramInput
    compute_method: str = 'qcop'  # available: ['qcop','chemcloud']
    compute_program: str = 'terachem'

    # node vars
    converged: bool = False
    do_climb: bool = False

    _cached_result: ProgramOutput = None

    is_a_molecule = True

    GLOBAL_RMSD_CUTOFF: float = 20.0
    FRAGMENT_RMSD_CUTOFF: float = 0.5
    KCAL_MOL_CUTOFF: float = 1.0
    BARRIER_THRE: float = 5  # kcal/mol

    def __repr__(self) -> str:
        return 'node3d_qcio'

    def compute(self) -> Callable:
        """
        stores compute function
        """
        if self.compute_method == 'chemcloud':
            client = CCClient()
            return client.compute
        elif self.compute_method == "qcop":
            return qcop.compute

        else:
            raise ValueError(
                f"Invalid compute method: {self.compute_method}. Available: {AVAIL_COMPUTE_METHODS}")

    # these next 3 properties are so I dont break my code. this highlights a need to refactor.
    @property
    def tdstructure(self) -> TDStructure:
        coords = self.structure.geometry_angstrom
        symbols = [atomic_number_to_symbol(n)
                   for n in self.structure.atomic_numbers]
        charge = self.structure.charge
        spinmult = self.structure.multiplicity
        return TDStructure.from_coords_symbols(coords=coords, symbols=symbols, tot_charge=charge,
                                               tot_spinmult=spinmult)

    @property
    def _cached_gradient(self):
        if self._cached_result is None:
            return None
        else:
            return self._cached_result.results.gradient

    @property
    def _cached_energy(self):
        if self._cached_result is None:
            return None
        else:
            return self._cached_result.results.energy

    @property
    def coords(self):
        return self.structure.geometry_angstrom

    @property
    def coords_bohr(self):
        return self.structure.geometry

    @property
    def energy(self):
        if self._cached_energy is not None:
            return self._cached_energy
        else:
            future_result = self.compute(
                self.compute_program,
                self.prog_inp,
                collect_files=True
            )

            self._cached_result = future_result.get()
            return self._cached_energy

    @property
    def gradient(self):
        if self.converged:
            return np.zeros_like(self.coords)

        else:
            if self._cached_gradient is not None:
                return self._cached_gradient
            else:
                grad = self.tdstructure.gradient_tc() * BOHR_TO_ANGSTROMS
                self._cached_gradient = grad
                return grad

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
        # return np.sum(first * second, axis=1).reshape(-1, 1)
        return np.tensordot(first, second)

    def get_nudged_pe_grad(self, unit_tangent, gradient):
        """
        Alessio to Jan: comment your functions motherfucker.
        """
        pe_grad = gradient
        pe_grad_nudged_const = self.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
        return pe_grad_nudged

    def copy(self):
        copy_node = Node3D_TC(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
        )

        copy_node._cached_energy = self._cached_energy

        return copy_node

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        copy_tdstruct.update_tc_parameters(td_ref=self.tdstructure)

        return Node3D_TC(
            tdstructure=copy_tdstruct,
            converged=self.converged,
            do_climb=self.do_climb,
            BARRIER_THRE=self.BARRIER_THRE,
            GLOBAL_RMSD_CUTOFF=self.GLOBAL_RMSD_CUTOFF,
            FRAGMENT_RMSD_CUTOFF=self.FRAGMENT_RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF,
        )

    # def opt_func(self, v=True):
    #     return self.tdstructure.tc_geom_optimization()

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @property
    def hessian(self: Node3D_TC):
        dr = 0.01  # displacement vector, Bohr
        numatoms = self.tdstructure.atomn
        approx_hess = []
        for n in range(numatoms):
            # for n in range(2):
            grad_n = []
            for coord_ind, coord_name in enumerate(["dx", "dy", "dz"]):
                # print(f"doing atom #{n} | {coord_name}")

                coords = np.array(self.coords, dtype="float64")

                # print(coord_ind, coords[n, coord_ind])
                coords[n, coord_ind] = coords[n, coord_ind] + dr
                # print(coord_ind, coords[n, coord_ind])

                node2 = self.copy()
                node2 = node2.update_coords(coords)

                delta_grad = (node2.gradient - self.gradient) / dr
                # print(delta_grad)
                grad_n.append(delta_grad.flatten())

            approx_hess.extend(grad_n)
        approx_hess = np.array(approx_hess)
        # raise AlessioError(f"{approx_hess.shape}")

        approx_hess_sym = 0.5 * (approx_hess + approx_hess.T)
        assert self.check_symmetric(
            approx_hess_sym, rtol=1e-3, atol=1e-3
        ), "Hessian not symmetric for some reason"

        return approx_hess_sym

    @staticmethod
    def calculate_energy_and_gradients_parallel(chain):
        ens_grads_lists = [None] * len(chain)
        traj = chain.to_trajectory()
        inds_converged = [i for i, node in enumerate(chain) if node.converged]
        inds_not_converged = [i for i, node in enumerate(
            chain) if not node.converged]
        for ind in inds_converged:
            ref_node = chain[ind]
            ens_grads_lists[ind] = (
                ref_node._cached_energy,
                np.zeros_like(ref_node.coords),
            )

        if len(inds_not_converged) >= 1:
            new_traj = Trajectory(
                [td for (i, td) in enumerate(traj) if i in inds_not_converged]
            )
            new_traj.update_tc_parameters(
                traj[0]
            )  # needed so we propagate the setting.... bad.....
            new_traj_ene_grads = new_traj.energies_and_gradients_tc()
            for (ene, grad, ind) in zip(
                new_traj_ene_grads[0], new_traj_ene_grads[1], inds_not_converged
            ):
                ens_grads_lists[ind] = (ene, grad * BOHR_TO_ANGSTROMS)

        return ens_grads_lists

    def do_geom_opt_trajectory(self) -> Trajectory:
        td_copy = self.tdstructure.copy()
        td_opt_traj = td_copy.run_tc_local(
            calculation="minimize", return_object=True)
        print(f"len opt traj: {len(td_opt_traj)}")
        td_opt_traj.update_tc_parameters(td_copy)
        return td_opt_traj

    def do_geometry_optimization(self) -> Node3D_TC:
        try:
            # td_opt_xtb = self.tdstructure.xtb_geom_optimization()
            # td_opt = td_opt_xtb.tc_geom_optimization()
            td_opt = self.tdstructure.tc_local_geom_optimization()
            # td_opt = td_opt_xtb.tc_local_geom_optimization()

        except Exception:

            td_opt_xtb = self.tdstructure.xtb_geom_optimization()
            # td_opt = td_opt_xtb.tc_geom_optimization()
            td_opt = td_opt_xtb.tc_local_geom_optimization()

            # td_opt = self.tdstructure.tc_geom_optimization()
            # td_opt = self.tdstructure.tc_local_geom_optimization() ### this is being done locally because it is too slow to use the chemcloud version.

        return Node3D_TC(
            tdstructure=td_opt,
            converged=self.converged,
            do_climb=self.do_climb,
            BARRIER_THRE=self.BARRIER_THRE,
            GLOBAL_RMSD_CUTOFF=self.GLOBAL_RMSD_CUTOFF,
            FRAGMENT_RMSD_CUTOFF=self.FRAGMENT_RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF,
        )

    def _is_connectivity_identical(self, other) -> bool:
        connectivity_identical = self.tdstructure.molecule_rp.is_bond_isomorphic_to(
            other.tdstructure.molecule_rp
        )
        return connectivity_identical

    def _is_conformer_identical(self, other) -> bool:
        if self._is_connectivity_identical(other):
            aligned_self = self.tdstructure.align_to_td(other.tdstructure)

            global_dist = RMSD(aligned_self.coords,
                               other.tdstructure.coords)[0]
            per_frag_dists = []
            self_frags = self.tdstructure.split_td_into_frags()
            other_frags = other.tdstructure.split_td_into_frags()
            for frag_self, frag_other in zip(self_frags, other_frags):
                aligned_frag_self = frag_self.align_to_td(frag_other)
                frag_dist = RMSD(aligned_frag_self.coords,
                                 frag_other.coords)[0]
                per_frag_dists.append(frag_dist)
            print(f"{per_frag_dists=}")
            print(f"{global_dist=}")

            en_delta = np.abs((self.energy - other.energy) * 627.5)

            global_rmsd_identical = global_dist <= self.GLOBAL_RMSD_CUTOFF
            fragment_rmsd_identical = max(
                per_frag_dists) <= self.FRAGMENT_RMSD_CUTOFF
            rmsd_identical = global_rmsd_identical and fragment_rmsd_identical
            energies_identical = en_delta < self.KCAL_MOL_CUTOFF
            # print(f"\nbarrier_to_conformer_rearr: {barrier} kcal/mol\n{en_delta=}\n")

            if rmsd_identical and energies_identical:  # and barrier_accessible:
                return True
            else:
                return False
        else:
            return False

    def is_identical(self, other) -> bool:

        return all(
            [
                self._is_connectivity_identical(other),
                self._is_conformer_identical(other),
            ]
        )
