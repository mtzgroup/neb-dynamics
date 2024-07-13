from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from neb_dynamics.tdstructure import TDStructure
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.Node import Node
from neb_dynamics.helper_functions import RMSD
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.elements import symbol_to_atomic_number, atomic_number_to_symbol

import multiprocessing as mp
import concurrent.futures
import os



@dataclass
class Node3D(Node):
    tdstructure: TDStructure
    converged: bool = False
    do_climb: bool = False
    _cached_energy: float | None = None
    _cached_gradient: np.array | None = None

    is_a_molecule = True
    # GLOBAL_RMSD_CUTOFF: float = 1.0
    # FRAGMENT_RMSD_CUTOFF: float = 0.5
    GLOBAL_RMSD_CUTOFF: float = 20.0
    FRAGMENT_RMSD_CUTOFF: float = 0.5

    KCAL_MOL_CUTOFF: float = 1.0
    BARRIER_THRE: float = 5  # kcal/mol

    def __eq__(self, other: Node3D) -> bool:
        return self.is_identical(other)

    def __repr__(self):
        return 'node3d'

    @property
    def coords(self):
        return self.tdstructure.coords

    @property
    def coords_bohr(self):
        return self.tdstructure.coords * ANGSTROM_TO_BOHR

    @staticmethod
    def en_func(node: Node3D):
        res = Node3D.run_xtb_calc(node.tdstructure)
        return res.get_energy()

    @staticmethod
    def grad_func(node: Node3D):
        res = Node3D.run_xtb_calc(node.tdstructure)
        return res.get_gradient() * BOHR_TO_ANGSTROMS

    @property
    def energy(self):
        if self._cached_energy is not None:
            return self._cached_energy
        else:
            ene = Node3D.run_xtb_calc(self.tdstructure).get_energy()
            self._cached_energy = ene
            return ene

    def do_geom_opt_trajectory(self) -> Trajectory:
        td_copy = self.tdstructure.copy()
        td_copy.tc_model_method = "gfn2xtb"
        td_copy.tc_model_basis = "gfn2xtb"
        td_opt_traj = td_copy.xtb_geom_optimization(return_traj=True)
        td_opt_traj.update_tc_parameters(td_copy)
        # td_opt = self.tdstructure.xtb_geom_optimization()
        return td_opt_traj

    def do_geometry_optimization(self) -> Node3D:
        td_copy = self.tdstructure.copy()
        td_copy.tc_model_method = "gfn2xtb"
        td_copy.tc_model_basis = "gfn2xtb"
        # td_opt = td_copy.tc_local_geom_optimization()
        # td_opt.update_tc_parameters(td_copy)
        td_opt = self.tdstructure.xtb_geom_optimization()
        return Node3D(
            tdstructure=td_opt,
            converged=self.converged,
            do_climb=self.do_climb,
            BARRIER_THRE=self.BARRIER_THRE,
            GLOBAL_RMSD_CUTOFF=self.GLOBAL_RMSD_CUTOFF,
            FRAGMENT_RMSD_CUTOFF=self.FRAGMENT_RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF,
        )

    def _is_connectivity_identical(self, other) -> bool:
        # print("different graphs")
        connectivity_identical = self.tdstructure.molecule_rp.remove_Hs().is_bond_isomorphic_to(
            other.tdstructure.molecule_rp.remove_Hs()
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

        # return self._is_connectivity_identical(other)
        return all(
            [
                self._is_connectivity_identical(other),
                self._is_conformer_identical(other),
            ]
        )

    @property
    def gradient(self):
        if self.converged:
            return np.zeros_like(self.coords)

        else:
            if self._cached_gradient is not None:
                return self._cached_gradient
            else:
                grad = (
                    Node3D.run_xtb_calc(self.tdstructure).get_gradient()
                    * BOHR_TO_ANGSTROMS
                )
                self._cached_gradient = grad
                return grad

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
        # # return np.sum(first * second, axis=1).reshape(-1, 1)
        # return np.tensordot(first, second)
        return np.dot(first.flatten(), second.flatten())

    # i want to cache the result of this but idk how caching works
    def run_xtb_calc(tdstruct: TDStructure):
        atomic_numbers = tdstruct.atomic_numbers
        calc = Calculator(
            get_method("GFN2-xTB"),
            numbers=np.array(atomic_numbers),
            positions=tdstruct.coords_bohr,
            charge=tdstruct.charge,
            uhf=tdstruct.spinmult - 1,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()
        return res

    def get_nudged_pe_grad(self, unit_tangent, gradient):
        """
        Returns the component of the gradient that acts perpendicular to the path tangent
        """
        pe_grad = gradient
        pe_grad_nudged_const = self.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
        return pe_grad_nudged

    def copy(self):
        copy_node = Node3D(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
            BARRIER_THRE=self.BARRIER_THRE,
            GLOBAL_RMSD_CUTOFF=self.GLOBAL_RMSD_CUTOFF,
            FRAGMENT_RMSD_CUTOFF=self.FRAGMENT_RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF,
        )
        # if copy_node.converged:
        copy_node._cached_energy = self._cached_energy
        # copy_node._cached_gradient = self._cached_gradient
        return copy_node

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        copy_tdstruct.update_tc_parameters(self.tdstructure)

        return Node3D(
            tdstructure=copy_tdstruct,
            converged=self.converged,
            do_climb=self.do_climb,
            BARRIER_THRE=self.BARRIER_THRE,
            GLOBAL_RMSD_CUTOFF=self.GLOBAL_RMSD_CUTOFF,
            FRAGMENT_RMSD_CUTOFF=self.FRAGMENT_RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF,
        )

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @property
    def hessian(self: Node3D):
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

    @property
    def input_tuple(self):
        return (
            self.tdstructure.atomic_numbers,
            self.tdstructure.coords_bohr,
            self.tdstructure.charge,
            self.tdstructure.spinmult,
        )

    @staticmethod
    def calc_xtb_ene_grad_from_input_tuple(tuple):
        atomic_numbers, coords_bohr, charge, spinmult, converged, prev_en = tuple
        if converged:
            return prev_en, np.zeros_like(coords_bohr)

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

        # from qcio import CalcType, ProgramInput, Structure
        # from qcop import compute
        # structure = Structure(
        #     symbols=[atomic_number_to_symbol(int(numb)) for numb in atomic_numbers],
        #     geometry=coords_bohr,
        # )

        # # Define the program input
        # prog_input = ProgramInput(
        #     structure=structure,
        #     calctype=CalcType.energy,
        #     model={"method": "GFN2xTB"},  # type: ignore
        #     keywords={"max_iterations": 500},
        # )

        # output = compute("xtb", prog_input, print_stdout=False)
        # return output.results.energy, output.results.gradient*BOHR_TO_ANGSTROMS

    @classmethod
    def calculate_energy_and_gradients_parallel(cls, chain) -> list[tuple[float, np.ndarray]]:
        """"""
        iterator = (
            (
                n.tdstructure.atomic_numbers,
                n.tdstructure.coords_bohr,
                n.tdstructure.charge,
                n.tdstructure.spinmult,
                n.converged,
                n._cached_energy,
            )
            for n in chain.nodes
        )

        # with mp.Pool() as p:
        #     ene_gradients = p.map(
        #         cls.calc_xtb_ene_grad_from_input_tuple, iterator)
        # return ene_gradients
        # NOTE: Using CPU count since these calls are CPU bound calls to xtb library
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            return list(executor.map(cls.calc_xtb_ene_grad_from_input_tuple, iterator))
