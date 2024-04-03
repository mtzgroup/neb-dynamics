from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from neb_dynamics.tdstructure import TDStructure
from xtb.ase.calculator import XTB
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.Node import Node
from neb_dynamics.helper_functions import RMSD
from neb_dynamics.trajectory import Trajectory
import multiprocessing as mp
from pathlib import Path



@dataclass
class Node3D(Node):
    tdstructure: TDStructure
    converged: bool = False
    do_climb: bool = False
    _cached_energy: float | None = None
    _cached_gradient: np.array | None = None

    is_a_molecule = True
    RMSD_CUTOFF: float = 0.5
    KCAL_MOL_CUTOFF: float = 0.1
    BARRIER_THRE: float = 15  # kcal/mol
    
    

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

    def do_geometry_optimization(self) -> Node3D:
        td_copy = self.tdstructure.copy()
        td_copy.tc_model_method = 'gfn2xtb'
        td_copy.tc_model_basis = 'gfn2xtb'
        td_opt = td_copy.tc_local_geom_optimization()
        td_opt.update_tc_parameters(td_copy)
        # td_opt = self.tdstructure.xtb_geom_optimization()
        return Node3D(tdstructure=td_opt,
            converged=self.converged, 
            do_climb=self.do_climb,
            BARRIER_THRE=self.BARRIER_THRE,
            RMSD_CUTOFF=self.RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF)
        # max_steps=8000
        # ss=.05
        # tol=0.0001
        # nsteps = 0
        # traj = []

        # node = self.copy()
        # while nsteps < max_steps:
        #     traj.append(node)
        #     grad = node.tdstructure.gradient_xtb()
        #     if np.linalg.norm(grad) / len(grad) < tol:
        #         break
        #     new_coords = node.coords - ss*grad
        #     node = node.update_coords(new_coords)
        #     # print(f"|grad|={np.linalg.norm(grad)}",end='\r')
        #     nsteps+=1

        # if np.linalg.norm(grad) / len(grad) < tol:
        #     print(f"\nConverged in {nsteps} steps!\n")
        # else:
        #     print(f"\nDid not converge in {nsteps} steps. grad={np.linalg.norm(grad) / len(grad)}\n")

        # return node

    def _is_connectivity_identical(self, other) -> bool:
        # print("different graphs")
        connectivity_identical = self.tdstructure.molecule_rp.is_bond_isomorphic_to(
            other.tdstructure.molecule_rp
        )
        return connectivity_identical

    def _is_conformer_identical(self, other) -> bool:
        if self._is_connectivity_identical(other):
            aligned_self = self.tdstructure.align_to_td(other.tdstructure)

            traj = Trajectory([aligned_self, other.tdstructure]).run_geodesic(
                nimages=10)
            barrier = max(
                traj.energies_xtb()
            )  # energies are given relative to start struct

            # dist = RMSD(aligned_self.coords, other.tdstructure.coords)[0]
            barrier_accessible =  barrier <= self.BARRIER_THRE
            en_delta = np.abs((self.energy - other.energy)*627.5)
            print(f"\nbarrier_to_conformer_rearr: {barrier} kcal/mol\n{en_delta=}\n")

            # rmsd_identical = dist < self.RMSD_CUTOFF
            energies_identical = en_delta < self.KCAL_MOL_CUTOFF
            # if rmsd_identical and energies_identical:
            #     conformer_identical = True
            if barrier_accessible and energies_identical:
                return True
            else:
                return False

            # if not rmsd_identical and energies_identical:
            #     # going to assume this is a rotation issue. Need To address.
            #     conformer_identical = False

            # if not rmsd_identical and not energies_identical:
            #     conformer_identical = False

            # if rmsd_identical and not energies_identical:
            #     conformer_identical = False
            # print(f"\nRMSD : {dist} // |∆en| : {en_delta}\n")
            # return conformer_identical
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
            RMSD_CUTOFF=self.RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF
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
            RMSD_CUTOFF=self.RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF
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

    @classmethod
    def calculate_energy_and_gradients_parallel(cls, chain):
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

        with mp.Pool() as p:
            ene_gradients = p.map(cls.calc_xtb_ene_grad_from_input_tuple, iterator)
        return ene_gradients
