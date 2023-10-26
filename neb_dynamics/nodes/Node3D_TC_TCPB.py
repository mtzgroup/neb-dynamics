from __future__ import annotations
import tempfile
from dataclasses import dataclass
from functools import cached_property
import subprocess
import numpy as np
from retropaths.abinitio.tdstructure import TDStructure
import multiprocessing as mp
import shutil
from tcparse import parse
from pathlib import Path
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.Node import Node
from neb_dynamics.helper_functions import RMSD
RMSD_CUTOFF = 0.5
KCAL_MOL_CUTOFF = 0.1


@dataclass
class Node3D_TC_TCPB(Node):
    tdstructure: TDStructure
    converged: bool = False
    do_climb: bool = False
    _cached_energy: float | None = None
    _cached_gradient: np.array | None = None


    is_a_molecule = True

    @property
    def coords(self):
        return self.tdstructure.coords

    @property
    def coords_bohr(self):
        return self.tdstructure.coords * ANGSTROM_TO_BOHR

    @staticmethod
    def en_func(node: Node3D_TC_TCPB):
        return node.tdstructure.energy_tc_tcpb()
        

    @staticmethod
    def grad_func(node: Node3D_TC_TCPB):
        return node.tdstructure.gradient_tc_tcpb()

    @cached_property
    def energy(self):
        if self._cached_energy is not None:
            return self._cached_energy
        else:
            return self.tdstructure.energy_tc_tcpb()

    @cached_property
    def gradient(self):
        if self._cached_gradient is not None:
            return self._cached_gradient
        else:
            return self.tdstructure.gradient_tc_tcpb()

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
        return np.tensordot(first, second)


    def get_nudged_pe_grad(self, unit_tangent, gradient):
        '''
        Alessio to Jan: comment your functions motherfucker.
        '''
        pe_grad = gradient
        pe_grad_nudged_const = self.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
        return pe_grad_nudged

    def copy(self):
        return Node3D_TC_TCPB(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        copy_tdstruct.update_tc_parameters(td_ref=self.tdstructure)
        
        return Node3D_TC_TCPB(tdstructure=copy_tdstruct, converged=self.converged, do_climb=self.do_climb)

    def opt_func(self, v=True):
        
        #### NOTE: THIS IS NOT DONE WITH TERACHEM SERVER MODE.
        #### IDK HOW TO GET TC SERVER TO GIVE ME GEOM OPTS.
        #### THIS OPTIMIZATION IS DONE BY BOOTING UP TERACHEM FROM SCRATCH
        
        return self.tdstructure.tc_local_geom_optimization()

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @property
    def hessian(self: Node3D_TC_TCPB):
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
        assert self.check_symmetric(approx_hess_sym, rtol=1e-3, atol=1e-3), "Hessian not symmetric for some reason"

        return approx_hess_sym
    
    @classmethod
    def calc_ene_grad(cls, input_path: str):    
        raise NotImplementedError
    
    @classmethod
    def calculate_energy_and_gradients_parallel(cls, chain):
        raise NotImplementedError
    
    def do_geometry_optimization(self) -> Node3D_TC_TCPB:
        td_opt = self.tdstructure.tc_local_geom_optimization()
        return Node3D_TC_TCPB(td_opt)
    
    def _is_connectivity_identical(self, other) -> bool:
        connectivity_identical =  self.tdstructure.molecule_rp.is_bond_isomorphic_to(
            other.tdstructure.molecule_rp
        )
        return connectivity_identical
    
    def _is_conformer_identical(self, other) -> bool:
        aligned_self = self.tdstructure.align_to_td(other.tdstructure)
        rmsd_identical = RMSD(aligned_self.coords, other.tdstructure.coords)[0] < RMSD_CUTOFF
        energies_identical = np.abs((self.energy - other.energy)*627.5) < KCAL_MOL_CUTOFF
        if rmsd_identical and energies_identical:
            conformer_identical = True
        
        if not rmsd_identical and energies_identical:
            # going to assume this is a permutation issue. To address later
            conformer_identical = True
        
        if not rmsd_identical and not energies_identical:
            conformer_identical = False

        return conformer_identical

    def is_identical(self, other) -> bool:

        return self._is_connectivity_identical(other)
        # return all([self._is_connectivity_identical(other), self._is_conformer_identical(other)])
        