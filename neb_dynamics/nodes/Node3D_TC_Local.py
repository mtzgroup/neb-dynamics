from __future__ import annotations
import tempfile
from dataclasses import dataclass
from functools import cached_property
import subprocess
import numpy as np
# from retropaths.abinitio.tdstructure import TDStructure
from neb_dynamics.tdstructure import TDStructure
import multiprocessing as mp
import shutil
from qcparse import parse
from pathlib import Path
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.Node import Node
from neb_dynamics.helper_functions import RMSD
import qcop
RMSD_CUTOFF = 0.5
# KCAL_MOL_CUTOFF = 0.1
KCAL_MOL_CUTOFF = 0.3


@dataclass
class Node3D_TC_Local(Node):
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
    def en_func(node: Node3D_TC_Local):
        return node.tdstructure.energy_tc_local()
        

    @staticmethod
    def grad_func(node: Node3D_TC_Local):
        return node.tdstructure.gradient_tc_local()*BOHR_TO_ANGSTROMS
    
    
    def compute_ene_grad(self):
        
        prog_input = self.tdstructure._prepare_input(method='gradient')

        output = qcop.compute('terachem', prog_input, propagate_wfn=True, collect_files=True)
        ene = output.results.energy
        grad = output.results.gradient*BOHR_TO_ANGSTROMS
        self._cached_gradient = grad
        self._cached_energy = ene

    @property
    def energy(self):
        if self._cached_energy is  None:
            self.compute_ene_grad()
        return self._cached_energy

    @property
    def gradient(self):
        if self._cached_gradient is None:
            self.compute_ene_grad()
        return self._cached_gradient
            
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
        return Node3D_TC_Local(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        copy_tdstruct.update_tc_parameters(td_ref=self.tdstructure)
        
        return Node3D_TC_Local(tdstructure=copy_tdstruct, converged=self.converged, do_climb=self.do_climb)

    def opt_func(self, v=True):
        return self.tdstructure.tc_local_geom_optimization()

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @property
    def hessian(self: Node3D_TC_Local):
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
        with tempfile.NamedTemporaryFile(suffix='.out',mode="w+", delete=False) as tmp_out:
            out = subprocess.run([f"terachem {input_path}"], shell=True, 
                                capture_output=True)
            tmp_out.write(out.stdout.decode())

        if not out.stdout.decode():
            raise ValueError(f"error found: {out.stderr.decode()}")

        result_obj = parse(tmp_out.name)
        if result_obj.success:
            ene, grad = result_obj.properties.return_energy, result_obj.properties.return_gradient

        else:
            ene = None
            grad = None


        Path(tmp_out.name).unlink()
        return ene, grad
    
    
    
    @classmethod
    def calculate_energy_and_gradients_parallel(cls, chain):
        all_geoms = []
        all_inps = []
        for n in chain.nodes:
            geo, inp = n.tdstructure.make_geom_and_inp_file()
            all_geoms.append(geo)
            all_inps.append(inp)


        iterator = all_inps
        with mp.Pool() as p:
            ene_gradients = p.map(cls.calc_ene_grad, iterator)



        [Path(g).unlink() for g in all_geoms]
        [shutil.rmtree(g[:-4]) for g in all_geoms]
        [Path(inp).unlink() for inp in all_inps]
        return ene_gradients
    
    def do_geometry_optimization(self) -> Node3D_TC_Local:
        td_opt = self.tdstructure.tc_local_geom_optimization()
        return Node3D_TC_Local(td_opt)
    
    def _is_connectivity_identical(self, other) -> bool:
        connectivity_identical =  self.tdstructure.molecule_rp.is_bond_isomorphic_to(
            other.tdstructure.molecule_rp
        )
        return connectivity_identical
    
    def _is_conformer_identical(self, other) -> bool:
        if self._is_connectivity_identical(other):
            aligned_self = self.tdstructure.align_to_td(other.tdstructure)
            dist = RMSD(aligned_self.coords, other.tdstructure.coords)[0]
            en_delta = np.abs((self.energy - other.energy)*627.5)
            
            
            rmsd_identical = dist < RMSD_CUTOFF
            energies_identical = en_delta < KCAL_MOL_CUTOFF
            if rmsd_identical and energies_identical:
                conformer_identical = True
            
            if not rmsd_identical and energies_identical:
                # going to assume this is a rotation issue. Need To address.
                conformer_identical = False
            
            if not rmsd_identical and not energies_identical:
                conformer_identical = False
            
            if rmsd_identical and not energies_identical:
                conformer_identical = False
            print(f"\nRMSD : {dist} // |∆en| : {en_delta}\n")
            return conformer_identical

    def is_identical(self, other) -> bool:

        # return self._is_connectivity_identical(other)
        return all([self._is_connectivity_identical(other), self._is_conformer_identical(other)])
        