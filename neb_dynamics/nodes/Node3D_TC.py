from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory



from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.Node import Node
from neb_dynamics.helper_functions import RMSD
RMSD_CUTOFF = 0.1
KCAL_MOL_CUTOFF = 0.1


@dataclass
class Node3D_TC(Node):
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
    def en_func(node: Node3D_TC):
        res = Node3D_TC.run_tc_calc(node.tdstructure)
        return res.get_energy()

    @staticmethod
    def grad_func(node: Node3D_TC):
        res = Node3D_TC.run_tc_calc(node.tdstructure)
        return res.get_gradient() * BOHR_TO_ANGSTROMS

    @property
    def energy(self):
        if self._cached_energy is not None:
            return self._cached_energy
        else:
            ene =  self.tdstructure.energy_tc()
            self._cached_energy = ene
            return ene

    @property
    def gradient(self):
        if self.converged:
            return np.zeros_like(self.coords)
        
        else:
            if self._cached_gradient is not None:
                return self._cached_gradient
            else:
                grad =  self.tdstructure.gradient_tc()
                self._cached_gradient = grad
                return grad

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
        # return np.sum(first * second, axis=1).reshape(-1, 1)
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
        return Node3D_TC(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        copy_tdstruct.update_tc_parameters(td_ref=self.tdstructure)
        
        
        return Node3D_TC(tdstructure=copy_tdstruct, converged=self.converged, do_climb=self.do_climb)

    def opt_func(self, v=True):
        return self.tdstructure.tc_geom_optimization()

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
        assert self.check_symmetric(approx_hess_sym, rtol=1e-3, atol=1e-3), "Hessian not symmetric for some reason"

        return approx_hess_sym
    
    
    @staticmethod
    def calculate_energy_and_gradients_parallel(chain):
        ens_grads_lists = [None]*len(chain)
        traj = chain.to_trajectory()
        inds_converged = [i for i, node in enumerate(chain) if node.converged]
        inds_not_converged = [i for i, node in enumerate(chain) if not node.converged]
        for ind in inds_converged:
            ref_node = chain[ind]
            ens_grads_lists[ind] = (ref_node._cached_energy, np.zeros_like(ref_node.coords))
        
        new_traj = Trajectory([td for (i, td) in enumerate(traj) if i in inds_not_converged])
        new_traj_ene_grads = new_traj.energies_and_gradients_tc()
        for (ene, grad, ind) in zip(new_traj_ene_grads[0], new_traj_ene_grads[1], inds_not_converged):
            ens_grads_lists[ind] = (ene, grad)
        
        
        return ens_grads_lists
    
    def do_geometry_optimization(self) -> Node3D_TC:
        # td_opt_xtb = self.tdstructure.xtb_geom_optimization()
        # td_opt = td_opt_xtb.tc_geom_optimization()
        td_opt = self.tdstructure.tc_geom_optimization()
        return Node3D_TC(tdstructure=td_opt)
    
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
        else:
            return False

    def is_identical(self, other) -> bool:

        return self._is_connectivity_identical(other)
        # return all([self._is_connectivity_identical(other), self._is_conformer_identical(other)])
        