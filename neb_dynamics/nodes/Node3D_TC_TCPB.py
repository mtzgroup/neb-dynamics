from __future__ import annotations
import tempfile
from dataclasses import dataclass
from functools import cached_property
import subprocess
import numpy as np
from retropaths.abinitio.tdstructure import TDStructure
import multiprocessing as mp
import shutil
from qcparse import parse
from pathlib import Path
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.Node import Node
from neb_dynamics.helper_functions import RMSD
import pytcpb as tc
import time



RMSD_CUTOFF = 0.5
# KCAL_MOL_CUTOFF = 0.1
KCAL_MOL_CUTOFF = 0.3


@dataclass
class Node3D_TC_TCPB(Node):
    tdstructure: TDStructure
    converged: bool = False
    do_climb: bool = False
    _cached_energy: float | None = None
    _cached_gradient: np.array | None = None
    
    
    _tc_server = False

    is_a_molecule = True
    
    

    @property
    def coords(self):
        return self.tdstructure.coords

    @property
    def coords_bohr(self):
        return self.tdstructure.coords * ANGSTROM_TO_BOHR

    @staticmethod
    def en_func(node: Node3D_TC_TCPB):
        return node.tdstructure.energy_tc_local()
        

    @staticmethod
    def grad_func(node: Node3D_TC_TCPB):
        return node.tdstructure.gradient_tc_local()
    
    
    def compute_ene_grad(self):
        if not self._tc_server:
            self._connect_to_server()
        
        ene, grad = self.compute_tc_tcpb(self)
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
        copy_node =  Node3D_TC_TCPB(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
        )
        copy_node._tc_server = self._tc_server
        return copy_node

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        copy_tdstruct.update_tc_parameters(td_ref=self.tdstructure)
        
        return Node3D_TC_TCPB(tdstructure=copy_tdstruct, converged=self.converged, do_climb=self.do_climb)

    def opt_func(self, v=True):
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
    def calculate_energy_and_gradients_parallel(cls, chain):
        ref = chain[0]
        
        if not ref._tc_server:
            ref._connect_to_server()
        
        
        ene_gradients = []
        structures = chain
        for i, structure in enumerate(structures):
            # this if statement is so the wavefunc is propagated for the following structures
            if i==0: 
                chosen_index=1 
            else: 
                chosen_index=0
                
                
            structure._tc_server = ref._tc_server
            ene_grad = cls.compute_tc_tcpb(structure_node=structure, index=chosen_index)
            ene_gradients.append(ene_grad)

        return ene_gradients
    
    def do_geometry_optimization(self) -> Node3D_TC_TCPB:
        td_opt = self.tdstructure.tc_local_geom_optimization()
        return Node3D_TC_TCPB(td_opt)
    
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
        
    
    def _connect_to_server(self):
        # Set information about the server
        td = self.tdstructure
        host = "localhost"

        # Get Port
        port = 8888

        str_inp = td._tcpb_input_string()

        fname = '/tmp/inputfile.in'
        with open(fname,'w') as f:
            f.write(str_inp)
            
        tcfile = fname
        
        
        structures = [self]

        qmattypes = self.tdstructure.symbols

        # Attempts to connect to the TeraChem server
        status = tc.connect(host, port)
        if status == 0:
            # print("Connected to TC server.")
            pass
        elif status == 1:
            raise ValueError("Connection to TC server failed.")
        elif status == 2:
            raise ValueError(
                "Connection to TC server succeeded, but the server is not available."
            )
        else:
            raise ValueError("Status on tc.connect function is not recognized!")
        

        # Setup TeraChem
        status = tc.setup(str(tcfile), qmattypes) 
        if status == 0:
            # print("TC setup completed with success.")
            pass
        elif status == 1:
            raise ValueError(
                "No options read from TC input file or mismatch in the input options!"
            )
        elif status == 2:
            raise ValueError("Failed to setup TC.")
        else:
            raise ValueError("Status on tc_setup function is not recognized!")
    
        self._tc_server = tc
        
    @classmethod
    def compute_tc_tcpb(cls, structure_node: Node3D_TC_TCPB, index=0):
        structure = structure_node.tdstructure
        qmcoords = structure.coords.flatten() * ANGSTROM_TO_BOHR
        qmcoords = qmcoords.tolist()
        qmattypes = structure.symbols
        
        wf_treatments = ['Cont','Cont_Reset','Reinit']
        wf_treatment_chosen = wf_treatments[index]
        
        globaltreatment = {"Cont": 0, "Cont_Reset": 1, "Reinit": 2}
        
        
        # Compute energy and gradient
        time.sleep(0.020)  # TCPB needs a small delay between calls
        
        totenergy, qmgrad, mmgrad, status = structure_node._tc_server.compute_energy_gradient(
            qmattypes, qmcoords, globaltreatment=globaltreatment[wf_treatment_chosen]
        )
            

        # print(f"Status: {status}")
        if status == 0:
            # print("Successfully computed energy and gradients")
            pass
        elif status == 1:
            raise ValueError("Mismatch in the variables passed to compute_energy_gradient")
        elif status == 2:
            raise ValueError("Error in compute_energy_gradient.")
        else:
            raise ValueError(
                "Status on compute_energy_gradient function is not recognized!"
            )

        return totenergy, (np.array(qmgrad).reshape(structure.coords.shape))*BOHR_TO_ANGSTROMS
