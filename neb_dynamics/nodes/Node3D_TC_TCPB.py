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
import pytcpb as tc
import time

# Get Port
# port = 8888
port = 12121


@dataclass
class Node3D_TC_TCPB(Node):
    tdstructure: TDStructure
    converged: bool = False
    do_climb: bool = False
    _cached_energy: float | None = None
    _cached_gradient: np.array | None = None

    _tc_server = False

    is_a_molecule = True
    GLOBAL_RMSD_CUTOFF: float = 1.0
    FRAGMENT_RMSD_CUTOFF: float = 0.5

    KCAL_MOL_CUTOFF: float = 1.0
    BARRIER_THRE: float = 5  # kcal/mol

    def __repr__(self):
        return 'node3d_tcpb'

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
        if self._cached_energy is not None:
            return self._cached_energy
        if self._cached_energy is None:
            self.compute_ene_grad()
        return self._cached_energy

    @property
    def gradient(self):
        if self.converged:
            return np.zeros_like(self.coords)

        else:
            if self._cached_gradient is not None:
                return self._cached_gradient
            else:
                self.compute_ene_grad()
        return self._cached_gradient

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
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
        copy_node = Node3D_TC_TCPB(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
            BARRIER_THRE=self.BARRIER_THRE,
            GLOBAL_RMSD_CUTOFF=self.GLOBAL_RMSD_CUTOFF,
            FRAGMENT_RMSD_CUTOFF=self.FRAGMENT_RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF,
        )
        copy_node._tc_server = self._tc_server

        copy_node._cached_energy = self._cached_energy
        # copy_node._cached_gradient = self._cached_gradient

        return copy_node

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        copy_tdstruct.update_tc_parameters(td_ref=self.tdstructure)

        return Node3D_TC_TCPB(
            tdstructure=copy_tdstruct,
            converged=self.converged,
            do_climb=self.do_climb,
            BARRIER_THRE=self.BARRIER_THRE,
            GLOBAL_RMSD_CUTOFF=self.GLOBAL_RMSD_CUTOFF,
            FRAGMENT_RMSD_CUTOFF=self.FRAGMENT_RMSD_CUTOFF,
            KCAL_MOL_CUTOFF=self.KCAL_MOL_CUTOFF,
        )

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
        assert self.check_symmetric(
            approx_hess_sym, rtol=1e-3, atol=1e-3
        ), "Hessian not symmetric for some reason"

        return approx_hess_sym

    @classmethod
    def calculate_energy_and_gradients_parallel(cls, chain):
        ref = chain[0]
        # print('input chain:',[node._cached_energy for node in chain])
        if not ref._tc_server:
            ref._connect_to_server()

        ene_gradients = [None] * len(chain)
        inds_converged = [i for i, node in enumerate(chain) if node.converged]
        inds_not_converged = [i for i, node in enumerate(chain) if not node.converged]
        for ind in inds_converged:
            ref_node = chain[ind]
            ene_gradients[ind] = (
                ref_node._cached_energy,
                np.zeros_like(ref_node.coords),
            )

        if len(inds_not_converged) >= 1:
            structures = [
                node for i, node in enumerate(chain) if i in inds_not_converged
            ]
            # print(f"\nLen of structs: {len(structures)} \\ {len(inds_not_converged)=}\n")
            for i, (list_ind, structure) in enumerate(
                zip(inds_not_converged, structures)
            ):
                # this if statement is so the wavefunc is propagated for the following structures
                if i == 0:
                    chosen_index = 1
                else:
                    chosen_index = 0

                structure._tc_server = ref._tc_server
                ene_grad = cls.compute_tc_tcpb(
                    structure_node=structure, index=chosen_index
                )
                ene_gradients[list_ind] = ene_grad

        # print(ene_gradients)
        return ene_gradients

    def do_geom_opt_trajectory(self):
        td_copy = self.tdstructure.copy()
        td_opt_traj = td_copy.run_tc_local(calculation="minimize", return_object=True)
        print(f"len opt traj: {len(td_opt_traj)}")
        td_opt_traj.update_tc_parameters(td_copy)
        return td_opt_traj

    def do_geometry_optimization(self) -> Node3D_TC_TCPB:
        td_opt = self.tdstructure.tc_local_geom_optimization()
        time.sleep(0.3)  # sleep time so server can reset
        return Node3D_TC_TCPB( tdstructure=td_opt,
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

            global_dist = RMSD(aligned_self.coords, other.tdstructure.coords)[0]
            per_frag_dists = []
            self_frags = self.tdstructure.split_td_into_frags()
            other_frags = other.tdstructure.split_td_into_frags()
            for frag_self, frag_other in zip(self_frags, other_frags):
                aligned_frag_self = frag_self.align_to_td(frag_other)
                frag_dist = RMSD(aligned_frag_self.coords, frag_other.coords)[0]
                per_frag_dists.append(frag_dist)
            print(f"{per_frag_dists=}")
            print(f"{global_dist=}")

            en_delta = np.abs((self.energy - other.energy) * 627.5)

            global_rmsd_identical = global_dist <= self.GLOBAL_RMSD_CUTOFF
            fragment_rmsd_identical = max(per_frag_dists) <= self.FRAGMENT_RMSD_CUTOFF
            rmsd_identical = global_rmsd_identical and fragment_rmsd_identical
            energies_identical = en_delta < self.KCAL_MOL_CUTOFF
            # print(f"\nbarrier_to_conformer_rearr: {barrier} kcal/mol\n{en_delta=}\n")

            if rmsd_identical and energies_identical:  #and barrier_accessible:
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

    def _connect_to_server(self):
        # Set information about the server
        td = self.tdstructure
        host = "localhost"

        str_inp = td._tcpb_input_string()

        fname = "/tmp/inputfile.in"
        with open(fname, "w") as f:
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
            # raise ValueError(
            #     "Connection to TC server succeeded, but the server is not available."
            # )
            time.sleep(3)
            self._connect_to_server()
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

        wf_treatments = ["Cont", "Cont_Reset", "Reinit"]
        wf_treatment_chosen = wf_treatments[index]

        globaltreatment = {"Cont": 0, "Cont_Reset": 1, "Reinit": 2}

        # Compute energy and gradient
        time.sleep(0.030)  # TCPB needs a small delay between calls

        (
            totenergy,
            qmgrad,
            mmgrad,
            status,
        ) = structure_node._tc_server.compute_energy_gradient(
            qmattypes, qmcoords, globaltreatment=globaltreatment[wf_treatment_chosen]
        )

        # print(f"Status: {status}")
        if status == 0:
            # print("Successfully computed energy and gradients")
            pass
        elif status == 1:
            raise ValueError(
                "Mismatch in the variables passed to compute_energy_gradient"
            )
        elif status == 2:
            time.sleep(5)  # so the terachem server can restart
            raise ValueError("Error in compute_energy_gradient.")

        else:
            raise ValueError(
                "Status on compute_energy_gradient function is not recognized!"
            )

        return (
            totenergy,
            (np.array(qmgrad).reshape(structure.coords.shape)) * BOHR_TO_ANGSTROMS,
        )
