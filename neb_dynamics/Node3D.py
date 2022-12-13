from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from retropaths.abinitio.tdstructure import TDStructure
from xtb.ase.calculator import XTB
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.Node import Node


@dataclass
class Node3D(Node):
    tdstructure: TDStructure
    converged: bool = False
    do_climb: bool = False
    _cached_energy: float | None = None
    _cached_gradient: np.array | None = None

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

    @cached_property
    def energy(self):
        if self._cached_energy is not None:
            return self._cached_energy
        else:
            return Node3D.run_xtb_calc(self.tdstructure).get_energy()

    @cached_property
    def gradient(self):
        if self._cached_gradient is not None:
            return self._cached_gradient
        else:
            return Node3D.run_xtb_calc(self.tdstructure).get_gradient() * BOHR_TO_ANGSTROMS

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
        # return np.sum(first * second, axis=1).reshape(-1, 1)
        return np.tensordot(first, second)

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
        '''
        Alessio to Jan: comment your functions motherfucker.
        '''
        pe_grad = gradient
        pe_grad_nudged_const = self.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
        return pe_grad_nudged

    def copy(self):
        return Node3D(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        return Node3D(tdstructure=copy_tdstruct, converged=self.converged, do_climb=self.do_climb)

    def opt_func(self, v=True):
        atoms = Atoms(
            symbols=self.tdstructure.symbols.tolist(),
            positions=self.coords,  # ASE works in angstroms
        )

        atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
        if not v:
            opt = LBFGS(atoms, logfile=None)
        else:
            opt = LBFGS(atoms)
        opt.run(fmax=0.1)

        opt_struct = TDStructure.from_coords_symbols(coords=atoms.positions, symbols=self.tdstructure.symbols, tot_charge=self.tdstructure.charge, tot_spinmult=self.tdstructure.spinmult)  # ASE works in agnstroms

        return opt_struct

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
        assert self.check_symmetric(approx_hess_sym, rtol=1e-3, atol=1e-3), "Hessian not symmetric for some reason"

        return approx_hess_sym

    @property
    def input_tuple(self):
        return (self.tdstructure.atomic_numbers,  self.tdstructure.coords_bohr, 
                self.tdstructure.charge, self.tdstructure.spinmult)