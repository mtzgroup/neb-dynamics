from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import new
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from scipy.signal import argrelextrema
from xtb.ase.calculator import XTB
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.helper_functions import pairwise, quaternionrmsd
# from neb_dynamics.tdstructure import TDStructure
from retropaths.abinitio.tdstructure import TDStructure
# from neb_dynamics.trajectory import Trajectory
from retropaths.abinitio.trajectory import Trajectory


@dataclass
class NoneConvergedException(Exception):
    trajectory: list[Chain]
    msg: str

    obj: NEB


@dataclass
class AlessioError(Exception):
    message: str


class Node(ABC):
    @abstractmethod
    def en_func(coords):
        ...

    @abstractmethod
    def grad_func(coords):
        ...

    @property
    @abstractmethod
    def energy(self):
        ...

    @abstractmethod
    def copy(self):
        ...

    @property
    @abstractmethod
    def gradient(self):
        ...

    @property
    @abstractmethod
    def coords(self):
        ...

    @property
    @abstractmethod
    def do_climb(self):
        ...

    @abstractmethod
    def dot_function(self, other):
        ...

    @abstractmethod
    def update_coords(self, coords):
        ...


@dataclass
class Node2D(Node):
    pair_of_coordinates: np.array
    converged: bool = False
    do_climb: bool = False

    @property
    def coords(self):
        return self.pair_of_coordinates

    @staticmethod
    def en_func(node: Node2D):
        x, y = node.coords
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    @staticmethod
    def en_func_arr(xy_vals):
        x, y = xy_vals
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    @staticmethod
    def grad_func(node: Node2D):
        x, y = node.coords
        dx = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
        dy = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
        return np.array([dx, dy])

    @staticmethod
    def hess_func(node: Node2D):
        x, y = node.coords
        dx2 = 12*x**2 + 4*y - 42
        dy2 = 4*x + 12*y**2 - 26
        
        dxdy = 4*(x + y)
        
        return np.array([
                        [dx2, dxdy],
                        [dxdy, dy2]
                        ])

    @property
    def energy(self) -> float:
        return self.en_func(self)

    @property
    def gradient(self) -> np.array:
        return self.grad_func(self)

    @property
    def hessian(self) -> np.array:
        return self.hess_func(self)

    @staticmethod
    def dot_function(self, other: Node2D) -> float:
        return np.dot(self, other)

    def copy(self):
        return Node2D(
            pair_of_coordinates=self.pair_of_coordinates,
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array):
        new_node = self.copy()
        new_node.pair_of_coordinates = coords
        return new_node


@dataclass
class Node2D_2(Node):
    pair_of_coordinates: np.array
    converged: bool = False
    do_climb: bool = False

    @property
    def coords(self):
        return self.pair_of_coordinates

    @staticmethod
    def en_func(node):
        x, y = node.coords

        A = np.cos(np.pi * x)
        B = np.cos(np.pi * y)
        C = np.pi * np.exp(-np.pi * x ** 2)
        D = (np.exp(-np.pi * (y - 0.8) ** 2)) + np.exp(-np.pi * (y + 0.8) ** 2)
        return A + B + C * D

    @staticmethod
    def en_func_arr(xy_vals):
        x, y = xy_vals
        A = np.cos(np.pi * x)
        B = np.cos(np.pi * y)
        C = np.pi * np.exp(-np.pi * x ** 2)
        D = (np.exp(-np.pi * (y - 0.8) ** 2)) + np.exp(-np.pi * (y + 0.8) ** 2)
        return A + B + C * D

    @staticmethod
    def grad_func(node):
        x, y = node.coords
        A_x = (-2 * np.pi ** 2) * (np.exp(-np.pi * (x ** 2)) * x)
        B_x = np.exp(-np.pi * ((y - 0.8) ** 2)) + np.exp(-np.pi * (y + 0.8) ** 2)
        C_x = -np.pi * np.sin(np.pi * x)

        dx = A_x * B_x + C_x

        A_y = np.pi * np.exp(-np.pi * x ** 2)
        B_y = (-2 * np.pi * np.exp(-np.pi * (y - 0.8) ** 2)) * (y - 0.8)
        C_y = -2 * np.pi * np.exp(-np.pi * (y + 0.8) ** 2) * (y + 0.8)
        D_y = -np.pi * np.sin(np.pi * y)

        dy = A_y * (B_y + C_y) + D_y

        return np.array([dx, dy])

    @staticmethod
    def hess_func(node: Node2D):
        x, y = node.coords


        A_xx = (4*np.pi**3)*np.exp(-np.pi*x**2)*x**2
        B_xx = (np.exp(-np.pi*(y - 0.8)**2) + np.exp(-np.pi*(y + 0.8)**2))
        C_xx = (2*np.pi**2)*(np.exp(-np.pi*x**2))*(np.exp(-np.pi*(y - 0.8)**2) + np.exp(-np.pi*(y+0.8)**2))
        D_xx = np.pi**2 * np.cos(np.pi*x)

        A_yy = np.pi*np.exp(-np.pi*x*2)
        B_yy = 4*np.pi**2 * np.exp(-np.pi*(y - 0.8)**2)*(y - 0.8)**2
        C_yy =4*np.pi**2*np.exp(-np.pi*(y+0.8)**2)*(y+0.8)**2
        D_yy = 2*np.pi*np.exp(-np.pi*(y-0.8)**2)
        E_yy = 2*np.pi*np.exp(-np.pi*(y+0.8)**2)
        F_yy = np.pi**2*np.cos(np.pi*y)


        dx2 = A_xx*B_xx - C_xx - D_xx
        dy2 = A_yy*(B_yy + C_yy - D_yy - E_yy) - F_yy
        
        dxdy = 2.22386*x*(np.exp(np.pi*(y+0.8)**2)*(y - 0.8) + (np.exp(np.pi*(y - 0.8)**2))*(y + 0.8))*np.exp(-3.14159*x**2 - 6.28319*y**2)
        
        return np.array([
                        [dx2, dxdy],
                        [dxdy, dy2]
                        ])

    @property
    def energy(self) -> float:
        return self.en_func(self)

    @property
    def gradient(self) -> np.array:
        return self.grad_func(self)

    @property 
    def hessian(self) -> np.array:
        return self.hess_func(self)

    @staticmethod
    def dot_function(self, other) -> float:
        return np.dot(self, other)

    def copy(self):
        return Node2D_2(
            pair_of_coordinates=self.pair_of_coordinates,
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array):
        new_node = self.copy()
        new_node.pair_of_coordinates = coords
        return new_node

@dataclass
class Node2D_ITM(Node):
    pair_of_coordinates: np.array
    converged: bool = False
    do_climb: bool = False

    @property
    def coords(self):
        return self.pair_of_coordinates

    @staticmethod
    def en_func(node: Node2D_ITM):
        x, y = node.coords
        Ax = 1
        Ay = 1
        return -1*(Ax*np.cos(2*np.pi*x) + Ay*np.cos(2*np.pi*y))

    @staticmethod
    def en_func_arr(xy_vals):
        x, y = xy_vals
        Ax = 1
        Ay = 1
        return -1*(Ax*np.cos(2*np.pi*x) + Ay*np.cos(2*np.pi*y))


    @staticmethod
    def grad_func(node: Node2D_ITM):
        x, y = node.coords
        Ax = 1
        Ay = 1
        dx = 2*Ax*np.pi*np.sin(2*np.pi*x)
        dy = 2*Ay*np.pi*np.sin(2*np.pi*y)
        return np.array([dx, dy])

    @property
    def energy(self) -> float:
        return self.en_func(self)

    @property
    def gradient(self) -> np.array:
        return self.grad_func(self)

    @staticmethod
    def dot_function(self, other: Node2D_ITM) -> float:
        return np.dot(self, other)

    def copy(self):
        return Node2D_ITM(
            pair_of_coordinates=self.pair_of_coordinates,
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array):
        new_node = self.copy()
        new_node.pair_of_coordinates = coords
        return new_node

@dataclass
class Node2D_LEPS(Node):
    pair_of_coordinates: np.array
    converged: bool = False
    do_climb: bool = False

    @property
    def coords(self):
        return self.pair_of_coordinates

    @staticmethod
    def coulomb(r, d, r0, alpha):
        return (d / 2) * ((3 / 2) * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))
    
    @staticmethod
    def exchange(r, d, r0, alpha):
        return (d / 4) * (np.exp(-2 * alpha * (r - r0)) - 6 * np.exp(-alpha * (r - r0)))


    @staticmethod
    def en_func(node: Node2D_LEPS):
        a=0.05
        b=0.30
        c=0.05
        d_ab=4.746
        d_bc=4.746
        d_ac=3.445
        r0=0.742
        alpha=1.942
        inp = node.pair_of_coordinates
        r_ab, r_bc = inp

        Q_AB = Node2D_LEPS.coulomb(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
        Q_BC = Node2D_LEPS.coulomb(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
        Q_AC = Node2D_LEPS.coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        J_AB = Node2D_LEPS.exchange(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
        J_BC = Node2D_LEPS.exchange(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
        J_AC = Node2D_LEPS.exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        result_Qs = (Q_AB / (1 + a)) + (Q_BC / (1 + b)) + (Q_AC / (1 + c))
        result_Js_1 = ((J_AB**2) / ((1 + a) ** 2)) + ((J_BC**2) / ((1 + b) ** 2)) + ((J_AC**2) / ((1 + c) ** 2))
        result_Js_2 = ((J_AB * J_BC) / ((1 + a) * (1 + b))) + ((J_AC * J_BC) / ((1 + c) * (1 + b))) + ((J_AB * J_AC) / ((1 + a) * (1 + c)))
        result_Js = result_Js_1 - result_Js_2

        result = result_Qs - (result_Js) ** (1 / 2)
        return result
    
    @staticmethod
    def grad_x(node: Node2D_LEPS):
        a=0.05
        b=0.30
        c=0.05
        d_ab=4.746
        d_bc=4.746
        d_ac=3.445
        r0=0.742
        alpha=1.942

        inp = node.pair_of_coordinates
        r_ab, r_bc = inp

        ealpha_x = np.exp(alpha * (r0 - r_ab))
        neg_ealpha_x = np.exp(alpha * (r_ab - r0))
        ealpha_y = np.exp(alpha * (r0 - r_bc))
        neg_ealpha_y = np.exp(alpha * (r_bc - r0))

        e2alpha_x = np.exp(2 * alpha * (r0 - r_ab))
        e2alpha_y = np.exp(2 * alpha * (r0 - r_bc))

        aDenom = 1 / (1 + a)
        bDenom = 1 / (1 + b)

        Qconst = Node2D_LEPS.coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
        Jconst = Node2D_LEPS.exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        cDenom = 1 / (1 + c)

        d = d_ab

        dx = (
            0.25
            * aDenom**2
            * alpha
            * d
            * ealpha_x
            * (
                -2 * (1 + a) * (-1 + 3 * ealpha_x)
                + ((-3 + ealpha_x) * (2 * d * ealpha_x * (-6 + ealpha_x) - (1 + a) * d * ealpha_y * (-6 + ealpha_y) * bDenom - 4 * (1 + a) * Jconst * cDenom))
                / (
                    np.sqrt(
                        (
                            ((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2)
                            + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2
                            - d**2 * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc)) * (-1 + 6 * neg_ealpha_x) * (-1 + 6 * neg_ealpha_y) * aDenom * bDenom
                        )
                        - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom
                        - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom)
                        + 16 * Jconst**2 * cDenom**2
                    )
                )
            )
        )

        return dx

    def grad_y(node: Node2D_LEPS):
        a=0.05
        b=0.30
        c=0.05
        d_ab=4.746
        d_bc=4.746
        d_ac=3.445
        r0=0.742
        alpha=1.942
        r_ab, r_bc = node.pair_of_coordinates

        ealpha_x = np.exp(alpha * (r0 - r_ab))
        neg_ealpha_x = np.exp(alpha * (r_ab - r0))
        ealpha_y = np.exp(alpha * (r0 - r_bc))
        neg_ealpha_y = np.exp(alpha * (r_bc - r0))

        e2alpha_x = np.exp(2 * alpha * (r0 - r_ab))
        e2alpha_y = np.exp(2 * alpha * (r0 - r_bc))

        aDenom = 1 / (1 + a)
        bDenom = 1 / (1 + b)

        Qconst = Node2D_LEPS.coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
        Jconst = Node2D_LEPS.exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        cDenom = 1 / (1 + c)

        d = d_bc

        dy = (
            0.25
            * bDenom**2
            * alpha
            * d
            * ealpha_y
            * (
                -2 * (1 + b) * (-1 + 3 * ealpha_y)
                + ((-3 + ealpha_y) * (2 * d * ealpha_y * (-6 + ealpha_y) - (1 + b) * d * ealpha_x * (-6 + ealpha_x) * aDenom - 4 * (1 + b) * Jconst * cDenom))
                / (
                    np.sqrt(
                        (
                            ((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2)
                            + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2
                            - d**2 * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc)) * (-1 + 6 * neg_ealpha_x) * (-1 + 6 * neg_ealpha_y) * aDenom * bDenom
                        )
                        - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom
                        - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom)
                        + 16 * Jconst**2 * cDenom**2
                    )
                )
            )
        )

        return dy

    def dQ_dr(d, alpha, r, r0):
        return (d / 2) * ((3 / 2) * (-2 * alpha * np.exp(-2 * alpha * (r - r0))) + alpha * np.exp(-alpha * (r - r0)))


    def dJ_dr(d, alpha, r, r0):
        return (d / 4) * (np.exp(-2 * alpha * (r - r0)) * (-2 * alpha) + 6 * alpha * np.exp(-alpha * (r - r0)))


    @staticmethod
    def grad_func(node: Node2D_LEPS):
        return np.array([Node2D_LEPS.grad_x(node), Node2D_LEPS.grad_y(node)])

    @property
    def energy(self) -> float:
        return self.en_func(self)

    @property
    def gradient(self) -> np.array:
        return self.grad_func(self)

    @staticmethod
    def dot_function(self, other: Node2D_LEPS) -> float:
        return np.dot(self, other)

    def copy(self):
        return Node2D_LEPS(
            pair_of_coordinates=self.pair_of_coordinates,
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array):
        new_node = self.copy()
        new_node.pair_of_coordinates = coords
        return new_node




@dataclass
class Node3D(Node):
    tdstructure: TDStructure
    converged: bool = False
    do_climb: bool = False

    @property
    def coords(self):
        return self.tdstructure.coords

    @property
    def coords_bohr(self):
        return self.tdstructure.coords*ANGSTROM_TO_BOHR

    @staticmethod
    def en_func(node: Node3D):
        res = Node3D.run_xtb_calc(node.tdstructure)
        return res.get_energy()

    @staticmethod
    def grad_func(node: Node3D):
        res = Node3D.run_xtb_calc(node.tdstructure)
        return res.get_gradient()*BOHR_TO_ANGSTROMS

    @cached_property
    def energy(self):
        return Node3D.run_xtb_calc(self.tdstructure).get_energy()

    @cached_property
    def gradient(self):
        return Node3D.run_xtb_calc(self.tdstructure).get_gradient()*BOHR_TO_ANGSTROMS

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

    def copy(self):
        return Node3D(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        return Node3D(
            tdstructure=copy_tdstruct, converged=self.converged, do_climb=self.do_climb
        )

    def opt_func(self, v=True):
        atoms = Atoms(
                symbols = self.tdstructure.symbols.tolist(),
                positions = self.coords, # ASE works in angstroms
            )

        atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
        if not v: opt = LBFGS(atoms, logfile=None)
        else: opt = LBFGS(atoms)
        opt.run(fmax=0.1)

        opt_struct = TDStructure.from_coords_symbols(
            coords=atoms.positions, # ASE works in agnstroms
            symbols=self.tdstructure.symbols,
            tot_charge=self.tdstructure.charge,
            tot_spinmult=self.tdstructure.spinmult)

        return opt_struct


    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @property
    def hessian(self: Node3D):
        dr = 0.01 # displacement vector, Bohr
        numatoms = self.tdstructure.atomn
        approx_hess = []
        for n in range(numatoms):
        # for n in range(2):
            grad_n = []
            for coord_ind, coord_name in enumerate(['dx', 'dy', 'dz']):
                # print(f"doing atom #{n} | {coord_name}")

                coords = np.array(self.coords, dtype='float64')

                # print(coord_ind, coords[n, coord_ind])
                coords[n, coord_ind] = coords[n, coord_ind] + dr
                # print(coord_ind, coords[n, coord_ind])

                node2 = self.copy()
                node2 = node2.update_coords(coords)

                delta_grad = (node2.gradient - self.gradient ) / dr
                # print(delta_grad)
                grad_n.append(delta_grad.flatten())


            approx_hess.extend(grad_n)
        approx_hess = np.array(approx_hess)
        # raise AlessioError(f"{approx_hess.shape}")
        
        approx_hess_sym = 0.5*(approx_hess + approx_hess.T)
        assert self.check_symmetric(approx_hess_sym, rtol=1e-3, atol=1e-3), 'Hessian not symmetric for some reason'
        
        return approx_hess_sym



@dataclass
class Chain:
    nodes: List[Node]
    k: Union[List[float], float]
    delta_k: float = 0
    step_size: float = 1
    velocity: np.array = np.zeros(1)
    node_class: Node = Node3D



    @classmethod
    def from_xyz(cls, fp: Path, k=0.1, delta_k=0, step_size=1, velocity=np.zeros(1), node_class=Node3D):
        traj = Trajectory.from_xyz(fp)
        chain = cls.from_traj(traj, k=k, delta_k=delta_k, step_size=step_size, velocity=velocity, node_class=node_class)
        return chain

    @property
    def integrated_path_length(self):
        
        cum_sums = [0]
        
        int_path_len = [0]
        for i, frame in enumerate(self.nodes):
            if i == len(self.nodes) -1 :
                continue
            next_frame = self.nodes[i+1]
            dist_vec = next_frame.coords - frame.coords
            cum_sums.append(cum_sums[-1]+np.linalg.norm(dist_vec))
            # proj = (frame.dot_function(dist_vec, endpoint_vec) / frame.dot_function(endpoint_vec, endpoint_vec))*endpoint_vec

            # proj_dist = np.linalg.norm(proj)
            # int_path_len.append(int_path_len[-1]+proj_dist)
        cum_sums = np.array(cum_sums)
        int_path_len = cum_sums / cum_sums[-1]
        return np.array(int_path_len)

    def neighs_grad_func(
        self, prev_node: Node, current_node: Node, next_node: Node
    ):

        vec_tan_path = self._create_tangent_path(
            prev_node=prev_node, current_node=current_node, next_node=next_node
        )
        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        if not current_node.do_climb:
            pe_grads_nudged = self.get_pe_grad_nudged(
                current_node=current_node, unit_tan_path=unit_tan_path
            )
            spring_forces_nudged = self.get_force_spring_nudged(
                prev_node=prev_node,
                current_node=current_node,
                next_node=next_node,
                unit_tan_path=unit_tan_path,
            )

        elif current_node.do_climb:

            pe_grad = current_node.gradient
            pe_along_path_const = current_node.dot_function(pe_grad, unit_tan_path)
            pe_along_path = pe_along_path_const * unit_tan_path

            climbing_grad = 2 * pe_along_path

            pe_grads_nudged = pe_grad - climbing_grad

            zero = np.zeros_like(pe_grad)
            spring_forces_nudged = zero
            

        else:
            raise ValueError(
                f"current_node.do_climb is not a boolean: {current_node.do_climb=}"
            )

        return pe_grads_nudged, spring_forces_nudged #, anti_kinking_grads

    def _k_between_nodes(
        self, node0: Node, node1: Node, e_ref: float, k_max: float, e_max: float
    ):
        e_i = max(node1.energy, node0.energy)
        if e_i > e_ref:
            new_k = k_max - self.delta_k * ((e_max - e_i) / (e_max - e_ref))
        elif e_i <= e_ref:
            new_k = k_max - self.delta_k
        return new_k

    def plot_chain(self):
        s = 8
        fs = 18
        f,ax = plt.subplots(figsize=(1.16*s, s))
        plt.plot(self.integrated_path_length, (self.energies-self.energies[0])*627.5, 'o--', label='neb')
        plt.ylabel("Energy (kcal/mol)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.show()

    def __getitem__(self, index):
        return self.nodes.__getitem__(index)

    def __len__(self):
        return len(self.nodes)

    def copy(self):
        list_of_nodes = [node.copy() for node in self.nodes]
        chain_copy = Chain(
            nodes=list_of_nodes,
            k=self.k,
            delta_k=self.delta_k,
            step_size=self.step_size,
            velocity=self.velocity,
        )
        return chain_copy

    def iter_triplets(self):
        for i in range(1, len(self.nodes) - 1):
            yield self.nodes[i - 1 : i + 2]

    @classmethod
    def from_traj(cls, traj, k, delta_k, step_size, node_class, velocity=None):
        if velocity == None:
            velocity = np.zeros_like([struct.coords for struct in traj])
        nodes = [node_class(s) for s in traj]
        return Chain(
            nodes, k=k, delta_k=delta_k, step_size=step_size, velocity=velocity
        )

    @classmethod
    def from_list_of_coords(
        cls,
        k,
        list_of_coords: List,
        node_class: Node,
        delta_k: float,
        step_size: float,
        velocity=None,
    ) -> Chain:

        if velocity == None:
            velocity = np.zeros_like([c for c in list_of_coords])
        nodes = [node_class(point) for point in list_of_coords]
        return cls(
            nodes=nodes, k=k, delta_k=delta_k, step_size=step_size, velocity=velocity
        )

    @property
    def path_distances(self):
        dist = []
        for i in range(len(self.nodes)):
            if i == 0: continue
            start = self.nodes[i-1]
            end = self.nodes[i]
            
            dist.append(quaternionrmsd(start.coords, end.coords))
        
        return np.array(dist)

    @cached_property
    def work(self) -> float:
        ens = self.energies
        ens -= ens[0]

        works = np.abs(ens[1:]*self.path_distances)
        tot_work = works.sum()
        return tot_work

    @cached_property
    def energies(self) -> np.array:
        return np.array([node.energy for node in self.nodes])

    @cached_property
    def gradients(self) -> np.array:
        pe_grads_nudged = []
        spring_forces_nudged = []
        # anti_kinking_grads = []
        for prev_node, current_node, next_node in self.iter_triplets():
            
            (
                pe_grad_nudged,
                spring_force_nudged) = self.neighs_grad_func(
                prev_node=prev_node, current_node=current_node, next_node=next_node
            )


            # anti_kinking_grads.append(anti_kinking_grad)
            if not current_node.converged:
                pe_grads_nudged.append(pe_grad_nudged)
                spring_forces_nudged.append(spring_force_nudged)
            else:
                zero = np.zeros_like(pe_grad_nudged)
                pe_grads_nudged.append(zero)
                spring_forces_nudged.append(zero)

        pe_grads_nudged = np.array(pe_grads_nudged)
        spring_forces_nudged = np.array(spring_forces_nudged)
        # anti_kinking_grads = np.array(anti_kinking_grads)
        
        grads = (pe_grads_nudged - spring_forces_nudged) #+ self.k * anti_kinking_grads

        zero = np.zeros_like(grads[0])
        grads = np.insert(grads, 0, zero, axis=0)
        grads = np.insert(grads, len(grads), zero, axis=0)

        # remove rotations and translations
        if grads.shape[1] >= 3:  # if we have at least 3 atoms
            grads[:, 0, :] = 0  # this atom cannot move
            grads[:, 1, :2] = 0  # this atom can only move in a line
            grads[:, 2, :1] = 0  # this atom can only move in a plane

        return grads

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

    def _create_tangent_path(
        self, prev_node: Node, current_node: Node, next_node: Node
    ):
        en_2 = next_node.energy
        en_1 = current_node.energy
        en_0 = prev_node.energy
        if en_2 > en_1 and en_1 > en_0:
            return next_node.coords - current_node.coords
        elif en_2 < en_1 and en_1 < en_0:
            return current_node.coords - prev_node.coords

        else:
            deltaV_max = max(np.abs(en_2 - en_1), np.abs(en_0 - en_1))
            deltaV_min = min(np.abs(en_2 - en_1), np.abs(en_0 - en_1))


            tau_plus = next_node.coords - current_node.coords
            tau_minus = current_node.coords - prev_node.coords
            if en_2 > en_0:
                tan_vec = deltaV_max*tau_plus  + deltaV_min*tau_minus
            elif en_2 < en_0:
                tan_vec = deltaV_min*tau_plus + deltaV_max*tau_minus

            else:
                return 0.5*(tau_minus + tau_plus)
                # raise ValueError(
                #     f"Energies adjacent to current node are identical. {en_2=} {en_0=}"
                # )

            return tan_vec

    def _get_nudged_pe_grad(self, node, unit_tangent):
        pe_grad = node.gradient
        pe_grad_nudged_const = node.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent

        return pe_grad_nudged

    def _get_anti_kink_switch_func(
        self, prev_node, current_node, next_node
    ):
        # ANTI-KINK FORCE
        vec_2_to_1 = next_node.coords - current_node.coords
        vec_1_to_0 = current_node.coords - prev_node.coords
        cos_phi = current_node.dot_function(vec_2_to_1, vec_1_to_0) / (
            np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0)
        )

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))
        return f_phi

    def get_pe_grad_nudged(self, current_node: Node, unit_tan_path):

        pe_grad_nudged = self._get_nudged_pe_grad(
            node=current_node, unit_tangent=unit_tan_path
        )
        return pe_grad_nudged

    def get_force_spring_nudged(
        self,
        prev_node: Node,
        current_node: Node,
        next_node: Node,
        unit_tan_path: np.array,
    ):
        
        k_max = max(self.k) if hasattr(self.k, "__iter__") else self.k
        e_ref = max(self.nodes[0].energy, self.nodes[-1].energy)
        e_max = max(self.energies)
        # print(f"***{e_max=}//{e_ref=}//{k_max=}")

        k01 = self._k_between_nodes(
                    node0=prev_node,
                    node1=current_node,
                    e_ref=e_ref,
                    k_max=k_max,
                    e_max=e_max,
                )

        k12 = self._k_between_nodes(
            node0=current_node,
            node1=next_node,
            e_ref=e_ref,
            k_max=k_max,
            e_max=e_max,
        )

        # print(f"***{k12=} // {k01=}")

        force_spring = k12*np.linalg.norm(next_node.coords - current_node.coords) - k01*np.linalg.norm(current_node.coords - prev_node.coords)
        return force_spring*unit_tan_path


@dataclass
class NEB:
    initial_chain: Chain

    redistribute: bool = False
    remove_folding: bool = False
    climb: bool = False
    en_thre: float = 0.0045/450
    rms_grad_thre: float = 450*(2/3)
    grad_thre: float = 0.0045
    # mag_grad_thre: float = 1e-4
    max_steps: float = 1000

    vv_force_thre: float = 1.0

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)
    v: bool= True

    def do_velvel(self, chain: Chain):
        max_force_on_node = max([np.linalg.norm(grad) for grad in chain.gradients])
        return max_force_on_node < self.vv_force_thre

    def _reset_node_convergence(self, chain):
        for node in chain:
            node.converged = False


    def set_climbing_nodes(self, chain: Chain):
        # reset node convergence
        self._reset_node_convergence(chain=chain)

        inds_maxima = argrelextrema(chain.energies, np.greater, order=2)[0]
        if self.v > 1: print(f"----->Setting {len(inds_maxima)} nodes to climb")

        for ind in inds_maxima:
            chain[ind].do_climb = True

    def optimize_chain(self):
        nsteps = 1
        chain_previous = self.initial_chain.copy()

        while nsteps < self.max_steps + 1:
            

            max_grad_val = np.max([np.linalg.norm(grad) for grad in chain_previous.gradients])
            if max_grad_val<= 3*self.grad_thre and self.climb: 
                self.set_climbing_nodes(chain=chain_previous)
                self.climb = False

            new_chain = self.update_chain(chain=chain_previous)
            if self.v > 1: 
                print(
                f"step {nsteps} // max |gradient| {np.max([np.linalg.norm(grad) for grad in new_chain.gradients])}"
                )
            if self.v: 
                print(f'on step {nsteps}                       ', end='\r')
            sys.stdout.flush()

            self.chain_trajectory.append(new_chain)
            self.gradient_trajectory.append(new_chain.gradients)

            if self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
                if self.v: print(f"\nChain converged!")
                original_chain_len = len(new_chain)

                if self.remove_folding:
                    new_chain = self.remove_chain_folding(chain=new_chain.copy())
                    self.chain_trajectory.append(new_chain)

                if self.redistribute:

                    new_chain = self.redistribute_chain(
                        chain=new_chain.copy(),
                        requested_length_of_chain=original_chain_len,
                    )
                    self.chain_trajectory.append(new_chain)



                self.optimized = new_chain
                return
            chain_previous = new_chain.copy()


            nsteps += 1

        new_chain = self.update_chain(chain=chain_previous)
        if not self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
            raise NoneConvergedException(
                trajectory=self.chain_trajectory,
                msg=f"\nChain did not converge at step {nsteps}",
                obj=self,
            )

    def get_chain_velocity(self, chain: Chain) -> np.array:
        prev_velocity = chain.velocity

        step = self.grad_thre/10 # make the step size rel. to threshold we want

        new_force = -(chain.gradients) * step

        directions = prev_velocity * new_force
        prev_velocity[
            directions < 0
        ] = 0  # zero the velocities for which we overshot the minima

        new_velocity = prev_velocity + new_force
        return new_velocity

    def update_chain(self, chain: Chain) -> Chain:
        from neb_dynamics import ALS

        do_vv = self.do_velvel(chain=chain)

        if do_vv:
            velocity = self.get_chain_velocity(chain=chain)
            new_chain_coordinates = chain.coordinates + velocity

        else:
            disp = ALS.ArmijoLineSearch(chain=chain, t=chain.step_size, alpha=0.01, beta=0.5, grad=chain.gradients)
            new_chain_coordinates = (
                chain.coordinates - chain.gradients * disp
            )

        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = Chain(
            new_nodes,
            k=chain.k,
            delta_k=chain.delta_k,
            step_size=chain.step_size,
            velocity=chain.velocity,
        )
        if do_vv:
            new_chain.velocity = velocity

        return new_chain

    def _update_node_convergence(self, chain: Chain, indices: np.array) -> None:
        for i, node in enumerate(chain):
            if i in indices:
                node.converged = True
            else: 
                node.converged = False

    def _check_en_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
        differences = np.abs(chain_new.energies - chain_prev.energies)
        indices_converged = np.where(differences < self.en_thre)

        return indices_converged[0], differences

    def _check_grad_converged(self, chain: Chain) -> bool:
        # delta_grad = np.abs(chain_prev.gradients - chain_new.gradients)
        # mag_grad = np.array([np.linalg.norm(grad) for grad in chain_new.gradients])
        bools = []
        max_grad_components = []
        gradients = chain.gradients
        for grad in gradients:
            max_grad = np.amax(grad)
            max_grad_components.append(max_grad)
            bools.append(max_grad < self.grad_thre)


        # grad_converged = np.where(delta_grad < self.grad_thre)
        # mag_converged = np.where(mag_grad < self.mag_grad_thre)
        # mag_converged = mag_grad < self.mag_grad_thre

        # return delta_converged[0], mag_converged[0], delta_grad, mag_grad
        # return delta_converged[0], delta_grad
        return np.where(bools), max_grad_components

    def _check_rms_grad_converged(self, chain: Chain):
        bools = []
        rms_grads = []
        grads = chain.gradients
        for grad in grads:
            rms_gradient= np.sqrt(np.mean(np.square(grad)))
            rms_grads.append(rms_gradient)
            rms_grad_converged = rms_gradient < self.rms_grad_thre
            bools.append(rms_grad_converged)

        return np.where(bools), rms_grads


    def _chain_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:

        rms_grad_conv_ind, max_rms_grads = self._check_rms_grad_converged(chain_new)
        en_converged_indices, en_deltas = self._check_en_converged(
            chain_prev=chain_prev, chain_new=chain_new
        )
        
        grad_conv_ind, max_grad_components = self._check_grad_converged(chain=chain_new)



        converged_nodes_indices = np.intersect1d(
            en_converged_indices, rms_grad_conv_ind
        )
        converged_nodes_indices = np.intersect1d(converged_nodes_indices,  grad_conv_ind)

        
        
        # [print(f"\t\tnode{i} | ∆E : {en_deltas[i]} | Max(∆Grad) : { np.amax(grad_deltas[i])} | |Grad| : {mag_grad_deltas[i]} | Converged? : {chain_new.nodes[i].converged}") for i in range(len(chain_new))]
        if self.v > 1: [print(f"\t\tnode{i} | ∆E : {en_deltas[i]} | Max(RMS Grad): {max_rms_grads[i]} | Max(Grad components): {max_grad_components[i]} | Converged? : {chain_new.nodes[i].converged}") for i in range(len(chain_new))]
        if self.v > 1: print(f"\t{len(converged_nodes_indices)} nodes have converged")

        self._update_node_convergence(chain=chain_new, indices=converged_nodes_indices)
        # return len(converged_node_indices) == len(chain_new.nodes)

        return len(converged_nodes_indices) == len(chain_new)

        # return np.all(
        #     [np.linalg.norm(grad) <= self.grad_thre for grad in chain_new.gradients]
        # )

        # return en_bool and grad_bool

    def remove_chain_folding(self, chain: Chain) -> Chain:
        not_converged = True
        count = 0
        points_removed = []
        while not_converged:
            if self.v > 1: print(f"anti-folding: on count {count}...")
            new_chain = []
            new_chain.append(chain[0])

            for prev_node, current_node, next_node in chain.iter_triplets():
                vec1 = current_node.coords - prev_node.coords
                vec2 = next_node.coords - current_node.coords

                if np.all(current_node.dot_function(vec1, vec2)) > 0:
                    new_chain.append(current_node)
                else:
                    points_removed.append(current_node)

            new_chain.append(chain[-1])
            new_chain = Chain(nodes=new_chain, k=chain.k)
            if self._check_dot_product_converged(new_chain):
                not_converged = False
            chain = new_chain.copy()
            count += 1

        return chain

    def _check_dot_product_converged(self, chain: Chain) -> bool:
        dps = []
        for prev_node, current_node, next_node in chain.iter_triplets():
            vec1 = current_node.coords - prev_node.coords
            vec2 = next_node.coords - current_node.coords
            dps.append(current_node.dot_function(vec1, vec2) > 0)

        return all(dps)

    def redistribute_chain(self, chain: Chain, requested_length_of_chain: int) -> Chain:
        # if len(chain) < requested_length_of_chain:
        #     fixed_chain = chain.copy()
        #     [
        #         fixed_chain.nodes.insert(1, fixed_chain[1])
        #         for _ in range(requested_length_of_chain - len(chain))
        #     ]
        #     chain = fixed_chain

        direction = np.array(
            [
                next_node.coords - current_node.coords
                for current_node, next_node in pairwise(chain)
            ]
        )
        distances = np.linalg.norm(direction, axis=1)
        tot_dist = np.sum(distances)
        cumsum = np.cumsum(distances)  # cumulative sum
        cumsum = np.insert(cumsum, 0, 0)

        distributed_chain = []
        for num in np.linspace(0, tot_dist, len(chain)):
            new_node = self.redistribution_helper(num=num, cum=cumsum, chain=chain)

            distributed_chain.append(new_node)

        distributed_chain[0] = chain[0]
        distributed_chain[-1] = chain[-1]

        return Chain(distributed_chain, k=chain.k)

    def redistribution_helper(self, num, cum, chain: Chain) -> Node:
        """
        num: the distance from first node to return output point to
        cum: cumulative sums
        new_chain: chain that we are considering

        """

        for ii, ((cum_sum_init, node_start), (cum_sum_end, node_end)) in enumerate(
            pairwise(zip(cum, chain))
        ):

            if cum_sum_init <= num < cum_sum_end:
                direction = node_end.coords - node_start.coords
                percentage = (num - cum_sum_init) / (cum_sum_end - cum_sum_init)

                new_node = node_start.copy()
                new_coords = node_start.coords + (direction * percentage)
                new_node = new_node.update_coords(new_coords)

                return new_node

    def write_to_disk(self, fp: Path):
        out_traj = Trajectory([node.tdstructure for node in self.chain_trajectory[-1].nodes])
        out_traj.write_trajectory(fp)


# @dataclass
# class Dimer:
#     initial_node: Node
#     delta_r: float
#     step_size: float
#     d_theta: float
#     optimized_dimer = None
#     en_thre: float = 1e-7


#     @property
#     def ts_node(self):
#         if self.optimized_dimer:
#             final_unit_dir = self.get_unit_dir(self.optimized_dimer)
#             r1, r2 = self.optimized_dimer
#             ts = self.initial_node.copy()
#             ts_coords = r1.coords + self.delta_r*final_unit_dir
#             ts = ts.update_coords(ts_coords)


#             return ts


#     def make_initial_dimer(self):
#         random_vec = np.random.rand(*self.initial_node.coords.shape)
#         random_unit_vec = random_vec / np.linalg.norm(random_vec)
        
#         r1_coords = self.initial_node.coords - self.delta_r*random_unit_vec
#         r2_coords = self.initial_node.coords + self.delta_r*random_unit_vec


#         r1 = self.initial_node.copy()
#         r1 = r1.update_coords(r1_coords)

#         r2 = self.initial_node.copy()
#         r2 = r2.update_coords(r2_coords)

#         dimer = np.array([r1, r2])

#         return dimer

#     def get_dimer_energy(self, dimer):
#         r1,r2 = dimer

#         return r1.energy + r2.energy


#     def force_func(self, node: Node):
#         return - node.gradient

#     def force_perp(self, r_vec: Node, unit_dir: np.array):
#         force_r_vec = - r_vec.gradient
#         return force_r_vec - r_vec.dot_function(force_r_vec, unit_dir)*unit_dir

#     def get_unit_dir(self, dimer):
#         r1, r2 = dimer
#         return (r2.coords - r1.coords)/np.linalg.norm(r2.coords-r1.coords)

#     def get_dimer_force_perp(self, dimer):
#         r1, r2 = dimer
#         unit_dir= self.get_unit_dir(dimer)
        
#         f_r1 = self.force_perp(r1, unit_dir=unit_dir)
#         f_r2 = self.force_perp(r2, unit_dir=unit_dir)
#         return f_r2 - f_r1



#     def _attempt_rot_step(self, dimer: np.array,theta_rot, t, unit_dir):
    
#         # update dimer endpoint
#         _, r2 = dimer
#         r2_prime_coords = r2.coords + (unit_dir*np.cos(t) + theta_rot*np.sin(t))*self.delta_r
#         r2_prime = self.initial_node.copy()
#         r2_prime = r2_prime.update_coords(r2_prime_coords)

#         # calculate new unit direction
#         midpoint_coords = r2.coords - unit_dir*self.delta_r
#         new_dir = (r2_prime_coords - midpoint_coords)
#         new_unit_dir = new_dir / np.linalg.norm(new_dir)
        
#         # remake dimer using new unit direciton
#         r1_prime_coords = r2_prime_coords - 2*self.delta_r*new_unit_dir
#         r1_prime = self.initial_node.copy()
#         r1_prime = r1_prime.update_coords(r1_prime_coords)

#         new_dimer = np.array([r1_prime, r2_prime])
#         new_grad = self.get_dimer_force_perp(new_dimer)
        
#         # en_struct_prime = np.linalg.norm(new_grad)
#         en_struct_prime = np.dot(new_grad, theta_rot)
#         # en_struct_prime = self.get_dimer_energy(new_dimer)
    
#         return en_struct_prime, t

#     def _rotate_img(self, r_vec: Node, unit_dir: np.array, theta_rot: float, dimer: np.array,dt,  
#     alpha=0.000001, beta=0.5):
        
#         # max_steps = 10
#         # count = 0
        


#         # grad = self.get_dimer_force_perp(dimer)
#         # # en_struct = np.linalg.norm(grad)
#         # en_struct = np.dot(grad, theta_rot)
#         # # en_struct = self.get_dimer_energy(dimer)

#         # en_struct_prime, t = self._attempt_rot_step(dimer=dimer, t=self.d_theta, unit_dir=unit_dir, theta_rot=theta_rot)

#         # condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

#         # while condition and count < max_steps:
#         #     t *= beta
#         #     count += 1
#         #     en_struct = en_struct_prime

#         #     en_struct_prime, t = self._attempt_rot_step(dimer=dimer, t=self.d_theta, unit_dir=unit_dir, theta_rot=theta_rot)

#         #     condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

#         # print(f"\t\t\t{t=} {count=} || force: {np.linalg.norm(grad)}")
#         # sys.stdout.flush()
        


#         # return r_vec.coords + (unit_dir*np.cos(t) + theta_rot*np.sin(t))*self.delta_r
#         return r_vec.coords + (unit_dir*np.cos(dt) + theta_rot*np.sin(dt))*self.delta_r
        

    

#     def _rotate_dimer(self, dimer):
#         """
#         from this paper https://aip.scitation.org/doi/pdf/10.1063/1.480097
#         """
#         _, r2 = dimer
#         unit_dir = self.get_unit_dir(dimer)
#         midpoint_coords = r2.coords - unit_dir*self.delta_r
#         midpoint = self.initial_node.copy()
#         midpoint = midpoint.update_coords(midpoint_coords)
        
#         dimer_force_perp = self.get_dimer_force_perp(dimer)
#         theta_rot = dimer_force_perp / np.linalg.norm(dimer_force_perp)




#         r2_prime_coords = self._rotate_img(r_vec=midpoint, unit_dir=unit_dir, theta_rot=theta_rot, dimer=dimer, dt=self.d_theta)
#         r2_prime = self.initial_node.copy()
#         r2_prime = r2_prime.update_coords(r2_prime_coords)
        
#         new_dir = (r2_prime_coords - midpoint_coords)
#         new_unit_dir = new_dir / np.linalg.norm(new_dir)
        
#         r1_prime_coords = r2_prime_coords - 2*self.delta_r*new_unit_dir
#         r1_prime = self.initial_node.copy()
#         r1_prime = r1_prime.update_coords(r1_prime_coords)


#         dimer_prime = np.array([r1_prime, r2_prime])
#         dimer_prime_force_perp = self.get_dimer_force_perp(dimer_prime)
#         theta_rot_prime = dimer_prime_force_perp / np.linalg.norm(dimer_prime_force_perp)
#         f_prime = (np.dot(dimer_prime_force_perp, theta_rot_prime) - np.dot(dimer_force_perp, theta_rot))/self.d_theta

#         # optimal_d_theta = (np.dot(dimer_force_perp, theta_rot) + np.dot(dimer_prime_force_perp, theta_rot_prime))/(-2*f_prime)


#         r1_p, r2_p = dimer_prime
#         f_r1_p = self.force_perp(r1_p, unit_dir=unit_dir)
#         f_r2_p = self.force_perp(r2_p, unit_dir=unit_dir)

#         r1, r2 = dimer
#         f_r1 = self.force_perp(r1, unit_dir=unit_dir)
#         f_r2 = self.force_perp(r2, unit_dir=unit_dir)



#         f_val = 0.5*(np.dot(f_r2_p - f_r1_p, theta_rot_prime) + np.dot(f_r2 - f_r1, theta_rot))
#         f_val_prime = (1/self.d_theta)*(np.dot(f_r2_p - f_r1_p, theta_rot_prime) - np.dot(f_r2 - f_r1, theta_rot))

#         optimal_d_theta = 0.5*np.arctan(2*f_val/f_val_prime) - self.d_theta/2
#         print(f"\t\t{optimal_d_theta=}")
#         r2_final_coords = self._rotate_img(r_vec=midpoint, unit_dir=unit_dir, theta_rot=theta_rot, dimer=dimer, dt=optimal_d_theta)
#         r2_final = self.initial_node.copy()
#         r2_final = r2_final.update_coords(r2_final_coords)

#         final_dir = (r2_final_coords - midpoint_coords)
#         final_unit_dir = final_dir / np.linalg.norm(final_dir)

#         r1_final_coords = r2_final_coords - 2*self.delta_r*final_unit_dir
#         r1_final = self.initial_node.copy()
#         r1_final = r1_final.update_coords(r1_final_coords)

#         dimer_final = np.array([r1_final, r2_final])


#         return dimer_final

#     def _translate_dimer(self, dimer):
#         dimer_0 = dimer
        
#         r1,r2 = dimer_0
#         force = self.get_climb_force(dimer_0)




#         r2_prime_coords = r2.coords + self.step_size*force
#         r2_prime = self.initial_node.copy()
#         r2_prime = r2_prime.update_coords(r2_prime_coords)


#         r1_prime_coords = r1.coords + self.step_size*force
#         r1_prime = self.initial_node.copy()
#         r1_prime = r1_prime.update_coords(r1_prime_coords)
        
        
#         dimer_1 = (r1_prime, r2_prime)

#         return dimer_1


#     def fully_update_dimer(self, dimer):
#         dimer_0 = dimer
#         en_0 = self.get_dimer_energy(dimer_0)
        
#         dimer_0_prime = self._rotate_dimer(dimer_0)
#         dimer_1 = self._translate_dimer(dimer_0_prime)
#         en_1 = self.get_dimer_energy(dimer_1)
        
#         n_counts = 0
#         while np.abs(en_1 - en_0) > self.en_thre and n_counts < 100000:
#             # print(f"{n_counts=} // |∆E|: {np.abs(en_1 - en_0)}")
#             dimer_0 = dimer_1
#             en_0 = self.get_dimer_energy(dimer_0)

#             dimer_0_prime = self._rotate_dimer(dimer_0)
#             dimer_1 = self._translate_dimer(dimer_0_prime)

#             en_1 = self.get_dimer_energy(dimer_1)
#             n_counts+=1
            
#         if np.abs(en_1 - en_0) <= self.en_thre: print(f"Optimization converged in {n_counts} steps!")
#         else: print(f"Optimization did not converge. Final |∆E|: {np.abs(en_1 - en_0)}")
#         return dimer_1



   
#     def get_climb_force(self, dimer):
#         r1, r2 = dimer
#         unit_path = self.get_unit_dir(dimer)
        
        
#         f_r1 = self.force_func(r1)
#         f_r2 = self.force_func(r2)
#         F_R = f_r1 + f_r2
        
        
#         f_parallel_r1 = self.initial_node.dot_function(f_r1, unit_path)*unit_path
#         f_parallel_r2 = self.initial_node.dot_function(f_r2, unit_path)*unit_path
#         F_Par = f_parallel_r1 + f_parallel_r2
        

#         return F_R - 2*F_Par
        


#     def find_ts(self):
#         dimer = self.make_initial_dimer()
#         opt_dimer = self.fully_update_dimer(dimer)
#         self.optimized_dimer = opt_dimer


@dataclass
class TS_PRFO:
    initial_node: Node
    max_nsteps: int = 2000
    dr: float = 0.1
    grad_thre: float = 1e-6

    @property
    def ts(self):
        ts, _ = self.ts_and_traj
        return ts

    @property
    def traj(self):
        _, traj = self.ts_and_traj
        return traj
        
    @cached_property
    def ts_and_traj(self):
        return self.find_ts()


    def get_lambda_n(self, all_eigenvalues, f_vector, tol=10e-3, break_limit=1e6):
        """
        f_vector and all_eigsenvalues need to be ordered
        from lowest to highest based on eigenvalue value
        """
        break_ind = 0
        lamb_guess = 0
        hist = [lamb_guess]
        while break_ind < break_limit:
            tot_lamb = 0
            for eigval_i,f_i in zip(all_eigenvalues, f_vector):
                tot_lamb+= (f_i**2)/(lamb_guess - eigval_i)
            if np.abs(tot_lamb - lamb_guess) <= tol:
                return tot_lamb, hist
            else:
                lamb_guess = tot_lamb
            break_ind+=1
            hist.append(lamb_guess)
        return None, hist
    

    def get_lambda_p(self, eigval, f_i):
        lambda_p = 0.5*eigval + 0.5*(np.sqrt(eigval**2 + 4*(f_i)**2))
        return lambda_p

    def prfo_step(self, node):
        grad = node.gradient
        H = node.hessian
        H_evals, H_evecs = np.linalg.eigh(H)
        orig_dim = grad.shape
        grad_reshaped = grad.reshape(-1,1)
        F_vec = np.dot(H_evecs.T, grad_reshaped)

        lambda_p = self.get_lambda_p(H_evals[0], F_vec[0])
        lambda_n, _ = self.get_lambda_n(all_eigenvalues=H_evals, f_vector=F_vec, break_limit=1e6)
        if type(lambda_n)==type(None): raise NoneConvergedException("lambda_n calculation failed. maybe start again with a different initial guess")

        h0 = (-1*F_vec[0]*H_evecs[:, 0])/(H_evals[0] - lambda_p)
        hrest = sum([(-1*F_vec[i]*H_evecs[:, i])/(H_evals[i] - lambda_n) for i in range(1, len(F_vec))])
        step =  h0+hrest
        step_reshaped = step.reshape(orig_dim)
        return step_reshaped


    def find_ts(self):
        traj = []
        traj.append(self.initial_node)
        
        steps_taken = 0
        converged = False
        
        start_node = self.initial_node
        
        grad_mag = np.linalg.norm(start_node.gradient)
        converged = grad_mag <= self.grad_thre 
        while steps_taken < self.max_nsteps and not converged:
            steps_taken+=1
            print(f'StepsTaken:{steps_taken}||grad_mag:{grad_mag}                 \r', end="")
            direction = self.prfo_step(start_node)

            new_point = start_node.coords + direction*self.dr
            new_node = start_node.copy()
            new_node = new_node.update_coords(new_point)
                        
            grad_mag = np.linalg.norm(new_node.gradient)
            converged = grad_mag <= self.grad_thre 

            if converged: 
                print(f"Converged in {steps_taken} steps!")
                new_node.converged = True
            
            traj.append(new_node)
            start_node = new_node
            
        if converged: return new_node, traj

        if not converged: 
            print(f"Did not converge in {steps_taken} steps.")
            return new_node, traj

    def plot_path(self):
        traj = self.traj
        node = traj[0]
        
        s=4
        fs=18

        en_func = node.en_func_arr

        min_val = -s
        max_val = s

        fig = 10
        f, _ = plt.subplots(figsize=(1.18 * fig, fig))
        x = np.linspace(start=min_val, stop=max_val, num=10)
        y = x.reshape(-1, 1)


        h = en_func([x, y])
        cs = plt.contourf(x, x, h, levels=10)
        # _ = f.colorbar(cs)


        plt.plot([x.coords[0] for x in traj], [x.coords[1] for x in traj], "*--", c="white", label="path", ms=15)
        # for inp, (evec0, evec1) in zip(traj, eigvecs):
        #     plt.arrow(inp[0], inp[1], evec0[0], evec0[1], color='red', width=.05)
        #     plt.arrow(inp[0], inp[1], evec1[0], evec1[1], color='green', width=.05)

        plt.yticks(fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.show()


