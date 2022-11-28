from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neb_dynamics.Node import Node


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
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    @staticmethod
    def en_func_arr(xy_vals):
        x, y = xy_vals
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    @staticmethod
    def grad_func(node: Node2D):
        x, y = node.coords
        dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
        return np.array([dx, dy])

    @staticmethod
    def hess_func(node: Node2D):
        x, y = node.coords
        dx2 = 12 * x**2 + 4 * y - 42
        dy2 = 4 * x + 12 * y**2 - 26

        dxdy = 4 * (x + y)

        return np.array([[dx2, dxdy], [dxdy, dy2]])

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
        C = np.pi * np.exp(-np.pi * x**2)
        D = (np.exp(-np.pi * (y - 0.8) ** 2)) + np.exp(-np.pi * (y + 0.8) ** 2)
        return A + B + C * D

    @staticmethod
    def en_func_arr(xy_vals):
        x, y = xy_vals
        A = np.cos(np.pi * x)
        B = np.cos(np.pi * y)
        C = np.pi * np.exp(-np.pi * x**2)
        D = (np.exp(-np.pi * (y - 0.8) ** 2)) + np.exp(-np.pi * (y + 0.8) ** 2)
        return A + B + C * D

    @staticmethod
    def grad_func(node):
        x, y = node.coords
        A_x = (-2 * np.pi**2) * (np.exp(-np.pi * (x**2)) * x)
        B_x = np.exp(-np.pi * ((y - 0.8) ** 2)) + np.exp(-np.pi * (y + 0.8) ** 2)
        C_x = -np.pi * np.sin(np.pi * x)

        dx = A_x * B_x + C_x

        A_y = np.pi * np.exp(-np.pi * x**2)
        B_y = (-2 * np.pi * np.exp(-np.pi * (y - 0.8) ** 2)) * (y - 0.8)
        C_y = -2 * np.pi * np.exp(-np.pi * (y + 0.8) ** 2) * (y + 0.8)
        D_y = -np.pi * np.sin(np.pi * y)

        dy = A_y * (B_y + C_y) + D_y

        return np.array([dx, dy])

    @staticmethod
    def hess_func(node: Node2D):
        x, y = node.coords

        A_xx = (4 * np.pi**3) * np.exp(-np.pi * x**2) * x**2
        B_xx = np.exp(-np.pi * (y - 0.8) ** 2) + np.exp(-np.pi * (y + 0.8) ** 2)
        C_xx = (2 * np.pi**2) * (np.exp(-np.pi * x**2)) * (np.exp(-np.pi * (y - 0.8) ** 2) + np.exp(-np.pi * (y + 0.8) ** 2))
        D_xx = np.pi**2 * np.cos(np.pi * x)

        A_yy = np.pi * np.exp(-np.pi * x * 2)
        B_yy = 4 * np.pi**2 * np.exp(-np.pi * (y - 0.8) ** 2) * (y - 0.8) ** 2
        C_yy = 4 * np.pi**2 * np.exp(-np.pi * (y + 0.8) ** 2) * (y + 0.8) ** 2
        D_yy = 2 * np.pi * np.exp(-np.pi * (y - 0.8) ** 2)
        E_yy = 2 * np.pi * np.exp(-np.pi * (y + 0.8) ** 2)
        F_yy = np.pi**2 * np.cos(np.pi * y)

        dx2 = A_xx * B_xx - C_xx - D_xx
        dy2 = A_yy * (B_yy + C_yy - D_yy - E_yy) - F_yy

        dxdy = 2.22386 * x * (np.exp(np.pi * (y + 0.8) ** 2) * (y - 0.8) + (np.exp(np.pi * (y - 0.8) ** 2)) * (y + 0.8)) * np.exp(-3.14159 * x**2 - 6.28319 * y**2)

        return np.array([[dx2, dxdy], [dxdy, dy2]])

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
        return -1 * (Ax * np.cos(2 * np.pi * x) + Ay * np.cos(2 * np.pi * y))

    @staticmethod
    def en_func_arr(xy_vals):
        x, y = xy_vals
        Ax = 1
        Ay = 1
        return -1 * (Ax * np.cos(2 * np.pi * x) + Ay * np.cos(2 * np.pi * y))

    @staticmethod
    def grad_func(node: Node2D_ITM):
        x, y = node.coords
        Ax = 1
        Ay = 1
        dx = 2 * Ax * np.pi * np.sin(2 * np.pi * x)
        dy = 2 * Ay * np.pi * np.sin(2 * np.pi * y)
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
        a = 0.05
        b = 0.30
        c = 0.05
        d_ab = 4.746
        d_bc = 4.746
        d_ac = 3.445
        r0 = 0.742
        alpha = 1.942
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
        a = 0.05
        b = 0.30
        c = 0.05
        d_ab = 4.746
        # d_bc = 4.746
        d_ac = 3.445
        r0 = 0.742
        alpha = 1.942

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

        # Qconst = Node2D_LEPS.coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
        Jconst = Node2D_LEPS.exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        cDenom = 1 / (1 + c)

        d = d_ab

        dx = 0.25 * aDenom**2 * alpha * d * ealpha_x * (-2 * (1 + a) * (-1 + 3 * ealpha_x) + ((-3 + ealpha_x) * (2 * d * ealpha_x * (-6 + ealpha_x) - (1 + a) * d * ealpha_y * (-6 + ealpha_y) * bDenom - 4 * (1 + a) * Jconst * cDenom)) / (np.sqrt((((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2) + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2 - d**2 * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc)) * (-1 + 6 * neg_ealpha_x) * (-1 + 6 * neg_ealpha_y) * aDenom * bDenom) - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom) + 16 * Jconst**2 * cDenom**2)))

        return dx

    def grad_y(node: Node2D_LEPS):
        a = 0.05
        b = 0.30
        c = 0.05
        # d_ab = 4.746
        d_bc = 4.746
        d_ac = 3.445
        r0 = 0.742
        alpha = 1.942
        r_ab, r_bc = node.pair_of_coordinates

        ealpha_x = np.exp(alpha * (r0 - r_ab))
        neg_ealpha_x = np.exp(alpha * (r_ab - r0))
        ealpha_y = np.exp(alpha * (r0 - r_bc))
        neg_ealpha_y = np.exp(alpha * (r_bc - r0))

        e2alpha_x = np.exp(2 * alpha * (r0 - r_ab))
        e2alpha_y = np.exp(2 * alpha * (r0 - r_bc))

        aDenom = 1 / (1 + a)
        bDenom = 1 / (1 + b)

        # Qconst = Node2D_LEPS.coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
        Jconst = Node2D_LEPS.exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

        cDenom = 1 / (1 + c)

        d = d_bc

        dy = 0.25 * bDenom**2 * alpha * d * ealpha_y * (-2 * (1 + b) * (-1 + 3 * ealpha_y) + ((-3 + ealpha_y) * (2 * d * ealpha_y * (-6 + ealpha_y) - (1 + b) * d * ealpha_x * (-6 + ealpha_x) * aDenom - 4 * (1 + b) * Jconst * cDenom)) / (np.sqrt((((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2) + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2 - d**2 * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc)) * (-1 + 6 * neg_ealpha_x) * (-1 + 6 * neg_ealpha_y) * aDenom * bDenom) - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom) + 16 * Jconst**2 * cDenom**2)))

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