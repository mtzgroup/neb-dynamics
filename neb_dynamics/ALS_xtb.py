from __future__ import annotations

import numpy as np

from neb_dynamics.NEB import Node3D
from neb_dynamics.tdstructure import TDStructure


ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROMS = 1 / ANGSTROM_TO_BOHR


def ArmijoLineSearch(node: Node3D, grad, alpha0=0.01, rho=0.5, c1=1e-4):
    """Minimize over alpha, the function ``f(xₖ + αpₖ)``.
    α > 0 is assumed to be a descent direction.

    Parameters
    --------------------
    f : callable
        Function to be minimized.
    xk : array
        Current point.
    gfk : array
        Gradient of `f` at point `xk`.
    phi0 : float
        Value of `f` at point `xk`.
    alpha0 : scalar
        Value of `alpha` at the start of the optimization.
    rho : float, optional
        Value of alpha shrinkage factor.
    c1 : float, optional
        Value to control stopping criterion.

    Returns
    --------------------
    alpha : scalar
        Value of `alpha` at the end of the optimization.
    phi : float
        Value of `f` at the new point `x_{k+1}`.
    """
    phi0 = node.energy

    # derphi0 = np.dot(gfk, pk)
    pk = -1 * grad
    derphi0 = np.sum(node.dot_function(grad, pk))

    # new_coords = xk + alpha0 * pk
    new_coords = node.coords + alpha0*derphi0
    new_tdstruct = node.tdstructure.copy()
    new_tdstruct.update_coords(new_coords)


    new_node = Node3D(new_tdstruct)

    phi_a0 = new_node.energy

    while not phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        # print(f"{phi0=}")
        alpha0 = alpha0 * rho
        new_coords = node.coords + alpha0 * pk
        new_tdstruct = node.tdstructure.copy()
        new_tdstruct.update_coords(new_coords)
        new_node = Node3D(new_tdstruct)

        phi_a0 = new_node.energy

    return alpha0



# def _attempt_step(node: Node3D, grad, t, f):
#     # print("t_init: ", t)

#     # try:
#     struct = node.tdstructure
#     new_coords_bohr = struct.coords_bohr - t * grad
#     new_coords = new_coords_bohr * BOHR_TO_ANGSTROMS


#     struct_prime = TDStructure.from_coords_symbs(coords=new_coords, symbs=struct.symbols, tot_charge=struct.charge, tot_spinmult=struct.spinmult)
#     node_prime = Node3D(struct_prime)

#     en_struct_prime = f(node_prime)

#     # except:
#         # t *= 0.1
#         # new_coords_bohr = struct.coords_bohr - t * grad
#         # new_coords = new_coords_bohr * BOHR_TO_ANGSTROMS

#         # struct_prime = TDStructure.from_coords_symbs(coords=new_coords, symbs=struct.symbols, tot_charge=struct.charge, tot_spinmult=struct.spinmult)

#         # en_struct_prime, t = _attempt_step(struct_prime, grad, t, f)
#     # print("t_final:",t)
#     return en_struct_prime, t


# def ArmijoLineSearch(node: Node3D, grad, t, alpha, beta, f):

#     max_steps = 10
#     count = 0
    
#     en_struct_prime, t = _attempt_step(node=node, grad=grad, t=t, f=f)

#     en_struct = f(node)
#     condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

#     while condition and count < max_steps:
#         t *= beta
#         count += 1

#         en_struct_prime, t = _attempt_step(node=node, grad=grad, t=t, f=f)

#         en_struct = f(node)

#         condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

#         # print(f"{t=} {count=}")
#     return t
