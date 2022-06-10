import numpy as np

from neb_dynamics.NEB import Node2D


def ArmijoLineSearch(node: Node2D, grad: np.array, alpha0=1, rho=0.8, c1=0.0001):
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
    derphi0 = np.dot(grad, pk)

    # new_coords = xk + alpha0 * pk
    new_coords = node.coords + alpha0*pk

    new_node = Node2D(new_coords)

    phi_a0 = new_node.energy

    while not phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        # print(f"{phi0=}")
        alpha0 *= rho
        new_coords = node.coords + alpha0 * pk
        new_node = Node2D(new_coords)

        phi_a0 = new_node.energy

    return alpha0

# def _attempt_step(node: Node2D, grad, t, f):
#     # print("t_init: ", t)

#     # try:
#     new_coords = node.coords - t * grad

#     node_prime = Node2D(new_coords)

#     en_node_prime = f(node_prime)

#     # except:
#         # t *= 0.1
#         # new_coords_bohr = struct.coords_bohr - t * grad
#         # new_coords = new_coords_bohr * BOHR_TO_ANGSTROMS

#         # struct_prime = TDStructure.from_coords_symbs(coords=new_coords, symbs=struct.symbols, tot_charge=struct.charge, tot_spinmult=struct.spinmult)

#         # en_struct_prime, t = _attempt_step(struct_prime, grad, t, f)
#     # print("t_final:",t)
#     return en_node_prime, t


# def ArmijoLineSearch(node: Node2D, grad, t, alpha, beta, f):

    # max_steps = 10
    # count = 0
    
    # en_node_prime, t = _attempt_step(node=node, grad=grad, t=t, f=f)

    # en_node = f(node)
    # condition = en_node - (en_node_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

    # while condition and count < max_steps:
    #     t *= beta
    #     count += 1

    #     en_node_prime, t = _attempt_step(node=node, grad=grad, t=t, f=f)

    #     en_node = f(node)

    #     condition = en_node - (en_node_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

    #     # print(f"{t=} {count=}")
    # return t