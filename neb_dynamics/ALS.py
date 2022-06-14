import numpy as np

from neb_dynamics.NEB import Node


def ArmijoLineSearch(node: Node, grad: np.array, alpha0=1, rho=0.5, c1=1e-4):
    """
    alpha0: step size
    rho: factor by which to decrease step size
    c1: convergence criterion

    """
    count_max = 20
    phi0 = node.energy

    if np.all(grad == 0):
        return 0

    else:
        pk = -1 * grad / np.linalg.norm(grad)
        derphi0 = np.sum(node.dot_function(grad, pk), axis=0) / np.linalg.norm(grad) # function change along direction

    new_coords = node.coords + alpha0 * pk
    new_node = node.copy()
    new_node.update_coords(new_coords)
    # print(f"\told_coords: {node.coords} / stepgrad: {-alpha0*grad} / new_coords: {new_node.coords}")
    phi_a0 = new_node.energy

    count = 0
    
    while not phi_a0 <= phi0 + c1 * alpha0 * derphi0 and count < count_max:
        # print(f"{phi_a0} <= {phi0} + {c1 * alpha0 * derphi0}")
        alpha0 *= rho
        
        new_coords = node.coords + alpha0 * pk
        new_node = node.copy()
        new_node.update_coords(new_coords)

        phi_a0 = new_node.energy
        count += 1

    # print(f"\t{alpha0=} // {count=} //{phi_a0} <= {phi0 + c1 * alpha0 * derphi0}")
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
