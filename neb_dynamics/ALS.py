import numpy as np

from neb_dynamics.NEB import Node, Node3D
ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROMS = 1 / ANGSTROM_TO_BOHR

# def ArmijoLineSearch(node: Node, grad: np.array, alpha0=1, rho=0.5, c1=1e-4):
#     """
#     alpha0: step size
#     rho: factor by which to decrease step size
#     c1: convergence criterion

#     """
#     count_max = 10
#     phi0 = node.energy

#     if np.all(grad == 0):
#         return 0

#     else:
#         pk = -1 * grad / np.linalg.norm(grad)
#         # derphi0 = np.sum(node.dot_function(grad, pk), axis=0) / np.linalg.norm(grad) # function change along direction
#         derphi0 = np.linalg.norm(grad)**2

#     new_coords = node.coords + alpha0 * pk
#     new_node = node.copy()
#     new_node.update_coords(new_coords)
#     # print(f"\told_coords: {node.coords} / stepgrad: {-alpha0*grad} / new_coords: {new_node.coords}")
#     phi_a0 = new_node.energy

#     count = 0
    
#     while not phi_a0 <= phi0 + c1 * alpha0 * derphi0 and count < count_max:
#         # print(f"{phi_a0} <= {phi0} + {c1 * alpha0 * derphi0}")
#         alpha0 *= rho
        
#         new_coords = node.coords + alpha0 * pk
#         new_node = node.copy()
#         new_node.update_coords(new_coords)

#         phi_a0 = new_node.energy
#         count += 1

#     # print(f"\t{alpha0=} // {count=} //{phi_a0} <= {phi0 + c1 * alpha0 * derphi0}")
#     return alpha0



def _attempt_step(node: Node3D, grad, t, f):
    # print("t_init: ", t)

    struct = node.tdstructure

    
    new_coords = struct.coords - t * grad

    new_node = node.update_coords(new_coords)

    en_struct_prime = f(new_node)

    failed_so_far = False
    
    return en_struct_prime, t

def ArmijoLineSearch(node: Node3D, grad, t, alpha, beta, f):

    max_steps = 30
    count = 0

    en_struct_prime, t = _attempt_step(node=node, grad=grad, t=t, f=f)

    en_struct = f(node)
    condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

    while condition and count < max_steps:
        t *= beta
        count += 1

        en_struct_prime, t = _attempt_step(node=node, grad=grad, t=t, f=f)

        en_struct = f(node)

        condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

    print(f"\t\t\t{t=} {count=}")
    return t