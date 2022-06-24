import numpy as np

from neb_dynamics.NEB import Node

def _attempt_step(node: Node, grad, t, f):
    
    new_coords = node.coords - t * grad
    new_node = node.update_coords(new_coords)

    en_struct_prime = f(new_node)
    
    return en_struct_prime, t

def ArmijoLineSearch(node: Node, grad, t, alpha, beta, f):

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