import numpy as np
import sys
from neb_dynamics.NEB import AlessioError, Node

def _attempt_step(node: Node, grad, grad_func, t, f, prev_node, next_node, k):
    
    # print(f'{grad=} // {t=}')
    
    new_coords = node.coords - t * grad
    new_node = node.update_coords(new_coords)


    pe_grad_nudged, spring_force_nudged_no_k = grad_func(prev_node=prev_node, current_node=new_node, next_node=next_node)
    new_grad = (pe_grad_nudged - k*spring_force_nudged_no_k) #+ k*anti_kinking_grad

    en_struct_prime = np.linalg.norm(new_grad)
    # en_struct_prime = new_node.energy
    
    return en_struct_prime, t

def ArmijoLineSearch(node: Node, prev_node: Node, next_node: Node, grad_func, t, alpha, beta, f, k, grad):

    max_steps = 10
    count = 0


    en_struct_prime, t = _attempt_step(node=node, grad=grad, grad_func=grad_func, t=t, f=f, prev_node=prev_node, next_node=next_node, k=k)

    # en_struct = node.energy
    en_struct = np.linalg.norm(grad)


    condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

    while condition and count < max_steps:
        t *= beta
        count += 1

        en_struct_prime, t = _attempt_step(node=node, grad=grad, t=t, f=f, grad_func=grad_func, prev_node=prev_node, next_node=next_node, k=k)

        pe_grad_nudged, spring_force_nudged_no_k = grad_func(prev_node=prev_node, current_node=node, next_node=next_node)
        grad = (pe_grad_nudged - k*spring_force_nudged_no_k) #+ k*anti_kinking_grad


        en_struct = np.linalg.norm(grad)
        # en_struct = node.energy

        condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

    print(f"\t\t\t{t=} {count=} || force: {np.linalg.norm(grad)}")
    sys.stdout.flush()
    return t