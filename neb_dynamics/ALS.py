import numpy as np
import sys
from neb_dynamics.NEB import Chain

# def _attempt_step(node: Node, grad, grad_func, t, prev_node, next_node, k):
    
#     # print(f'{grad=} // {t=}')
    
#     new_coords = node.coords - t * grad
#     new_node = node.update_coords(new_coords)

#     # print("\t\tALS: calling grad_func...")
#     pe_grad_nudged, spring_force_nudged = grad_func(prev_node=prev_node, current_node=new_node, next_node=next_node)
#     new_grad = (pe_grad_nudged - spring_force_nudged) #+ k*anti_kinking_grad


    
#     en_struct_prime = np.linalg.norm(new_grad)
#     # en_struct_prime = new_node.energy
#     # raise AlessioError("FOO")
#     return en_struct_prime, t

# def ArmijoLineSearch(node: Node, prev_node: Node, next_node: Node, grad_func, t, alpha, beta, f, k, grad):

#     max_steps = 10
#     count = 0

#     en_struct_prime, t =  _attempt_step(node=node, grad=grad, grad_func=grad_func, t=t, prev_node=prev_node, next_node=next_node, k=k)
#     # en_struct = node.energy 
#     en_struct = np.linalg.norm(grad)


#     condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

#     while condition and count < max_steps:
#         t *= beta
#         count += 1

#         en_struct_prime, t = _attempt_step(node=node, grad=grad, t=t, grad_func=grad_func, prev_node=prev_node, next_node=next_node, k=k)
#         condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

#     print(f"\t\t\t{t=} {count=} || force0: {en_struct} || force1: {en_struct_prime}")
#     sys.stdout.flush()
#     return t

# # def backtrack(node: Node, prev_node: Node, next_node: Node, grad_func, t, alpha, beta, f, k, grad):


# #     while fun(x - t*grad(x)) > f(x) - (t/2)np.linalg.norm(grad(x)**2):
# #         t = beta*t
# #     return t
def _attempt_step(chain: Chain, t):
    
    # print(f'{grad=} // {t=}')
    
    new_chain_coordinates = (chain.coordinates - chain.gradients * t)
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
    
    new_grad = new_chain.gradients
    en_struct_prime = np.sqrt(np.mean(np.square(new_grad)))
    # en_struct_prime = np.sum(new_chain.energies)
    return en_struct_prime, t

def ArmijoLineSearch(chain: Chain, t, alpha, beta,grad):
    max_steps = 10
    count = 0

    en_struct_prime, t =  _attempt_step(chain=chain, t=t)
    en_struct = np.sqrt(np.mean(np.square(grad)))
    # en_struct = np.sum(chain.energies)
    condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

    while condition and count < max_steps:
        t *= beta
        count += 1

        en_struct_prime, t = _attempt_step(chain=chain, t=t)
        condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

    print(f"\t\t\t{t=} {count=} || force0: {en_struct} || force1: {en_struct_prime}")
    sys.stdout.flush()
    return t

