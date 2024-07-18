import numpy as np
import sys
from neb_dynamics.neb import Chain


def _attempt_step(chain: Chain, t):

    new_chain_coordinates = chain.coordinates - chain.gradients * t
    new_nodes = []
    for node, new_coords in zip(chain.nodes, new_chain_coordinates):

        new_nodes.append(node.update_coords(new_coords))

    new_chain = Chain(new_nodes, parameters=chain.parameters)
    new_grad = new_chain.gradients
    # en_struct_prime = np.linalg.norm(new_grad)
    en_struct_prime = new_chain.get_maximum_grad_magnitude()
    assert all([en is not None for en in new_chain.energies])

    return en_struct_prime, t


def ArmijoLineSearch(chain: Chain, t, alpha, beta, grad, max_steps):
    count = 0
    # en_struct = np.linalg.norm(grad)
    orig_grad = chain.gradients
    en_struct = chain.get_maximum_grad_magnitude()
    t *= 1 / beta
    condition = True
    while condition and count < max_steps:
        try:
            t *= beta
            count += 1

            en_struct_prime, t = _attempt_step(chain=chain, t=t)
            condition = (
                en_struct - (en_struct_prime + alpha * t * (np.dot(grad, orig_grad)))
                < 0
            )
            # condition = en_struct_prime <= en_struct + alpha * t * (np.linalg.norm(grad) ** 2)

        except:
            t *= 0.5 * beta
    sys.stdout.flush()
    print(f"\ndid {count} ALS steps\n")
    return t
