from __future__ import annotations

import numpy as np

from neb_dynamics.NEB import Node3D
from neb_dynamics.tdstructure import TDStructure


ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROMS = 1 / ANGSTROM_TO_BOHR

def _attempt_step(node: Node3D, grad, t, f):
    # print("t_init: ", t)

    # try:
    struct = node.tdstructure
    new_coords_bohr = struct.coords_bohr - t * grad
    new_coords = new_coords_bohr * BOHR_TO_ANGSTROMS


    struct_prime = TDStructure.from_coords_symbs(coords=new_coords, symbs=struct.symbols, tot_charge=struct.charge, tot_spinmult=struct.spinmult)
    node_prime = Node3D(struct_prime)

    en_struct_prime = f(node_prime)

    # except:
        # t *= 0.1
        # new_coords_bohr = struct.coords_bohr - t * grad
        # new_coords = new_coords_bohr * BOHR_TO_ANGSTROMS

        # struct_prime = TDStructure.from_coords_symbs(coords=new_coords, symbs=struct.symbols, tot_charge=struct.charge, tot_spinmult=struct.spinmult)

        # en_struct_prime, t = _attempt_step(struct_prime, grad, t, f)
    # print("t_final:",t)
    return en_struct_prime, t


def ArmijoLineSearch(node: Node3D, grad, t, alpha, beta, f):

    max_steps = 10
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

        # print(f"{t=} {count=}")
    return t
