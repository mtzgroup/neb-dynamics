import numpy as np
from tdstructure import TDStructure

ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROMS = 1 / ANGSTROM_TO_BOHR


def _attempt_step(struct, grad, t, f):
    # print("t_init: ", t)

    try:
        new_coords_bohr = struct.coords_bohr - t * grad
        new_coords = new_coords_bohr * BOHR_TO_ANGSTROMS

        struct_prime = TDStructure.from_coords_symbs(coords=new_coords, symbs=struct.symbols, tot_charge=struct.charge, tot_spinmult=struct.spinmult)

        en_struct_prime = f(struct_prime)

    except:
        t *= 0.1
        new_coords_bohr = struct.coords_bohr - t * grad
        new_coords = new_coords_bohr * BOHR_TO_ANGSTROMS

        struct_prime = TDStructure.from_coords_symbs(coords=new_coords, symbs=struct.symbols, tot_charge=struct.charge, tot_spinmult=struct.spinmult)

        en_struct_prime, t = _attempt_step(struct_prime, grad, t, f)
    print("t_final:",t)
    return en_struct_prime, t


def ArmijoLineSearch(struct, grad, t, alpha, beta, f):

    max_steps = 10
    count = 0

    en_struct_prime, t = _attempt_step(struct=struct, grad=grad, t=t, f=f)

    en_struct = f(struct)
    condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

    while condition and count < max_steps:
        t *= beta
        count += 1

        en_struct_prime, t = _attempt_step(struct=struct, grad=grad, t=t, f=f)

        en_struct = f(struct)

        condition = en_struct - (en_struct_prime + alpha * t * (np.linalg.norm(grad) ** 2)) < 0

        # print(f"{t=} {count=}")
    return t
