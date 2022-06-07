import multiprocessing as mp

import numpy as np
from foobar import double, howmany_within_range
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory

from ALS_xtb import ArmijoLineSearch
from NEB_xtb import neb

traj = Trajectory.from_xyz("./example_cases/PDDA_geodesic.xyz")


def update_chain(chain, k, en_func, grad_func, ideal_dist, nodes_converged):

    chain_copy = np.zeros_like(chain)
    chain_copy[0] = chain[0]
    chain_copy[-1] = chain[-1]

    for i in range(1, len(chain) - 1):
        if nodes_converged[i] == True:
            chain_copy[i] = chain[i]
            continue

        print(f"updating node {i}...")
        view = chain[i - 1 : i + 2]
        # print(f"{view=}")
        grad = self.spring_grad_neb(view, k=k, ideal_distance=ideal_dist, grad_func=grad_func, en_func=en_func)
        # dr = 0.01

        dr = ArmijoLineSearch(struct=chain[i], grad=grad, t=1, alpha=0.3, beta=0.8, f=en_func)

        coords_new = chain[i].coords - grad * dr

        p_new = TDStructure.from_coords_symbs(coords=coords_new, symbs=chain[i].symbols, tot_charge=chain[i].charge, tot_spinmult=chain[i].spinmult)

        chain_copy[i] = p_new

    return chain_copy


# +
n = neb()


chain = traj
grad_func = n.grad_func
en_func = n.en_func
k = 10
en_thre = 0.0005
grad_thre = 0.0005
max_steps = 2


chain_traj = []
nsteps = 0
ideal_dist = np.linalg.norm(np.array(chain[-1].coords) - np.array(chain[0].coords)) / len(chain)
chain_previous = chain.copy()
nodes_converged = np.zeros(len(chain))

# while nsteps < max_steps:
# print(f"--->On step {nsteps}")

# -


def get_new_node(view, k, ideal_distance, grad_func, en_func):
    neighs = view[[0, 2]]

    en_2 = en_func(view[2])
    en_1 = en_func(view[1])
    en_0 = en_func(view[0])

    if en_2 > en_1 and en_1 > en_0:
        return view[2].coords - view[1].coords
    elif en_2 < en_1 and en_1 < en_2:
        return view[1].coords - view[0].coords

    else:
        deltaV_max = max(np.abs(en_2 - en_1), np.abs(en_0 - en_1))
        deltaV_min = min(np.abs(en_2 - en_1), np.abs(en_0 - en_1))

        if en_2 > en_0:
            tan_vec = (view[2].coords - view[1].coords) * deltaV_max + (view[1].coords - view[0].coords) * deltaV_min
        elif en_2 < en_0:
            tan_vec = (view[2].coords - view[1].coords) * deltaV_min + (view[1].coords - view[0].coords) * deltaV_max
        else:
            print("Chain must have blown up in covergence. Check step size.")
    vec_tan_path = tan_vec
    unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

    pe_grad = grad_func(view[1])

    pe_grad_nudged_const = np.sum(pe_grad * unit_tan_path, axis=1).reshape(-1, 1)  # Nx1 matrix
    pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path

    grads_neighs = []
    force_springs = []

    for neigh in neighs:
        dist = np.abs(neigh.coords - view[1].coords)
        force_spring = -k * (dist - ideal_distance)  # magnitude tells me if it's attractive or repulsive

        # check if force vector is pointing towards neighbor
        direction = np.sum((neigh.coords - view[1].coords) * force_spring, axis=1)

        # flip vectors that weren't pointing towards neighbor
        force_spring[direction < 0] *= -1

        force_springs.append(force_spring)

        force_spring_nudged_const = np.sum(force_spring * unit_tan_path, axis=1).reshape(-1, 1)
        force_spring_nudged = force_spring - force_spring_nudged_const * unit_tan_path

        grads_neighs.append(force_spring_nudged)

    tot_grads_neighs = np.sum(grads_neighs, axis=0)

    ### ANTI-KINK FORCE
    # print(f"{force_springs=}")
    force_springs = np.sum(force_springs, axis=0)
    # print(f"{force_springs=}")

    vec_2_to_1 = view[2].coords - view[1].coords
    vec_1_to_0 = view[1].coords - view[0].coords
    cos_phi = np.sum(vec_2_to_1 * vec_1_to_0, axis=1).reshape(-1, 1) / (np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0))

    f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))

    proj_force_springs = force_springs - np.sum(force_springs * unit_tan_path, axis=1).reshape(-1, 1) * unit_tan_path

    # print(f"{pe_grad_nudged=} {tot_grads_neighs=} {f_phi=} {proj_force_springs=}")

    grad = (pe_grad_nudged - tot_grads_neighs) + f_phi * (proj_force_springs)

    dr = 0.01

    # dr  = ArmijoLineSearch(
    #     struct=chain[i], grad=grad, t=1, alpha=0.3, beta=0.8, f=en_func
    # )

    coords_new = chain[i].coords - grad * dr

    p_new = TDStructure.from_coords_symbs(coords=coords_new, symbs=chain[i].symbols, tot_charge=chain[i].charge, tot_spinmult=chain[i].spinmult)

    return p_new


from time import time

# +
import numpy as np

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()
data[:5]


# -


def driver_func():
    PROCESSES = mp.cpu_count()
    print(f"{PROCESSES=}")
    with mp.Pool(PROCESSES) as pool:
        params = [(1,), (2,), (3,), (4,)]
        results = [pool.apply(double, p) for p in params]

        for r in results:
            print("\t", r)


driver_func()

# +
chain_copy = np.zeros_like(chain)
chain_copy[0] = chain[0]
chain_copy[-1] = chain[-1]

PROCESSES = mp.cpu_count()
print(f"{PROCESSES=}")
with mp.Pool(PROCESSES) as pool:

    results = [pool.apply(get_new_node, kwds={"view": chain[i - 1 : i + 2], "k": k, "ideal_dist": ideal_dist, "grad_func": grad_func, "en_func": en_func}) for i in range(1, len(chain) - 1)]

for r in results:
    print("\t", r)


# for i in range(1, len(chain) - 1):
#     if nodes_converged[i] == True:
#         chain_copy[i] = chain[i]
#         continue
#     print(f"updating node {i}...")
#     view = chain[i - 1 : i + 2]
#     results.append(
#         pool.apply(get_new_node, args=(view))
#     )
#     # print(f"{view=}")


# new_chain = chain_copy


# chain_traj.append(new_chain)
# nodes_converged = self._chain_converged(
#     chain_previous=chain_previous,
#     new_chain=new_chain,
#     en_func=en_func,
#     en_thre=en_thre,
#     grad_func=grad_func,
#     grad_thre=grad_thre,
# )
# if False not in nodes_converged:
#     print("Chain converged!")
#     return new_chain, chain_traj

# chain_previous = new_chain.copy()
# nsteps += 1

# print("Chain did not converge...")
# -
